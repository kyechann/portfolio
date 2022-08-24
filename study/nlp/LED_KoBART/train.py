import logging
import os
import math
import pandas as pd
import numpy as np
import json
import gc
import re
import copy
import random
from tqdm import tqdm

from dataclasses import dataclass, field
from datasets import load_dataset
from datasets import load_metric
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast

from Library.bart_to_long import LongformerEncoderDecoderForConditionalGeneration
from Library.data_preprocess import *

from torchinfo import summary
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 5
epochs = 2
attention_window = 512
max_pos = 4104
max_seq_len = 4096

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def accuracy_function(real, pred):
    accuracies = torch.eq(real, torch.argmax(pred, dim=2))
    mask = torch.logical_not(torch.eq(real, -100))
    accuracies = torch.logical_and(mask, accuracies)
    accuracies = accuracies.clone().detach()
    mask = mask.clone().detach()

    return torch.sum(accuracies)/torch.sum(mask)

def loss_function(real, pred):
    mask = torch.logical_not(torch.eq(real, -100))
    loss_ = criterion(pred.permute(0,2,1), real)
    mask = mask.clone().detach()
    loss_ = mask * loss_

    return torch.sum(loss_)/torch.sum(mask)




def train():
    saved_model = "../LED_KoBART/model"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(saved_model)
    model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(saved_model)
    model = model.to(device)

    train = pd.read_csv("../LED_KoBART/data/train_data.csv", encoding = 'utf-8', index_col= False)
    sample_set = train.groupby('Subject',group_keys=False).apply(sampling_func, sample_pct = 0.07)
    sample_set.reset_index(inplace=True)
    del sample_set['index']

    # train,test에 본문 토큰 길이와 러프한 내용 분류 추가
    sample_set['con_token_len'] = sample_set['Content'].apply(token_len)
    tar_train = sample_set[sample_set['con_token_len']<=4096]
    del tar_train['con_token_len']

    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_data, eval_data = train_test_split(tar_train, test_size=0.1, shuffle=True, stratify=tar_train['Subject'], random_state=42)
    train_dataset = SummaryDataset(train_data, max_seq_len, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    eval_dataset = SummaryDataset(eval_data, max_seq_len, tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=4, shuffle=True)


    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('LED_KoBART/board')
    min_loss = 100

    loss_plot, val_loss_plot = [], []
    acc_plot, val_acc_plot = [], []
    writer = SummaryWriter()

    def train_step(batch_item, epoch, batch, training):
        
        input_ids = batch_item['input_ids'].to(device)
        attention_mask = batch_item['attention_mask'].to(device)
        decoder_input_ids = batch_item['decoder_input_ids'].to(device)
        decoder_attention_mask = batch_item['decoder_attention_mask'].to(device)
        labels = batch_item['labels'].to(device)
        
        if training is True:
            model.train()
            model.model.encoder.config.gradient_checkpointing = True
            model.model.decoder.config.gradient_checkpointing = True
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            labels=labels, use_cache=False, return_dict=True)
                
                loss = output.loss
                # loss = loss_function(labels, output.logits)
            
            acc = accuracy_function(labels, output.logits)
            
            loss.backward()
            
            optimizer.step()
            
            lr = optimizer.param_groups[0]["lr"]
            return loss, acc, round(lr, 10)
        else:
            model.eval()
            with torch.no_grad():
                output = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            labels=labels, return_dict=True)
                loss = output.loss
                # loss = loss_function(labels, output.logits)
            
            acc = accuracy_function(labels, output.logits)
            
            return loss, acc
    
    for epoch in range(epochs):
        gc.collect()
        total_loss, total_val_loss = 0, 0
        total_acc, total_val_acc = 0, 0

        tqdm_dataset = tqdm(enumerate(train_dataloader))
        training = True
        for batch, batch_item in tqdm_dataset:
            train_step(batch_item, epoch, batch, training)
            batch_loss, batch_acc, lr = train_step(batch_item, epoch, batch, training)
            total_loss += batch_loss
            total_acc += batch_acc

            
            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'LR' : lr,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Total Loss' : '{:06f}'.format(total_loss/(batch+1)),
                'Total ACC' : '{:06f}'.format(total_acc/(batch+1))
            })  
        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Acc/train", total_acc, epoch)
        loss_plot.append(total_loss/(batch+1))
        acc_plot.append(total_acc/(batch+1))

        torch.save(model.state_dict(), 'epoch {} weight.ckpt'.format(epoch + 1))


        tqdm_dataset = tqdm(enumerate(eval_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss, batch_acc = train_step(batch_item, epoch, batch, training)
            total_val_loss += batch_loss
            total_val_acc += batch_acc

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                'Total Val ACC' : '{:06f}'.format(total_val_acc/(batch+1))
            })
        writer.add_scalar("Loss/val", total_val_loss, epoch)
        writer.add_scalar("Acc/val", total_val_acc, epoch)
        val_loss_plot.append(total_val_loss/(batch+1)) 
        val_acc_plot.append(total_val_acc/(batch+1))
        writer.close()
        
if __name__=='__main__':
    train()