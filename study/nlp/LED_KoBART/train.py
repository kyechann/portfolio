import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import pandas as pd
import numpy as np
import argparse
import json
import gc
import re
import copy
import random
import datetime
nowdate = datetime.datetime.now()
from tqdm import tqdm
from IPython import get_ipython

from dataclasses import dataclass, field
from datasets import load_dataset
from datasets import load_metric
from typing import Dict, List, Optional, Tuple

from Library.bart_to_long import *
from Library.data_preprocess import *

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast, LEDForConditionalGeneration, AutoModel
from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers import get_linear_schedule_with_warmup, AdamW, TrainingArguments

from torchinfo import summary
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import *
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import random_split
from pytorch_lightning import loggers as pl_loggers
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import pytorch_lightning.metrics.functional as FM
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, MLFlowLogger

import mlflow
from pyngrok import ngrok
from mlflow.tracking import MlflowClient


device = torch.device('gpu : 0' if torch.cuda.is_available() else 'cpu')

batch_size = 5
epochs = 2
attention_window = 512
max_pos = 4104
max_seq_len = 4096

class LongformerKobart(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_path = '../LED_KoBART/model'
        self.model = LongformerBartForConditionalGeneration.from_pretrained(self.save_path)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.save_path)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = 3
        self.ignore_token_id= -100
        self.lr = 1e4

    def forward(self, inputs) : 
        input_ids = inputs['input_ids'].to( device)
        attention_mask = inputs['attention_mask'].to( device)
        decoder_input_ids = inputs['decoder_input_ids'].to( device)
        decoder_attention_mask = inputs['decoder_attention_mask'].to( device)
        labels = inputs['labels'].to( device)

        with torch.cuda.amp.autocast():
            output = self.model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels, use_cache=False, return_dict=True)

        return output

    def training_step(self, batch, batch_idx) :
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx) : 
        outs = self(batch)
        loss = outs.loss
        self.log("val_loss",loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        return self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
    
    def test_step(self, batch, batch_idx) :
        _, _, _, label_str = batch
        outs = self(batch)
        loss = outs.loss
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def test_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        return self.log('test_loss', torch.stack(losses).mean(), prog_bar=True)

    def configure_optimizers(self):
        # Prepare optimizer
        optimizer = AdamW(self.model.parameters(), lr=1e-4, correct_bias=True)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        lr_scheduler = {"scheduler" : lr_scheduler,
        "monitor" : "loss",
        "interval" : "step",
        "frequency" : 1}

        return [optimizer], [lr_scheduler]
    
def load_data():
    tar_train = pd.read_csv('../LED_KoBART/data/tar_train.csv', encoding='utf-8', index_col = False)
    train_data, val_data = train_test_split(tar_train, test_size=0.3, shuffle=True, stratify=tar_train['Subject'], random_state=42)
    return train_data, val_data

if __name__ == "__main__":
    filename = 'KoBART_LED' + '_' + nowdate.strftime('%Y-%m-%d')
    wandb_logger = WandbLogger(project = filename,name=f'Longformer') 
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join('../LED_KoBART/Logger', 'tb_logs'))
    client = MlflowClient()
    exp_id = client.create_experiment(filename)
    client.set_experiment_tag(exp_id, "nlp flow", "Longformer KoBART Model")
    exp_name = client.get_experiment(exp_id)
    print("Name: {}".format(exp_name.name))
    print("Experiment_id: {}".format(exp_name.experiment_id))
    print("Artifact Location: {}".format(exp_name.artifact_location))
    print("Tags: {}".format(exp_name.tags))
    print("Lifecycle_stage: {}".format(exp_name.lifecycle_stage))
    mlf_logger = MLFlowLogger(experiment_name=exp_name, tracking_uri="file:./mlruns")
    
    
    train_data, val_data = load_data()
    model = LongformerKobart()
    
    gc.collect()
    torch.cuda.empty_cache()
    dm = KobartLedDataModule(batch_size = 1, 
                            num_workers = 10,
                            tokenizer = tokenizer,
                            train_data = train_data,
                            val_data = val_data)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor = 'train_loss', dirpath = '../LED_KoBART/Logger/checkpoints',
                                                        filename='LEDKoBART/{epoch:02d}-{val_accuracy:.3f}',
                                                        verbose = True, save_last = True, 
                                                        mode = 'min', save_top_k = -1)
    torch.cuda.empty_cache()
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer(
        default_root_dir='../LED_KoBART/Logger/checkpoints',
        logger = [wandb_logger, tb_logger, mlf_logger],
        #gpus = -1,
        accumulate_grad_batches = 16,
        gradient_clip_val=0.0, 
        log_every_n_steps=20, # logging frequency in training step
        val_check_interval=0.1, # validation 몇번마다 돌릴것인지? > 0.1 epoch에 1번) 
        callbacks = checkpoint_callback,
        max_epochs=2,
    )
    # 학습 진행
    with mlflow.start_run() as run:

        # Terminate open tunnels if exis
        trainer.fit(model, dm)
        mlflow.pytorch.log_model(model,"models")
        print("Logged data and model in run {}".format(run.info.run_id))
        
mlflow.end_run()