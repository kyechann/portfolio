import logging
import os
import math
import pandas as pd
import numpy as np


import re
import copy
import random
from tqdm import tqdm


from dataclasses import dataclass, field
from datasets import load_dataset
from datasets import load_metric
from typing import Dict, List, Optional, Tuple
from transformers import TrainingArguments, Trainer
from transformers import pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BartConfig, LEDTokenizerFast
from transformers import PreTrainedTokenizerFast
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from Library.bart_to_long import LongformerEncoderDecoderForConditionalGeneration


from torchinfo import summary
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
tokenizer = LEDTokenizerFast.from_pretrained("../LED_KoBART/model")

# 데이터 샘플 추출
def sampling_func(data, sample_pct):
    np.random.seed(123)
    N = len(data)
    sample_n = int(len(data)*sample_pct) # integer    
    sample = data.take(np.random.permutation(N)[:sample_n])
    return sample

# context 토큰 길이 
def token_len(text):
    return len(tokenizer.tokenize(text))

# 데이터셋  Summary
class SummaryDataset(Dataset):
    def __init__(self, dataframe, max_seq_len, tokenizer) -> None:
        self.dataframe = dataframe
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len
        
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataframe.shape[0]

    def make_input_id_mask(self, tokens, index):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens) 
        attention_mask = [1] * len(input_id) 
        if len(input_id) < self.max_seq_len:   
            while len(input_id) < self.max_seq_len: 
                input_id += [self.tokenizer.pad_token_id] 
                attention_mask += [0]
        else:
            input_id = input_id[:self.max_seq_len - 1] + [   
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return input_id, attention_mask


    def __getitem__(self, index):
        target_row = self.dataframe.iloc[index]
        context, summary = target_row['Content'], target_row['Title']
        context_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(context) + [self.eos_token]
        summary_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(summary) + [self.eos_token]
        encoder_input_id, encoder_attention_mask = self.make_input_id_mask(
            context_tokens, index)
        decoder_input_id, decoder_attention_mask = self.make_input_id_mask(
            summary_tokens, index)
        labels = self.tokenizer.convert_tokens_to_ids(
            summary_tokens[1:(self.max_seq_len + 1)])
        if len(labels) < self.max_seq_len:
            while len(labels) < self.max_seq_len:
                # for cross entropy loss masking
                  labels += [-100]

        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'attention_mask': np.array(encoder_attention_mask, dtype=np.float_),
                'decoder_input_ids': np.array(decoder_input_id[:1024], dtype=np.int_),
                'decoder_attention_mask': np.array(decoder_attention_mask[:1024], dtype=np.float_),
                'labels': np.array(labels[:1024], dtype=np.int_)}
        
