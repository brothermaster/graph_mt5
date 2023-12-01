import argparse
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from data_loader import T5DataModule # src.
from model_training import T5FineTuner # src.
from my_416_t5 import T5ForConditionalGeneration as ModT5

device = torch.device("cuda")
model_name="model_path/t5-small"
tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
# model = T5ForConditionalGeneration.from_pretrained(f"{model_name}",output_loading_info=True, return_dict=True)


modified_model,massage_dict = ModT5.from_pretrained(f"{model_name}",output_loading_info=True, return_dict=True)
input_ids = torch.randint(0,100,size=[10,6])
decoder_input_ids = torch.randint(0,100,size=[10,6])
output =  modified_model(input_ids=input_ids,decoder_input_ids=decoder_input_ids)
print()
#   "feed_forward_proj": "relu",