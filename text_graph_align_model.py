
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
import dgl.nn as dglnn
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

torch.cuda.empty_cache()
pl.seed_everything(84)

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()

        perious_h = heads[0]
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, perious_h, activation=F.elu)
        )

        for h in heads[1:]:
            self.gat_layers.append(
                dglnn.GATConv(
                    hid_size * perious_h,
                    hid_size,
                    h,
                    residual=True,
                    activation=F.elu,
                )
            )
            perious_h = h

        self.liner = nn.Linear(hid_size,out_size)

    def forward(self, g, inputs):
        h = inputs # [3144, 50])
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h) # [3144, 4, 256] 
            if i == 2:  # last layer
                h = h.mean(1) # [3144, 121]
            else:  # other layer(s)
                h = h.flatten(-2) # [3144, 1024] 
        return h
    
class T5FineTune:
    """
    This class is using for fine-tune T5 based models
    """

    def __init__(self, model_type="t5", model_name="t5-base") -> None:
        """ Initiates T5FineTune class and loads T5, MT5, ByT5, t0, or flan-t5 model for fine-tuning """

        if model_type in ["t5", "flan-t5"]:
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "byt5":
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "t0":
            self.tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                f"{model_name}", return_dict=True
            )

    def train(
            self,
            train_df: pd.DataFrame,
            eval_df: pd.DataFrame,
            args: argparse.Namespace = argparse.Namespace(),):
        """

        :param train_df: Dataframe must have 2 column --> "source_text" and "target_text":
        :param eval_df: Dataframe must have 2 column --> "source_text" and "target_text":
        :param args: arguments
        :return: trained model
        """
        self.data_module = T5DataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=args.batch_size,
            source_max_token_length=args.source_max_token_length,
            target_max_token_length=args.target_max_token_length,
            num_workers=args.dataloader_num_workers)

        self.t5_model = T5FineTuner(args, tokenizer=self.tokenizer, model=self.model)

        callbacks = [TQDMProgressBar(refresh_rate=1)]

        if args.early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                                patience=args.early_stopping_patience_epochs, verbose=True, mode="min")
            callbacks.append(early_stop_callback)

        gpus = 1 if args.use_gpu else 0

        # add logger
        loggers = True if args.logger == "default" else args.logger

        # prepare trainer
        trainer = pl.Trainer(logger=loggers, callbacks=callbacks, max_epochs=args.max_epochs, gpus=gpus,
                             precision=args.precision, log_every_n_steps=1)

        # fit trainer
        trainer.fit(self.t5_model, self.data_module)

    def load_model(self, model_type: str = "t5", model_dir: str = "outputs", use_gpu: bool = False):
        """
        This function is using for load trained models
        :param model_type: model type
        :param model_dir: trained model directory
        :param use_gpu: gpu usage
        :return: loaded model
        """
        if model_type in ["t5", "flan-t5"]:
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "mt5":
            self.model = MT5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "byt5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "t0":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}")

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def predict(
            self,
            source_text: str,
            max_length: int = 512,
            num_return_sequences: int = 1,
            num_beams: int = 2,
            top_k: int = 10,
            top_p: float = 0.95,
            do_sample: bool = True,
            repetition_penalty: float = 2.5,
            length_penalty: float = 1.0,
            early_stopping: bool = True,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True,
    ):

        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]
        return preds