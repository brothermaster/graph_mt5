"""
This is data loader class file
"""
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset

class T5DataModule(pl.LightningDataModule):
    """ T5 data class """

    def __init__(self,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int = 4,
                 source_max_token_length: int = 512,
                 target_max_token_length: int = 512,
                 num_workers: int = 2
                 ):
        """ initiates a T5 Data Module """
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_length = source_max_token_length
        self.target_max_token_length = target_max_token_length
        self.num_workers = num_workers

    def setup(self,stage=None):
        """ getting train_dataset, test_dataset """

        self.train_dataset = PyTorchDataModule(self.train_df, self.tokenizer, self.source_max_token_length, self.target_max_token_length)
        self.val_dataset = PyTorchDataModule(self.val_df, self.tokenizer, self.source_max_token_length, self.target_max_token_length)

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


class PyTorchDataModule(Dataset):
    """  PyTorch Dataset class  """

    def __init__(self, data, tokenizer, source_max_token_length: int = 512, target_max_token_length: int = 512):
        """ initiates a PyTorch Dataset Module for input data """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_length = source_max_token_length
        self.target_max_token_length = target_max_token_length

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"]
        labels[labels == 0] = -100  # to make sure we have correct labels for T5 text generation

        return dict(
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )
