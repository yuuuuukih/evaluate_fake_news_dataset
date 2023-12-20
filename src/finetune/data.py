import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset

class FakeNewsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, tokenizer, batch_size: int = 32, num_workers: int = 0, max_length: int = 512, mode: str = 'base'):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.mode = mode

    def setup(self, stage=None):
        # load dataset
        dataset = load_dataset('json', data_files={
            'train': os.path.join(self.data_dir, 'train.json'),
            'val': os.path.join(self.data_dir, 'val.json'),
            # 'test': os.path.join(self.data_dir, 'test.json')
        }, field='data')
        self.train_dataset, self.val_dataset = dataset['train'], dataset['val']

        # tokenization.
        self.train_dataset = self.train_dataset.map(
            self.tokenize, batched=True
        )

        self.val_dataset = self.val_dataset.map(
            self.tokenize, batched=True
        )

    def tokenize(self, example):
        src_texts = example['src']
        tgt_texts = example['tgt']
        # tokenize src text and tgt text.
        tokenized_src_texts = self.tokenizer(src_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt', add_special_tokens=True)
        # tokenized_tgt_texts = self.tokenizer(tgt_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        input_ids = tokenized_src_texts['input_ids']
        attention_mask = tokenized_src_texts['attention_mask']
        labels = torch.tensor(tgt_texts, dtype=torch.long)
        # labels = tokenized_tgt_texts['input_ids']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=default_data_collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=default_data_collator)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=default_data_collator)