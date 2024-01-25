from typing import Any, Dict, Literal
import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from transformers import AutoConfig

from const.special_token import TARGET_TOKEN

class BinaryClassifierByTARGET(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=2e-5, dropout_rate=0.1, batch_size=32, concat_or_mean: Literal['concat', 'mean'] ='concat', save_transformer_model_path=None):
        super().__init__()
        # learning rate
        self.lr = lr
        # batch_size
        self.batch_size = batch_size
        # Load pre-trained BERT model for sequence classification
        config = AutoConfig.from_pretrained(
            model_name,
            # num_labels=1,
            num_hidden_layers=12,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            )
        self.model = AutoModel.from_pretrained(model_name, config=config)
        # instantiate tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # add special tokens
        self.tokenizer.add_tokens([TARGET_TOKEN])
        self.model.resize_token_embeddings(len(self.tokenizer))

        # concat or mean (2 [TARGET] token)
        # self.concat_or_mean = concat_or_mean
        self.set_concat_or_mean(concat_or_mean)

        # classifier (concatenate 2 [TARGET] token)
        hidden_size_scale_factor = 2 if self.concat_or_mean == 'concat' else 1
        self.classifier = nn.Linear(self.model.config.hidden_size * hidden_size_scale_factor, 1)

        # sigmoid https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        self.criteria = nn.BCEWithLogitsLoss()

        self.model_dir_path = save_transformer_model_path

    def set_concat_or_mean(self, concat_or_mean: Literal['concat', 'mean']):
        self.concat_or_mean = concat_or_mean

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        # Find the position of <target> token.
        # target_token_indices = (input_ids == self.tokenizer.convert_tokens_to_ids(TARGET_TOKEN)).nonzero(as_tuple=True)
        target_token_indices = torch.tensor([]).cuda()
        for i in range(len(input_ids)):
            indices = (input_ids[i] == self.tokenizer.convert_tokens_to_ids(TARGET_TOKEN)).nonzero().squeeze()
            # If only one [TARGET] token is included, repeat the index (tensor(148) -> tensor([148, 148])).
            if indices.nelement() == 1:
                indices = indices.repeat(2)
            elif indices.nelement() == 0:
                raise Exception('target_token_indices is empty.')

            target_token_indices = torch.cat((target_token_indices, indices.unsqueeze(0)), dim=0)
        # convert tartget_token_indicies to int type
        target_token_indices = target_token_indices.to(torch.int64)

        target_states = torch.tensor([]).cuda()
        for i, tti in enumerate(target_token_indices):
            if len(tti) != 2:
                raise Exception('target_token_indices is not 2.')
            if self.concat_or_mean == 'concat':
                concat_last_state = torch.cat([last_hidden_state[i][tti[0]], last_hidden_state[i][tti[1]]], dim=0)
                target_states = torch.cat([target_states, concat_last_state.unsqueeze(0)], dim=0)
            elif self.concat_or_mean == 'mean':
                mean_last_state = torch.mean(torch.stack([last_hidden_state[i][tti[0]], last_hidden_state[i][tti[1]]]), dim=0)
                target_states = torch.cat([target_states, mean_last_state.unsqueeze(0)], dim=0)

        logits = self.classifier(target_states)
        return logits.squeeze()

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = self.criteria(logits, batch['labels'].float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = self.criteria(logits, batch['labels'].float())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = self.criteria(logits, batch['labels'].float())
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.model_dir_path is not None:
            self.model.save_pretrained(self.model_dir_path)
        return super().on_save_checkpoint(checkpoint)