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
        self.classifier = nn.Linear(self.model.config.hidden_size * hidden_size_scale_factor, batch_size)

        # sigmoid https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        self.criteria = nn.BCEWithLogitsLoss()

        self.model_dir_path = save_transformer_model_path

    def set_concat_or_mean(self, concat_or_mean: Literal['concat', 'mean']):
        self.concat_or_mean = concat_or_mean

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        # Find the position of <target> token.
        target_token_indicies = (input_ids == self.tokenizer.convert_tokens_to_ids(TARGET_TOKEN)).nonzero(as_tuple=True)
        if len(target_token_indicies) == 2:
            target_token_states = last_hidden_state[target_token_indicies[0], target_token_indicies[1]]
        else:
            raise Exception('target_token_indicies is not 2.')

        # concat or mean
        if self.concat_or_mean == 'concat':
            combined_target_token_states = torch.cat([target_token_states[0], target_token_states[1]], dim=0)
        elif self.concat_or_mean == 'mean':
            combined_target_token_states = torch.mean(torch.stack([target_token_states[0], target_token_states[1]]), dim=0)

        logits = self.classifier(combined_target_token_states)
        return logits

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