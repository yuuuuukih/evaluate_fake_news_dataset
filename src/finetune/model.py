import torch
from torch import nn
import pytorch_lightning as pl
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig

class BertBinaryClassifier(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=1e-3, dropout_rate=0.1, add_special_token=True, save_transformer_model_path=None):
        super().__init__()
        # learning rate
        self.lr = lr
        # Load pre-trained BERT model for sequence classification
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=1,
            num_hidden_layers=12,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            )
        self.model = BertForSequenceClassification.from_pretrained(model_name, config=config)
        # instantiate tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # add special tokens
        if add_special_token:
            self.tokenizer.add_tokens(['<target>'])
            self.model.resize_token_embeddings(len(self.tokenizer))

        # classifier
        self.model.classifier = nn.Linear(self.model.config.hidden_size, self.model.config.num_labels)
        self.model.classifier.apply(self.model._init_weights)
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels=None):
        # Get the classifier's output
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        proba = self.sigmoid(output.logits).squeeze()
        return proba

    def training_step(self, batch, batch_idx):
        # Forward pass
        # proba = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
        proba = self.forward(batch['input_ids'], batch['attention_mask'])
        # Calculate loss
        loss = nn.BCELoss()(proba, batch['labels'].float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        proba = self.forward(batch['input_ids'], batch['attention_mask'])
        # Calculate loss
        loss = nn.BCELoss()(proba, batch['labels'].float())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Forward pass
        proba = self.forward(batch['input_ids'], batch['attention_mask'])
        # Calculate loss
        loss = nn.BCELoss()(proba, batch['labels'].float())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
