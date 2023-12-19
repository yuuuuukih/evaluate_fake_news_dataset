import torch
import pytorch_lightning as pl
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig

class BertForSequenceClassificationModel(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=1e-3, dropout_rate=0.1, add_special_token=True, save_transformer_model_path=None):
        super().__init__()
        # self.save_hyperparameters()
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=2,
            num_hidden_layers=12,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            )
        self.model = BertForSequenceClassification.from_pretrained(model_name, config=config)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        if add_special_token:
            self.tokenizer.add_special_tokens(['<target>'])
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output