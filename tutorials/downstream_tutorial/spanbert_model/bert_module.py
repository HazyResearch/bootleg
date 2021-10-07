import os

from pytorch_pretrained_bert.modeling import BertModel
from torch import nn


class BertModule(nn.Module):
    def __init__(self, bert_model_name, dropout_prob=0.1, cache_dir="./cache/"):
        super().__init__()

        # Create cache directory if not exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.bert_model = BertModel.from_pretrained(
            bert_model_name, cache_dir=cache_dir
        )

    def forward(self, token_ids, token_type_ids=None, attention_mask=None):
        encoded_layers, pooled_output = self.bert_model(
            token_ids, token_type_ids, attention_mask
        )
        return encoded_layers, pooled_output
