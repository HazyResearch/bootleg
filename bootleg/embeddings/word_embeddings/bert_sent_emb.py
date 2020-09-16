"""BERT sentence embedding class"""

import os

import torch
from bootleg.embeddings.word_embeddings import BaseSentEmbedding
from bootleg.layers.layers import MLP
from bootleg.utils import model_utils
from bootleg.utils.classes.dotted_dict import DottedDict


class BERTSentEmbedding(BaseSentEmbedding):
    """
    Pretrained BERT sentence embedding class that uses the BERT encoder to contextualize the word embeddings.
    """
    def __init__(self, emb_args, main_args, word_emb_dim, word_symbols):
        super(BERTSentEmbedding, self).__init__(emb_args, main_args, word_emb_dim, word_symbols)
        # TO LOAD AND SAVE BERT
        # import torch
        # import os
        # from transformers import BertModel
        # cache_dir = "pretrained_bert_models"
        # model = BertModel.from_pretrained('bert-base-cased', cache_dir=cache_dir)
        # torch.save(model.encoder, os.path.join(cache_dir, "bert_base_cased_encoder.pt"))
        # model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        # torch.save(model.encoder, os.path.join(cache_dir, "bert_base_uncased_encoder.pt"))

        # Even if sent_emb is frozen, we may need to keep track of gradients if word_emb is NOT frozen
        # If both are frozen, we can use torch.no_grad in the forward pass to save memory
        self.requires_grad = not emb_args.freeze_sent_emb or not emb_args.freeze_word_emb
        self.num_layers = emb_args.layers
        cache_dir = emb_args.cache_dir
        if emb_args.use_lower_case:
            self.encoder = torch.load(os.path.join(cache_dir, "bert_base_uncased_encoder.pt"))
        else:
            self.encoder = torch.load(os.path.join(cache_dir, "bert_base_cased_encoder.pt"))
        self.encoder.layer = self.encoder.layer[:emb_args.layers]
        self._key = "sentence"
        self._dim = word_emb_dim
        self.attention_weights = {}

    def forward(self, word_package):
        attention_mask = word_package.mask
        word_vectors = word_package.tensor
        head_mask = [None] * self.num_layers
        if self.requires_grad:
            output = self.encoder(word_vectors, attention_mask, head_mask)[0]
        else:
            with torch.no_grad():
                output = self.encoder(word_vectors, attention_mask, head_mask)[0]
        emb = DottedDict(
            tensor=output,
            downstream_mask=word_package.downstream_mask,
            key=self.get_key(),
            dim=self.get_dim()
        )
        return emb