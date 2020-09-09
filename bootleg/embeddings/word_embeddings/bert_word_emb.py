"""BERT word embedding class"""

import os
import torch

from bootleg.embeddings.word_embeddings import BaseWordEmbedding
from bootleg.utils.classes.dotted_dict import DottedDict

class BERTWordEmbedding(BaseWordEmbedding):
    """
    Pretrained BERT word embedding class.
    """
    def __init__(self, emb_args, main_args, word_symbols):
        super(BERTWordEmbedding, self).__init__(emb_args, main_args, word_symbols)
        # TO LOAD AND SAVE BERT
        # import torch
        # import os
        # from transformers import BertModel
        # cache_dir = os.path.join("embs", "pretrained_bert_models")
        # model = BertModel.from_pretrained('bert-base-cased', cache_dir=cache_dir)
        # torch.save(model.embeddings, os.path.join(cache_dir, "bert_base_cased_embedding.pt"))
        # model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        # torch.save(model.embeddings, os.path.join(cache_dir, "bert_base_uncased_embedding.pt"))
        cache_dir = os.path.join(emb_args.cache_dir, "pretrained_bert_models")
        if emb_args.use_lower_case:
            self.embeddings = torch.load(os.path.join(cache_dir, "bert_base_uncased_embedding.pt"))
        else:
            self.embeddings = torch.load(os.path.join(cache_dir, "bert_base_cased_embedding.pt"))
        self._dim = self.embeddings.word_embeddings.embedding_dim
        self.requires_grad = not emb_args.freeze_word_emb

    def get_dim(self):
        return self._dim

    def get_key(self):
        return "bert_word_emb"

    # BERT does this internally in the entire model: (1-mask)*-1000 and adds that to the attention output before softmaxing
    # So, we are replicating it.
    # Taken from https://github.com/huggingface/transformers/
    def get_bert_mask(self, word_indices):
        attention_mask = (word_indices != self.pad_id).long()
        assert attention_mask.sum() != 0
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def forward(self, word_indices, requires_grad=None, token_id=0):
        if requires_grad is None:
            requires_grad = self.requires_grad
        token_type_ids = torch.ones_like(word_indices)*token_id
        if requires_grad:
            out = self.embeddings(word_indices, token_type_ids=token_type_ids)
        else:
            with torch.no_grad():
                out = self.embeddings(word_indices, token_type_ids=token_type_ids)
        packed_emb = DottedDict(
            tensor = out,
            mask = self.get_bert_mask(word_indices),
            downstream_mask = self.get_downstream_mask(word_indices),
            key = self.get_key(),
            dim = self.get_dim()
        )
        return packed_emb