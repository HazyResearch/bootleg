"""Standard word embedding class."""

import torch
import torch.nn as nn
import torchtext

from bootleg.embeddings.word_embeddings import BaseWordEmbedding
from bootleg.layers.layers import MLP, PositionalEncoding
from bootleg.utils.classes.dotted_dict import DottedDict
from bootleg.utils import model_utils


class StandardWordEmb(BaseWordEmbedding):
    """
    Word embedding layer that takes word indices and outputs the embedding vectors.  This is mimicking the BERT word layer.
    Like BERT, we add word positions, dropout, and layer norm.
    """
    def __init__(self, args, main_args, word_symbols):
        super(StandardWordEmb, self).__init__(args, main_args, word_symbols)
        if args.custom_proj_size != -1:
            self._dim = args.custom_proj_size
            self.use_proj = True
        else:
            self._dim = word_symbols.word_embedding_dim
            self.use_proj = False
        self.layers = args.layers
        # this is stored for making sure we don't collect gradients during the forward pass
        # Gradients are set to False in the base_model
        self.requires_grad = not args.freeze_word_emb

        self.position_words = PositionalEncoding(self._dim)

        # +2 is for the pad and the unk row
        self.num_words_with_pad_unk = word_symbols.num_words+2
        self.word_embedding = nn.Embedding(self.num_words_with_pad_unk,
            word_symbols.word_embedding_dim, padding_idx=-1, sparse=True)
        # set rows to zero so UNK row will be zero
        self.word_embedding.weight.data.fill_(0)
        # we leave the first row for a zero UNK row, and the last row for a zero PAD row
        if args.custom_vocab_embedding_file != "":
            word_embeddings_vecs = torchtext.vocab.Vectors(args.custom_vocab_embedding_file,
                cache=main_args.data_config.emb_dir)
        else:
            word_embeddings_vecs = torchtext.vocab.GloVe(cache=main_args.data_config.emb_dir)
        self.word_embedding.weight.data[1:-1].copy_(word_embeddings_vecs.vectors)

        self.layer_norm = nn.LayerNorm(self._dim)
        # self.layer_norm.apply(model_utils.init_weights)
        self.dropout = nn.Dropout(main_args.train_config.dropout)

        # projection to larger dimension if arg
        # mainly used for unit testing
        if self.use_proj:
            self.proj = MLP(word_symbols.word_embedding_dim, 0, self._dim, num_layers=1, residual=False, activation=None)

    def get_dim(self):
        return self._dim

    def get_key(self):
        return "stand_word_emb"

    def forward(self, word_indices, requires_grad=None):
        if requires_grad is None:
            requires_grad = self.requires_grad
        (batch_size, seq_length) = word_indices.shape
        # num_words_with_pad_unk-1  because index starts at 0
        word_indices_pos = torch.where(word_indices >= 0, word_indices,
                                           (torch.ones_like(word_indices,
                                           dtype=torch.long)*(self.num_words_with_pad_unk-1)))

        if requires_grad:
            word_vectors = self.word_embedding(word_indices_pos)
        else:
            with torch.no_grad():
                word_vectors = self.word_embedding(word_indices_pos)
        if self.use_proj:
            word_vectors = self.proj(word_vectors)
        word_vectors = self.position_words(word_vectors, torch.arange(0, word_indices.shape[1]).repeat(batch_size, 1))
        word_vectors = self.layer_norm(word_vectors)
        word_vectors = self.dropout(word_vectors)
        packed_emb = DottedDict(
            tensor = word_vectors,
            # The mask for the standard word embedding is the same as the downstream mask
            mask = self.get_downstream_mask(word_indices),
            downstream_mask = self.get_downstream_mask(word_indices),
            key = self.get_key(),
            dim = self.get_dim()
        )
        return packed_emb