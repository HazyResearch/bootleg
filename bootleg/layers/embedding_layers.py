"""Embedding layers"""
import copy
from torch import nn

from bootleg.utils.classes.dotted_dict import DottedDict
from bootleg.utils.utils import import_class
from bootleg.utils import logging_utils
from bootleg.layers.layers import *

class EmbeddingLayer(nn.Module):
    """
    Embedding layer class. This queries each embedding class, words and entities, before being sent to embedding_combiner
    to form the entity payload. Projects sentence embedding to hidden size.
    """
    def __init__(self, args, model_device, entity_symbols, word_symbols):
        super(EmbeddingLayer, self).__init__()
        self.logger = logging_utils.get_logger(args)
        self.num_entities_with_pad_and_nocand = entity_symbols.num_entities_with_pad_and_nocand

        # Word Embedding (passed to Sentence and Entity Embedding classess)
        mod, load_class = import_class("bootleg.embeddings.word_embeddings",
            args.data_config.word_embedding.load_class)
        self.word_emb = getattr(mod, load_class)(
            args.data_config.word_embedding, args, word_symbols)

        # Sentence Embedding
        mod, load_class = import_class("bootleg.embeddings.word_embeddings",
            args.data_config.word_embedding.sent_class)
        self.sent_emb = getattr(mod, load_class)(
            args.data_config.word_embedding, args, self.word_emb.get_dim(), word_symbols)

        # Entity Embedding
        self.entity_embs = nn.ModuleDict()
        self.logger.info('Loading embeddings...')
        for emb in args.data_config.ent_embeddings:
            try:
                emb_args = emb.args
            except:
                emb_args = None
            assert "load_class" in emb, "You must specify a load_class in the embedding config: {load_class: ..., key: ...}"
            assert "key" in emb, "You must specify a key in the embedding config: {load_class: ..., key: ...}"
            mod, load_class = import_class("bootleg.embeddings", emb.load_class)
            emb_obj = getattr(mod, load_class)(main_args=args,
                emb_args=emb_args, model_device=model_device, entity_symbols=entity_symbols, word_symbols=word_symbols,
                word_emb=self.word_emb, key=emb.key)
            self.entity_embs[emb.key] = emb_obj
        self.logger.info('Finished loading embeddings.')

        # Track the dimensions of different embeddings
        self.emb_sizes = {}
        for emb in self.entity_embs.values():
            key = emb.key
            dim = emb.get_dim()
            assert not key in self.emb_sizes, f"Can't have duplicate keys in your embeddings and {key} is already here"
            self.emb_sizes[key] = dim
        self.sent_emb_size = self.sent_emb._dim

        self.project_sent = MLP(input_size=self.sent_emb_size,
                        num_hidden_units=0, output_size=args.model_config.hidden_size,
                        num_layers=1, dropout=0, residual=False, activation=None)

    def forward(self, word_indices, entity_package, batch_prepped_data, batch_on_the_fly_data):
        word_package = self.word_emb(word_indices)
        sent_emb = self.sent_emb(word_package)
        sent_tensor = self.project_sent(sent_emb.tensor)
        # WordPackages have two masks. One is used in the sentence embedding module (param mask) and one is used in our attention network (param downstream mask).
        # At this point, the mask we always want to use is the downstream mask (as we are beyond sentence embedding stage).
        # Hence we sent both mask and downstream mask to be the same.
        sent_emb = DottedDict(
            tensor=sent_tensor,
            mask=sent_emb.downstream_mask,
            key=sent_emb.key,
            dim=sent_emb.dim
        )
        entity_embs = []
        for entity_emb in self.entity_embs.values():
            forward_package = entity_package
            forward_package.key = entity_emb.key
            emb_package = entity_emb(forward_package, batch_prepped_data, batch_on_the_fly_data, sent_emb)
            # If some embeddings do not want to be added to the payload, they will return an empty emb_package
            # This happens for the kg bias (KGIndices) class for our kg attention network
            if len(emb_package) == 0:
                continue
            emb_package.tensor = model_utils.emb_1d_dropout(entity_emb.training, entity_emb.dropout_perc, emb_package.tensor)
            emb_package.tensor = model_utils.emb_2d_dropout(entity_emb.training, entity_emb.mask_perc, emb_package.tensor)
            entity_embs.append(emb_package)
        return sent_emb, entity_embs


# Used for the BERT baseline
class EmbeddingLayerNoProj(EmbeddingLayer):
    """Embedding layer that does not project sentence embedding to hidden size. Used for BERTNED baseline."""
    def __init__(self, args, model_device, entity_symbols, word_symbols):
        super(EmbeddingLayerNoProj, self).__init__(args, model_device, entity_symbols, word_symbols)

    def forward(self, word_indices, entity_package, batch_prepped_data, batch_on_the_fly_data):
        word_package = self.word_emb(word_indices)
        sent_emb = self.sent_emb(word_package)
        entity_embs = []
        for entity_emb in self.entity_embs.values():
            emb_package = entity_emb(entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb)
            # If some embeddings do not want to be added to the payload, they will return an empty emb_package
            # This happens for the kg bias (KGIndices) class for our kg attention network
            if len(emb_package) == 0:
                continue
            emb_package.tensor = model_utils.emb_1d_dropout(entity_emb.training, entity_emb.dropout_perc, emb_package.tensor)
            emb_package.tensor = model_utils.emb_2d_dropout(entity_emb.training, entity_emb.mask_perc, emb_package.tensor)
            entity_embs.append(emb_package)
        return sent_emb, entity_embs