"""Base model class."""
import torch.nn as nn
import torch

from bootleg.layers.embedding_layers import EmbeddingLayer, EmbeddingLayerNoProj
from bootleg.layers.slice_heads import SliceHeadsSBL, NoSliceHeads
from bootleg.layers.emb_combiner import EmbCombinerProj
from bootleg.layers.layers import TypePred
from bootleg.utils import logging_utils, train_utils
from bootleg.utils.classes.dotted_dict import DottedDict
from bootleg.utils.utils import import_class

from bootleg.symbols.constants import DISAMBIG, TYPEPRED, FINAL_LOSS


class Model(nn.Module):
    def __init__(self, args, model_device, entity_symbols, word_symbols):
        super(Model, self).__init__()
        self.model_device = model_device
        self.num_entities_with_pad_and_nocand = entity_symbols.num_entities_with_pad_and_nocand
        self.logger = logging_utils.get_logger(args)
        # embeddings
        self.emb_layer = EmbeddingLayer(args, self.model_device, entity_symbols, word_symbols)
        self.type_pred = False
        if args.data_config.type_prediction.use_type_pred:
            self.type_pred = True
            # Add 1 for pad type
            self.type_prediction = TypePred(args.model_config.hidden_size, args.data_config.type_prediction.dim, args.data_config.type_prediction.num_types+1)
        self.emb_combiner = EmbCombinerProj(args, self.emb_layer.emb_sizes,
            self.emb_layer.sent_emb_size, word_symbols, entity_symbols)
        # attention network
        mod, load_class = import_class("bootleg.layers.attn_networks", args.model_config.attn_load_class)
        self.attn_network = getattr(mod, load_class)(args, self.emb_layer.emb_sizes, self.emb_layer.sent_emb_size, entity_symbols, word_symbols)
        # slice heads
        self.slice_heads = self.get_slice_method(args, entity_symbols)
        self.freeze_components(args)

    def get_slice_method(self, args, entity_symbols):
        if args.train_config.slice_method == "SBL":
            return SliceHeadsSBL(args, entity_symbols)
        elif args.train_config.slice_method == "Normal":
            return NoSliceHeads(args, entity_symbols)
        else:
            raise ValueError('Slice method must be one of [SBL, Normal].')

    def freeze_components(self, args):
        # freeze embeddings if needed
        # we freeze embeddings if freeze is explicitly set for that entity embedding (opt-in)
        self.logger.debug('=====Checking freeze on entity embedding parameters=====')
        for emb in args.data_config.ent_embeddings:
            if 'freeze' in emb and emb.freeze is True:
                print(emb.key)
                self.emb_layer.entity_embs[emb.key].freeze_params()

        self.logger.debug('=====Checking freeze on word embedding parameters=====')
        if args.data_config.word_embedding.freeze_word_emb:
            self.emb_layer.word_emb.freeze_params()

        self.logger.debug('=====Checking freeze on sentence embedding parameters=====')
        if args.data_config.word_embedding.freeze_sent_emb:
            self.emb_layer.sent_emb.freeze_params()

    def forward(self, alias_idx_pair_sent, word_indices, alias_indices,
        entity_indices, batch_prepped_data, batch_on_the_fly_data):
        # mask out padded candidates
        mask = entity_indices == -1
        entity_indices = torch.where(entity_indices >= 0, entity_indices,
                                           (torch.ones_like(entity_indices, dtype=torch.long)*(self.num_entities_with_pad_and_nocand-1)))
        entity_package = DottedDict(tensor=entity_indices, pos_in_sent=alias_idx_pair_sent, alias_indices=alias_indices, mask=mask)
        sent_emb, entity_embs = self.emb_layer(word_indices, entity_package, batch_prepped_data, batch_on_the_fly_data)
        if self.type_pred:
            entity_embs, batch_type_pred = self.type_prediction(sent_emb, entity_package, entity_embs)
        sent_emb, entity_embs = self.emb_combiner(sent_embedding=sent_emb,
            alias_idx_pair_sent=alias_idx_pair_sent, entity_embedding=entity_embs, entity_mask=mask)
        context_matrix_dict, backbone_out = self.attn_network(alias_idx_pair_sent, sent_emb, entity_embs, batch_prepped_data, batch_on_the_fly_data)
        res, final_entity_embs = self.slice_heads(context_matrix_dict, alias_idx_pair_sent=alias_idx_pair_sent,
            entity_pack=entity_package, sent_emb=sent_emb, batch_prepped=batch_prepped_data,
            raw_entity_emb=entity_embs)
        # update output dictionary with backbone out
        res[DISAMBIG].update(backbone_out[DISAMBIG])
        if self.type_pred:
            res[TYPEPRED] = {train_utils.get_type_head_name(): batch_type_pred}
        return res, entity_package, final_entity_embs


class BaselineModel(Model):
    def __init__(self, args, model_device, entity_symbols, word_symbols):
        super(BaselineModel, self).__init__(args, model_device, entity_symbols, word_symbols)
        self.model_device = model_device
        self.logger = logging_utils.get_logger(args)
        mod, load_class = import_class("bootleg.layers.attn_networks", args.model_config.attn_load_class)
        self.emb_layer = EmbeddingLayerNoProj(args, self.model_device, entity_symbols, word_symbols)
        self.attn_network = getattr(mod, load_class)(args, self.emb_layer.emb_sizes, self.emb_layer.sent_emb_size, entity_symbols, word_symbols)
        self.num_entities_with_pad_and_nocand = entity_symbols.num_entities_with_pad_and_nocand
        self.freeze_components(args)

    def forward(self, alias_idx_pair_sent, word_indices, alias_indices,
        entity_indices, batch_prepped_data, batch_on_the_fly_data):
        # mask out padded candidates (last row)
        mask = entity_indices == -1
        entity_indices = torch.where(entity_indices >= 0, entity_indices,
                                           (torch.ones_like(entity_indices, dtype=torch.long)*(self.num_entities_with_pad_and_nocand-1)))
        entity_package = DottedDict(tensor=entity_indices, pos_in_sent=alias_idx_pair_sent, alias_indices=alias_indices, mask=mask, normalize=False, key=None, dim=None)
        sent_emb, entity_embs = self.emb_layer(word_indices, entity_package, batch_prepped_data, batch_on_the_fly_data)
        res = self.attn_network(alias_idx_pair_sent, sent_emb, entity_embs, batch_prepped_data, batch_on_the_fly_data)
        return res, entity_package, None