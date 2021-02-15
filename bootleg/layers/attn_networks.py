"""Attention networks."""
import logging

import torch.nn as nn

import bootleg.utils.model_utils
from bootleg.layers.helper_modules import *
from bootleg.symbols.constants import (
    BERT_WORD_DIM,
    DISAMBIG,
    FINAL_LOSS,
    KG_BIAS_LOAD_CLASS,
    MAIN_CONTEXT_MATRIX,
    REL_EMB_INDICES_KEY,
    REL_INDICES_KEY,
)
from bootleg.utils import eval_utils, model_utils
from bootleg.utils.embedding_utils import get_max_candidates

logger = logging.getLogger(__name__)


class AttnNetwork(nn.Module):
    """Base attention network.

    Args:
        args: args
        entity_symbols: entity symbols
    """

    def __init__(self, args, entity_symbols):
        super(AttnNetwork, self).__init__()
        self.num_entities_with_pad_and_nocand = (
            entity_symbols.num_entities_with_pad_and_nocand
        )
        # Number of candidates
        self.K = get_max_candidates(entity_symbols, args.data_config)
        # Number of aliases
        self.M = args.data_config.max_aliases
        self.hidden_size = args.model_config.hidden_size
        self.num_heads = args.model_config.num_heads
        self.num_model_stages = args.model_config.num_model_stages
        assert (
            self.num_model_stages > 0
        ), f"You must have > 0 model stages. You have {self.num_model_stages}"
        self.num_fc_layers = args.model_config.num_fc_layers
        self.ff_inner_size = args.model_config.ff_inner_size

    def forward(
        self,
        sent_embedding,
        sent_embedding_mask,
        entity_embedding,
        entity_embedding_mask,
        start_span_idx,
        end_span_idx,
        batch_on_the_fly_data,
    ):
        """Model forward.

        Args:
            sent_embedding: sentence embedding (B x N x L)
            sent_embedding_mask: sentence embedding mask (B x N)
            entity_embedding: entity embedding (B x M x K x H)
            entity_embedding_mask: entity embedding mask (B x M x K)
            start_span_idx: start mention index into sentence (B x M)
            end_span_idx: end mention index into sentence (B x M)
            batch_on_the_fly_data: batch on the fly dictionary with values (B x (M*K) x (M*K)) of KG adjacency matrices

        Returns: Dict of Dict of intermediate layer candidate scores (B x M x K),
                         Dict of all output entity embeddings from each KG matrix (B x M x K x H)
        """
        raise NotImplementedError


class Bootleg(AttnNetwork):
    """Bootleg attention network V1.

    Args:
        args: args
        entity_symbols: entity symbols
    """

    def __init__(self, args, entity_symbols):
        super(Bootleg, self).__init__(args, entity_symbols)
        self.dropout = args.train_config.dropout

        # For each stage, instantiate a transformer block for phrase (entity_word) and co-occurrence (self_entity) modules
        self.attention_modules = nn.ModuleDict()
        self.combine_modules = nn.ModuleDict()
        for i in range(self.num_model_stages):
            self.attention_modules[f"stage_{i}_entity_word"] = AttnBlock(
                size=self.hidden_size,
                ff_inner_size=args.model_config.ff_inner_size,
                dropout=self.dropout,
                num_heads=self.num_heads,
            )
            self.attention_modules[f"stage_{i}_self_entity"] = SelfAttnBlock(
                size=self.hidden_size,
                ff_inner_size=args.model_config.ff_inner_size,
                dropout=self.dropout,
                num_heads=self.num_heads,
            )
            self.combine_modules[f"stage_{i}_combine"] = NormAndSum(self.hidden_size)

        # For the KG bias module
        self.kg_bias_list = []
        self.kg_bias_keys = []
        for emb in args.data_config.ent_embeddings:
            if emb.load_class == KG_BIAS_LOAD_CLASS:
                # self.kg_bias_weights[emb.key] = torch.nn.Parameter(torch.tensor(2.0))
                setattr(self, emb.key, torch.nn.Parameter(torch.tensor(2.0)))
                self.kg_bias_list.append(getattr(self, emb.key))
                self.kg_bias_keys.append(emb.key)
        self.kg_bias_keys = sorted(self.kg_bias_keys)
        # If we have kg bias terms, we want to take the average of those context matrices when generating the final context matrix to be returned.
        # The no_kg_key is used for the context matrix without kg_bias terms added. If we use the key ending in _nokg, it will not be averaged
        # in the final result.
        # If we do not have kg bias terms, we want the nokg context matrix to be the final matrix. MAIN_CONTEXT_MATRIX key allows for this.
        if len(self.kg_bias_keys) > 0:
            self.no_kg_key = "context_matrix_nokg"
        else:
            self.no_kg_key = MAIN_CONTEXT_MATRIX
        self.kg_softmax = nn.Softmax(dim=2)

        # Two things to note, the attn mask is a block diagonal matrix prevent an alias from paying attention to its own K candidates in the attention layer
        # This works because the original input is added to the output of this attention, meaning an alias becomes its
        # original embedding plus the contributions of the other aliases in the sentence.
        # Second, the attn mask is added to the attention before softmax (added to Q dot V^T) -- softmax makes e^(-1e9+old_value) become zero
        # When setting it to be -inf, you can get nans in the loss if all entities end up being masked out (eg only one alias in the sentence)
        self.e2e_entity_mask = torch.zeros((self.K * self.M, self.K * self.M))
        for i in range(self.M):
            self.e2e_entity_mask[
                i * self.K : (i + 1) * self.K, i * self.K : (i + 1) * self.K
            ] = 1.0
            # Must manually move this to the device as it's not part of a module...we can probably fix this
        self.e2e_entity_mask = self.e2e_entity_mask.masked_fill(
            (self.e2e_entity_mask == 1), float(-1e9)
        )

        # Track attention weights
        self.attention_weights = {}

        # Prediction layers: each stage except the last gets a prediction layer
        # Last layer's prediction head is added in slice heads
        disambig_task = nn.ModuleDict()
        for i in range(self.num_model_stages - 1):
            disambig_task[bootleg.utils.model_utils.get_stage_head_name(i)] = MLP(
                self.hidden_size, self.hidden_size, 1, self.num_fc_layers, self.dropout
            )
        self.predict_layers = {DISAMBIG: disambig_task}
        self.predict_layers = nn.ModuleDict(self.predict_layers)

    def forward(
        self,
        sent_embedding,
        sent_embedding_mask,
        entity_embedding,
        entity_embedding_mask,
        start_span_idx,
        end_span_idx,
        batch_on_the_fly_data,
    ):
        """Model forward.

        Args:
            sent_embedding: sentence embedding (B x N x L)
            sent_embedding_mask: sentence embedding mask (B x N)
            entity_embedding: entity embedding (B x M x K x H)
            entity_embedding_mask: entity embedding mask (B x M x K)
            start_span_idx: start mention index into sentence (B x M)
            end_span_idx: end mention index into sentence (B x M)
            batch_on_the_fly_data: batch on the fly dictionary with values (B x (M*K) x (M*K)) of KG adjacency matrices

        Returns: Dict of Dict of intermediate layer candidate scores (B x M x K),
                         Dict of all output entity embeddings from each KG matrix (B x M x K x H)
        """
        batch_size = sent_embedding.shape[0]
        out = {DISAMBIG: {}}

        # Create KG bias matrices for each kg bias key
        kg_bias_norms = {}
        for key in self.kg_bias_keys:
            bias_weight = getattr(self, key)  # self.kg_bias_weights[key]
            kg_bias = (
                batch_on_the_fly_data[key]
                .float()
                .to(sent_embedding.device)
                .reshape(batch_size, self.M * self.K, self.M * self.K)
            )
            kg_bias_diag = kg_bias + bias_weight * torch.eye(self.M * self.K).repeat(
                batch_size, 1, 1
            ).view(batch_size, self.M * self.K, self.M * self.K).to(kg_bias.device)
            kg_bias_norm = self.kg_softmax(
                kg_bias_diag.masked_fill((kg_bias_diag == 0), float(-1e9))
            )
            kg_bias_norms[key] = kg_bias_norm
        sent_tensor = sent_embedding.transpose(0, 1)

        # Resize the alias embeddings and the entity mask from B x M x K x D -> B x (M*K) x D
        entity_embedding = entity_embedding.contiguous().view(
            batch_size, self.M * self.K, self.hidden_size
        )
        entity_embedding = entity_embedding.transpose(0, 1)  # reshape for attention
        key_padding_mask_entities = entity_embedding_mask.contiguous().view(
            batch_size, self.M * self.K
        )

        # Iterate through stages
        query_tensor = entity_embedding
        for stage_index in range(self.num_model_stages):
            # As we are adding a residual in the attention modules, we can make embs empty
            embs = []
            context_mat_dict = {}
            # ============================================================================
            # Phrase module: compute attention between entities and words
            # ============================================================================
            word_entity_attn_context, word_entity_attn_weights = self.attention_modules[
                f"stage_{stage_index}_entity_word"
            ](
                q=query_tensor,
                x=sent_tensor,
                key_mask=sent_embedding_mask,
                attn_mask=None,
            )
            # Add embeddings to be merged in the output
            embs.append(word_entity_attn_context)
            # Save the attention weights
            self.attention_weights[
                f"stage_{stage_index}_entity_word"
            ] = word_entity_attn_weights

            # ============================================================================
            # Co-occurrence module: compute self attention over entities
            # ============================================================================
            # Move entity mask to device
            # TODO: move to device in init?
            self.e2e_entity_mask = self.e2e_entity_mask.to(
                key_padding_mask_entities.device
            )

            entity_attn_context, entity_attn_weights = self.attention_modules[
                f"stage_{stage_index}_self_entity"
            ](
                x=query_tensor,
                key_mask=key_padding_mask_entities,
                attn_mask=self.e2e_entity_mask,
            )
            # Mask out MxK of single aliases, alias_indices is batch x M, mask is true when single alias
            non_null_aliases = (
                self.K
                - key_padding_mask_entities.reshape(batch_size, self.M, self.K).sum(-1)
            ) != 0
            entity_attn_post_mask = (
                (non_null_aliases.sum(1) == 1)
                .unsqueeze(1)
                .expand(batch_size, self.K * self.M)
                .transpose(0, 1)
            )
            entity_attn_post_mask = entity_attn_post_mask.unsqueeze(-1).expand_as(
                entity_attn_context
            )
            entity_attn_context = torch.where(
                entity_attn_post_mask,
                torch.zeros_like(entity_attn_context),
                entity_attn_context,
            )

            # Add embeddings to be merged in the output
            embs.append(entity_attn_context)
            # Save the attention weights
            self.attention_weights[
                f"stage_{stage_index}_self_entity"
            ] = entity_attn_weights

            # Combine module output
            context_matrix_nokg = self.combine_modules[f"stage_{stage_index}_combine"](
                embs
            )
            context_mat_dict[self.no_kg_key] = context_matrix_nokg.transpose(
                0, 1
            ).reshape(batch_size, self.M, self.K, self.hidden_size)
            # ============================================================================
            # KG module: add in KG connectivity bias
            # ============================================================================
            for key in self.kg_bias_keys:
                context_matrix_kg = torch.bmm(
                    kg_bias_norms[key], context_matrix_nokg.transpose(0, 1)
                ).transpose(0, 1)
                context_matrix_kg = (context_matrix_nokg + context_matrix_kg) / 2
                context_mat_dict[f"context_matrix_{key}"] = context_matrix_kg.transpose(
                    0, 1
                ).reshape(batch_size, self.M, self.K, self.hidden_size)

            if stage_index < self.num_model_stages - 1:
                score = model_utils.max_score_context_matrix(
                    context_mat_dict,
                    self.predict_layers[DISAMBIG][
                        bootleg.utils.model_utils.get_stage_head_name(stage_index)
                    ],
                )
                out[DISAMBIG][
                    f"{bootleg.utils.model_utils.get_stage_head_name(stage_index)}"
                ] = score

            # This will take the average of the context matrices that do not end in the key "_nokg"; if there are not kg bias terms, it will
            # select the context_matrix_nokg (as it's key, in this setting, will not end in _nokg)
            query_tensor = (
                model_utils.generate_final_context_matrix(
                    context_mat_dict, ending_key_to_exclude="_nokg"
                )
                .reshape(batch_size, self.M * self.K, self.hidden_size)
                .transpose(0, 1)
            )
        return {
            "intermed_scores": out,
            "ent_embs": context_mat_dict,
            "final_scores": None,
        }


class BootlegM2E(AttnNetwork):
    """Bootleg attention network with a mention to entity canidate tensformer
    layer.

    Args:
        args: args
        entity_symbols: entity symbols
    """

    def __init__(self, args, entity_symbols):
        super(BootlegM2E, self).__init__(args, entity_symbols)
        self.dropout = args.train_config.dropout

        # For each stage, instantiate a transformer block for phrase (entity_word) and co-occurrence (self_entity) modules
        self.attention_modules = nn.ModuleDict()
        self.combine_modules = nn.ModuleDict()
        for i in range(self.num_model_stages):
            self.attention_modules[f"stage_{i}_entity_word"] = AttnBlock(
                size=self.hidden_size,
                ff_inner_size=args.model_config.ff_inner_size,
                dropout=self.dropout,
                num_heads=self.num_heads,
            )
            self.attention_modules[f"stage_{i}_self_entity"] = SelfAttnBlock(
                size=self.hidden_size,
                ff_inner_size=args.model_config.ff_inner_size,
                dropout=self.dropout,
                num_heads=self.num_heads,
            )
            self.attention_modules[f"stage_{i}_mention_entity"] = AttnBlock(
                size=self.hidden_size,
                ff_inner_size=args.model_config.ff_inner_size,
                dropout=self.dropout,
                num_heads=self.num_heads,
            )
            self.combine_modules[f"stage_{i}_combine"] = NormAndSum(self.hidden_size)

        # For the KG bias module
        # self.kg_bias_weights = nn.ParameterDict()  # ParameterDicts are buggy in DataParallel
        # self.kg_bias_list = []
        self.kg_bias_keys = []
        for emb in args.data_config.ent_embeddings:
            if emb.load_class == KG_BIAS_LOAD_CLASS:
                # self.kg_bias_weights[emb.key] = torch.nn.Parameter(torch.tensor(2.0))
                # setattr(self, emb.key, torch.nn.Parameter(torch.tensor(2.0)))
                # self.kg_bias_list.append(getattr(self, emb.key))
                self.kg_bias_keys.append(emb.key)
        self.kg_bias_keys = sorted(self.kg_bias_keys)
        # If we have kg bias terms, we want to take the average of those context matrices when generating the final context matrix to be returned.
        # The no_kg_key is used for the context matrix without kg_bias terms added. If we use the key ending in _nokg, it will not be averaged
        # in the final result.
        # If we do not have kg bias terms, we want the nokg context matrix to be the final matrix. MAIN_CONTEXT_MATRIX key allows for this.
        if len(self.kg_bias_keys) > 0:
            self.no_kg_key = "context_matrix_nokg"
        else:
            self.no_kg_key = MAIN_CONTEXT_MATRIX
        self.kg_softmax = nn.Softmax(dim=2)

        # Two things to note, the attn mask is a block diagonal matrix prevent an alias from paying attention to its own K candidates in the attention layer
        # This works because the original input is added to the output of this attention, meaning an alias becomes its
        # original embedding plus the contributions of the other aliases in the sentence.
        # Second, the attn mask is added to the attention before softmax (added to Q dot V^T) -- softmax makes e^(-1e9+old_value) become zero
        # When setting it to be -inf, you can get nans in the loss if all entities end up being masked out (eg only one alias in the sentence)
        self.e2e_entity_mask = torch.zeros((self.K * self.M, self.K * self.M))
        for i in range(self.M):
            self.e2e_entity_mask[
                i * self.K : (i + 1) * self.K, i * self.K : (i + 1) * self.K
            ] = 1.0
            # Must manually move this to the device as it's not part of a module...we can probably fix this
        self.e2e_entity_mask = self.e2e_entity_mask.masked_fill(
            (self.e2e_entity_mask == 1), float(-1e9)
        )

        # Track attention weights
        self.attention_weights = {}

        # Prediction layers: each stage except the last gets a prediction layer
        # Last layer's prediction head is added in slice heads
        disambig_task = nn.ModuleDict()
        for i in range(self.num_model_stages - 1):
            disambig_task[bootleg.utils.model_utils.get_stage_head_name(i)] = MLP(
                self.hidden_size, self.hidden_size, 1, self.num_fc_layers, self.dropout
            )
        self.predict_layers = {DISAMBIG: disambig_task}
        self.predict_layers = nn.ModuleDict(self.predict_layers)

    def forward(
        self,
        sent_embedding,
        sent_embedding_mask,
        entity_embedding,
        entity_embedding_mask,
        start_span_idx,
        end_span_idx,
        batch_on_the_fly_data,
    ):
        """Model forward.

        Args:
            sent_embedding: sentence embedding (B x N x L)
            sent_embedding_mask: sentence embedding mask (B x N)
            entity_embedding: entity embedding (B x M x K x H)
            entity_embedding_mask: entity embedding mask (B x M x K)
            start_span_idx: start mention index into sentence (B x M)
            end_span_idx: end mention index into sentence (B x M)
            batch_on_the_fly_data: batch on the fly dictionary with values (B x (M*K) x (M*K)) of KG adjacency matrices

        Returns: Dict of Dict of intermediate layer candidate scores (B x M x K),
                         Dict of all output entity embeddings from each KG matrix (B x M x K x H)
        """
        batch_size = sent_embedding.shape[0]
        out = {DISAMBIG: {}}

        # Create KG bias matrices for each kg bias key
        kg_bias_norms = {}
        for key in self.kg_bias_keys:
            kg_bias_norms[key] = (
                batch_on_the_fly_data[key]
                .float()
                .reshape(batch_size, self.M * self.K, self.M * self.K)
            )

        # get mention embedding
        # average words in mention; batch x M x dim
        mention_tensor_start = model_utils.select_alias_word_sent(
            start_span_idx, sent_embedding
        )
        mention_tensor_end = model_utils.select_alias_word_sent(
            end_span_idx, sent_embedding
        )
        mention_tensor = (mention_tensor_start + mention_tensor_end) / 2

        # reshape for alias attention where each mention attends to its K candidates
        # query = batch*M x 1 x dim, key = value = batch*M x K x dim
        # softmax(QK^T) -> batch*M x 1 x K
        # softmax(QK^T)V -> batch*M x 1 x dim
        mention_tensor = mention_tensor.reshape(
            batch_size * self.M, 1, self.hidden_size
        ).transpose(0, 1)

        # get sentence embedding; move batch to middle
        sent_tensor = sent_embedding.transpose(0, 1)

        # Resize the alias embeddings and the entity mask from B x M x K x D -> B x (M*K) x D
        entity_embedding = entity_embedding.contiguous().view(
            batch_size, self.M * self.K, self.hidden_size
        )
        entity_embedding = entity_embedding.transpose(0, 1)  # reshape for attention
        key_padding_mask_entities = entity_embedding_mask.contiguous().view(
            batch_size, self.M * self.K
        )
        key_padding_mask_entities_mention = entity_embedding_mask.contiguous().view(
            batch_size * self.M, self.K
        )
        # Mask of aliases; key_padding_mask_entities_mention of True means mask. We want to find aliases with all masked entities
        key_padding_mask_mentions = (
            torch.sum(~key_padding_mask_entities_mention, dim=-1) == 0
        )
        # Unmask these aliases to avoid nan in attention
        key_padding_mask_entities_mention[key_padding_mask_mentions] = False
        # Iterate through stages
        query_tensor = entity_embedding
        for stage_index in range(self.num_model_stages):
            # As we are adding a residual in the attention modules, we can make embs empty
            embs = []
            context_mat_dict = {}
            key_tensor_mention = (
                query_tensor.transpose(0, 1)
                .contiguous()
                .reshape(batch_size, self.M, self.K, self.hidden_size)
                .reshape(batch_size * self.M, self.K, self.hidden_size)
                .transpose(0, 1)
            )
            # ============================================================================
            # Phrase module: compute attention between entities and words
            # ============================================================================
            word_entity_attn_context, word_entity_attn_weights = self.attention_modules[
                f"stage_{stage_index}_entity_word"
            ](
                q=query_tensor,
                x=sent_tensor,
                key_mask=sent_embedding_mask,
                attn_mask=None,
            )
            # Add embeddings to be merged in the output
            embs.append(word_entity_attn_context)
            # Save the attention weights
            self.attention_weights[
                f"stage_{stage_index}_entity_word"
            ] = word_entity_attn_weights

            # ============================================================================
            # Co-occurrence module: compute self attention over entities
            # ============================================================================
            # Move entity mask to device
            # TODO: move to device in init?
            self.e2e_entity_mask = self.e2e_entity_mask.to(
                key_padding_mask_entities.device
            )

            entity_attn_context, entity_attn_weights = self.attention_modules[
                f"stage_{stage_index}_self_entity"
            ](
                x=query_tensor,
                key_mask=key_padding_mask_entities,
                attn_mask=self.e2e_entity_mask,
            )
            # Mask out MxK of single aliases, alias_indices is batch x M, mask is true when single alias
            non_null_aliases = (
                self.K
                - key_padding_mask_entities.reshape(batch_size, self.M, self.K).sum(-1)
            ) != 0
            entity_attn_post_mask = (
                (non_null_aliases.sum(1) == 1)
                .unsqueeze(1)
                .expand(batch_size, self.K * self.M)
                .transpose(0, 1)
            )
            entity_attn_post_mask = entity_attn_post_mask.unsqueeze(-1).expand_as(
                entity_attn_context
            )
            entity_attn_context = torch.where(
                entity_attn_post_mask,
                torch.zeros_like(entity_attn_context),
                entity_attn_context,
            )

            # Add embeddings to be merged in the output
            embs.append(entity_attn_context)
            # Save the attention weights
            self.attention_weights[
                f"stage_{stage_index}_self_entity"
            ] = entity_attn_weights

            # ============================================================================
            # Mention module: compute attention between entities and mentions
            # ============================================================================
            # output is 1 x batch*M x dim
            (
                mention_entity_attn_context,
                mention_entity_attn_weights,
            ) = self.attention_modules[f"stage_{stage_index}_mention_entity"](
                q=mention_tensor,
                x=key_tensor_mention,
                key_mask=key_padding_mask_entities_mention,
                attn_mask=None,
            )
            # key_padding_mask_mentions mentions have all padded candidates, meaning their row in the context matrix are all nan
            mention_entity_attn_context[key_padding_mask_mentions.unsqueeze(0)] = 0
            mention_entity_attn_context = (
                mention_entity_attn_context.expand(
                    self.K, batch_size * self.M, self.hidden_size
                )
                .transpose(0, 1)
                .reshape(batch_size, self.M * self.K, self.hidden_size)
                .transpose(0, 1)
            )
            # Add embeddings to be merged in the output
            embs.append(mention_entity_attn_context)
            # Save the attention weights
            self.attention_weights[
                f"stage_{stage_index}_mention_entity"
            ] = mention_entity_attn_weights

            # Combine module output
            context_matrix_nokg = self.combine_modules[f"stage_{stage_index}_combine"](
                embs
            )
            context_mat_dict[self.no_kg_key] = context_matrix_nokg.transpose(
                0, 1
            ).reshape(batch_size, self.M, self.K, self.hidden_size)
            # ============================================================================
            # KG module: add in KG connectivity bias
            # ============================================================================
            for key in self.kg_bias_keys:
                context_matrix_kg = torch.bmm(
                    kg_bias_norms[key], context_matrix_nokg.transpose(0, 1)
                ).transpose(0, 1)
                context_matrix_kg = (context_matrix_nokg + context_matrix_kg) / 2
                context_mat_dict[f"context_matrix_{key}"] = context_matrix_kg.transpose(
                    0, 1
                ).reshape(batch_size, self.M, self.K, self.hidden_size)

            if stage_index < self.num_model_stages - 1:
                score = model_utils.max_score_context_matrix(
                    context_mat_dict,
                    self.predict_layers[DISAMBIG][
                        bootleg.utils.model_utils.get_stage_head_name(stage_index)
                    ],
                )
                out[DISAMBIG][
                    f"{bootleg.utils.model_utils.get_stage_head_name(stage_index)}"
                ] = score

            # This will take the average of the context matrices that do not end in the key "_nokg"; if there are not kg bias terms, it will
            # select the context_matrix_nokg (as it's key, in this setting, will not end in _nokg)
            query_tensor = (
                model_utils.generate_final_context_matrix(
                    context_mat_dict, ending_key_to_exclude="_nokg"
                )
                .reshape(batch_size, self.M * self.K, self.hidden_size)
                .transpose(0, 1)
            )
        return {
            "intermed_scores": out,
            "ent_embs": context_mat_dict,
            "final_scores": None,
        }


class BERTNED(AttnNetwork):
    """NED Baseline model using BERT.

    Args:
        args: args
        entity_symbols: entity symbols
    """

    def __init__(self, args, entity_symbols):
        super(BERTNED, self).__init__(args, entity_symbols)
        self.dropout = args.train_config.dropout
        self.span_proj = MLP(
            input_size=2 * BERT_WORD_DIM,
            num_hidden_units=None,
            output_size=self.hidden_size,
            num_layers=1,
        )
        # Prediction layers
        disambig_task = nn.ModuleDict()
        disambig_task["final"] = MLP(
            self.hidden_size, self.hidden_size, 1, self.num_fc_layers, self.dropout
        )
        self.predict_layers = {DISAMBIG: disambig_task}
        self.predict_layers = nn.ModuleDict(self.predict_layers)

    def forward(
        self,
        sent_embedding,
        sent_embedding_mask,
        entity_embedding,
        entity_embedding_mask,
        start_span_idx,
        end_span_idx,
        batch_on_the_fly_data,
    ):
        """Model forward.

        Args:
            sent_embedding: sentence embedding (B x N x L)
            sent_embedding_mask: sentence embedding mask (B x N)
            entity_embedding: entity embedding (B x M x K x H)
            entity_embedding_mask: entity embedding mask (B x M x K)
            start_span_idx: start mention index into sentence (B x M)
            end_span_idx: end mention index into sentence (B x M)
            batch_on_the_fly_data: batch on the fly dictionary with values (B x (M*K) x (M*K)) of KG adjacency matrices

        Returns: Dict of Dict of intermediate output layer scores (will be empty for this model),
                 Output entity embeddings (B x M x K x H),
                 Candidate scores (B x M x K)
        """
        out = {DISAMBIG: {}}
        context_mat_dict = {}

        batch_size, M, K, emb_dim = entity_embedding.shape
        alias_start_idx_sent = start_span_idx
        alias_end_idx_sent = end_span_idx
        assert (
            emb_dim == self.hidden_size
        ), f"BERT NED requires the learned entity embedding dim be the same as the hidden size"
        assert alias_start_idx_sent.shape == alias_end_idx_sent.shape

        # Get alias words from sent embedding then cat and proj
        alias_start_word_tensor = model_utils.select_alias_word_sent(
            alias_start_idx_sent, sent_embedding
        )
        alias_end_word_tensor = model_utils.select_alias_word_sent(
            alias_end_idx_sent, sent_embedding
        )
        alias_pair_word_tensor = torch.cat(
            [alias_start_word_tensor, alias_end_word_tensor], dim=-1
        )
        alias_emb = (
            self.span_proj(alias_pair_word_tensor)
            .unsqueeze(2)
            .expand(batch_size, M, self.K, self.hidden_size)
        )
        alias_emb = (
            alias_emb.contiguous()
            .reshape((batch_size * M * self.K), self.hidden_size)
            .unsqueeze(1)
        )

        # entity_embedding_mask: if I don't have 30 candidates, use a mask to fill the rest of the
        # matrix for empty candidates
        entity_embedding_zeroed = torch.where(
            entity_embedding_mask.unsqueeze(-1),
            torch.zeros_like(entity_embedding),
            entity_embedding,
        )
        entity_embedding_tensor = (
            entity_embedding_zeroed.contiguous()
            .reshape((batch_size * M * self.K), self.hidden_size)
            .unsqueeze(-1)
        )

        # Performs batch wise dot produce across each dim=0 dimension
        score = (
            torch.bmm(alias_emb, entity_embedding_tensor)
            .unsqueeze(-1)
            .reshape(batch_size, M, self.K)
        )
        context_mat_dict[DISAMBIG] = entity_embedding_tensor.reshape(
            batch_size, M, self.K, self.hidden_size
        )
        return {
            "intermed_scores": out,
            "ent_embs": context_mat_dict,
            "final_scores": score,
        }
