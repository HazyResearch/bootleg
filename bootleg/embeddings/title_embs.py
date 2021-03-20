"""Title embeddings."""
import logging
import os
import time

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from bootleg import log_rank_0_debug
from bootleg.embeddings import EntityEmb
from bootleg.symbols.constants import BERT_WORD_DIM, MAX_BERT_TOKEN_LEN
from bootleg.utils import data_utils, model_utils, utils
from bootleg.utils.embedding_utils import get_max_candidates

logger = logging.getLogger(__name__)


def load_tokenizer(data_config):
    return BertTokenizer.from_pretrained(
        data_config.word_embedding.bert_model,
        do_lower_case=True
        if "uncased" in data_config.word_embedding.bert_model
        else False,
        cache_dir=data_config.word_embedding.cache_dir,
    )


class TitleEmb(EntityEmb):
    """Title entity embeddings class.

    As the title embedding requires being sent through the BERT encoder to be contextualized,
    these get added to the BertEncoder class (see bert_encoder.py).

    This class therefore functions as the pre and postprocessing step to this BERT forward call.
    Given a batch of entity candidates, the forward is responsable for retrieving the word token ids for each entity.
    Then, after the forward, it can do any postprocessing on the outs, e.g., projection layer.

    Add the following to your config to use::

        ent_embeddings:
           - key: title
             load_class: TitleEmb
             freeze: false # Freeze the projection layer or not
             send_through_bert: true
             args:
               proj: 128
               requires_grad: false

    Args:
        main_args: main args
        emb_args: specific embedding args
        entity_symbols: entity symbols
        key: unique embedding key
        cpu: bool of if one cpu or not
        normalize: bool if normalize embeddings or not
        dropout1d_perc: 1D dropout percent
        dropout2d_perc: 2D dropout percent
    """

    def __init__(
        self,
        main_args,
        emb_args,
        entity_symbols,
        key,
        cpu,
        normalize,
        dropout1d_perc,
        dropout2d_perc,
    ):
        super(TitleEmb, self).__init__(
            main_args=main_args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )
        allowable_keys = {"proj", "requires_grad"}
        correct, bad_key = utils.assert_keys_in_dict(allowable_keys, emb_args)
        if not correct:
            raise ValueError(f"The key {bad_key} is not in {allowable_keys}")
        self.orig_dim = BERT_WORD_DIM
        self.merge_func = self.average_titles
        self.M = main_args.data_config.max_aliases
        self.K = get_max_candidates(entity_symbols, main_args.data_config)
        self._dim = main_args.model_config.hidden_size
        if "proj" in emb_args:
            self._dim = emb_args.proj
        self.requires_grad = True
        if "requires_grad" in emb_args:
            self.requires_grad = emb_args.requires_grad
        self.title_proj = torch.nn.Linear(self.orig_dim, self._dim)
        log_rank_0_debug(
            logger,
            f'Setting the "proj" parameter to {self._dim} and the "requires_grad" parameter to {self.requires_grad}',
        )

        (
            entity2titleid_table,
            entity2titlemask_table,
            entity2tokentypeid_table,
        ) = self.prep(
            data_config=main_args.data_config,
            entity_symbols=entity_symbols,
        )
        self.register_buffer(
            "entity2titleid_table", entity2titleid_table, persistent=False
        )
        self.register_buffer(
            "entity2titlemask_table", entity2titlemask_table, persistent=False
        )
        self.register_buffer(
            "entity2tokentypeid_table", entity2tokentypeid_table, persistent=False
        )

    @classmethod
    def prep(cls, data_config, entity_symbols):
        """Prep the title data.

        Args:
            data_config: data config
            entity_symbols: entity symbols

        Returns: torch tensor EID to title token IDs, EID to title token mask, EID to title token type ID (for BERT)
        """
        prep_dir = data_utils.get_emb_prep_dir(data_config)
        prep_file_token_ids = os.path.join(
            prep_dir, f"title_token_ids_{data_config.word_embedding.bert_model}.pt"
        )
        prep_file_attn_mask = os.path.join(
            prep_dir, f"title_attn_mask_{data_config.word_embedding.bert_model}.pt"
        )
        prep_file_token_type_ids = os.path.join(
            prep_dir, f"title_token_type_ids_{data_config.word_embedding.bert_model}.pt"
        )
        utils.ensure_dir(os.path.dirname(prep_file_token_ids))
        log_rank_0_debug(
            logger, f"Looking for title table mapping in {prep_file_token_ids}"
        )
        if (
            not data_config.overwrite_preprocessed_data
            and os.path.exists(prep_file_token_ids)
            and os.path.exists(prep_file_attn_mask)
            and os.path.exists(prep_file_token_type_ids)
        ):
            log_rank_0_debug(
                logger, f"Loading existing title table from {prep_file_token_ids}"
            )
            start = time.time()
            entity2titleid = torch.load(prep_file_token_ids)
            entity2titlemask = torch.load(prep_file_attn_mask)
            entity2tokentypeid = torch.load(prep_file_token_type_ids)
            log_rank_0_debug(
                logger,
                f"Loaded existing title table in {round(time.time() - start, 2)}s",
            )
        else:
            start = time.time()
            log_rank_0_debug(logger, f"Loading tokenizer")
            tokenizer = load_tokenizer(data_config)
            (
                entity2titleid,
                entity2titlemask,
                entity2tokentypeid,
            ) = cls.build_title_table(
                tokenizer=tokenizer, entity_symbols=entity_symbols
            )
            torch.save(entity2titleid, prep_file_token_ids)
            torch.save(entity2titlemask, prep_file_attn_mask)
            torch.save(entity2tokentypeid, prep_file_token_type_ids)
            log_rank_0_debug(
                logger,
                f"Finished building and saving title table in {round(time.time() - start, 2)}s.",
            )
        return entity2titleid, entity2titlemask, entity2tokentypeid

    @classmethod
    def build_title_table(cls, tokenizer, entity_symbols):
        """Builds the table from EID to title IDs.

        Args:
            tokenizer: tokenizer
            entity_symbols: entity symbols

        Returns: torch tensor EID to title token IDs, EID to title token mask, EID to title token type ID (for BERT)
        """
        titles = []
        tokenized_titles = []
        attention_masks = []
        eids = []
        max_token_size = 0
        token_type_id = 1
        for q in tqdm(
            entity_symbols.get_all_qids(),
            total=len(entity_symbols.get_all_titles()),
            desc="Itearting over entities",
        ):
            cur_title = entity_symbols.get_title(q)
            eids.append(entity_symbols.get_eid(q))
            titles.append(cur_title)
            batch_inputs = tokenizer(
                cur_title, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = batch_inputs["input_ids"][:MAX_BERT_TOKEN_LEN]
            attention_mask = batch_inputs["attention_mask"][:MAX_BERT_TOKEN_LEN]
            max_token_size = max(inputs.shape[-1], max_token_size)
            tokenized_titles.extend(inputs)
            attention_masks.extend(attention_mask)

        assert len(eids) == len(titles)
        entity2titleid = torch.zeros(
            entity_symbols.num_entities_with_pad_and_nocand, max_token_size
        )
        entity2attnmask = torch.zeros(
            entity_symbols.num_entities_with_pad_and_nocand, max_token_size
        )
        entity2tokentypeid = torch.zeros(
            entity_symbols.num_entities_with_pad_and_nocand, max_token_size
        )

        for i in tqdm(range(0, len(tokenized_titles)), desc="Building table"):
            eid = eids[i]
            tokens = tokenized_titles[i]
            mask = attention_masks[i]
            entity2titleid[eid, : len(tokens)] = tokens
            entity2attnmask[eid, : len(mask)] = mask
            entity2tokentypeid[eid, : len(mask)] = torch.ones(len(mask)) * token_type_id
        return entity2titleid.long(), entity2attnmask.long(), entity2tokentypeid.long()

    def get_dim(self):
        return self._dim

    def average_titles(self, subset_mask, title_emb):
        """Take the average title embedding, respecting unk embeddings.

        Args:
            subset_mask: mask of unk embeddings (True means we remove)
            title_emb: title embedding

        Returns: average title embedding
        """
        # subset_mask is downstream Pytorch mask where True means remove. Averages requires True to mean we keep
        embs = model_utils.selective_avg(mask=~subset_mask, embeds=title_emb)
        return embs

    def get_subset_title_embs(self, title_token_ids, title_mask, title_token_type):
        """Gets the subset of title embeddings that are for non-null entities
        as an optimization.

        Args:
            title_token_ids: token IDs
            title_mask: token masks
            title_token_type: token type IDs

        Returns: subset token IDs, subset mask, subset token type IDs,
                 index of kept rows in original data (for reconstruction)
        """
        # Bert takes 2-D input so reshape
        flat_title_token_ids = title_token_ids.reshape(-1, title_token_ids.shape[-1])
        flat_mask = title_mask.reshape(-1, title_token_ids.shape[-1])
        flat_token_types = title_token_type.reshape(-1, title_token_ids.shape[-1])
        # only get the subset of words that you need (ignore 0 mask ids) -- remember,
        # this is BERT mask so 1 means what to pay attention to (all 0s is ignore)

        # subset_idx = flat_mask.sum(-1) != 0
        # subset_title_token_ids = flat_title_token_ids[subset_idx]
        # subset_mask = flat_mask[subset_idx]
        # subset_token_types = flat_token_types[subset_idx]
        subset_idx = flat_mask.sum(-1).nonzero().squeeze(-1)
        subset_title_token_ids = flat_title_token_ids.index_select(0, subset_idx)
        subset_mask = flat_mask.index_select(0, subset_idx)
        subset_token_types = flat_token_types.index_select(0, subset_idx)
        return subset_title_token_ids, subset_mask, subset_token_types, subset_idx

    def forward(self, entity_cand_eid, batch_on_the_fly_data):
        """Model forward.

        Args:
            entity_cand_eid:  entity candidate EIDs (B x M x K)
            batch_on_the_fly_data: dict of batch on the fly embeddings

        Returns: B x M x K x <num_token_ids> tensor of word token IDs for each candidate title
        """
        title_token_ids = self.entity2titleid_table[entity_cand_eid]
        title_mask = self.entity2titlemask_table[entity_cand_eid]
        title_token_type = self.entity2tokentypeid_table[entity_cand_eid]
        (
            subset_title_token_ids,
            subset_mask,
            subset_token_types,
            subset_idx,
        ) = self.get_subset_title_embs(title_token_ids, title_mask, title_token_type)
        return (
            subset_title_token_ids,
            subset_mask,
            subset_token_types,
            self.requires_grad,
            subset_idx,
            entity_cand_eid.shape[0],
        )

    def postprocess_embedding(self, title_emb, subset_mask, subset_idx, batch_size):
        """This gets called after the forward for the BERT encoder. This
        reconstructs the title embeddings after being subsetted and projects
        them.

        Args:
            title_emb: title embedding after BERT
            subset_mask: subset title mask
            subset_idx: subset index back into original embedding matrix
            batch_size: batch size

        Returns: batch x M x K x dim title embedding
        """
        assert type(batch_size) is int
        # import time
        # st = time.time()
        # scatter averaged embs to original locations
        final_title_emb = torch.empty(
            batch_size * self.M * self.K, self._dim, device=title_emb.device
        ).fill_(0)
        # print(f"EMTPY", time.time()-st)
        # st = time.time()
        title_emb = self.title_proj(title_emb)
        # print(f"PROJ", time.time()-st)
        # st = time.time()
        average_title_emb = self.merge_func(subset_mask, title_emb)
        # print(f"MERGE", time.time()-st)
        # st = time.time()
        average_title_emb = self.normalize_and_dropout_emb(average_title_emb)
        # print(f"NORM", time.time()-st)
        # st = time.time()
        final_title_emb = final_title_emb.index_copy_(0, subset_idx, average_title_emb)
        # final_title_emb[subset_idx] = average_title_emb
        # print(f"FILL", time.time()-st)
        # st = time.time()
        final_title_emb = final_title_emb.reshape(batch_size, self.M, self.K, self._dim)
        # print(f"RESHAPE", time.time()-st)
        # st = time.time()
        assert list(final_title_emb.shape) == [batch_size, self.M, self.K, self._dim]
        return final_title_emb
