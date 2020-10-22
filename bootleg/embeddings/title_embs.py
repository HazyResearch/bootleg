"""Title embeddings"""
import os
import time
import torch

from bootleg.embeddings import EntityEmb
from bootleg.layers.layers import MLP, SoftAttn
from bootleg.symbols.constants import CLS_BERT, SEP_BERT
from bootleg.utils import logging_utils, data_utils, model_utils
from bootleg.utils.model_utils import selective_avg

class AvgTitleEmb(EntityEmb):
    """
    Title embedding class. In the forward, it get the title embedding over each entity's title from the BERT word embeddings. This is not
    sent through the BERT encoder for efficiency.

    Attributes:
        requires_grad_title: if gradients should be computed from the forward of the word embedding layer.
    """

    def __init__(self, main_args, emb_args, model_device, entity_symbols, word_symbols, word_emb, key):
        super(AvgTitleEmb, self).__init__(main_args=main_args, emb_args=emb_args, model_device=model_device,
                                          entity_symbols=entity_symbols, word_symbols=word_symbols, word_emb=word_emb,
                                          key=key)
        # self.word_emb = word_emb
        self.logger = logging_utils.get_logger(main_args)
        self.model_device = model_device
        self.word_emb = word_emb
        self.word_symbols = word_symbols
        self.orig_dim = word_emb.get_dim()
        self.merge_func = self.average_titles
        self.normalize = True
        self._dim = main_args.model_config.hidden_size
        self.requires_grad_title = word_emb.requires_grad
        if "freeze_word_emb_for_titles" in emb_args:
            self.requires_grad_title = not emb_args.freeze_word_emb_for_titles
        assert not self.requires_grad_title or word_emb.requires_grad,\
            "Inconsistent Args: You have to not freeze word embeddings for titles but freeze word embeddings"
        self.entity2titleid_table = self.prep(main_args=main_args, emb_args=emb_args, entity_symbols=entity_symbols,
            word_symbols=word_symbols, log_func=self.logger.debug)
        self.entity2titleid_table = self.entity2titleid_table.to(model_device)

    @classmethod
    def prep(cls, main_args, emb_args, entity_symbols, word_symbols, log_func=print):
        prep_dir = data_utils.get_emb_prep_dir(main_args)
        prep_file = os.path.join(prep_dir,
            f'avg_title_table_{main_args.data_config.word_embedding.word_symbols}.pt')
        if (not main_args.data_config.overwrite_preprocessed_data
            and os.path.exists(prep_file)):
            log_func(f'Loading existing title table from {prep_file}')
            start = time.time()
            entity2titleid_table = torch.load(prep_file)
            log_func(f'Loaded existing title table in {round(time.time() - start, 2)}s')
        else:
            start = time.time()
            log_func(f'Building title table')
            entity2titleid_table = cls.build_title_table(word_symbols=word_symbols,
            entity_symbols=entity_symbols)
            torch.save(entity2titleid_table, prep_file)
            log_func(f"Finished building and saving title table in {round(time.time() - start, 2)}s.")
        return entity2titleid_table

    @classmethod
    def build_title_table(cls, word_symbols, entity_symbols):
        """Builds the table of the word indices associated with each title."""
        max_title_size = max(len(word_symbols.tokenize(t)) for t in entity_symbols.get_all_titles())
        if word_symbols.is_bert:
            # add the CLS and SEP tags
            max_title_size += 2
            assert max_title_size < 512
        entity2titleid_table = torch.ones(entity_symbols.num_entities_with_pad_and_nocand, max_title_size) * word_symbols.pad_id
        for qid in entity_symbols.get_all_qids():
            title = entity_symbols.get_title(qid)
            eid = entity_symbols.get_eid(qid)
            tokens = word_symbols.tokenize(title)
            if word_symbols.is_bert:
                # add the CLS and SEP tags
                tokens = [CLS_BERT] + tokens + [SEP_BERT]
            entity2titleid_table[eid, :len(tokens)] = torch.tensor(word_symbols.convert_tokens_to_ids(tokens))
        return entity2titleid_table.long()

    def freeze_params(self):
        """Override freeze params from base_emb. As the only named paramters for the title embedding are the word embedding,
        we don't want to freeze those here. The word embedding will get frozen elsewhere."""
        return

    def unfreeze_params(self):
        return

    def average_titles(self, entity_package, subset_idx, batch_title_ids, mask_subset, batch_title_emb, sent_emb):
        # mask of True means it's valid so we invert
        embs = selective_avg(ids=batch_title_ids, mask=~mask_subset, embeds=batch_title_emb)
        return embs

    def get_subset_title_embs(self, word_title_ids):
        """Gets the subset of title embeddings that are for non-null entities as an optimizatoion"""
        max_title_size = self.entity2titleid_table.shape[1]
        # Mask of True means discard
        mask = (word_title_ids == self.word_symbols.pad_id) | \
               ((self.word_symbols.unk_id is not None) and (word_title_ids == self.word_symbols.unk_id))
        # bert doesn't take 4-D input so reshape input
        word_title_ids = word_title_ids.reshape(-1, max_title_size)
        flat_mask = mask.reshape(-1, max_title_size)
        # only get the subset of words that you need (ignore null titles)
        subset_idx = (flat_mask.shape[-1] - flat_mask.sum(-1)) != 0
        word_title_ids_subset = word_title_ids[subset_idx]
        mask_subset = flat_mask[subset_idx]
        embs_subset = self.word_emb(word_title_ids_subset, requires_grad=self.requires_grad_title).tensor
        return subset_idx, word_title_ids_subset, mask_subset, embs_subset

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        batch, M, K = entity_package.tensor.shape
        # We want to limit the amount of data needing to be averaged so subselect embeddings based on non-null entities
        # We reconstruct the embeddings
        word_title_ids = self.entity2titleid_table[entity_package.tensor.long()]
        batch_title_emb = torch.zeros(batch, M, K, self.orig_dim).to(self.model_device)
        subset_idx, word_title_ids_subset, mask_subset, embs_subset = self.get_subset_title_embs(word_title_ids)
        embs_subset = self.merge_func(entity_package, subset_idx, word_title_ids_subset, mask_subset, embs_subset, sent_emb)
        # scatter averaged embs to original locations
        batch_title_emb[subset_idx.reshape(batch, M, K)] = embs_subset
        assert list(batch_title_emb.shape) == [batch, M, K, self.orig_dim]
        res = self._package(tensor=batch_title_emb,
                            pos_in_sent=entity_package.pos_in_sent,
                            alias_indices=entity_package.alias_indices,
                            mask=entity_package.mask)
        return res

    def get_dim(self):
        return self.orig_dim

class AvgTitleEmbProj(AvgTitleEmb):
    """Class that projects the title embedding to the hidden dimension before being added to the entity payload embedding."""
    def __init__(self, main_args, emb_args, model_device, entity_symbols, word_symbols, word_emb, key):
        super(AvgTitleEmbProj, self).__init__(main_args=main_args, emb_args=emb_args, model_device=model_device,
                                          entity_symbols=entity_symbols, word_symbols=word_symbols, word_emb=word_emb,
                                          key=key)
        self.proj = MLP(input_size=self.orig_dim,
                        num_hidden_units=0, output_size=self._dim,
                        num_layers=1, dropout=0, residual=False, activation=None)

    def freeze_params(self):
        """Only freeze the projection layer, not the word embeddings."""
        for name, param in self.named_parameters():
            if 'proj' in name:
                param.requires_grad = False
                self.logger.debug(f'Freezing {name}')

    def forward(self, entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb):
        res = super(AvgTitleEmbProj, self).forward(entity_package, batch_prepped_data, batch_on_the_fly_data, sent_emb).tensor
        res = self.proj(res)
        res = self._package(tensor=res,
                            pos_in_sent=entity_package.pos_in_sent,
                            alias_indices=entity_package.alias_indices,
                            mask=entity_package.mask)
        return res

    def get_dim(self):
        return self._dim