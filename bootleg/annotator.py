"""Annotator"""
from collections import OrderedDict
import logging
import numpy as np
import os
import torch
import ujson

from bootleg.utils import data_utils, sentence_utils, eval_utils
from bootleg.utils.parser_utils import get_full_config
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.alias_entity_table import AliasEntityTable
from bootleg.symbols.constants import *
from bootleg.model import Model
from bootleg.extract_mentions import find_aliases_in_sentence_tag, get_all_aliases
from bootleg.utils.utils import import_class


class Annotator():
    """
    Annotator class: convenient wrapper of preprocessing and model eval to allow for
    annotating single sentences at a time for quick experimentation, e.g. in notebooks.
    """
    def __init__(self, config_args, device='cuda', max_alias_len=6,
        cand_map=None, threshold=0.0):
        self.args = config_args
        self.device = device
        self.entity_db =  EntitySymbols(os.path.join(self.args.data_config.entity_dir,
                                                     self.args.data_config.entity_map_dir),
                                        alias_cand_map_file=self.args.data_config.alias_cand_map)
        self.word_db = data_utils.load_wordsymbols(self.args.data_config, is_writer=True, distributed=False)
        self.model = self._load_model()
        self.max_alias_len = max_alias_len
        if cand_map is None:
            alias_map = self.entity_db._alias2qids
        else:
            alias_map = ujson.load(open(cand_map))
        self.all_aliases_trie = get_all_aliases(alias_map, logger=logging.getLogger())
        self.alias_table = AliasEntityTable(args=self.args, entity_symbols=self.entity_db)

        # minimum probability of prediction to return mention
        self.threshold = threshold

        self.batch_on_the_fly_embs = {}
        for emb in self.args.data_config.ent_embeddings:
            if 'batch_on_the_fly' in emb and emb['batch_on_the_fly'] is True:
                mod, load_class = import_class("bootleg.embeddings", emb.load_class)
                try:
                    self.batch_on_the_fly_embs[emb.key] = getattr(mod, load_class)(main_args=self.args,
                        emb_args=emb['args'], entity_symbols=self.entity_db, model_device=None, word_symbols=None)
                except AttributeError as e:
                    print(f'No prep method found for {emb.load_class} with error {e}')
                except Exception as e:
                    print("ERROR", e)

    def _load_model(self):
        model_state_dict = torch.load(self.args.run_config.init_checkpoint,
            map_location=lambda storage, loc: storage)['model']
        model = Model(args=self.args, model_device=self.device,
            entity_symbols=self.entity_db, word_symbols=self.word_db).to(self.device)
        # remove distributed naming if it exists
        if not self.args.run_config.distributed:
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                if 'module.' in k and k[:len('module.')] == 'module.':
                    name = k[len('module.'):]
                    new_state_dict[name] = v
            # we renamed all layers due to distributed naming
            if len(new_state_dict) == len(model_state_dict):
                model_state_dict = new_state_dict
            else:
                assert len(new_state_dict) == 0
        model.load_state_dict(model_state_dict, strict=True)
        # must set model in eval mode
        model.eval()
        return model

    def extract_mentions(self, text):
        found_aliases, found_spans = find_aliases_in_sentence_tag(text,
            self.all_aliases_trie, self.max_alias_len)
        return {'sentence': text,
                'aliases': found_aliases,
                'spans': found_spans,
                # we don't know the true QID
                'qids': ['Q-1'for i in range(len(found_aliases))],
                'anchor': [True for i in range(len(found_aliases))]}

    def set_threshold(self, value):
        self.threshold = value

    def label_mentions(self, text):
        sample = self.extract_mentions(text)
        idxs_arr, aliases_to_predict_per_split, spans_arr, phrase_tokens_arr = sentence_utils.split_sentence(
            max_aliases=self.args.data_config.max_aliases,
            phrase=sample['sentence'],
            spans=sample['spans'],
            aliases=sample['aliases'],
            aliases_seen_by_model=[i for i in range(len(sample['aliases']))],
            seq_len=self.args.data_config.max_word_token_len,
            word_symbols=self.word_db)
        aliases_arr = [[sample['aliases'][idx] for idx in idxs] for idxs in idxs_arr]
        qids_arr = [[sample['qids'][idx] for idx in idxs] for idxs in idxs_arr]
        word_indices_arr = [self.word_db.convert_tokens_to_ids(pt) for pt in phrase_tokens_arr]

        if len(idxs_arr) > 1:
            #TODO: support sentences that overflow due to long sequence length or too many mentions
            raise ValueError('Overflowing sentences not currently supported in Annotator')

        # iterate over each sample in the split
        for sub_idx in range(len(idxs_arr)):
            example_aliases = np.ones(self.args.data_config.max_aliases, dtype=np.int) * PAD_ID
            example_true_entities = np.ones(self.args.data_config.max_aliases) * PAD_ID
            example_aliases_locs_start = np.ones(self.args.data_config.max_aliases) * PAD_ID
            example_aliases_locs_end = np.ones(self.args.data_config.max_aliases) * PAD_ID

            aliases = aliases_arr[sub_idx]
            for mention_idx, alias in enumerate(aliases):
                # get aliases
                alias_trie_idx = self.entity_db.get_alias_idx(alias)
                alias_qids = np.array(self.entity_db.get_qid_cands(alias))
                example_aliases[mention_idx] = alias_trie_idx

                # alias_idx_pair
                span_idx = spans_arr[sub_idx][mention_idx]
                span_start_idx = int(span_idx.split(":")[0])
                span_end_idx = int(span_idx.split(":")[1])
                example_aliases_locs_start[mention_idx] = span_start_idx
                example_aliases_locs_end[mention_idx] = span_end_idx

            # get word indices
            word_indices = word_indices_arr[sub_idx]

            # entity indices from alias table (these are the candidates)
            entity_indices = self.alias_table(example_aliases)

            # all CPU embs have to retrieved on the fly
            batch_on_the_fly_data = {}
            for emb_name, emb in self.batch_on_the_fly_embs.items():
                batch_on_the_fly_data[emb_name] = torch.tensor(emb.batch_prep(example_aliases,
                                                                              entity_indices), device=self.device)


            outs, entity_pack, _ = self.model(
                alias_idx_pair_sent = [torch.tensor(example_aliases_locs_start, device=self.device).unsqueeze(0),
                                       torch.tensor(example_aliases_locs_end, device=self.device).unsqueeze(0)],
                word_indices = torch.tensor([word_indices], device=self.device),
                alias_indices = torch.tensor(example_aliases, device=self.device).unsqueeze(0),
                entity_indices = torch.tensor(entity_indices, device=self.device).unsqueeze(0),
                batch_prepped_data = {},
                batch_on_the_fly_data = batch_on_the_fly_data)

            entity_cands = eval_utils.map_aliases_to_candidates(self.args.data_config.train_in_candidates,
                                                                self.entity_db,
                                                                aliases)
            # recover predictions
            probs = torch.exp(eval_utils.masked_class_logsoftmax(pred=outs[DISAMBIG][FINAL_LOSS],
                                                                 mask=~entity_pack.mask, dim=2))
            max_probs, max_probs_indices = probs.max(2)

            pred_cands = []
            pred_probs = []
            titles = []
            for alias_idx in range(len(aliases)):
                pred_idx = max_probs_indices[0][alias_idx]
                pred_prob = max_probs[0][alias_idx].item()
                pred_qid = entity_cands[alias_idx][pred_idx]
                if pred_prob > self.threshold:
                    pred_cands.append(pred_qid)
                    pred_probs.append(pred_prob)
                    titles.append(self.entity_db.get_title(pred_qid) if pred_qid != 'NC' else 'NC')

            return pred_cands, pred_probs, titles
