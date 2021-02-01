"""Annotator"""
from collections import OrderedDict
import logging
import numpy as np
import os
import torch
import ujson
from tqdm import tqdm
import logging

from bootleg.utils import data_utils, sentence_utils, eval_utils
from bootleg.utils.parser_utils import get_full_config
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.alias_entity_table import AliasEntityTable
from bootleg.symbols.constants import *
from bootleg.model import Model
from bootleg.extract_mentions import find_aliases_in_sentence_tag, get_all_aliases
from bootleg.utils.utils import import_class

logger = logging.getLogger(__name__)

class Annotator():
    """
    Annotator class: convenient wrapper of preprocessing and model eval to allow for
    annotating single sentences at a time for quick experimentation, e.g. in notebooks.
    """

    def __init__(self, config_args, device='cuda', max_alias_len=6,
                 cand_map=None, threshold=0.0):
        self.args = config_args
        self.device = device
        logger.info("Reading entity database")
        self.entity_db = EntitySymbols(os.path.join(self.args.data_config.entity_dir,
                                                    self.args.data_config.entity_map_dir),
                                       alias_cand_map_file=self.args.data_config.alias_cand_map)
        logger.info("Reading word tokenizers")
        self.word_db = data_utils.load_wordsymbols(self.args.data_config, is_writer=True, distributed=False)
        logger.info("Loading model")
        self.model = self._load_model()
        self.max_alias_len = max_alias_len
        if cand_map is None:
            alias_map = self.entity_db._alias2qids
        else:
            logger.info(f"Loading candidate map")
            alias_map = ujson.load(open(cand_map))
        self.all_aliases_trie = get_all_aliases(alias_map, logger=logging.getLogger())
        logger.info("Reading in alias table")
        self.alias_table = AliasEntityTable(args=self.args, entity_symbols=self.entity_db)

        # minimum probability of prediction to return mention
        self.threshold = threshold

        # get batch_on_the_fly embeddings _and_ the batch_prep embeddings
        self.batch_on_the_fly_embs = {}
        for i, emb in enumerate(self.args.data_config.ent_embeddings):
            if 'batch_prep' in emb and emb['batch_prep'] is True:
                self.args.data_config.ent_embeddings[i]['batch_on_the_fly'] = True
                del self.args.data_config.ent_embeddings[i]['batch_prep']
            if 'batch_on_the_fly' in emb and emb['batch_on_the_fly'] is True:
                mod, load_class = import_class("bootleg.embeddings", emb.load_class)
                try:
                    self.batch_on_the_fly_embs[emb.key] = getattr(mod, load_class)(main_args=self.args,
                                                                                   emb_args=emb['args'],
                                                                                   entity_symbols=self.entity_db,
                                                                                   model_device=None, word_symbols=None)
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
        found_aliases, found_spans = find_aliases_in_sentence_tag(text, self.all_aliases_trie, self.max_alias_len)
        return {'sentence': text,
                'aliases': found_aliases,
                'spans': found_spans,
                # we don't know the true QID
                'qids': ['Q-1' for i in range(len(found_aliases))],
                'gold': [True for i in range(len(found_aliases))]}

    def set_threshold(self, value):
        self.threshold = value

    def label_mentions(self, text_list):
        if type(text_list) is str:
            text_list = [text_list]
        else:
            assert type(text_list) is list and len(text_list) > 0 and type(
                text_list[0]) is str, f"We only accept inputs of strings and lists of strings"

        ebs = self.args.run_config.eval_batch_size
        total_start_exs = 0
        total_final_exs = 0
        dropped_by_thresh = 0

        final_char_spans = []

        batch_example_aliases = []
        batch_example_aliases_locs_start = []
        batch_example_aliases_locs_end = []
        batch_example_alias_list_pos = []
        batch_example_true_entities = []
        batch_word_indices = []
        batch_spans_arr = []
        batch_aliases_arr = []
        batch_idx_unq = []
        batch_subsplit_idx = []
        for idx_unq, text in tqdm(enumerate(text_list), desc="Prepping data", total=len(text_list)):
            sample = self.extract_mentions(text)
            total_start_exs += len(sample['aliases'])
            char_spans = self.get_char_spans(sample['spans'], text)

            final_char_spans.append(char_spans)

            idxs_arr, aliases_to_predict_per_split, spans_arr, phrase_tokens_arr, pos_idxs = sentence_utils.split_sentence(
                max_aliases=self.args.data_config.max_aliases,
                phrase=sample['sentence'],
                spans=sample['spans'],
                aliases=sample['aliases'],
                aliases_seen_by_model=[i for i in range(len(sample['aliases']))],
                seq_len=self.args.data_config.max_word_token_len,
                word_symbols=self.word_db)
            aliases_arr = [[sample['aliases'][idx] for idx in idxs] for idxs in idxs_arr]
            old_spans_arr = [[sample['spans'][idx] for idx in idxs] for idxs in idxs_arr]
            qids_arr = [[sample['qids'][idx] for idx in idxs] for idxs in idxs_arr]
            word_indices_arr = [self.word_db.convert_tokens_to_ids(pt) for pt in phrase_tokens_arr]
            # iterate over each sample in the split

            for sub_idx in range(len(idxs_arr)):
                # ====================================================
                # GENERATE MODEL INPUTS
                # ====================================================
                aliases_to_predict_arr = aliases_to_predict_per_split[sub_idx]
                assert len(aliases_to_predict_arr) >= 0, f'There are no aliases to predict for an example. This should not happen at this point.'
                assert len(aliases_arr[
                               sub_idx]) <= self.args.data_config.max_aliases, f'Each example should have no more that {self.args.data_config.max_aliases} max aliases. {sample} does.'

                example_aliases = np.ones(self.args.data_config.max_aliases) * PAD_ID
                example_aliases_locs_start = np.ones(self.args.data_config.max_aliases) * PAD_ID
                example_aliases_locs_end = np.ones(self.args.data_config.max_aliases) * PAD_ID
                example_alias_list_pos = np.ones(self.args.data_config.max_aliases) * PAD_ID
                example_true_entities = np.ones(self.args.data_config.max_aliases) * PAD_ID

                for mention_idx, alias in enumerate(aliases_arr[sub_idx]):
                    span_start_idx, span_end_idx = spans_arr[sub_idx][mention_idx]
                    # generate indexes into alias table.
                    alias_trie_idx = self.entity_db.get_alias_idx(alias)
                    alias_qids = np.array(self.entity_db.get_qid_cands(alias))
                    if not qids_arr[sub_idx][mention_idx] in alias_qids:
                        # assert not data_args.train_in_candidates
                        if not self.args.data_config.train_in_candidates:
                            # set class label to be "not in candidate set"
                            true_entity_idx = 0
                        else:
                            true_entity_idx = -2
                    else:
                        # Here we are getting the correct class label for training.
                        # Our training is "which of the max_entities entity candidates is the right one (class labels 1 to max_entities) or is it none of these (class label 0)".
                        # + (not discard_noncandidate_entities) is to ensure label 0 is reserved for "not in candidate set" class
                        true_entity_idx = np.nonzero(alias_qids == qids_arr[sub_idx][mention_idx])[0][0] + (
                            not self.args.data_config.train_in_candidates)
                    example_aliases[mention_idx] = alias_trie_idx
                    example_aliases_locs_start[mention_idx] = span_start_idx
                    # The span_idxs are [start, end). We want [start, end]. So subtract 1 from end idx.
                    example_aliases_locs_end[mention_idx] = span_end_idx - 1
                    example_alias_list_pos[mention_idx] = idxs_arr[sub_idx][mention_idx]
                    # leave as -1 if it's not an alias we want to predict; we get these if we split a sentence and need to only predict subsets
                    if mention_idx in aliases_to_predict_arr:
                        example_true_entities[mention_idx] = true_entity_idx

                # get word indices
                word_indices = word_indices_arr[sub_idx]

                batch_example_aliases.append(example_aliases)
                batch_example_aliases_locs_start.append(example_aliases_locs_start)
                batch_example_aliases_locs_end.append(example_aliases_locs_end)
                batch_example_alias_list_pos.append(example_alias_list_pos)
                batch_example_true_entities.append(example_true_entities)
                batch_word_indices.append(word_indices)
                batch_aliases_arr.append(aliases_arr[sub_idx])
                # Add the orginal sample spans because spans_arr is w.r.t BERT subword token
                batch_spans_arr.append(old_spans_arr[sub_idx])
                batch_idx_unq.append(idx_unq)
                batch_subsplit_idx.append(sub_idx)

        batch_example_aliases = torch.tensor(batch_example_aliases).long()
        batch_example_aliases_locs_start = torch.tensor(batch_example_aliases_locs_start, device=self.device)
        batch_example_aliases_locs_end = torch.tensor(batch_example_aliases_locs_end, device=self.device)
        batch_example_true_entities = torch.tensor(batch_example_true_entities, device=self.device)
        batch_word_indices = torch.tensor(batch_word_indices, device=self.device)

        final_pred_cands = [[] for _ in range(len(text_list))]
        final_all_cands = [[] for _ in range(len(text_list))]
        final_cand_probs = [[] for _ in range(len(text_list))]
        final_pred_probs = [[] for _ in range(len(text_list))]
        final_titles = [[] for _ in range(len(text_list))]
        final_spans = [[] for _ in range(len(text_list))]
        final_aliases = [[] for _ in range(len(text_list))]
        for b_i in tqdm(range(0, batch_example_aliases.shape[0], ebs), desc="Evaluating model"):
            # entity indices from alias table (these are the candidates)
            batch_entity_indices = self.alias_table(batch_example_aliases[b_i:b_i + ebs])

            # all CPU embs have to retrieved on the fly
            batch_on_the_fly_data = {}
            for emb_name, emb in self.batch_on_the_fly_embs.items():
                batch_prep = []
                for j in range(b_i, min(b_i + ebs, batch_example_aliases.shape[0])):
                    batch_prep.append(emb.batch_prep(batch_example_aliases[j], batch_entity_indices[j - b_i]))
                batch_on_the_fly_data[emb_name] = torch.tensor(batch_prep, device=self.device)

            alias_idx_pair_sent = [batch_example_aliases_locs_start[b_i:b_i + ebs], batch_example_aliases_locs_end[b_i:b_i + ebs]]
            word_indices = batch_word_indices[b_i:b_i + ebs]
            alias_indices = batch_example_aliases[b_i:b_i + ebs]
            entity_indices = torch.tensor(batch_entity_indices, device=self.device)

            outs, entity_pack, _ = self.model(
                alias_idx_pair_sent=alias_idx_pair_sent,
                word_indices=word_indices,
                alias_indices=alias_indices,
                entity_indices=entity_indices,
                batch_prepped_data={},
                batch_on_the_fly_data=batch_on_the_fly_data)

            # ====================================================
            # EVALUATE MODEL OUTPUTS
            # ====================================================

            final_loss_vals = outs[DISAMBIG][FINAL_LOSS]
            # recover predictions
            probs = torch.exp(eval_utils.masked_class_logsoftmax(pred=final_loss_vals,
                                                                 mask=~entity_pack.mask, dim=2))
            max_probs, max_probs_indices = probs.max(2)
            for ex_i in range(final_loss_vals.shape[0]):
                idx_unq = batch_idx_unq[b_i + ex_i]
                subsplit_idx = batch_subsplit_idx[b_i + ex_i]
                entity_cands = eval_utils.map_aliases_to_candidates(self.args.data_config.train_in_candidates,
                                                                    self.entity_db,
                                                                    batch_aliases_arr[b_i + ex_i])

                # batch size is 1 so we can reshape
                probs_ex = probs[ex_i].detach().cpu().numpy().reshape(self.args.data_config.max_aliases, probs.shape[2])
                for alias_idx, true_entity_pos_idx in enumerate(batch_example_true_entities[b_i + ex_i]):
                    if true_entity_pos_idx != PAD_ID:
                        pred_idx = max_probs_indices[ex_i][alias_idx]
                        pred_prob = max_probs[ex_i][alias_idx].item()
                        all_cands = entity_cands[alias_idx]
                        pred_qid = all_cands[pred_idx]
                        if pred_prob > self.threshold:
                            final_all_cands[idx_unq].append(all_cands)
                            final_cand_probs[idx_unq].append(probs_ex[alias_idx])
                            final_pred_cands[idx_unq].append(pred_qid)
                            final_pred_probs[idx_unq].append(pred_prob)
                            final_aliases[idx_unq].append(batch_aliases_arr[b_i + ex_i][alias_idx])
                            final_spans[idx_unq].append(batch_spans_arr[b_i + ex_i][alias_idx])
                            final_titles[idx_unq].append(self.entity_db.get_title(pred_qid) if pred_qid != 'NC' else 'NC')
                            total_final_exs += 1
                        else:
                            dropped_by_thresh += 1
        assert total_final_exs + dropped_by_thresh == total_start_exs, f"Something went wrong and we have predicted fewer mentions than extracted. Start {total_start_exs}, Out {total_final_exs}, No cand {dropped_by_thresh}"
        return final_pred_cands, final_pred_probs, final_titles, final_all_cands, final_cand_probs, final_spans, final_aliases

    def get_char_spans(self, spans, text):
        query_toks = text.split()
        char_spans = []
        for span in spans:
            space_btwn_toks = len(' '.join(query_toks[0:span[0] + 1])) - \
                              len(' '.join(query_toks[0:span[0]])) - \
                              len(
                query_toks[span[0]])
            char_b = len(' '.join(query_toks[0:span[0]])) + space_btwn_toks
            char_e = char_b + len(' '.join(query_toks[span[0]:span[1]]))
            char_spans.append([char_b, char_e])
        return char_spans
