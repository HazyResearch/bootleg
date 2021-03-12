"""BootlegAnnotator."""
import logging
import os
import tarfile
import urllib
from pathlib import Path

import numpy as np
import torch
import ujson
from tqdm import tqdm
from transformers import BertTokenizer

import emmental
from bootleg.data import get_dataloader_embeddings
from bootleg.end2end.annotator_utils import DownloadProgressBar
from bootleg.end2end.extract_mentions import (
    find_aliases_in_sentence_tag,
    get_all_aliases,
)
from bootleg.layers.alias_to_ent_encoder import AliasEntityTable
from bootleg.symbols.constants import PAD_ID
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.task_config import NED_TASK, NED_TASK_TO_LABEL, TYPE_PRED_TASK
from bootleg.tasks import ned_task, type_pred_task
from bootleg.utils import eval_utils, sentence_utils
from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args
from bootleg.utils.utils import load_yaml_file
from emmental.model import EmmentalModel

logger = logging.getLogger(__name__)

BOOTLEG_MODEL_PATHS = {
    "bootleg_cased": "https://bootleg-data.s3.amazonaws.com/models/latest/bootleg_cased.tar.gz",
    "bootleg_cased_mini": "https://bootleg-data.s3.amazonaws.com/models/latest/bootleg_cased_mini.tar.gz",
    "bootleg_uncased": "https://bootleg-data.s3.amazonaws.com/models/latest/bootleg_uncased.tar.gz",
    "bootleg_uncased_mini": "https://bootleg-data.s3.amazonaws.com/models/latest/bootleg_uncased_mini.tar.gz",
}


def get_default_cache():
    """Gets default cache directory for saving Bootleg data.

    Returns:
    """
    try:
        from torch.hub import _get_torch_home

        torch_cache_home = _get_torch_home()
    except ImportError:
        torch_cache_home = os.path.expanduser(
            os.getenv(
                "TORCH_HOME",
                os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"),
            )
        )
    return Path(torch_cache_home) / "bootleg"


def create_config(model_path, data_path, model_name):
    """Creates Bootleg config.

    Args:
        model_path: model directory
        data_path: data directory
        model_name: model name

    Returns: updated config
    """
    config_file = model_path / model_name / "bootleg_config.yaml"
    config_args = load_yaml_file(config_file)

    # set the model checkpoint path
    config_args["emmental"]["model_path"] = str(
        model_path / model_name / "bootleg_wiki.pth"
    )

    # set the path for the entity db and candidate map
    config_args["data_config"]["entity_dir"] = str(data_path / "entity_db")
    config_args["data_config"]["alias_cand_map"] = "alias2qids.json"

    # set the embedding paths
    config_args["data_config"]["emb_dir"] = str(data_path / "entity_db")
    config_args["data_config"]["word_embedding"]["cache_dir"] = str(
        data_path / "pretrained_bert_models"
    )

    # set log path
    config_args["emmental"]["log_path"] = str(data_path / "log_dir")

    config_args = parse_boot_and_emm_args(config_args)
    return config_args


def create_sources(model_path, data_path, model_name):
    """Downloads Bootleg data and saves in log dir.

    Args:
        model_path: model directory
        data_path: data directory
        model_name: model name to download

    Returns:
    """
    download_path = BOOTLEG_MODEL_PATHS[model_name]
    if not (model_path / model_name).exists():
        print(
            f"{model_path / model_name} not found. Downloading from {download_path}.."
        )
        urllib.request.urlretrieve(
            download_path,
            filename=str(model_path / f"{model_name}.tar.gz"),
            reporthook=DownloadProgressBar(),
        )
        print(f"Downloaded. Decompressing...")
        tar = tarfile.open(str(model_path / f"{model_name}.tar.gz"), "r:gz")
        tar.extractall(model_path)
        tar.close()

    if not (data_path / "entity_db").exists():
        print(f"{data_path / 'entity_db'} not found. Downloading..")
        urllib.request.urlretrieve(
            "https://bootleg-data.s3.amazonaws.com/data/latest/entity_db.tar.gz",
            filename=str(data_path / "entity_db.tar.gz"),
            reporthook=DownloadProgressBar(),
        )
        print(f"Downloaded. Decompressing...")
        tar = tarfile.open(str(data_path / "entity_db.tar.gz"), "r:gz")
        tar.extractall(data_path)
        tar.close()


class BootlegAnnotator(object):
    """BootlegAnnotator class: convenient wrapper of preprocessing and model
    eval to allow for annotating single sentences at a time for quick
    experimentation, e.g. in notebooks.

    Args:
        config: model config (default None)
        device: model device, -1 for CPU (default None)
        max_alias_len: maximum alias length (default 6)
        cand_map: alias candidate map (default None)
        threshold: probability threshold (default 0.0)
        cache_dir: cache directory (default None)
        model_name: model name (default None)
        verbose: verbose boolean (default False)
    """

    def __init__(
        self,
        config=None,
        device=None,
        max_alias_len=6,
        cand_map=None,
        threshold=0.0,
        cache_dir=None,
        model_name=None,
        verbose=False,
    ):
        self.max_alias_len = (
            max_alias_len  # minimum probability of prediction to return mention
        )
        self.verbose = verbose
        self.threshold = threshold

        if not cache_dir:
            self.cache_dir = get_default_cache()
            self.model_path = self.cache_dir / "models"
            self.data_path = self.cache_dir / "data"
        else:
            self.cache_dir = Path(cache_dir)
            self.model_path = self.cache_dir / "models"
            self.data_path = self.cache_dir / "data"

        if not model_name:
            model_name = "bootleg_uncased"

        assert model_name in {
            "bootleg_cased",
            "bootleg_cased_mini",
            "bootleg_uncased",
            "bootleg_uncased_mini",
        }, f"model_name must be one of [bootleg_cased, bootleg_cased_mini, bootleg_uncased_mini, bootleg_uncased]. You have {model_name}."

        if not config:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.model_path.mkdir(parents=True, exist_ok=True)
            self.data_path.mkdir(parents=True, exist_ok=True)
            create_sources(self.model_path, self.data_path, model_name)
            self.config = create_config(self.model_path, self.data_path, model_name)
        else:
            if "emmental" in config:
                config = parse_boot_and_emm_args(config)
            self.config = config
            # Ensure some of the critical annotator args are the correct type
            self.config.data_config.max_aliases = int(
                self.config.data_config.max_aliases
            )
            self.config.run_config.eval_batch_size = int(
                self.config.run_config.eval_batch_size
            )
            self.config.data_config.max_seq_len = int(
                self.config.data_config.max_seq_len
            )
            self.config.data_config.train_in_candidates = bool(
                self.config.data_config.train_in_candidates
            )

        if not device:
            device = 0 if torch.cuda.is_available() else -1

        self.torch_device = (
            torch.device(device) if device != -1 else torch.device("cpu")
        )
        self.config.model_config.device = device

        log_level = logging.getLevelName(self.config["run_config"]["log_level"].upper())
        emmental.init(
            log_dir=self.config["meta_config"]["log_path"],
            config=self.config,
            use_exact_log_path=self.config["meta_config"]["use_exact_log_path"],
            level=log_level,
        )

        logger.debug("Reading entity database")
        self.entity_db = EntitySymbols.load_from_cache(
            os.path.join(
                self.config.data_config.entity_dir,
                self.config.data_config.entity_map_dir,
            ),
            alias_cand_map_file=self.config.data_config.alias_cand_map,
        )
        logger.debug("Reading word tokenizers")
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config.data_config.word_embedding.bert_model,
            do_lower_case=True
            if "uncased" in self.config.data_config.word_embedding.bert_model
            else False,
            cache_dir=self.config.data_config.word_embedding.cache_dir,
        )

        # Create tasks
        tasks = [NED_TASK]
        if self.config.data_config.type_prediction.use_type_pred is True:
            tasks.append(TYPE_PRED_TASK)
        self.task_to_label_dict = {t: NED_TASK_TO_LABEL[t] for t in tasks}

        # Create tasks
        self.model = EmmentalModel(name="Bootleg")
        self.model.add_task(ned_task.create_task(self.config, self.entity_db))
        if TYPE_PRED_TASK in tasks:
            self.model.add_task(type_pred_task.create_task(self.config, self.entity_db))
            # Add the mention type embedding to the embedding payload
            type_pred_task.update_ned_task(self.model)

        logger.debug("Loading model")
        # Load the best model from the pretrained model
        assert (
            self.config["model_config"]["model_path"] is not None
        ), f"Must have a model to load in the model_path for the BootlegAnnotator"
        self.model.load(self.config["model_config"]["model_path"])
        self.model.eval()
        if cand_map is None:
            alias_map = self.entity_db.get_alias2qids()
        else:
            logger.debug(f"Loading candidate map")
            alias_map = ujson.load(open(cand_map))

        self.all_aliases_trie = get_all_aliases(alias_map, verbose)

        logger.debug("Reading in alias table")
        self.alias2cands = AliasEntityTable(
            data_config=self.config.data_config, entity_symbols=self.entity_db
        )

        # get batch_on_the_fly embeddings
        self.batch_on_the_fly_embs = get_dataloader_embeddings(
            self.config, self.entity_db
        )

    def extract_mentions(self, text, label_func):
        """Wrapper function for mention extraction.

        Args:
            text: text to extract mentions from
            label_func: function that performs extraction (input is (text, alias trie, max alias length) -> output is list of found aliases and found spans

        Returns: JSON object of sentence to be used in eval
        """
        found_aliases, found_spans = label_func(
            text, self.all_aliases_trie, self.max_alias_len
        )
        return {
            "sentence": text,
            "aliases": found_aliases,
            "spans": found_spans,
            # we don't know the true QID
            "qids": ["Q-1" for i in range(len(found_aliases))],
            "gold": [True for i in range(len(found_aliases))],
        }

    def set_threshold(self, value):
        """Sets threshold.

        Args:
            value: threshold value

        Returns:
        """
        self.threshold = value

    def label_mentions(self, text_list, label_func=find_aliases_in_sentence_tag):
        """Extracts mentions and runs disambiguation.

        Args:
            text_list: list of text to disambiguate (or single sentence)
            label_func: mention extraction funciton (optional)

        Returns: Dict of

            * ``qids``: final predicted QIDs,
            * ``probs``: final predicted probs,
            * ``titles``: final predicted titles,
            * ``cands``: all entity canddiates,
            * ``cand_probs``: probabilities of all candidates,
            * ``spans``: final extracted word spans,
            * ``aliases``: final extracted aliases,
        """
        if type(text_list) is str:
            text_list = [text_list]
        else:
            assert (
                type(text_list) is list
                and len(text_list) > 0
                and type(text_list[0]) is str
            ), f"We only accept inputs of strings and lists of strings"

        ebs = int(self.config.run_config.eval_batch_size)
        self.config.data_config.max_aliases = int(self.config.data_config.max_aliases)
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
        for idx_unq, text in tqdm(
            enumerate(text_list),
            desc="Prepping data",
            total=len(text_list),
            disable=not self.verbose,
        ):
            sample = self.extract_mentions(text, label_func)
            total_start_exs += len(sample["aliases"])
            char_spans = self.get_char_spans(sample["spans"], text)

            final_char_spans.append(char_spans)

            (
                idxs_arr,
                aliases_to_predict_per_split,
                spans_arr,
                phrase_tokens_arr,
                pos_idxs,
            ) = sentence_utils.split_sentence(
                max_aliases=self.config.data_config.max_aliases,
                phrase=sample["sentence"],
                spans=sample["spans"],
                aliases=sample["aliases"],
                aliases_seen_by_model=list(range(len(sample["aliases"]))),
                seq_len=self.config.data_config.max_seq_len,
                is_bert=True,
                tokenizer=self.tokenizer,
            )
            aliases_arr = [
                [sample["aliases"][idx] for idx in idxs] for idxs in idxs_arr
            ]
            old_spans_arr = [
                [sample["spans"][idx] for idx in idxs] for idxs in idxs_arr
            ]
            qids_arr = [[sample["qids"][idx] for idx in idxs] for idxs in idxs_arr]
            word_indices_arr = [
                self.tokenizer.convert_tokens_to_ids(pt) for pt in phrase_tokens_arr
            ]
            # iterate over each sample in the split

            for sub_idx in range(len(idxs_arr)):
                # ====================================================
                # GENERATE MODEL INPUTS
                # ====================================================
                aliases_to_predict_arr = aliases_to_predict_per_split[sub_idx]

                assert (
                    len(aliases_to_predict_arr) >= 0
                ), f"There are no aliases to predict for an example. This should not happen at this point."
                assert (
                    len(aliases_arr[sub_idx]) <= self.config.data_config.max_aliases
                ), f"Each example should have no more that {self.config.data_config.max_aliases} max aliases. {sample} does."

                example_aliases = np.ones(self.config.data_config.max_aliases) * PAD_ID
                example_aliases_locs_start = (
                    np.ones(self.config.data_config.max_aliases) * PAD_ID
                )
                example_aliases_locs_end = (
                    np.ones(self.config.data_config.max_aliases) * PAD_ID
                )
                example_alias_list_pos = (
                    np.ones(self.config.data_config.max_aliases) * PAD_ID
                )
                example_true_entities = (
                    np.ones(self.config.data_config.max_aliases) * PAD_ID
                )

                for mention_idx, alias in enumerate(aliases_arr[sub_idx]):
                    span_start_idx, span_end_idx = spans_arr[sub_idx][mention_idx]
                    # generate indexes into alias table.
                    alias_trie_idx = self.entity_db.get_alias_idx(alias)
                    alias_qids = np.array(self.entity_db.get_qid_cands(alias))
                    if not qids_arr[sub_idx][mention_idx] in alias_qids:
                        # assert not data_args.train_in_candidates
                        if not self.config.data_config.train_in_candidates:
                            # set class label to be "not in candidate set"
                            true_entity_idx = 0
                        else:
                            true_entity_idx = -2
                    else:
                        # Here we are getting the correct class label for training.
                        # Our training is "which of the max_entities entity candidates is the right one (class labels 1 to max_entities) or is it none of these (class label 0)".
                        # + (not discard_noncandidate_entities) is to ensure label 0 is reserved for "not in candidate set" class
                        true_entity_idx = np.nonzero(
                            alias_qids == qids_arr[sub_idx][mention_idx]
                        )[0][0] + (not self.config.data_config.train_in_candidates)
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
        batch_example_aliases_locs_start = torch.tensor(
            batch_example_aliases_locs_start, device=self.torch_device
        )
        batch_example_aliases_locs_end = torch.tensor(
            batch_example_aliases_locs_end, device=self.torch_device
        )
        batch_example_true_entities = torch.tensor(
            batch_example_true_entities, device=self.torch_device
        )
        batch_word_indices = torch.tensor(batch_word_indices, device=self.torch_device)

        final_pred_cands = [[] for _ in range(len(text_list))]
        final_all_cands = [[] for _ in range(len(text_list))]
        final_cand_probs = [[] for _ in range(len(text_list))]
        final_pred_probs = [[] for _ in range(len(text_list))]
        final_titles = [[] for _ in range(len(text_list))]
        final_spans = [[] for _ in range(len(text_list))]
        final_aliases = [[] for _ in range(len(text_list))]
        for b_i in tqdm(
            range(0, batch_example_aliases.shape[0], ebs),
            desc="Evaluating model",
            disable=not self.verbose,
        ):
            start_span_idx = batch_example_aliases_locs_start[b_i : b_i + ebs]
            end_span_idx = batch_example_aliases_locs_end[b_i : b_i + ebs]
            word_indices = batch_word_indices[b_i : b_i + ebs]
            alias_indices = batch_example_aliases[b_i : b_i + ebs]
            x_dict = self.get_forward_batch(
                start_span_idx, end_span_idx, word_indices, alias_indices
            )
            x_dict["guid"] = torch.arange(b_i, b_i + ebs, device=self.torch_device)

            (uid_bdict, _, prob_bdict, _) = self.model(  # type: ignore
                uids=x_dict["guid"],
                X_dict=x_dict,
                Y_dict=None,
                task_to_label_dict=self.task_to_label_dict,
                return_action_outputs=False,
            )
            # ====================================================
            # EVALUATE MODEL OUTPUTS
            # ====================================================
            # recover predictions
            probs = prob_bdict[NED_TASK]
            max_probs = probs.max(2)
            max_probs_indices = probs.argmax(2)
            for ex_i in range(probs.shape[0]):
                idx_unq = batch_idx_unq[b_i + ex_i]
                subsplit_idx = batch_subsplit_idx[b_i + ex_i]
                entity_cands = eval_utils.map_aliases_to_candidates(
                    self.config.data_config.train_in_candidates,
                    self.config.data_config.max_aliases,
                    self.entity_db.get_alias2qids(),
                    batch_aliases_arr[b_i + ex_i],
                )
                # batch size is 1 so we can reshape
                probs_ex = probs[ex_i].reshape(
                    self.config.data_config.max_aliases, probs.shape[2]
                )
                for alias_idx, true_entity_pos_idx in enumerate(
                    batch_example_true_entities[b_i + ex_i]
                ):
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
                            final_aliases[idx_unq].append(
                                batch_aliases_arr[b_i + ex_i][alias_idx]
                            )
                            final_spans[idx_unq].append(
                                batch_spans_arr[b_i + ex_i][alias_idx]
                            )
                            final_titles[idx_unq].append(
                                self.entity_db.get_title(pred_qid)
                                if pred_qid != "NC"
                                else "NC"
                            )
                            total_final_exs += 1
                        else:
                            dropped_by_thresh += 1
        assert (
            total_final_exs + dropped_by_thresh == total_start_exs
        ), f"Something went wrong and we have predicted fewer mentions than extracted. Start {total_start_exs}, Out {total_final_exs}, No cand {dropped_by_thresh}"
        res_dict = {
            "qids": final_pred_cands,
            "probs": final_pred_probs,
            "titles": final_titles,
            "cands": final_all_cands,
            "cand_probs": final_cand_probs,
            "spans": final_spans,
            "aliases": final_aliases,
        }
        return res_dict

    def get_forward_batch(self, start_span_idx, end_span_idx, token_ids, alias_idx):
        """Preps the forward batch for disambiguation.

        Args:
            start_span_idx: start span tensor
            end_span_idx: end span tensor
            token_ids: word token tensor
            alias_idx: alias index used for extracting candidate eids

        Returns: X_dict used in Emmental
        """
        entity_cand_eid = self.alias2cands(alias_idx).long()
        entity_cand_eid_mask = entity_cand_eid == -1
        entity_cand_eid_noneg = torch.where(
            entity_cand_eid >= 0,
            entity_cand_eid,
            (
                torch.ones_like(entity_cand_eid, dtype=torch.long)
                * (self.entity_db.num_entities_with_pad_and_nocand - 1)
            ),
        )

        kg_prepped_embs = {}
        for emb_key in self.batch_on_the_fly_embs:
            kg_adj = self.batch_on_the_fly_embs[emb_key]["kg_adj"]
            prep_func = self.batch_on_the_fly_embs[emb_key]["kg_adj_process_func"]
            batch_prep = []
            for j in range(entity_cand_eid_noneg.shape[0]):
                batch_prep.append(
                    prep_func(entity_cand_eid_noneg[j].cpu(), kg_adj).reshape(1, -1)
                )
            kg_prepped_embs[emb_key] = torch.tensor(
                batch_prep, device=self.torch_device
            )

        X_dict = {
            "guids": [],
            "start_span_idx": start_span_idx,
            "end_span_idx": end_span_idx,
            "token_ids": token_ids,
            "entity_cand_eid": entity_cand_eid_noneg,
            "entity_cand_eid_mask": entity_cand_eid_mask,
            "batch_on_the_fly_kg_adj": kg_prepped_embs,
        }
        return X_dict

    def get_char_spans(self, spans, text):
        """Helper function to get character spans instead of default word
        spans.

        Args:
            spans: word spans
            text: text

        Returns: character spans
        """
        query_toks = text.split()
        char_spans = []
        for span in spans:
            space_btwn_toks = (
                len(" ".join(query_toks[0 : span[0] + 1]))
                - len(" ".join(query_toks[0 : span[0]]))
                - len(query_toks[span[0]])
            )
            char_b = len(" ".join(query_toks[0 : span[0]])) + space_btwn_toks
            char_e = char_b + len(" ".join(query_toks[span[0] : span[1]]))
            char_spans.append([char_b, char_e])
        return char_spans
