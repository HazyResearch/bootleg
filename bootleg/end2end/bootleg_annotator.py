"""BootlegAnnotator."""
import logging
import os
import tarfile
import urllib
from pathlib import Path

import emmental
import numpy as np
import torch
from emmental.model import EmmentalModel
from tqdm import tqdm
from transformers import AutoTokenizer

from bootleg.dataset import extract_context_windows, get_entity_string
from bootleg.end2end.annotator_utils import DownloadProgressBar
from bootleg.end2end.extract_mentions import find_aliases_in_sentence_tag
from bootleg.symbols.constants import PAD_ID
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.kg_symbols import KGSymbols
from bootleg.symbols.type_symbols import TypeSymbols
from bootleg.task_config import NED_TASK
from bootleg.tasks import ned_task
from bootleg.utils import data_utils
from bootleg.utils.eval_utils import get_char_spans
from bootleg.utils.model_utils import get_max_candidates
from bootleg.utils.parser.parser_utils import parse_boot_and_emm_args
from bootleg.utils.utils import load_yaml_file

logger = logging.getLogger(__name__)

BOOTLEG_MODEL_PATHS = {
    "bootleg_uncased": "https://bootleg-data.s3-us-west-2.amazonaws.com/models/latest/bootleg_uncased.tar.gz",
}


def get_default_cache():
    """Get default cache directory for saving Bootleg data."""
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
    """Create Bootleg config.

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
    config_args["data_config"]["word_embedding"]["cache_dir"] = str(
        data_path / "pretrained_bert_models"
    )

    # set log path
    config_args["emmental"]["log_path"] = str(data_path / "log_dir")

    config_args = parse_boot_and_emm_args(config_args)
    return config_args


def create_sources(model_path, data_path, model_name):
    """Download Bootleg data and saves in log dir.

    Args:
        model_path: model directory
        data_path: data directory
        model_name: model name to download
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
        print("Downloaded. Decompressing...")
        tar = tarfile.open(str(model_path / f"{model_name}.tar.gz"), "r:gz")
        tar.extractall(model_path)
        tar.close()

    if not (data_path / "entity_db").exists():
        print(f"{data_path / 'entity_db'} not found. Downloading..")
        urllib.request.urlretrieve(
            "https://bootleg-data.s3-us-west-2.amazonaws.com/data/latest/entity_db.tar.gz",
            filename=str(data_path / "entity_db.tar.gz"),
            reporthook=DownloadProgressBar(),
        )
        print("Downloaded. Decompressing...")
        tar = tarfile.open(str(data_path / "entity_db.tar.gz"), "r:gz")
        tar.extractall(data_path)
        tar.close()


class BootlegAnnotator(object):
    """
    Bootleg on-the-fly annotator.

    BootlegAnnotator class: convenient wrapper of preprocessing and model
    eval to allow for annotating single sentences at a time for quick
    experimentation, e.g. in notebooks.

    Args:
        config: model config (default None)
        device: model device, -1 for CPU (default None)
        min_alias_len: minimum alias length (default 1)
        max_alias_len: maximum alias length (default 6)
        threshold: probability threshold (default 0.0)
        cache_dir: cache directory (default None)
        model_name: model name (default None)
        entity_emb_file: entity embedding file (default None)
        return_embs: whether to return embeddings or not (default False)
        verbose: verbose boolean (default False)
    """

    def __init__(
        self,
        config=None,
        device=None,
        min_alias_len=1,
        max_alias_len=6,
        threshold=0.0,
        cache_dir=None,
        model_name=None,
        entity_emb_file=None,
        return_embs=False,
        verbose=False,
    ):
        """Bootleg annotator initializer."""
        self.min_alias_len = min_alias_len
        self.max_alias_len = max_alias_len
        self.verbose = verbose
        self.threshold = threshold
        self.return_embs = return_embs
        self.entity_emb_file = entity_emb_file

        if self.entity_emb_file is not None:
            assert Path(
                self.entity_emb_file
            ).exists(), f"{self.entity_emb_file} must exist."

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
            "bootleg_uncased",
        }, f"model_name must be bootleg_uncased. You have {model_name}."

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
            self.config.run_config.eval_batch_size = int(
                self.config.run_config.eval_batch_size
            )
            self.config.data_config.max_seq_len = int(
                self.config.data_config.max_seq_len
            )
            self.config.data_config.train_in_candidates = bool(
                self.config.data_config.train_in_candidates
            )

        if entity_emb_file is not None:
            assert Path(entity_emb_file).exists(), f"{entity_emb_file} does not exist"

        if not device:
            device = 0 if torch.cuda.is_available() else -1

        if self.verbose:
            self.config["run_config"]["log_level"] = "DEBUG"
        else:
            self.config["run_config"]["log_level"] = "INFO"

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
            alias_cand_map_fld=self.config.data_config.alias_cand_map,
            alias_idx_fld=self.config.data_config.alias_idx_map,
        )
        self.all_aliases_trie = self.entity_db.get_allalias_vocabtrie()

        add_entity_type = self.config.data_config.entity_type_data.use_entity_types
        self.type_symbols = None
        # If we do not have self.entity_emb_file, then need to generate entity encoder input with metadata
        if add_entity_type and self.entity_emb_file is None:
            logger.debug("Reading entity type database")
            self.type_symbols = TypeSymbols.load_from_cache(
                os.path.join(
                    self.config.data_config.entity_dir,
                    self.config.data_config.entity_type_data.type_symbols_dir,
                )
            )
        add_entity_kg = self.config.data_config.entity_kg_data.use_entity_kg
        self.kg_symbols = None
        # If we do not have self.entity_emb_file, then need to generate entity encoder input with metadata
        if add_entity_kg and self.entity_emb_file is None:
            logger.debug("Reading entity kg database")
            self.kg_symbols = KGSymbols.load_from_cache(
                os.path.join(
                    self.config.data_config.entity_dir,
                    self.config.data_config.entity_kg_data.kg_symbols_dir,
                )
            )
        logger.debug("Reading word tokenizers")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.data_config.word_embedding.bert_model,
            do_lower_case=True
            if "uncased" in self.config.data_config.word_embedding.bert_model
            else False,
            cache_dir=self.config.data_config.word_embedding.cache_dir,
        )
        data_utils.add_special_tokens(self.tokenizer)

        # Create tasks
        self.task_to_label_dict = {NED_TASK: None}

        # Create tasks
        self.model = EmmentalModel(name="Bootleg")
        task_to_add = ned_task.create_task(
            self.config,
            use_batch_cands=False,
            len_context_tok=len(self.tokenizer),
            entity_emb_file=self.entity_emb_file,
        )
        # As we manually keep track of the aliases for scoring, we only need the embeddings as action outputs
        task_to_add.action_outputs = [("entity_encoder", 0)]
        self.model.add_task(task_to_add)

        logger.debug("Loading model")
        # Load the best model from the pretrained model
        assert (
            self.config["model_config"]["model_path"] is not None
        ), "Must have a model to load in the model_path for the BootlegAnnotator"
        self.model.load(self.config["model_config"]["model_path"])
        self.model.eval()

    def extract_mentions(self, text, label_func):
        """Mention extraction wrapper.

        Args:
            text: text to extract mentions from
            label_func: function that performs extraction (input is (text, alias trie, max alias length) ->
                        output is list of found aliases and found spans

        Returns: JSON object of sentence to be used in eval
        """
        found_aliases, found_spans = label_func(
            text, self.all_aliases_trie, self.min_alias_len, self.max_alias_len
        )
        return {
            "sentence": text,
            "aliases": found_aliases,
            "spans": found_spans,
            "cands": [self.entity_db.get_qid_cands(al) for al in found_aliases],
            # we don't know the true QID
            "qids": ["Q-1" for i in range(len(found_aliases))],
            "gold": [True for i in range(len(found_aliases))],
        }

    def set_threshold(self, value):
        """Set threshold.

        Args:
            value: threshold value
        """
        self.threshold = value

    def label_mentions(
        self,
        text_list=None,
        label_func=find_aliases_in_sentence_tag,
        extracted_examples=None,
    ):
        """Extract mentions and runs disambiguation.

        If user provides extracted_examples, we will ignore text_list.

        Args:
            text_list: list of text to disambiguate (or single string) (can be None if extracted_examples is not None)
            label_func: mention extraction funciton (optional)
            extracted_examples: List of Dicts of keys "sentence", "aliases", "spans", "cands" (QIDs) (optional)

        Returns: Dict of

            * ``qids``: final predicted QIDs,
            * ``probs``: final predicted probs,
            * ``titles``: final predicted titles,
            * ``cands``: all entity candidates,
            * ``cand_probs``: probabilities of all candidates,
            * ``spans``: final extracted word spans,
            * ``aliases``: final extracted aliases,
            * ``embs``: final entity contextualized embeddings (if return_embs is True)
            * ``cand_embs``: final candidate entity contextualized embeddings (if return_embs is True)
        """
        # Check inputs are sane
        do_extract_mentions = True
        if extracted_examples is not None:
            do_extract_mentions = False
            assert (
                type(extracted_examples) is list
            ), "Must provide a list of Dics for extracted_examples"
            check_ex = extracted_examples[0]
            assert (
                len(
                    {"sentence", "aliases", "spans", "cands"}.intersection(
                        check_ex.keys()
                    )
                )
                == 4
            ), (
                f"You must have keys of sentence, aliases, spans, and cands for extracted_examples. You have"
                f"{extracted_examples.keys()}"
            )
        else:
            assert (
                text_list is not None
            ), "If you do not provide extracted_examples you must provide text_list"

        if text_list is None:
            assert (
                extracted_examples is not None
            ), "If you do not provide text_list you must provide extracted_exampels"
        else:
            if type(text_list) is str:
                text_list = [text_list]
            else:
                assert (
                    type(text_list) is list
                    and len(text_list) > 0
                    and type(text_list[0]) is str
                ), "We only accept inputs of strings and lists of strings"

        # Get number of examples
        if extracted_examples is not None:
            num_exs = len(extracted_examples)
        else:
            num_exs = len(text_list)

        ebs = int(self.config.run_config.eval_batch_size)
        total_start_exs = 0
        total_final_exs = 0
        dropped_by_thresh = 0

        batch_example_qid_cands = []
        batch_example_eid_cands = []
        batch_example_true_entities = []
        batch_word_indices = []
        batch_word_token_types = []
        batch_word_attention = []
        batch_ent_indices = []
        batch_ent_token_types = []
        batch_ent_attention = []
        batch_spans_arr = []
        batch_char_spans_arr = []
        batch_example_aliases = []
        batch_idx_unq = []
        for idx_unq in tqdm(
            range(num_exs),
            desc="Prepping data",
            total=num_exs,
            disable=not self.verbose,
        ):
            if do_extract_mentions:
                sample = self.extract_mentions(text_list[idx_unq], label_func)
            else:
                sample = extracted_examples[idx_unq]
                # Add the unk qids and gold values
                sample["qids"] = ["Q-1" for _ in range(len(sample["aliases"]))]
                sample["gold"] = [True for _ in range(len(sample["aliases"]))]
            total_start_exs += len(sample["aliases"])
            char_spans_arr = get_char_spans(sample["spans"], sample["sentence"])
            for men_idx in range(len(sample["aliases"])):
                # ====================================================
                # GENERATE TEXT INPUTS
                # ====================================================
                inputs = self.get_sentence_tokens(sample, men_idx)

                # ====================================================
                # GENERATE CANDIDATE INPUTS
                # ====================================================
                example_qid_cands = [
                    "-1"
                    for _ in range(
                        get_max_candidates(self.entity_db, self.config.data_config)
                    )
                ]
                example_eid_cands = [
                    -1
                    for _ in range(
                        get_max_candidates(self.entity_db, self.config.data_config)
                    )
                ]
                # generate indexes into alias table.
                alias_qids = np.array(sample["cands"][men_idx])
                # first entry is the non candidate class (NC and eid 0) - used when train in cands is false
                # if we train in candidates, this gets overwritten
                example_qid_cands[0] = "NC"
                example_qid_cands[
                    (not self.config.data_config.train_in_candidates) : len(alias_qids)
                    + (not self.config.data_config.train_in_candidates)
                ] = sample["cands"][men_idx]
                example_eid_cands[0] = 0
                example_eid_cands[
                    (not self.config.data_config.train_in_candidates) : len(alias_qids)
                    + (not self.config.data_config.train_in_candidates)
                ] = [self.entity_db.get_eid(q) for q in sample["cands"][men_idx]]
                if not sample["qids"][men_idx] in alias_qids:
                    # assert not data_args.train_in_candidates
                    if not self.config.data_config.train_in_candidates:
                        # set class label to be "not in candidate set"
                        true_entity_idx = 0
                    else:
                        true_entity_idx = -2
                else:
                    # Here we are getting the correct class label for training.
                    # Our training is "which of the max_entities entity candidates is the right one
                    # (class labels 1 to max_entities) or is it none of these (class label 0)".
                    # + (not discard_noncandidate_entities) is to ensure label 0 is
                    # reserved for "not in candidate set" class
                    true_entity_idx = np.nonzero(alias_qids == sample["qids"][men_idx])[
                        0
                    ][0] + (not self.config.data_config.train_in_candidates)

                # Get candidate tokens
                example_cand_input_ids = []
                example_cand_token_type_ids = []
                example_cand_attention_mask = []
                if self.entity_emb_file is None:
                    entity_tokens = [
                        self.get_entity_tokens(cand_qid)
                        for cand_qid in example_qid_cands
                    ]
                    example_cand_input_ids = [
                        ent_toks["input_ids"] for ent_toks in entity_tokens
                    ]
                    example_cand_token_type_ids = [
                        ent_toks["token_type_ids"] for ent_toks in entity_tokens
                    ]
                    example_cand_attention_mask = [
                        ent_toks["attention_mask"] for ent_toks in entity_tokens
                    ]

                # ====================================================
                # ACCUMULATE
                # ====================================================
                batch_example_qid_cands.append(example_qid_cands)
                batch_example_eid_cands.append(example_eid_cands)
                batch_example_true_entities.append(true_entity_idx)
                batch_word_indices.append(inputs["input_ids"])
                batch_word_token_types.append(inputs["token_type_ids"])
                batch_word_attention.append(inputs["attention_mask"])
                batch_ent_indices.append(example_cand_input_ids)
                batch_ent_token_types.append(example_cand_token_type_ids)
                batch_ent_attention.append(example_cand_attention_mask)
                batch_example_aliases.append(sample["aliases"][men_idx])
                # Add the orginal sample spans because spans_arr is w.r.t BERT subword token
                batch_spans_arr.append(sample["spans"][men_idx])
                batch_char_spans_arr.append(char_spans_arr[men_idx])
                batch_idx_unq.append(idx_unq)

        batch_example_eid_cands = torch.tensor(batch_example_eid_cands).long()
        batch_example_true_entities = torch.tensor(batch_example_true_entities)

        final_pred_cands = [[] for _ in range(num_exs)]
        final_all_cands = [[] for _ in range(num_exs)]
        final_cand_probs = [[] for _ in range(num_exs)]
        final_pred_probs = [[] for _ in range(num_exs)]
        final_entity_embs = [[] for _ in range(num_exs)]
        final_entity_cand_embs = [[] for _ in range(num_exs)]
        final_titles = [[] for _ in range(num_exs)]
        final_spans = [[] for _ in range(num_exs)]
        final_char_spans = [[] for _ in range(num_exs)]
        final_aliases = [[] for _ in range(num_exs)]
        for b_i in tqdm(
            range(0, len(batch_word_indices), ebs),
            desc="Evaluating model",
            disable=not self.verbose,
        ):
            x_dict = self.get_forward_batch(
                input_ids=torch.tensor(batch_word_indices)[b_i : b_i + ebs],
                token_type_ids=torch.tensor(batch_word_token_types)[b_i : b_i + ebs],
                attention_mask=torch.tensor(batch_word_attention)[b_i : b_i + ebs],
                entity_token_ids=torch.tensor(batch_ent_indices)[b_i : b_i + ebs],
                entity_type_ids=torch.tensor(batch_ent_token_types)[b_i : b_i + ebs],
                entity_attention_mask=torch.tensor(batch_ent_attention)[
                    b_i : b_i + ebs
                ],
                entity_cand_eid=batch_example_eid_cands[b_i : b_i + ebs],
                generate_entity_inputs=(self.entity_emb_file is None),
            )
            x_dict["guid"] = torch.arange(b_i, b_i + ebs, device=self.torch_device)
            with torch.no_grad():
                res = self.model(  # type: ignore
                    uids=x_dict["guid"],
                    X_dict=x_dict,
                    Y_dict=None,
                    task_to_label_dict=self.task_to_label_dict,
                    return_loss=False,
                    return_probs=True,
                    return_action_outputs=self.return_embs,
                )
            del x_dict
            if self.return_embs:
                (uid_bdict, _, prob_bdict, _, out_bdict) = res
                output_embs = out_bdict[NED_TASK][
                    "entity_encoder_0"
                    if (self.entity_emb_file is None)
                    else "entity_encoder_static_0"
                ]
            else:
                output_embs = None
                (uid_bdict, _, prob_bdict, _) = res
            # ====================================================
            # EVALUATE MODEL OUTPUTS
            # ====================================================
            # recover predictions
            probs = prob_bdict[NED_TASK]
            max_probs = probs.max(1)
            max_probs_indices = probs.argmax(1)
            for ex_i in range(probs.shape[0]):
                idx_unq = batch_idx_unq[b_i + ex_i]
                entity_cands = batch_example_qid_cands[b_i + ex_i]
                # batch size is 1 so we can reshape
                probs_ex = probs[ex_i]
                true_entity_pos_idx = batch_example_true_entities[b_i + ex_i]
                if true_entity_pos_idx != PAD_ID:
                    pred_idx = max_probs_indices[ex_i]
                    pred_prob = max_probs[ex_i].item()
                    pred_qid = entity_cands[pred_idx]
                    if pred_prob > self.threshold:
                        final_all_cands[idx_unq].append(entity_cands)
                        final_cand_probs[idx_unq].append(probs_ex)
                        final_pred_cands[idx_unq].append(pred_qid)
                        final_pred_probs[idx_unq].append(pred_prob)
                        if self.return_embs:
                            final_entity_embs[idx_unq].append(
                                output_embs[ex_i][pred_idx]
                            )
                            final_entity_cand_embs[idx_unq].append(output_embs[ex_i])
                        final_aliases[idx_unq].append(batch_example_aliases[b_i + ex_i])
                        final_spans[idx_unq].append(batch_spans_arr[b_i + ex_i])
                        final_char_spans[idx_unq].append(
                            batch_char_spans_arr[b_i + ex_i]
                        )
                        final_titles[idx_unq].append(
                            self.entity_db.get_title(pred_qid)
                            if pred_qid != "NC"
                            else "NC"
                        )
                        total_final_exs += 1
                    else:
                        dropped_by_thresh += 1
        assert total_final_exs + dropped_by_thresh == total_start_exs, (
            f"Something went wrong and we have predicted fewer mentions than extracted. "
            f"Start {total_start_exs}, Out {total_final_exs}, No cand {dropped_by_thresh}"
        )
        res_dict = {
            "qids": final_pred_cands,
            "probs": final_pred_probs,
            "titles": final_titles,
            "cands": final_all_cands,
            "cand_probs": final_cand_probs,
            "spans": final_spans,
            "char_spans": final_char_spans,
            "aliases": final_aliases,
        }
        if self.return_embs:
            res_dict["embs"] = final_entity_embs
            res_dict["cand_embs"] = final_entity_cand_embs
        return res_dict

    def get_sentence_tokens(self, sample, men_idx):
        """
        Get context tokens.

        Args:
            sample: Dict sample after extraction
            men_idx: mention index to select

        Returns: Dict of tokenized outputs
        """
        span = sample["spans"][men_idx]
        tokens = sample["sentence"].split()
        prev_context, next_context = extract_context_windows(
            span, tokens, self.config.data_config.max_seq_window_len
        )
        context_tokens = (
            prev_context
            + ["[ent_start]"]
            + tokens[span[0] : span[1]]
            + ["[ent_end]"]
            + next_context
        )
        context = " ".join(context_tokens)
        inputs = self.tokenizer(
            context,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.data_config.max_seq_len,
            return_overflowing_tokens=False,
        )
        return inputs

    def get_entity_tokens(self, qid):
        """
        Get entity tokens.

        Args:
            qid: entity QID

        Returns:
            Dict of input tokens for forward pass.
        """
        constants = {
            "max_ent_len": self.config.data_config.max_ent_len,
            "max_ent_type_len": self.config.data_config.entity_type_data.max_ent_type_len,
            "max_ent_kg_len": self.config.data_config.entity_kg_data.max_ent_kg_len,
            "use_types": self.config.data_config.entity_type_data.use_entity_types,
            "use_kg": self.config.data_config.entity_kg_data.use_entity_kg,
            "use_desc": self.config.data_config.use_entity_desc,
        }
        ent_str, title_spans, over_type_len, over_kg_len = get_entity_string(
            qid,
            constants,
            self.entity_db,
            self.kg_symbols,
            self.type_symbols,
        )
        inputs = self.tokenizer(
            ent_str,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            max_length=constants["max_ent_len"],
        )
        return inputs

    def get_forward_batch(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        entity_token_ids,
        entity_type_ids,
        entity_attention_mask,
        entity_cand_eid,
        generate_entity_inputs,
    ):
        """Generate emmental batch.

        Args:
            input_ids: word token ids
            token_type_ids: word token type ids
            attention_mask: work attention mask
            entity_token_ids: entity token ids
            entity_type_ids: entity type ids
            entity_attention_mask: entity attention mask
            entity_cand_eid: entity candidate eids
            generate_entity_inputs: whether to generate entity id inputs

        Returns: X_dict for emmental
        """
        entity_cand_eval_mask = entity_cand_eid == -1
        entity_cand_eid_noneg = torch.where(
            entity_cand_eid >= 0,
            entity_cand_eid,
            (
                torch.ones_like(entity_cand_eid, dtype=torch.long)
                * (self.entity_db.num_entities_with_pad_and_nocand - 1)
            ),
        )
        X_dict = {
            "guids": [],
            "input_ids": input_ids.to(self.torch_device),
            "token_type_ids": token_type_ids.to(self.torch_device),
            "attention_mask": attention_mask.to(self.torch_device),
            "entity_cand_eid": entity_cand_eid_noneg.to(self.torch_device),
            "entity_cand_eval_mask": entity_cand_eval_mask.to(self.torch_device),
        }
        if generate_entity_inputs:
            X_dict["entity_cand_input_ids"] = entity_token_ids.to(self.torch_device)
            X_dict["entity_cand_token_type_ids"] = entity_type_ids.to(self.torch_device)
            X_dict["entity_cand_attention_mask"] = entity_attention_mask.to(
                self.torch_device
            )
        return X_dict
