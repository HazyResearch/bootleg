import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings

import numpy as np
import scipy.sparse
import torch
import ujson
from tqdm import tqdm

from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.layers.alias_to_ent_encoder import AliasEntityTable
from bootleg.symbols.constants import ANCHOR_KEY, PAD_ID
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import data_utils, sentence_utils, utils
from emmental.data import EmmentalDataset

warnings.filterwarnings(
    "ignore",
    message="Could not import the lzma module. Your installed Python is incomplete. "
    "Attempting to use lzma compression will result in a RuntimeError.",
)
warnings.filterwarnings(
    "ignore",
    message="FutureWarning: Passing (type, 1) or '1type'*",
)

logger = logging.getLogger(__name__)
# Removes warnings about TOKENIZERS_PARALLELISM
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InputExample(object):
    """A single training/test example for prediction."""

    def __init__(
        self,
        sent_idx,
        subsent_idx,
        alias_list_pos,
        aliases_to_predict,
        train_aliases_to_predict_arr,
        spans,
        phrase_tokens,
        aliases,
        qids,
    ):
        assert (
            type(sent_idx) is int
        ), f"We need the sentence index is an int. You have {type(sent_idx)}"
        self.sent_idx = int(sent_idx)
        self.subsent_idx = subsent_idx
        self.alias_list_pos = alias_list_pos
        self.aliases_to_predict = aliases_to_predict
        self.train_aliases_to_predict_arr = train_aliases_to_predict_arr
        self.spans = spans
        self.phrase_tokens = phrase_tokens
        self.aliases = aliases
        self.qids = qids

    def to_dict(self):
        return {
            "sent_idx": self.sent_idx,
            "subsent_idx": self.subsent_idx,
            "alias_list_pos": self.alias_list_pos,
            "aliases_to_predict": self.aliases_to_predict,
            "train_aliases_to_predict_arr": self.train_aliases_to_predict_arr,
            "spans": self.spans,
            "phrase_tokens": self.phrase_tokens,
            "aliases": self.aliases,
            "qids": self.qids,
        }

    @classmethod
    def from_dict(cls, in_dict):
        return cls(
            in_dict["sent_idx"],
            in_dict["subsent_idx"],
            in_dict["alias_list_pos"],
            in_dict["aliases_to_predict"],
            in_dict["train_aliases_to_predict_arr"],
            in_dict["spans"],
            in_dict["phrase_tokens"],
            in_dict["aliases"],
            in_dict["qids"],
        )


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        start_idx_in_sent,
        end_idx_in_sent,
        alias_idx,
        word_indices,
        gold_cand_K_idx,
        gold_eid,
        for_dump_gold_cand_K_idx_train,
        alias_list_pos,
        sent_idx,
        subsent_idx,
        guid,
    ):
        self.start_idx_in_sent = start_idx_in_sent
        self.end_idx_in_sent = end_idx_in_sent
        self.alias_idx = alias_idx
        self.word_indices = word_indices
        self.gold_cand_K_idx = gold_cand_K_idx
        self.gold_eid = gold_eid
        self.for_dump_gold_cand_K_idx_train = for_dump_gold_cand_K_idx_train
        self.alias_list_pos = alias_list_pos
        self.sent_idx = sent_idx
        self.subsent_idx = subsent_idx
        self.guid = guid

    def to_dict(self):
        return {
            "start_idx_in_sent": self.start_idx_in_sent,
            "end_idx_in_sent": self.end_idx_in_sent,
            "alias_idx": self.alias_idx,
            "word_indices": self.word_indices,
            "gold_cand_K_idx": self.gold_cand_K_idx,
            "gold_eid": self.gold_eid,
            "for_dump_gold_cand_K_idx_train": self.for_dump_gold_cand_K_idx_train,
            "alias_list_pos": self.alias_list_pos,
            "sent_idx": self.sent_idx,
            "subsent_idx": self.subsent_idx,
            "guid": self.guid,
        }

    @classmethod
    def from_dict(cls, in_dict):
        return cls(
            in_dict["start_idx_in_sent"],
            in_dict["end_idx_in_sent"],
            in_dict["alias_idx"],
            in_dict["word_indices"],
            in_dict["gold_cand_K_idx"],
            in_dict["gold_eid"],
            in_dict["for_dump_gold_cand_K_idx_train"],
            in_dict["alias_list_pos"],
            in_dict["sent_idx"],
            in_dict["subsent_idx"],
            in_dict["guid"],
        )


def read_in_types(data_config):
    """Reads in type mapping from QID -> list of types ids and converts
    dictionary of EIDs to single type id (taking first)

    Args:
        data_config: data config

    Returns: dictionary of EID to type id
    """
    emb_dir = data_config.emb_dir
    coarse_type_file = data_config.type_prediction.type_labels
    coarse_type_vocab_file = data_config.type_prediction.type_vocab
    type_file = os.path.join(emb_dir, coarse_type_file)
    vocab_file = os.path.join(emb_dir, coarse_type_vocab_file)
    entitysymbols = EntitySymbols.load_from_cache(
        load_dir=os.path.join(data_config.entity_dir, data_config.entity_map_dir),
        alias_cand_map_file=data_config.alias_cand_map,
        alias_idx_file=data_config.alias_idx_map,
    )
    with open(vocab_file, "r") as in_f:
        vocab = ujson.load(in_f)

    log_rank_0_debug(logger, f"Building type labels from {type_file} and {vocab_file}.")
    all_type_ids = set(vocab.values())
    assert (
        0 not in all_type_ids
    ), f"We assume type indices start at 1. 0 is reserved for UNK type. You have index 0."
    with open(type_file) as in_f:
        # take the first type; UNK type is 0
        eid2type = {}
        for k, v in ujson.load(in_f).items():
            # Happens if have QIDs that are not in save
            if not entitysymbols.qid_exists(k):
                continue
            mapped_values = []
            for sub_item in v:
                if type(sub_item) is str:
                    sub_item_id = vocab[sub_item]
                else:
                    sub_item_id = sub_item
                mapped_values.append(sub_item_id)
                assert (
                    sub_item_id in all_type_ids
                ), f"Type id {sub_item_id} is not in all type ids for {vocab_file}."
            eid = entitysymbols.get_eid(k)
            # Store strings as keys for making a Tri later
            if len(mapped_values) > 0:
                eid2type[str(eid)] = mapped_values[0]
            else:
                eid2type[str(eid)] = 0
    # We assume types are indexed from 1. So, 6 types will have indices 1 - 6. Max type will get 6.
    assert len(all_type_ids) == data_config.type_prediction.num_types, (
        f"{data_config.type_prediction.num_types} from args.data_config.type_prediction.num_types must "
        f"match our computed number {len(all_type_ids)}"
    )
    return eid2type


def create_examples_initializer(
    tokenizer, is_bert, use_weak_label, split, max_aliases, max_seq_len
):
    global tokenizer_global
    tokenizer_global = tokenizer
    global constants_global
    constants_global = {
        "is_bert": is_bert,
        "use_weak_label": use_weak_label,
        "split": split,
        "max_aliases": max_aliases,
        "max_seq_len": max_seq_len,
    }


def create_examples(
    dataset,
    create_ex_indir,
    create_ex_outdir,
    meta_file,
    data_config,
    dataset_threads,
    use_weak_label,
    split,
    is_bert,
    tokenizer,
):
    """Creates examples from the raw input data.

    Args:
        dataset: data file to read
        create_ex_indir: temporary directory where input files are stored
        create_ex_outdir: temporary directory to store output files from method
        meta_file: metadata file to save the file names/paths for the next step in prep pipeline
        data_config: data config
        dataset_threads: number of threads
        use_weak_label: whether to use weak labeling or not
        split: data split
        is_bert: is the tokenizer a BERT one
        tokenizer: tokenizer

    Returns:
    """
    start = time.time()
    num_processes = min(dataset_threads, int(0.8 * multiprocessing.cpu_count()))

    log_rank_0_debug(logger, f"Counting lines")
    total_input = sum(1 for _ in open(dataset))

    if num_processes == 1:
        constants_dict = {
            "is_bert": is_bert,
            "use_weak_label": use_weak_label,
            "split": split,
            "max_aliases": data_config.max_aliases,
            "max_seq_len": data_config.max_seq_len,
        }
        out_file_name = os.path.join(create_ex_outdir, os.path.basename(dataset))
        print("INPUT", dataset, "OUTPUT", out_file_name)
        res = create_examples_single(
            in_file_name=dataset,
            in_file_lines=total_input,
            out_file_name=out_file_name,
            constants_dict=constants_dict,
            tokenizer=tokenizer,
        )
        files_and_counts = {}
        total_output = res["total_lines"]
        files_and_counts[res["output_filename"]] = res["total_lines"]
    else:
        log_rank_0_info(
            logger, f"Starting to extract examples using {num_processes} processes"
        )
        chunk_input = int(np.ceil(total_input / num_processes))
        log_rank_0_debug(
            logger,
            f"Chunking up {total_input} lines into subfiles of size {chunk_input} lines",
        )
        total_input_from_chunks, input_files_dict = utils.chunk_file(
            dataset, create_ex_indir, chunk_input
        )

        input_files = list(input_files_dict.keys())
        input_file_lines = [input_files_dict[k] for k in input_files]
        output_files = [
            in_file_name.replace(create_ex_indir, create_ex_outdir)
            for in_file_name in input_files
        ]
        assert (
            total_input == total_input_from_chunks
        ), f"Lengths of files {total_input} doesn't mathc {total_input_from_chunks}"
        log_rank_0_debug(logger, f"Done chunking files. Starting pool.")

        pool = multiprocessing.Pool(
            processes=num_processes,
            initializer=create_examples_initializer,
            initargs=[
                tokenizer,
                is_bert,
                use_weak_label,
                split,
                data_config.max_aliases,
                data_config.max_seq_len,
            ],
        )

        total_output = 0
        input_args = list(zip(input_files, input_file_lines, output_files))
        # Store output files and counts for saving in next step
        files_and_counts = {}
        for res in pool.imap_unordered(create_examples_hlp, input_args, chunksize=1):
            total_output += res["total_lines"]
            files_and_counts[res["output_filename"]] = res["total_lines"]
        pool.close()
    utils.dump_json_file(
        meta_file, {"num_mentions": total_output, "files_and_counts": files_and_counts}
    )
    log_rank_0_debug(
        logger,
        f"Done with extracting examples in {time.time()-start}. "
        f"Total lines seen {total_input}. Total lines kept {total_output}.",
    )
    return


def create_examples_hlp(args):
    """Create examples multiprocessing helper."""
    in_file_name, in_file_lines, out_file_name = args
    return create_examples_single(
        in_file_name, in_file_lines, out_file_name, constants_global, tokenizer_global
    )


def create_examples_single(
    in_file_name, in_file_lines, out_file_name, constants_dict, tokenizer
):
    split = constants_dict["split"]
    max_aliases = constants_dict["max_aliases"]
    max_seq_len = constants_dict["max_seq_len"]
    is_bert = constants_dict["is_bert"]
    use_weak_label = constants_dict["use_weak_label"]
    with open(out_file_name, "w", encoding="utf-8") as out_f:
        total_subsents = 0
        total_lines = 0
        for ex in tqdm(
            open(in_file_name, "r", encoding="utf-8"),
            total=in_file_lines,
            desc=f"Reading in {in_file_name}",
        ):
            total_lines += 1
            line = ujson.loads(ex)
            assert "sent_idx_unq" in line
            assert "aliases" in line
            assert "qids" in line
            assert "spans" in line
            assert "sentence" in line
            assert ANCHOR_KEY in line
            sent_idx = line["sent_idx_unq"]
            # aliases are assumed to be lower-cased in candidate map
            aliases = [alias.lower() for alias in line["aliases"]]
            qids = line["qids"]
            spans = line["spans"]
            phrase = line["sentence"]
            assert (
                len(spans) == len(aliases) == len(qids)
            ), "lengths of alias-related values not equal"
            # For datasets, we see all aliases, unless use_weak_label is turned off
            aliases_seen_by_model = [i for i in range(len(aliases))]
            anchor = [True for i in range(len(aliases))]
            if ANCHOR_KEY in line:
                anchor = line[ANCHOR_KEY]
                assert len(aliases) == len(anchor)
                assert all(isinstance(a, bool) for a in anchor)

            for span in spans:
                assert (
                    len(span) == 2
                ), f"Span should be len 2. Your span {span} is {len(span)}"
                assert span[1] <= len(
                    phrase.split()
                ), f"You have span {span} that is beyond the length of the sentence {phrase}"
            if not use_weak_label:
                aliases = [aliases[i] for i in range(len(anchor)) if anchor[i] is True]
                qids = [qids[i] for i in range(len(anchor)) if anchor[i] is True]
                spans = [spans[i] for i in range(len(anchor)) if anchor[i] is True]
                aliases_seen_by_model = [i for i in range(len(aliases))]
                anchor = [True for i in range(len(aliases))]
            # Happens if use weak labels is False
            if len(aliases) == 0:
                continue
            # Extract the subphrase in the sentence
            # idxs_arr is list of lists of indexes of qids, aliases, spans that are part of each subphrase
            # aliases_seen_by_model represents the aliases to be sent through the model
            # The output of aliases_to_predict_per_split represents which aliases are to be scored in which subphrase.
            # Ex:
            # If I have 15 aliases and M = 10, then I can split it into two chunks of 10 aliases each
            # idxs_arr = [0,1,2,3,4,5,6,7,8,9] and [5,6,7,8,9,10,11,12,13,14]
            # However, I do not want to score aliases 5-9 twice. Therefore, aliases_to_predict_per_split
            # represents which ones to score
            # aliases_to_predict_per_split = [0,1,2,3,4,5,6,7] and [3,4,5,6,7,8,9]
            # These are indexes into the idx_arr (the first aliases to be scored in the second list is idx = 3,
            # representing the 8th aliases in the original sequence.
            (
                idxs_arr,
                aliases_to_predict_per_split,
                spans_arr,
                phrase_tokens_arr,
                phrase_tokens_pos_arr,
            ) = sentence_utils.split_sentence(
                max_aliases=max_aliases,
                phrase=phrase,
                spans=spans,
                aliases=aliases,
                aliases_seen_by_model=aliases_seen_by_model,
                seq_len=max_seq_len,
                is_bert=is_bert,
                tokenizer=tokenizer,
            )
            aliases_arr = [[aliases[idx] for idx in idxs] for idxs in idxs_arr]
            anchor_arr = [[anchor[idx] for idx in idxs] for idxs in idxs_arr]
            qids_arr = [[qids[idx] for idx in idxs] for idxs in idxs_arr]
            # write out results to json lines file
            for subsent_idx in range(len(idxs_arr)):
                # This contains the mapping of aliases seen by model to the ones to be scored, pre false anchor
                # dataset filtering. We need this mapping during eval to create on "prediction" per alias,
                # True or False anchor.
                train_aliases_to_predict_arr = aliases_to_predict_per_split[
                    subsent_idx
                ][:]
                # aliases_to_predict_arr is an index into idxs_arr/anchor_arr/aliases_arr.
                # It should only include true anchors if eval dataset.
                # During training want to backpropagate on false anchors as well
                if split != "train":
                    aliases_to_predict_arr = [
                        a2p_ex
                        for a2p_ex in aliases_to_predict_per_split[subsent_idx]
                        if anchor_arr[subsent_idx][a2p_ex] is True
                    ]
                else:
                    aliases_to_predict_arr = aliases_to_predict_per_split[subsent_idx]
                total_subsents += 1
                out_f.write(
                    ujson.dumps(
                        InputExample(
                            sent_idx=sent_idx,
                            subsent_idx=subsent_idx,
                            alias_list_pos=idxs_arr[subsent_idx],
                            aliases_to_predict=aliases_to_predict_arr,
                            train_aliases_to_predict_arr=train_aliases_to_predict_arr,
                            spans=spans_arr[subsent_idx],
                            phrase_tokens=phrase_tokens_arr[subsent_idx],
                            aliases=aliases_arr[subsent_idx],
                            qids=qids_arr[subsent_idx],
                        ).to_dict()
                    )
                    + "\n"
                )
    return {"total_lines": total_subsents, "output_filename": out_file_name}


def convert_examples_to_features_and_save_initializer(
    tokenizer, data_config, save_dataset_name, save_labels_name, X_storage, Y_storage
):
    global tokenizer_global
    tokenizer_global = tokenizer
    global entitysymbols_global
    entitysymbols_global = EntitySymbols.load_from_cache(
        load_dir=os.path.join(data_config.entity_dir, data_config.entity_map_dir),
        alias_cand_map_file=data_config.alias_cand_map,
        alias_idx_file=data_config.alias_idx_map,
    )
    global mmap_file_global
    mmap_file_global = np.memmap(save_dataset_name, dtype=X_storage, mode="r+")
    global mmap_label_file_global
    mmap_label_file_global = np.memmap(save_labels_name, dtype=Y_storage, mode="r+")


def convert_examples_to_features_and_save(
    meta_file,
    guid_dtype,
    data_config,
    dataset_threads,
    use_weak_label,
    split,
    is_bert,
    save_dataset_name,
    save_labels_name,
    X_storage,
    Y_storage,
    tokenizer,
    entity_symbols,
):
    """Converts the prepped examples into input features and saves in memmap
    files. These are used in the __get_item__ method.

    Args:
        meta_file: metadata file where input file paths are saved
        guid_dtype: unique identifier dtype
        data_config: data config
        dataset_threads: number of threads
        use_weak_label: whether to use weak labeling or not
        split: data split
        is_bert: is the tokenizer a BERT tokenizer
        save_dataset_name: data features file name to save
        save_labels_name: data labels file name to save
        X_storage: data features storage type (for memmap)
        Y_storage: data labels storage type (for memmap)
        tokenizer: tokenizer
        entity_symbols: entity symbols

    Returns:
    """
    start = time.time()
    num_processes = min(dataset_threads, int(0.8 * multiprocessing.cpu_count()))

    total_input = utils.load_json_file(meta_file)["num_mentions"]
    files_and_counts = utils.load_json_file(meta_file)["files_and_counts"]

    # IMPORTANT: for distributed writing to memmap files, you must create them in w+ mode before being opened in r+
    memmap_file = np.memmap(
        save_dataset_name, dtype=X_storage, mode="w+", shape=(total_input,), order="C"
    )
    # Save -1 in sent_idx to check that things are loaded correctly later
    memmap_file["sent_idx"][:] = -1
    memmap_label_file = np.memmap(
        save_labels_name, dtype=Y_storage, mode="w+", shape=(total_input,), order="C"
    )

    input_args = []
    # Saves where in memap file to start writing
    offset = 0
    for i, in_file_name in enumerate(files_and_counts.keys()):
        input_args.append(
            {
                "file_name": in_file_name,
                "in_file_lines": files_and_counts[in_file_name],
                "save_file_offset": offset,
                "ex_print_mod": int(np.ceil(total_input / 20)),
                "guid_dtype": guid_dtype,
                "is_bert": is_bert,
                "use_weak_label": use_weak_label,
                "split": split,
                "max_seq_len": data_config.max_seq_len,
                "train_in_candidates": data_config.train_in_candidates,
                "max_aliases": data_config.max_aliases,
                "pred_examples": data_config.print_examples_prep,
            }
        )
        offset += files_and_counts[in_file_name]

    if num_processes == 1:
        assert len(input_args) == 1
        total_output = convert_examples_to_features_and_save_single(
            input_args[0], tokenizer, entity_symbols, memmap_file, memmap_label_file
        )
    else:
        log_rank_0_debug(
            logger,
            "Initializing pool. This make take a few minutes.",
        )
        pool = multiprocessing.Pool(
            processes=num_processes,
            initializer=convert_examples_to_features_and_save_initializer,
            initargs=[
                tokenizer,
                data_config,
                save_dataset_name,
                save_labels_name,
                X_storage,
                Y_storage,
            ],
        )

        total_output = 0
        for res in pool.imap_unordered(
            convert_examples_to_features_and_save_hlp, input_args, chunksize=1
        ):
            total_output += res
        pool.close()

    # Verify that sentences are unique and saved correctly
    mmap_file = np.memmap(save_dataset_name, dtype=X_storage, mode="r")
    all_uniq_ids = set()
    for i in tqdm(range(total_input), desc="Checking sentence uniqueness"):
        assert mmap_file["sent_idx"][i] != -1, f"Index {i} has -1 sent idx"
        uniq_id_without_al = str(
            f"{mmap_file['sent_idx'][i]}.{mmap_file['subsent_idx'][i]}"
        )
        assert (
            uniq_id_without_al not in all_uniq_ids
        ), f"Idx {uniq_id_without_al} is not unique and already in data"
        all_uniq_ids.add(uniq_id_without_al)

    log_rank_0_debug(
        logger,
        f"Done with extracting examples in {time.time()-start}. "
        f"Total lines seen {total_input}. Total lines kept {total_output}",
    )
    return


def convert_examples_to_features_and_save_hlp(input_dict):
    return convert_examples_to_features_and_save_single(
        input_dict,
        tokenizer_global,
        entitysymbols_global,
        mmap_file_global,
        mmap_label_file_global,
    )


def convert_examples_to_features_and_save_single(
    input_dict, tokenizer, entitysymbols, mmap_file, mmap_label_file
):
    """Convert examples to features multiprocessing helper."""
    file_name = input_dict["file_name"]
    in_file_lines = input_dict["in_file_lines"]
    save_file_offset = input_dict["save_file_offset"]
    ex_print_mod = input_dict["ex_print_mod"]
    guid_dtype = input_dict["guid_dtype"]
    print_examples = input_dict["pred_examples"]
    max_aliases = input_dict["max_aliases"]
    max_seq_len = input_dict["max_seq_len"]
    split = input_dict["split"]
    train_in_candidates = input_dict["train_in_candidates"]
    total_saved_features = 0
    for idx, in_line in tqdm(
        enumerate(open(file_name, "r", encoding="utf-8")),
        total=in_file_lines,
        desc=f"Processing {file_name}",
    ):
        example = InputExample.from_dict(ujson.loads(in_line))
        example_idx = save_file_offset + idx
        train_aliases_to_predict_arr = (
            example.train_aliases_to_predict_arr
        )  # Stores all aliases to be scored for all anchor labels (gold or not)
        aliases_to_predict = (
            example.aliases_to_predict
        )  # Stores all aliases to be scored for all anchor labels (if train) and for true anchors (if eval)
        spans = example.spans
        word_indices = tokenizer.convert_tokens_to_ids(example.phrase_tokens)
        aliases = example.aliases
        qids = example.qids
        alias_list_pos = example.alias_list_pos
        assert (
            len(aliases_to_predict) >= 0
        ), f"There are no aliases to predict for an example. This should not happen at this point."
        assert (
            len(aliases) <= max_aliases
        ), f"Each example should have no more that {max_aliases} max aliases. {example} does."
        example_aliases = np.ones(max_aliases) * PAD_ID
        example_aliases_locs_start = np.ones(max_aliases) * PAD_ID
        example_aliases_locs_end = np.ones(max_aliases) * PAD_ID
        example_alias_list_pos = np.ones(max_aliases) * PAD_ID
        # this stores the true entities we use to compute loss - (all anchors for train and true anchors for dev/test)
        example_true_cand_positions_for_loss = np.ones(max_aliases) * PAD_ID
        # this stores the true entity ids - used for building type data
        example_true_eids_for_loss = np.ones(max_aliases) * PAD_ID
        # this stores the true entities for all aliases seen by model - all anchors for both train and eval
        example_true_cand_positions_for_train = np.ones(max_aliases) * PAD_ID
        # used to keep track of original alias index in the list
        for idx, (alias, span_idx, qid, alias_pos) in enumerate(
            zip(aliases, spans, qids, alias_list_pos)
        ):
            span_start_idx, span_end_idx = span_idx
            assert (
                span_start_idx >= 0
            ), f"{span_start_idx} is not supposed to be less that zero"
            assert (
                span_end_idx <= max_seq_len
            ), f"{span_end_idx} is beyond max len {max_seq_len}."
            # generate indexes into alias table.
            assert entitysymbols.alias_exists(
                alias
            ), f"Alias {alias} not in alias mapping"
            alias_trie_idx = entitysymbols.get_alias_idx(alias)
            alias_qids = np.array(entitysymbols.get_qid_cands(alias))
            # When doing eval, we allow for QID to be "Q-1" so we can predict anyways -
            # as this QID isn't in our alias_qids, the assert below verifies that this will happen only for test/dev
            eid = -1
            if entitysymbols.qid_exists(qid):
                eid = entitysymbols.get_eid(qid)
            if qid not in alias_qids:
                # assert not data_args.train_in_candidates
                if not train_in_candidates:
                    # set class label to be "not in candidate set"
                    gold_cand_K_idx = 0
                else:
                    # if we are not using a NC (no candidate) but are in eval mode, we let the gold candidate
                    # not be in the candidate set we give in a true index of -2, meaning our model will
                    # always get this example incorrect
                    assert split in ["test", "dev",], (
                        f"Expected split of 'test' or 'dev'. If you are training, "
                        f"the QID must be in the candidate list for data_args.train_in_candidates to be True"
                    )
                    gold_cand_K_idx = -2
            else:
                # Here we are getting the correct class label for training.
                # Our training is "which of the max_entities entity candidates is the right one
                # (class labels 1 to max_entities) or is it none of these (class label 0)".
                # + (not discard_noncandidate_entities) is to ensure label 0 is
                # reserved for "not in candidate set" class
                gold_cand_K_idx = np.nonzero(alias_qids == qid)[0][0] + (
                    not train_in_candidates
                )
            assert gold_cand_K_idx < entitysymbols.max_candidates + int(
                not train_in_candidates
            ), (
                f"The qid {qid} and alias {alias} is not in the top {entitysymbols.max_candidates} max candidates. "
                f"The QID must be within max candidates."
            )
            example_aliases[idx : idx + 1] = alias_trie_idx
            example_aliases_locs_start[idx : idx + 1] = span_start_idx
            # The span_idxs are [start, end). We want [start, end]. So subtract 1 from end idx.
            example_aliases_locs_end[idx : idx + 1] = span_end_idx - 1
            example_alias_list_pos[idx : idx + 1] = alias_pos
            # leave as -1 if it's not an alias we want to predict;
            # we get these if we split a sentence and need to only predict subsets
            if idx in aliases_to_predict:
                example_true_cand_positions_for_loss[idx : idx + 1] = gold_cand_K_idx
                example_true_eids_for_loss[idx : idx + 1] = eid
            if idx in train_aliases_to_predict_arr:
                example_true_cand_positions_for_train[idx : idx + 1] = gold_cand_K_idx
        # drop example if we have nothing to predict (no valid aliases)
        if all(example_aliases == PAD_ID):
            logging.error(
                f"There were 0 aliases in this example {example}. This shouldn't happen."
            )
            sys.exit(0)
        total_saved_features += 1
        feature = InputFeatures(
            start_idx_in_sent=example_aliases_locs_start,
            end_idx_in_sent=example_aliases_locs_end,
            alias_idx=example_aliases,
            word_indices=word_indices,
            gold_cand_K_idx=example_true_cand_positions_for_loss,
            gold_eid=example_true_eids_for_loss,
            for_dump_gold_cand_K_idx_train=example_true_cand_positions_for_train,
            alias_list_pos=example_alias_list_pos,
            sent_idx=int(example.sent_idx),
            subsent_idx=int(example.subsent_idx),
            guid=np.array(
                [
                    (
                        int(example.sent_idx),
                        int(example.subsent_idx),
                        example_alias_list_pos,
                    )
                ],
                dtype=guid_dtype,
            ),
        )
        # Write feature
        # We are storing mmap file in column format, so column name first
        mmap_file["sent_idx"][example_idx] = feature.sent_idx
        mmap_file["subsent_idx"][example_idx] = feature.subsent_idx
        mmap_file["guids"][example_idx] = feature.guid
        mmap_file["start_span_idx"][example_idx] = feature.start_idx_in_sent
        mmap_file["end_span_idx"][example_idx] = feature.end_idx_in_sent
        mmap_file["alias_idx"][example_idx] = feature.alias_idx
        mmap_file["token_ids"][example_idx] = feature.word_indices
        mmap_file["alias_orig_list_pos"][example_idx] = feature.alias_list_pos
        mmap_file["for_dump_gold_cand_K_idx_train"][
            example_idx
        ] = feature.for_dump_gold_cand_K_idx_train
        mmap_file["gold_eid"][example_idx] = feature.gold_eid
        mmap_label_file["gold_cand_K_idx"][example_idx] = feature.gold_cand_K_idx
        if example_idx % ex_print_mod == 0:
            # Make one string for distributed computation consistency
            output_str = ""
            output_str += "*** Example ***" + "\n"
            output_str += (
                f"guid:                            {example.sent_idx} subsent {example.subsent_idx}"
                + "\n"
            )
            output_str += (
                f"examples:                        {' '.join([str(x) for x in example.phrase_tokens])}"
                + "\n"
            )
            output_str += f"spans:                           {example.spans}" + "\n"
            output_str += (
                f"aliases_to_predict:              {example.aliases_to_predict}" + "\n"
            )
            output_str += (
                f"train_aliases_to_predict_arr:    {example.train_aliases_to_predict_arr}"
                + "\n"
            )
            output_str += (
                f"alias_list_pos:                  {example.alias_list_pos}" + "\n"
            )
            output_str += f"aliases:                         {example.aliases}" + "\n"
            output_str += f"qids:                            {example.qids}" + "\n"
            output_str += "*** Feature ***" + "\n"
            output_str += (
                f"start_idx_in_sent:               {feature.start_idx_in_sent}" + "\n"
            )
            output_str += (
                f"end_idx_in_sent:                 {feature.end_idx_in_sent}" + "\n"
            )
            output_str += (
                f"gold_cand_K_idx:                 {feature.gold_cand_K_idx}" + "\n"
            )
            output_str += f"gold_eid:                        {feature.gold_eid}" + "\n"
            output_str += (
                f"for_dump_gold_cand_K_idx_train:  {feature.for_dump_gold_cand_K_idx_train}"
                + "\n"
            )
            output_str += f"guid:                            {feature.guid}" + "\n"
            if print_examples:
                print(output_str)
    mmap_file.flush()
    mmap_label_file.flush()
    return total_saved_features


def build_and_save_type_features_initializer(
    data_config,
    temp_qidfile,
    save_type_labels_name,
    Y_type_storage,
    X_gold_eid,
    Y_gold_cand_K_idx,
):
    global eid2type_global
    eid2type_global = utils.load_single_item_trie(temp_qidfile)
    global mmap_label_file_global
    mmap_label_file_global = np.memmap(
        save_type_labels_name, dtype=Y_type_storage, mode="r+"
    )
    global train_in_candidates_global
    train_in_candidates_global = data_config.train_in_candidates
    global Y_gold_cand_K_idx_global
    Y_gold_cand_K_idx_global = Y_gold_cand_K_idx
    global X_gold_eid_global
    X_gold_eid_global = X_gold_eid


def build_and_save_type_features(
    save_type_labels_name,
    Y_type_storage,
    X_gold_eid,
    Y_gold_cand_K_idx,
    data_config,
    dataset_threads,
    eid2type,
):
    """Using the already prepped training data, generates the labels for the
    type prediction task.

    Args:
        save_type_labels_name: memmap filename to save the type labels
        Y_type_storage: storage type for memmap file
        X_gold_eid: the gold entity eid (saved in X_dict) -- the eid is only use in this function
        Y_gold_cand_K_idx: the gold label for disambiguation
        data_config: data config
        dataset_threads: number of threads
        eid2type: Dict of eid to type ids

    Returns:
    """
    num_processes = min(dataset_threads, int(0.8 * multiprocessing.cpu_count()))

    # IMPORTANT: for distributed writing to memmap files, you must create them in w+ mode before being opened in r+
    memfile = np.memmap(
        save_type_labels_name,
        dtype=Y_type_storage,
        mode="w+",
        shape=(len(Y_gold_cand_K_idx),),
        order="C",
    )
    # We'll use the -1 to check that things were written correctly later because at the end, there should be no -1

    memfile["gold_type_id"][:] = -2

    if num_processes == 1:
        input_idxs = list(range(len(Y_gold_cand_K_idx)))
        build_and_save_type_features_single(
            input_idxs,
            data_config.train_in_candidates,
            X_gold_eid,
            Y_gold_cand_K_idx,
            memfile,
            eid2type,
        )
    else:
        temp_qidfile = tempfile.NamedTemporaryFile()
        log_rank_0_info(
            logger,
            f"Creating type prediction labeled data using {num_processes} threads",
        )
        utils.create_single_item_trie(eid2type, temp_qidfile.name)

        input_args = list(range(len(Y_gold_cand_K_idx)))
        chunk_size = int(np.ceil(len(input_args) / num_processes))
        input_chunks = [
            input_args[i : i + chunk_size]
            for i in range(0, len(input_args), chunk_size)
        ]

        log_rank_0_debug(logger, f"Starting pool with {num_processes} processes")
        pool = multiprocessing.Pool(
            processes=num_processes,
            initializer=build_and_save_type_features_initializer,
            initargs=[
                data_config,
                temp_qidfile.name,
                save_type_labels_name,
                Y_type_storage,
                X_gold_eid,
                Y_gold_cand_K_idx,
            ],
        )
        cnt = 0
        for res in tqdm(
            pool.imap_unordered(
                build_and_save_type_features_hlp, input_chunks, chunksize=1
            ),
            total=len(input_chunks),
            desc="Building type data",
        ):
            cnt += res
        pool.close()
        temp_qidfile.close()

    memfile = np.memmap(save_type_labels_name, dtype=Y_type_storage, mode="r")
    for i in tqdm(range(len(Y_gold_cand_K_idx)), desc="Verifying type labels"):
        assert all(memfile["gold_type_id"][i] != -2), f"Memfile at {i} is -2."
    return


def build_and_save_type_features_hlp(input_idxs):
    return build_and_save_type_features_single(
        input_idxs,
        train_in_candidates_global,
        X_gold_eid_global,
        Y_gold_cand_K_idx_global,
        mmap_label_file_global,
        eid2type_global,
    )


def build_and_save_type_features_single(
    input_idxs,
    train_in_candidates,
    X_gold_eid,
    Y_gold_cand_K_idx,
    mmap_label_file,
    eid2type,
):
    for i in tqdm(input_idxs, desc="Processing types"):
        gold_eids = X_gold_eid[i]
        for j, gold_cand in enumerate(Y_gold_cand_K_idx[i]):
            # If padded entity or special eval padded entity
            if gold_cand == -1:
                type_id = -1
            # If the "not in candidate" entity
            elif gold_cand == 0 and not train_in_candidates:
                type_id = 0
            # Otherwise, the qid is the gold_cand[j] index into the list of QID candidates
            else:
                # str() for marisa tri
                gold_eid = str(gold_eids[j])
                type_id = 0
                if gold_eid in eid2type:
                    if isinstance(eid2type, dict):
                        type_id = eid2type[gold_eid]
                    else:
                        # The [0] is because this is a record trie and it returns lists of elements
                        type_id = eid2type[gold_eid][0][0]
            mmap_label_file["gold_type_id"][i][j] = type_id
    mmap_label_file.flush()
    return len(input_idxs)


class BootlegDataset(EmmentalDataset):
    """Bootleg Dataset class to be used in dataloader.

    Args:
        main_args: input config
        name: internal dataset name
        dataset: dataset file
        use_weak_label: whether to use weakly labeled mentions or not
        tokenizer: sentence tokenizer
        entity_symbols: entity database class
        dataset_threads: number of threads to use
        split: data split
        is_bert: is the tokenizer a BERT or not
        batch_on_the_fly_kg_adj: special dictionary for stories KG adjacency information
                                 that needs to be prepped in the _get_item_ method

    Returns:
    """

    def __init__(
        self,
        main_args,
        name,
        dataset,
        use_weak_label,
        tokenizer,
        entity_symbols,
        dataset_threads,
        split="train",
        is_bert=True,
        batch_on_the_fly_kg_adj=None,
    ):
        if batch_on_the_fly_kg_adj is None:
            batch_on_the_fly_kg_adj = {}
        log_rank_0_info(
            logger,
            f"Starting to build data for {split} from {dataset}",
        )
        global_start = time.time()
        data_config = main_args.data_config
        spawn_method = main_args.run_config.spawn_method
        log_rank_0_debug(logger, f"Setting spawn method to be {spawn_method}")
        orig_spawn = multiprocessing.get_start_method()
        multiprocessing.set_start_method(spawn_method, force=True)

        # Unique identifier is sentence index, subsentence index (due to sentence splitting), and aliases in split
        guid_dtype = np.dtype(
            [
                ("sent_idx", "i8", 1),
                ("subsent_idx", "i8", 1),
                ("alias_orig_list_pos", "i8", (data_config.max_aliases,)),
            ]
        )
        # Storage for saving the data. entity_cand_eid and batch_on_the_fly_kg_adj get filled in in __get_item__
        self.X_storage, self.Y_storage, self.Y_type_storage = (
            [
                ("guids", guid_dtype, 1),
                ("sent_idx", "i8", 1),
                ("subsent_idx", "i8", 1),
                ("start_span_idx", "i8", (data_config.max_aliases,)),
                ("end_span_idx", "i8", (data_config.max_aliases,)),
                ("alias_idx", "i8", (data_config.max_aliases,)),
                ("token_ids", "i8", data_config.max_seq_len),
                ("alias_orig_list_pos", "i8", (data_config.max_aliases,)),
                (
                    "gold_eid",
                    "i8",
                    (data_config.max_aliases,),
                ),  # What the eid of the gold entity is -- just used for building our type table
                (
                    "for_dump_gold_cand_K_idx_train",
                    "i8",
                    (data_config.max_aliases,),
                )  # Which of the K candidates is correct. Only used in dump_pred to stitch sub-sentences together
                # ("entity_cand_eid", 'i8', data_config.max_aliases*<num_candidates_per_alias>) (see __get_item__)
                # ("entity_cand_eid_mask", 'i8', data_config.max_aliases*<num_candidates_per_alias>) (see __get_item__)
                # ("batch_on_the_fly_kg_adj", 'i8', ???) (see __get_item__). Shape depends on implementing class.
            ],
            [
                (
                    "gold_cand_K_idx",
                    "i8",
                    (data_config.max_aliases,),
                ),  # Which of the K candidates is correct.
            ],
            [("gold_type_id", "i8", (data_config.max_aliases,))],
        )

        # Table to map from alias_idx to entity_cand_eid used in the __get_item__
        self.alias2cands_model = AliasEntityTable(
            data_config=data_config, entity_symbols=entity_symbols
        )
        # Total number of entities used in the __get_item__
        self.num_entities_with_pad_and_nocand = (
            entity_symbols.num_entities_with_pad_and_nocand
        )

        self.raw_filename = dataset
        # Folder for all mmap saved files
        save_dataset_folder = data_utils.get_save_data_folder(
            data_config, use_weak_label, self.raw_filename
        )
        utils.ensure_dir(save_dataset_folder)
        # Folder for temporary output files
        temp_output_folder = os.path.join(
            data_config.data_dir,
            data_config.data_prep_dir,
            f"prep_{split}_dataset_files",
        )
        utils.ensure_dir(temp_output_folder)
        # Input step 1
        create_ex_indir = os.path.join(temp_output_folder, "create_examples_input")
        utils.ensure_dir(create_ex_indir)
        # Input step 2
        create_ex_outdir = os.path.join(temp_output_folder, "create_examples_output")
        utils.ensure_dir(create_ex_outdir)
        # Meta data saved files
        meta_file = os.path.join(temp_output_folder, "meta_data.json")
        # File for standard training data
        self.save_dataset_name = os.path.join(save_dataset_folder, "ned_data.bin")
        # File for standard labels
        self.save_labels_name = os.path.join(save_dataset_folder, "ned_label.bin")
        # File for type labels
        self.save_type_labels_name = None
        # =======================================================================================
        # =======================================================================================
        # =======================================================================================
        # STANDARD DISAMBIGUATION
        # =======================================================================================
        # =======================================================================================
        # =======================================================================================
        log_rank_0_debug(
            logger,
            f"Seeing if {self.save_dataset_name} exists and {self.save_labels_name} exists",
        )
        if (
            data_config.overwrite_preprocessed_data
            or (not os.path.exists(self.save_dataset_name))
            or (not os.path.exists(self.save_labels_name))
        ):
            st_time = time.time()
            log_rank_0_info(
                logger,
                f"Building dataset from scratch. Saving to {save_dataset_folder}.",
            )
            create_examples(
                dataset,
                create_ex_indir,
                create_ex_outdir,
                meta_file,
                data_config,
                dataset_threads,
                use_weak_label,
                split,
                is_bert,
                tokenizer,
            )
            try:
                convert_examples_to_features_and_save(
                    meta_file,
                    guid_dtype,
                    data_config,
                    dataset_threads,
                    use_weak_label,
                    split,
                    is_bert,
                    self.save_dataset_name,
                    self.save_labels_name,
                    self.X_storage,
                    self.Y_storage,
                    tokenizer,
                    entity_symbols,
                )
                log_rank_0_debug(
                    logger,
                    f"Finished prepping disambig training data in {time.time() - st_time}",
                )
            except Exception as e:
                tb = traceback.TracebackException.from_exception(e)
                logger.error(e)
                logger.error(traceback.format_exc())
                logger.error("\n".join(tb.stack.format()))
                os.remove(self.save_dataset_name)
                os.remove(self.save_labels_name)
                shutil.rmtree(save_dataset_folder, ignore_errors=True)
                # shutil.rmtree(temp_output_folder, ignore_errors=True)
                raise

        log_rank_0_info(
            logger,
            f"Loading data from {self.save_dataset_name} and {self.save_labels_name}",
        )
        X_dict, Y_dict = self.build_data_dicts(
            self.save_dataset_name,
            self.save_labels_name,
            self.X_storage,
            self.Y_storage,
        )

        # =======================================================================================
        # =======================================================================================
        # =======================================================================================
        # TYPE PREDICTION
        # =======================================================================================
        # =======================================================================================
        # =======================================================================================
        self.add_type_pred = False
        if data_config.type_prediction.use_type_pred:
            self.add_type_pred = True
            self.save_type_labels_name = os.path.join(
                save_dataset_folder,
                f"{os.path.splitext(os.path.basename(data_config.type_prediction.type_labels))[0]}_type_pred_label.bin",
            )
            log_rank_0_debug(logger, f"Seeing if {self.save_type_labels_name} exists")
            if data_config.overwrite_preprocessed_data or (
                not os.path.exists(self.save_type_labels_name)
            ):
                st_time = time.time()
                log_rank_0_info(logger, f"Building type labels from scatch.")
                eid2type = read_in_types(data_config)
                log_rank_0_debug(logger, f"Finished reading in type data")
                try:
                    # Creating/saving data
                    gold_eid_memmap = np.memmap(
                        self.save_dataset_name, dtype=self.X_storage, mode="r"
                    )["gold_eid"]
                    gold_cand_K_idx_memmap = np.memmap(
                        self.save_labels_name, dtype=self.Y_storage, mode="r"
                    )["gold_cand_K_idx"]
                    build_and_save_type_features(
                        self.save_type_labels_name,
                        self.Y_type_storage,
                        gold_eid_memmap,
                        gold_cand_K_idx_memmap,
                        data_config,
                        dataset_threads,
                        eid2type,
                    )
                    log_rank_0_debug(
                        logger, f"Finished prepping data in {time.time() - st_time}"
                    )
                except Exception as e:
                    tb = traceback.TracebackException.from_exception(e)
                    logger.error(e)
                    logger.error(traceback.format_exc())
                    logger.error("\n".join(tb.stack.format()))
                    os.remove(self.save_labels_name)
                    raise

            type_Y_dict = self.build_data_type_dicts(
                self.save_type_labels_name, self.Y_type_storage
            )
            for k, val in type_Y_dict.items():
                assert (
                    k not in Y_dict
                ), f"{k} is already in Y_dict but this is a key for the types"
                Y_dict[k] = val

        # =======================================================================================
        # =======================================================================================
        # =======================================================================================
        # KG EMBEDDINGS (embeddings that need to be prepped in the get_item method)
        # =======================================================================================
        # =======================================================================================
        # =======================================================================================
        # Dicts (batch_on_the_fly_kg_adj and batch_in_data_prep_kg_adj) are created in data.py
        # Each of these is a dictionary of emb_key (passed on config) -> dict of
        # kg_adj: adjacency matrix to pass into kg_adj_process_func for prep
        # kg_adj_process_func: function to process kg_adj
        # prep_file: file used to prep/save the kg_adj matrix. Used for loading/saving state.
        self.batch_on_the_fly_kg_adj = batch_on_the_fly_kg_adj

        log_rank_0_debug(logger, f"Removing temporary output files")
        shutil.rmtree(temp_output_folder, ignore_errors=True)
        log_rank_0_info(
            logger,
            f"Final data initialization time for {split} is {time.time() - global_start}s",
        )
        # Set spawn back to original/default, which is "fork" or "spawn".
        # This is needed for the Meta.config to be correctly passed in the collate_fn.
        multiprocessing.set_start_method(orig_spawn, force=True)
        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="guids")

    @classmethod
    def build_data_dicts(
        cls, save_dataset_name, save_labels_name, X_storage, Y_storage
    ):
        """Returns the X_dict and Y_dict of inputs and labels for the entity
        disambiguation task.

        Args:
            save_dataset_name: memmap file name with inputs
            save_labels_name: memmap file name with labels
            X_storage: memmap storage for inputs
            Y_storage: memmap storage labels

        Returns: X_dict of inputs and Y_dict of labels for Emmental datasets
        """
        X_dict, Y_dict = (
            {
                "guids": [],
                "sent_idx": [],
                "subsent_idx": [],
                "start_span_idx": [],
                "end_span_idx": [],
                "alias_idx": [],
                "token_ids": [],
                "alias_orig_list_pos": [],  # list of original position in the alias list this example is (see eval)
                "gold_eid": [],  # List of gold entity eids (used for building type label table and debugging)
                "for_dump_gold_cand_K_idx_train": []  # list of gold indices without subsentence masking (see eval)
                # "entity_cand_eid": [] --- this gets filled in in the __get_item__ method
                # "entity_cand_eid_mask": [] --- this gets filled in in the __get_item__ method
                # "batch_on_the_fly_kg_adj": [] --- this gets filled in in the __get_item__ method
            },
            {
                "gold_cand_K_idx": [],
            },
        )
        mmap_file = np.memmap(save_dataset_name, dtype=X_storage, mode="r")
        mmap_label_file = np.memmap(save_labels_name, dtype=Y_storage, mode="r")
        X_dict["sent_idx"] = torch.from_numpy(mmap_file["sent_idx"])
        X_dict["subsent_idx"] = torch.from_numpy(mmap_file["subsent_idx"])
        X_dict["guids"] = mmap_file["guids"]  # uid doesn't need to be tensor
        X_dict["start_span_idx"] = torch.from_numpy(mmap_file["start_span_idx"])
        X_dict["end_span_idx"] = torch.from_numpy(mmap_file["end_span_idx"])
        X_dict["alias_idx"] = torch.from_numpy(mmap_file["alias_idx"])
        X_dict["token_ids"] = torch.from_numpy(mmap_file["token_ids"])
        X_dict["alias_orig_list_pos"] = torch.from_numpy(
            mmap_file["alias_orig_list_pos"]
        )
        X_dict["gold_eid"] = torch.from_numpy(mmap_file["gold_eid"])
        X_dict["for_dump_gold_cand_K_idx_train"] = torch.from_numpy(
            mmap_file["for_dump_gold_cand_K_idx_train"]
        )
        Y_dict["gold_cand_K_idx"] = torch.from_numpy(mmap_label_file["gold_cand_K_idx"])
        return X_dict, Y_dict

    @classmethod
    def build_data_type_dicts(cls, save_labels_name, Y_storage):
        """Returns the Y_dict of labels for the type prediction task.

        Args:
            save_labels_name: memmap file name with type labels
            Y_storage: memmap storage type

        Returns: Dict of labels
        """
        Y_dict = {"gold_type_id": []}
        mmap_label_file = np.memmap(save_labels_name, dtype=Y_storage, mode="r")
        Y_dict["gold_type_id"] = torch.from_numpy(mmap_label_file["gold_type_id"])
        return Y_dict

    def __getitem__(self, index):
        r"""Get item by index.

        Args:
          index(index): The index of the item.
        Returns:
          Tuple[Dict[str, Any], Dict[str, Tensor]]: Tuple of x_dict and y_dict
        """
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}

        # Get the entity_cand_eid
        entity_cand_eid = self.alias2cands_model(x_dict["alias_idx"]).long()
        x_dict["entity_cand_eid_mask"] = entity_cand_eid == -1
        # Handles the index errors with -1 indexing into an embedding
        x_dict["entity_cand_eid"] = torch.where(
            entity_cand_eid >= 0,
            entity_cand_eid,
            (
                torch.ones_like(entity_cand_eid, dtype=torch.long)
                * (self.num_entities_with_pad_and_nocand - 1)
            ),
        )
        # Load the batch on the fly kg embeddings
        kg_prepped_embs = {}
        for emb_key in self.batch_on_the_fly_kg_adj:
            kg_adj = self.batch_on_the_fly_kg_adj[emb_key]["kg_adj"]
            prep_func = self.batch_on_the_fly_kg_adj[emb_key]["kg_adj_process_func"]
            kg_prepped_embs[emb_key] = torch.from_numpy(
                prep_func(x_dict["entity_cand_eid"].cpu(), kg_adj).reshape(1, -1)
            )
        x_dict["batch_on_the_fly_kg_adj"] = kg_prepped_embs
        return x_dict, y_dict

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["X_dict"]
        del state["Y_dict"]
        for emb_key in self.batch_on_the_fly_kg_adj:
            assert "kg_adj" in self.batch_on_the_fly_kg_adj[emb_key], (
                f"Something went wrong with saving state and self.batch_on_the_fly_kg_adj with {emb_key}. "
                f"It does not have kg_adj key."
            )
            # We will nullify now and reload later
            self.batch_on_the_fly_kg_adj[emb_key]["kg_adj"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.X_dict, self.Y_dict = self.build_data_dicts(
            self.save_dataset_name,
            self.save_labels_name,
            self.X_storage,
            self.Y_storage,
        )
        if self.add_type_pred:
            type_Y_dict = self.build_data_type_dicts(
                self.save_type_labels_name, self.Y_type_storage
            )
            for k, val in type_Y_dict.items():
                assert (
                    k not in self.Y_dict
                ), f"{k} is already in Y_dict but this is a key for the types"
                self.Y_dict[k] = val
        for emb_key in self.batch_on_the_fly_kg_adj:
            assert "prep_file" in self.batch_on_the_fly_kg_adj[emb_key], (
                f"Something went wrong with loading state and self.batch_on_the_fly_kg_adj with {emb_key}. "
                f"It does not have prep_file key."
            )
            self.batch_on_the_fly_kg_adj[emb_key]["kg_adj"] = scipy.sparse.load_npz(
                self.batch_on_the_fly_kg_adj[emb_key]["prep_file"]
            )
        return state

    def __repr__(self):
        return (
            f"Bootleg Dataset. Data at {self.save_dataset_name}. "
            f"Labels at {self.save_labels_name}. Use type pred is {self.add_type_pred}."
        )
