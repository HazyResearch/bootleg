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
import torch
import ujson
from tqdm import tqdm

from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.layers.alias_to_ent_encoder import AliasEntityTable
from bootleg.symbols.constants import ANCHOR_KEY, PAD_ID, STOP_WORDS
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import data_utils, utils
from bootleg.utils.data_utils import read_in_relations, read_in_types
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
        alias_to_predict,
        span,
        phrase,
        alias,
        qid,
        qid_cnt_mask_score,
    ):
        assert (
            type(sent_idx) is int
        ), f"We need the sentence index is an int. You have {type(sent_idx)}"
        self.sent_idx = int(sent_idx)
        self.subsent_idx = subsent_idx
        self.alias_list_pos = alias_list_pos
        self.alias_to_predict = alias_to_predict
        self.span = span
        self.phrase = phrase
        self.alias = alias
        self.qid = qid
        self.qid_cnt_mask_score = qid_cnt_mask_score

    def to_dict(self):
        return {
            "sent_idx": self.sent_idx,
            "subsent_idx": self.subsent_idx,
            "alias_list_pos": self.alias_list_pos,
            "alias_to_predict": self.alias_to_predict,
            "span": self.span,
            "phrase": self.phrase,
            "alias": self.alias,
            "qid": self.qid,
            "qid_cnt_mask_score": self.qid_cnt_mask_score,
        }

    @classmethod
    def from_dict(cls, in_dict):
        return cls(
            in_dict["sent_idx"],
            in_dict["subsent_idx"],
            in_dict["alias_list_pos"],
            in_dict["alias_to_predict"],
            in_dict["span"],
            in_dict["phrase"],
            in_dict["alias"],
            in_dict["qid"],
            in_dict["qid_cnt_mask_score"],
        )


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        alias_idx,
        word_input_ids,
        word_token_type_ids,
        word_attention_mask,
        word_qid_cnt_mask_score,
        gold_eid,
        gold_cand_K_idx,
        for_dump_gold_cand_K_idx_train,
        alias_list_pos,
        sent_idx,
        subsent_idx,
        guid,
    ):
        self.alias_idx = alias_idx
        self.word_input_ids = word_input_ids
        self.word_token_type_ids = word_token_type_ids
        self.word_attention_mask = word_attention_mask
        self.word_qid_cnt_mask_score = word_qid_cnt_mask_score
        self.gold_eid = gold_eid
        self.gold_cand_K_idx = gold_cand_K_idx
        self.for_dump_gold_cand_K_idx_train = for_dump_gold_cand_K_idx_train
        self.alias_list_pos = alias_list_pos
        self.sent_idx = sent_idx
        self.subsent_idx = subsent_idx
        self.guid = guid

    def to_dict(self):
        return {
            "alias_idx": self.alias_idx,
            "word_input_ids": self.word_input_ids,
            "word_token_type_ids": self.word_token_type_ids,
            "word_attention_mask": self.word_attention_mask,
            "word_qid_cnt_mask_score": self.word_qid_cnt_mask_score,
            "gold_eid": self.gold_eid,
            "gold_cand_K_idx": self.gold_cand_K_idx,
            "for_dump_gold_cand_K_idx_train": self.for_dump_gold_cand_K_idx_train,
            "alias_list_pos": self.alias_list_pos,
            "sent_idx": self.sent_idx,
            "subsent_idx": self.subsent_idx,
            "guid": self.guid,
        }

    @classmethod
    def from_dict(cls, in_dict):
        return cls(
            in_dict["alias_idx"],
            in_dict["word_input_ids"],
            in_dict["word_token_type_ids"],
            in_dict["word_attention_mask"],
            in_dict["word_qid_cnt_mask_score"],
            in_dict["gold_eid"],
            in_dict["gold_cand_K_idx"],
            in_dict["for_dump_gold_cand_K_idx_train"],
            in_dict["alias_list_pos"],
            in_dict["sent_idx"],
            in_dict["subsent_idx"],
            in_dict["guid"],
        )


def extract_context_windows(span, tokens, max_seq_window_len):
    """
    Extracts the left and right context window around a span

    Args:
        span: span (left and right values)
        tokens: tokens
        max_seq_window_len: maximum window length around a span

    Returns: left context, right context

    """
    # If more tokens to the right, shift weight there
    if span[0] < len(tokens) - span[1]:
        prev_context = tokens[max(0, span[0] - max_seq_window_len // 2) : span[0]]
        next_context = tokens[
            span[1] : span[1] + max_seq_window_len - len(prev_context)
        ]
    else:
        next_context = tokens[span[1] : span[1] + max_seq_window_len // 2]
        prev_context = tokens[
            max(0, span[0] - (max_seq_window_len - len(next_context))) : span[0]
        ]
    return prev_context, next_context


def get_structural_entity_str(items, max_tok_len, sep_tok):
    """For structural resources in items. Returns sep_tok joined list of items such
    that the number of words is less than max tok len.

    Args:
        items: list of structural resources
        max_tok_len: maximum token length
        sep_tok: token to separate out resources

    Returns:
        result string, number of items that went beyond ``max_tok_len``

    """
    i = 1
    over_len = 0
    while True:
        res = f" {sep_tok} " + f" {sep_tok} ".join(items[:i])
        if len(res.split()) > max_tok_len or i > len(items):
            if i < len(items):
                over_len = 1
            res = f" {sep_tok} " + f" {sep_tok} ".join(items[: max(1, i - 1)])
            break
        i += 1
    return res, over_len


def get_entity_string(
    qid,
    constants,
    entity_symbols,
    qid2relations,
    qid2typenames,
):
    """
    For each entity, generates a string that is fed into a language model to generate an entity embedding. Returns
    all tokens that are the title of the entity (even if in the description)

    Args:
        qid: QID
        constants: Dict of constants
        entity_symbols: entity symbols
        qid2relations: Dict of QID to list of relations
        qid2typenames: Dict of QID to list of types

    Returns: entity strings, number of types over max length, number of relations over max length

    """
    over_kg_len = 0
    over_type_len = 0
    desc_str = (
        "[ent_desc] " + entity_symbols.get_desc(qid) if constants["use_desc"] else ""
    )
    title_str = entity_symbols.get_title(qid) if entity_symbols.qid_exists(qid) else ""

    # To add kgs, sep by "[ent_kg]" and then truncate to max_ent_kg_len
    # Then merge with description text
    if constants["use_kg"]:
        kg_str, over_len = get_structural_entity_str(
            qid2relations.get(qid, []),
            constants["max_ent_kg_len"],
            "[ent_kg]",
        )
        over_kg_len += over_len
        desc_str = " ".join([kg_str, desc_str])
    # To add types, sep by "[ent_type]" and then truncate to max_type_ent_len
    # Then merge with description text
    if constants["use_types"]:
        type_str, over_len = get_structural_entity_str(
            qid2typenames.get(qid, []),
            constants["max_ent_type_len"],
            "[ent_type]",
        )
        over_type_len += over_len
        desc_str = " ".join([type_str, desc_str])
    ent_str = " ".join([title_str, desc_str])
    # Remove double spaces
    ent_split = ent_str.split()
    ent_str = " ".join(ent_split)
    title_spans = []
    if len(title_str) > 0:
        # Find all occurrences of title words in the ent_str (helps if description has abbreviated name)
        # Make sure you don't mask any types or kg relations
        title_pieces = set(title_str.split())
        to_skip = False
        for e_id, ent_w in enumerate(ent_split):
            if ent_w == "[ent_type]":
                to_skip = True
            if ent_w == "[ent_desc]":
                to_skip = False
            if to_skip:
                continue
            if ent_w in title_pieces and ent_w not in STOP_WORDS:
                title_spans.append(e_id)
        # all_title_occ = re.finditer(f"({title_str})", ent_str)
        # all_spaces = np.array([m.start() for m in re.finditer("\s", ent_str)])
        # for match in all_title_occ:
        #     start_w = np.sum(all_spaces < match.start())
        #     end_w = np.sum(all_spaces <= match.end())
        #     for i in range(start_w, end_w):
        #         title_spans.append(i)
    return ent_str, title_spans, over_type_len, over_kg_len


def create_examples_initializer(constants_dict):
    global constants_global
    constants_global = constants_dict


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
    qidcnt_file = os.path.join(
        data_config.entity_dir, data_config.entity_map_dir, data_config.qid_cnt_map
    )
    log_rank_0_debug(logger, f"Counting lines")
    total_input = sum(1 for _ in open(dataset))
    constants_dict = {
        "is_bert": is_bert,
        "use_weak_label": use_weak_label,
        "split": split,
        "qidcnt_file": qidcnt_file,
        "max_seq_len": data_config.max_seq_len,
        "max_seq_window_len": data_config.max_seq_window_len,
    }
    if num_processes == 1:
        out_file_name = os.path.join(create_ex_outdir, os.path.basename(dataset))
        res = create_examples_single(
            in_file_idx=0,
            in_file_name=dataset,
            in_file_lines=total_input,
            out_file_name=out_file_name,
            constants_dict=constants_dict,
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
                constants_dict,
            ],
        )

        total_output = 0
        input_args = list(
            zip(
                list(range(len(input_files))),
                input_files,
                input_file_lines,
                output_files,
            )
        )
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
        f"Done with extracting examples in {time.time() - start}. "
        f"Total lines seen {total_input}. Total lines kept {total_output}.",
    )
    return


def create_examples_hlp(args):
    """Create examples multiprocessing helper."""
    in_file_idx, in_file_name, in_file_lines, out_file_name = args
    return create_examples_single(
        in_file_idx,
        in_file_name,
        in_file_lines,
        out_file_name,
        constants_global,
    )


def create_examples_single(
    in_file_idx,
    in_file_name,
    in_file_lines,
    out_file_name,
    constants_dict,
):
    split = constants_dict["split"]
    max_seq_window_len = constants_dict["max_seq_window_len"]
    use_weak_label = constants_dict["use_weak_label"]

    qidcnt_file = constants_dict["qidcnt_file"]
    qid2cnt = {}
    quantile_buckets = [float(i / 100) for i in list(range(0, 101, 5))]
    # If not qid2cnt, the quantile_bucket will be 100
    quants = np.array([-1 for _ in quantile_buckets])
    quants[-1] = 0
    if os.path.exists(qidcnt_file):
        qid2cnt = ujson.load(open(qidcnt_file))
        quants = np.quantile(list(qid2cnt.values()), quantile_buckets)

    with open(out_file_name, "w", encoding="utf-8") as out_f:
        total_subsents = 0
        total_lines = 0
        for ex in tqdm(
            open(in_file_name, "r", encoding="utf-8"),
            total=in_file_lines,
            desc=f"{in_file_idx}",
            position=in_file_idx,
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
            # aliases_seen_by_model = [i for i in range(len(aliases))]
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
                # aliases_seen_by_model = [i for i in range(len(aliases))]
                anchor = [True for i in range(len(aliases))]
            # Happens if use weak labels is False
            if len(aliases) == 0:
                continue

            for subsent_idx in range(len(aliases)):
                span = spans[subsent_idx]
                alias_anchor = anchor[subsent_idx]
                alias = aliases[subsent_idx]
                qid = qids[subsent_idx]
                tokens = phrase.split()
                prev_context, next_context = extract_context_windows(
                    span, tokens, max_seq_window_len
                )
                # Set mention to be MASKed by popularity
                mention_toks = tokens[span[0] : span[1]]
                # Get the percentile bucket between [0, 100]
                qid_cnt_mask_score = quantile_buckets[sum(qid2cnt.get(qid, 0) > quants)]
                assert 0 <= qid_cnt_mask_score <= 100

                context_tokens = (
                    prev_context
                    + ["[ent_start]"]
                    + mention_toks
                    + ["[ent_end]"]
                    + next_context
                )
                new_span = [
                    context_tokens.index("[ent_start]"),
                    context_tokens.index("[ent_end]") + 1,
                ]
                context = " ".join(context_tokens)

                # alias_to_predict_arr is an index into idxs_arr/anchor_arr/aliases_arr.
                # It should only include true anchors if eval dataset.
                # During training want to backpropagate on false anchors as well
                if split != "train":
                    alias_to_predict = 0 if alias_anchor is True else -1
                else:
                    alias_to_predict = 0
                total_subsents += 1
                out_f.write(
                    ujson.dumps(
                        InputExample(
                            sent_idx=sent_idx,
                            subsent_idx=subsent_idx,
                            alias_list_pos=subsent_idx,
                            alias_to_predict=alias_to_predict,
                            span=new_span,
                            phrase=context,
                            alias=alias,
                            qid=qid,
                            qid_cnt_mask_score=qid_cnt_mask_score,
                        ).to_dict()
                    )
                    + "\n"
                )
    return {"total_lines": total_subsents, "output_filename": out_file_name}


def convert_examples_to_features_and_save_initializer(
    tokenizer,
    data_config,
    save_dataset_name,
    save_labels_name,
    X_storage,
    Y_storage,
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

    # One example per mention per candidate
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
                "in_file_idx": i,
                "in_file_lines": files_and_counts[in_file_name],
                "save_file_offset": offset,
                "ex_print_mod": int(np.ceil(total_input / 20)),
                "guid_dtype": guid_dtype,
                "is_bert": is_bert,
                "use_weak_label": use_weak_label,
                "split": split,
                "max_seq_len": data_config.max_seq_len,
                "train_in_candidates": data_config.train_in_candidates,
                "print_examples": data_config.print_examples_prep,
            }
        )
        offset += files_and_counts[in_file_name]

    if num_processes == 1:
        assert len(input_args) == 1
        total_output = convert_examples_to_features_and_save_single(
            input_args[0],
            tokenizer,
            entity_symbols,
            memmap_file,
            memmap_label_file,
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
            c = res
            total_output += c
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
        f"Done with extracting examples in {time.time() - start}. Total lines seen {total_input}. "
        f"Total lines kept {total_output}.",
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
    input_dict,
    tokenizer,
    entitysymbols,
    mmap_file,
    mmap_label_file,
):
    """Convert examples to features multiprocessing helper."""
    file_name = input_dict["file_name"]
    in_file_idx = input_dict["in_file_idx"]
    in_file_lines = input_dict["in_file_lines"]
    save_file_offset = input_dict["save_file_offset"]
    ex_print_mod = input_dict["ex_print_mod"]
    guid_dtype = input_dict["guid_dtype"]
    print_examples = input_dict["print_examples"]
    max_seq_len = input_dict["max_seq_len"]
    split = input_dict["split"]
    train_in_candidates = input_dict["train_in_candidates"]
    # if not train_in_candidates:
    #     raise NotImplementedError("train_in_candidates of False is not fully supported yet")
    max_total_input_len = max_seq_len
    total_saved_features = 0
    for idx, in_line in tqdm(
        enumerate(open(file_name, "r", encoding="utf-8")),
        total=in_file_lines,
        desc=f"Processing {file_name}",
        position=in_file_idx,
    ):
        example = InputExample.from_dict(ujson.loads(in_line))
        example_idx = save_file_offset + idx
        alias_to_predict = (
            example.alias_to_predict
        )  # Stores -1 if dev data and false anchor
        alias_list_pos = example.alias_list_pos
        span_start_idx, span_end_idx = example.span
        alias = example.alias
        qid = example.qid

        candidate_sentence_input_ids = (
            np.ones(max_total_input_len) * tokenizer.pad_token_id
        )
        candidate_sentence_attn_msks = np.ones(max_total_input_len) * 0
        candidate_sentence_token_type_ids = np.ones(max_total_input_len) * 0
        candidate_mention_cnt_ratio = np.ones(max_total_input_len) * -1
        # ===========================================================
        # GET GOLD LABEL
        # ===========================================================
        # generate indexes into alias table; -2 if unk
        if not entitysymbols.alias_exists(alias):
            # if we do not have this alias in our set, we give it an index of -2, meaning we will
            # always get it wrong in eval
            assert split in ["test", "dev",], (
                f"Expected split of 'test' or 'dev'. If you are training, "
                f"the alias {alias} must be in our entity dump"
            )
            alias_trie_idx = -2
            alias_qids = []
        else:
            alias_trie_idx = entitysymbols.get_alias_idx(alias)
            alias_qids = entitysymbols.get_qid_cands(alias)
        # When doing eval, we allow for QID to be "Q-1" so we can predict anyways -
        # as this QID isn't in our alias_qids, the assert below verifies that this will happen only for test/dev
        eid = -1
        if entitysymbols.qid_exists(qid):
            eid = entitysymbols.get_eid(qid)
        if qid not in alias_qids:
            # if we are not training in candidates, we only assign 0 correct id if the alias is in our map;
            # otherwise we assign -2
            if not train_in_candidates and alias_trie_idx != -2:
                # set class label to be "not in candidate set"
                gold_cand_K_idx = 0
            else:
                # if we are not using a NC (no candidate) but are in eval mode, we let the gold
                # candidate not be in the candidate set we give in a true index of -2,
                # meaning our model will always get this example incorrect
                assert split in ["test", "dev",], (
                    f"Expected split of 'test' or 'dev' in sent {example.sent_idx}. If you are training, "
                    f"the QID {qid} must be in the candidate list for data_args.train_in_candidates to be True"
                )
                gold_cand_K_idx = -2
        else:
            # Here we are getting the correct class label for training.
            # Our training is "which of the max_entities entity candidates is the right one
            # (class labels 1 to max_entities) or is it none of these (class label 0)".
            # + (not discard_noncandidate_entities) is to ensure label 0 is
            # reserved for "not in candidate set" class
            gold_cand_K_idx = np.nonzero(np.array(alias_qids) == qid)[0][0] + (
                not train_in_candidates
            )
        assert gold_cand_K_idx < entitysymbols.max_candidates + int(
            not train_in_candidates
        ), (
            f"The qid {qid} and alias {alias} is not in the top {entitysymbols.max_candidates} max candidates. "
            f"The QID must be within max candidates."
        )

        # Create input IDs here to ensure each entity is truncated properly
        inputs = tokenizer(
            example.phrase.split(),
            is_split_into_words=True,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            max_length=max_seq_len,
            return_overflowing_tokens=False,
        )
        # In the rare case that the pre-context goes beyond max_seq_len, retokenize strating from
        # ent start to guarantee the start/end tok will be there
        start_tok = inputs.word_to_tokens(span_start_idx)
        if start_tok is None:
            new_phrase = example.phrase.split()[span_start_idx:]
            # Adjust spans
            span_dist = span_end_idx - span_start_idx
            span_start_idx = 0
            span_end_idx = span_start_idx + span_dist
            inputs = tokenizer(
                new_phrase,
                is_split_into_words=True,
                padding="max_length",
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_len,
                return_overflowing_tokens=False,
            )
            if inputs.word_to_tokens(span_start_idx) is None:
                print("REALLY BAD")
                print(example)
            new_span_start = inputs.word_to_tokens(span_start_idx).start + 1
        else:
            # Includes the [ent_start]; we do not want to mask that so +1
            new_span_start = start_tok.start + 1
        # -1 to index the [ent_end] token, not the token after
        end_tok = inputs.word_to_tokens(span_end_idx - 1)
        if end_tok is None:
            # -1 for CLS token
            new_span_end = len(inputs["input_ids"]) - 1
        else:
            new_span_end = end_tok.start
        final_toks = tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        assert (
            final_toks[new_span_start - 1] == "[ent_start]"
        ), f"{final_toks} {new_span_start} {new_span_end} {span_start_idx} {span_end_idx}"
        assert (new_span_end == len(inputs["input_ids"]) - 1) or final_toks[
            new_span_end
        ] == "[ent_end]", f"{final_toks} {new_span_start} {new_span_end} {span_start_idx} {span_end_idx}"
        candidate_sentence_input_ids[: len(inputs["input_ids"])] = inputs["input_ids"]
        candidate_mention_cnt_ratio[new_span_start:new_span_end] = [
            example.qid_cnt_mask_score for _ in range(new_span_start, new_span_end)
        ]

        candidate_sentence_attn_msks[: len(inputs["attention_mask"])] = inputs[
            "attention_mask"
        ]
        candidate_sentence_token_type_ids[: len(inputs["token_type_ids"])] = inputs[
            "token_type_ids"
        ]

        # this stores the true entity pos in the candidate list we use to compute loss -
        # all anchors for train and true anchors for dev/test
        # leave as -1 if it's not an alias we want to predict; we get these if we split a
        # sentence and need to only predict subsets
        example_true_cand_positions_for_loss = PAD_ID
        # this stores the true entity pos in the candidate list for all alias seen by model -
        # all anchors for both train and eval
        example_true_entity_eid = PAD_ID
        # checks if alias is gold or not - alias_to_predict will be -1 for non gold aliases for eval
        if alias_to_predict == 0:
            example_true_cand_positions_for_loss = gold_cand_K_idx
            example_true_entity_eid = eid
        example_true_cand_positions_for_train = gold_cand_K_idx
        # drop example if we have nothing to predict (no valid aliases) -- make sure this doesn't cause
        # problems when we start using unk aliases...
        if alias_trie_idx == PAD_ID:
            logging.error(
                f"There were 0 aliases in this example {example}. This shouldn't happen."
            )
            sys.exit(0)
        total_saved_features += 1
        feature = InputFeatures(
            alias_idx=alias_trie_idx,
            word_input_ids=candidate_sentence_input_ids,
            word_token_type_ids=candidate_sentence_token_type_ids,
            word_attention_mask=candidate_sentence_attn_msks,
            word_qid_cnt_mask_score=candidate_mention_cnt_ratio,
            gold_eid=example_true_entity_eid,
            gold_cand_K_idx=example_true_cand_positions_for_loss,
            for_dump_gold_cand_K_idx_train=example_true_cand_positions_for_train,
            alias_list_pos=alias_list_pos,
            sent_idx=int(example.sent_idx),
            subsent_idx=int(example.subsent_idx),
            guid=np.array(
                [
                    (
                        int(example.sent_idx),
                        int(example.subsent_idx),
                        [alias_list_pos],
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
        mmap_file["alias_idx"][example_idx] = feature.alias_idx
        mmap_file["input_ids"][example_idx] = feature.word_input_ids
        mmap_file["token_type_ids"][example_idx] = feature.word_token_type_ids
        mmap_file["attention_mask"][example_idx] = feature.word_attention_mask
        mmap_file["word_qid_cnt_mask_score"][
            example_idx
        ] = feature.word_qid_cnt_mask_score
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
            output_str += f"phrase toks:                        {example.phrase}" + "\n"
            output_str += (
                f"alias_to_predict:              {example.alias_to_predict}" + "\n"
            )
            output_str += (
                f"alias_list_pos:                  {example.alias_list_pos}" + "\n"
            )
            output_str += f"aliases:                         {example.alias}" + "\n"
            output_str += f"qids:                            {example.qid}" + "\n"
            output_str += "*** Feature ***" + "\n"
            output_str += (
                f"gold_cand_K_idx:                 {feature.gold_cand_K_idx}" + "\n"
            )
            output_str += f"gold_eid:                        {feature.gold_eid}" + "\n"
            output_str += (
                f"for_dump_gold_cand_K_idx_train:  {feature.for_dump_gold_cand_K_idx_train}"
                + "\n"
            )
            output_str += (
                f"input_ids:                        {' '.join([str(x) for x in feature.word_input_ids])}"
                + "\n"
            )
            output_str += (
                f"token_type_ids:                        {' '.join([str(x) for x in feature.word_token_type_ids])}"
                + "\n"
            )
            output_str += (
                f"attention_mask:                        {' '.join([str(x) for x in feature.word_attention_mask])}"
                + "\n"
            )
            output_str += (
                f"word_qid_cnt_mask_score:               {' '.join([str(x) for x in feature.word_qid_cnt_mask_score])}"
                + "\n"
            )
            output_str += f"guid:                            {feature.guid}" + "\n"
            if print_examples:
                print(output_str)
    mmap_file.flush()
    mmap_label_file.flush()
    return total_saved_features


def build_and_save_entity_inputs_initializer(
    constants,
    data_config,
    save_entity_dataset_name,
    X_entity_storage,
    qid2typenames_file,
    qid2relations_file,
    tokenizer,
):
    global qid2typenames_global
    qid2typenames_global = ujson.load(open(qid2typenames_file))
    global qid2relations_global
    qid2relations_global = ujson.load(open(qid2relations_file))
    global mmap_entity_file_global
    mmap_entity_file_global = np.memmap(
        save_entity_dataset_name, dtype=X_entity_storage, mode="r+"
    )
    global constants_global
    constants_global = constants
    global tokenizer_global
    tokenizer_global = tokenizer
    global entitysymbols_global
    entitysymbols_global = EntitySymbols.load_from_cache(
        load_dir=os.path.join(data_config.entity_dir, data_config.entity_map_dir),
        alias_cand_map_file=data_config.alias_cand_map,
        alias_idx_file=data_config.alias_idx_map,
    )


def build_and_save_entity_inputs(
    save_entity_dataset_name,
    X_entity_storage,
    data_config,
    dataset_threads,
    tokenizer,
    entity_symbols,
):
    """Generates data for the entity encoder input.

    Args:
        save_entity_dataset_name: memmap filename to save the entity data
        X_entity_storage: storage type for memmap file
        data_config: data config
        dataset_threads: number of threads
        tokenizer: tokenizer
        entity_symbols: entity symbols

    Returns:
    """
    add_entity_type = data_config.entity_type_data.use_entity_types
    qid2typenames = {}
    if add_entity_type:
        qid2typenames = read_in_types(data_config, entity_symbols)

    add_entity_kg = data_config.entity_kg_data.use_entity_kg
    qid2relations = {}
    if add_entity_kg:
        qid2relations = read_in_relations(data_config, entity_symbols)

    num_processes = min(dataset_threads, int(0.8 * multiprocessing.cpu_count()))

    # IMPORTANT: for distributed writing to memmap files, you must create them in w+
    # mode before being opened in r+ mode by workers
    memfile = np.memmap(
        save_entity_dataset_name,
        dtype=X_entity_storage,
        mode="w+",
        shape=(entity_symbols.num_entities_with_pad_and_nocand,),
        order="C",
    )
    # We'll use the -1 to check that things were written correctly later because at
    # the end, there should be no -1
    memfile["entity_token_type_ids"][:] = -1

    # The memfile corresponds to eids. As eid 0 and -1 are reserved for UNK/PAD
    # we need to set the values. These get a single [SEP] for title [SEP] rest of entity
    empty_ent = tokenizer(
        "[SEP]",
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
        max_length=data_config.max_ent_len,
    )
    memfile["entity_input_ids"][0] = empty_ent["input_ids"][:]
    memfile["entity_token_type_ids"][0] = empty_ent["token_type_ids"][:]
    memfile["entity_attention_mask"][0] = empty_ent["attention_mask"][:]
    memfile["entity_to_mask"][0] = [0 for _ in range(len(empty_ent["input_ids"]))]

    memfile["entity_input_ids"][-1] = empty_ent["input_ids"][:]
    memfile["entity_token_type_ids"][-1] = empty_ent["token_type_ids"][:]
    memfile["entity_attention_mask"][-1] = empty_ent["attention_mask"][:]
    memfile["entity_to_mask"][-1] = [0 for _ in range(len(empty_ent["input_ids"]))]

    constants = {
        "train_in_candidates": data_config.train_in_candidates,
        "max_ent_len": data_config.max_ent_len,
        "max_ent_type_len": data_config.entity_type_data.max_ent_type_len,
        "max_ent_kg_len": data_config.entity_kg_data.max_ent_kg_len,
        "use_types": data_config.entity_type_data.use_entity_types,
        "use_kg": data_config.entity_kg_data.use_entity_kg,
        "use_desc": data_config.use_entity_desc,
        "print_examples_prep": data_config.print_examples_prep,
    }
    if num_processes == 1:
        input_qids = list(entity_symbols.get_all_qids())
        num_qids, overflowed = build_and_save_entity_inputs_single(
            input_qids,
            constants,
            memfile,
            qid2typenames,
            qid2relations,
            tokenizer,
            entity_symbols,
        )
    else:
        qid2typenames_file = tempfile.NamedTemporaryFile()
        with open(qid2typenames_file.name, "w") as out_f:
            ujson.dump(qid2typenames, out_f)

        qid2relations_file = tempfile.NamedTemporaryFile()
        with open(qid2relations_file.name, "w") as out_f:
            ujson.dump(qid2relations, out_f)

        input_qids = list(entity_symbols.get_all_qids())
        chunk_size = int(np.ceil(len(input_qids) / num_processes))
        input_chunks = [
            input_qids[i : i + chunk_size]
            for i in range(0, len(input_qids), chunk_size)
        ]

        log_rank_0_debug(logger, f"Starting pool with {num_processes} processes")
        pool = multiprocessing.Pool(
            processes=num_processes,
            initializer=build_and_save_entity_inputs_initializer,
            initargs=[
                constants,
                data_config,
                save_entity_dataset_name,
                X_entity_storage,
                qid2typenames_file.name,
                qid2relations_file.name,
                tokenizer,
            ],
        )
        cnt = 0
        overflowed = 0
        for res in tqdm(
            pool.imap_unordered(
                build_and_save_entity_inputs_hlp, input_chunks, chunksize=1
            ),
            total=len(input_chunks),
            desc="Building entity data",
        ):
            c, overfl = res
            cnt += c
            overflowed += overfl
        pool.close()
        qid2typenames_file.close()
        qid2relations_file.close()

    log_rank_0_debug(
        logger,
        f"{overflowed} out of {len(input_qids)} were overflowed",
    )

    memfile = np.memmap(save_entity_dataset_name, dtype=X_entity_storage, mode="r")
    for i in tqdm(
        range(entity_symbols.num_entities_with_pad_and_nocand),
        desc="Verifying entity data",
    ):
        assert all(memfile["entity_token_type_ids"][i] != -1), f"Memfile at {i} is -1."
    memfile = None
    return


def build_and_save_entity_inputs_hlp(input_qids):
    return build_and_save_entity_inputs_single(
        input_qids,
        constants_global,
        mmap_entity_file_global,
        qid2typenames_global,
        qid2relations_global,
        tokenizer_global,
        entitysymbols_global,
    )


def build_and_save_entity_inputs_single(
    input_qids,
    constants,
    memfile,
    qid2typenames,
    qid2relations,
    tokenizer,
    entity_symbols,
):
    printed = 0
    num_overflow = 0
    for qid in tqdm(input_qids, desc="Processing entities"):
        ent_str, title_spans, over_type_len, over_kg_len = get_entity_string(
            qid,
            constants,
            entity_symbols,
            qid2relations,
            qid2typenames,
        )
        inputs = tokenizer(
            ent_str.split(),
            is_split_into_words=True,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            max_length=constants["max_ent_len"],
        )
        to_mask = [0 for _ in range(len(inputs["input_ids"]))]
        for title_sp in title_spans:
            title_toks = inputs.word_to_tokens(title_sp)
            if title_toks is None:
                continue
            for i in range(title_toks.start, title_toks.end):
                to_mask[i] = 1
        # Heuristic function to compute this
        if inputs["input_ids"][-1] == tokenizer.sep_token_id:
            num_overflow += 1

        if printed < 8 and constants["print_examples_prep"]:
            print("QID:", qid)
            print("TITLE:", entity_symbols.get_title(qid))
            print("ENT STR:", ent_str)
            print("INPUTS:", inputs)
            print("TITLE SPANS:", title_spans)
            print("TO MASK:", to_mask)
            print(tokenizer.convert_ids_to_tokens(np.array(inputs["input_ids"])))
            printed += 1

        eid = entity_symbols.get_eid(qid)
        for k, value in inputs.items():
            memfile[f"entity_{k}"][eid] = value
        memfile["entity_to_mask"][eid] = to_mask
    memfile.flush()
    return len(input_qids), num_overflow


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
    ):
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
                ("alias_orig_list_pos", "i8", (1,)),
            ]
        )
        max_total_input_len = data_config.max_seq_len
        # Storage for saving the data.
        self.X_storage, self.Y_storage, self.X_entity_storage = (
            [
                ("guids", guid_dtype, 1),
                ("sent_idx", "i8", 1),
                ("subsent_idx", "i8", 1),
                ("alias_idx", "i8", 1),
                (
                    "input_ids",
                    "i8",
                    (max_total_input_len,),
                ),
                (
                    "token_type_ids",
                    "i8",
                    (max_total_input_len,),
                ),
                (
                    "attention_mask",
                    "i8",
                    (max_total_input_len,),
                ),
                (
                    "word_qid_cnt_mask_score",
                    "float",
                    (max_total_input_len,),
                ),
                ("alias_orig_list_pos", "i8", 1),
                (
                    "gold_eid",
                    "i8",
                    1,
                ),  # What the eid of the gold entity is
                (
                    "for_dump_gold_cand_K_idx_train",
                    "i8",
                    1,
                ),  # Which of the K candidates is correct. Only used in dump_pred to stitch sub-sentences together
            ],
            [
                (
                    "gold_cand_K_idx",
                    "i8",
                    1,
                ),  # Which of the K candidates is correct.
            ],
            [
                ("entity_input_ids", "i8", (data_config.max_ent_len)),
                ("entity_token_type_ids", "i8", (data_config.max_ent_len)),
                ("entity_attention_mask", "i8", (data_config.max_ent_len)),
                ("entity_to_mask", "i8", (data_config.max_ent_len)),
            ],
        )
        self.split = split
        self.popularity_mask = data_config.popularity_mask
        self.context_mask_perc = data_config.context_mask_perc
        self.tokenizer = tokenizer

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

        # Folder for entity mmap saved files
        save_entity_folder = data_utils.get_emb_prep_dir(data_config)
        utils.ensure_dir(save_entity_folder)

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
        self.save_entity_dataset_name = None
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
        # ENTITY TOKENS
        # =======================================================================================
        # =======================================================================================
        # =======================================================================================
        self.save_entity_dataset_name = os.path.join(
            save_entity_folder,
            f"entity_data"
            f"_type{int(data_config.entity_type_data.use_entity_types)}"
            f"_kg{int(data_config.entity_kg_data.use_entity_kg)}"
            f"_desc{int(data_config.use_entity_desc)}.bin",
        )
        log_rank_0_debug(logger, f"Seeing if {self.save_entity_dataset_name} exists")
        if data_config.overwrite_preprocessed_data or (
            not os.path.exists(self.save_entity_dataset_name)
        ):
            st_time = time.time()
            log_rank_0_info(logger, f"Building entity data from scatch.")
            try:
                # Creating/saving data
                build_and_save_entity_inputs(
                    self.save_entity_dataset_name,
                    self.X_entity_storage,
                    data_config,
                    dataset_threads,
                    tokenizer,
                    entity_symbols,
                )
                log_rank_0_debug(
                    logger, f"Finished prepping data in {time.time() - st_time}"
                )
            except Exception as e:
                tb = traceback.TracebackException.from_exception(e)
                logger.error(e)
                logger.error(traceback.format_exc())
                logger.error("\n".join(tb.stack.format()))
                os.remove(self.save_entity_dataset_name)
                raise

        X_entity_dict = self.build_data_entity_dicts(
            self.save_entity_dataset_name, self.X_entity_storage
        )
        self.X_entity_dict = X_entity_dict

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
                "alias_idx": [],
                "input_ids": [],
                "token_type_ids": [],
                "attention_mask": [],
                "word_qid_cnt_mask_score": [],
                "alias_orig_list_pos": [],  # list of original position in the alias list this example is (see eval)
                "gold_eid": [],  # List of gold entity eids
                "for_dump_gold_cand_K_idx_train": [],  # list of gold indices without subsentence masking (see eval)
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
        X_dict["alias_idx"] = torch.from_numpy(mmap_file["alias_idx"])
        X_dict["input_ids"] = torch.from_numpy(mmap_file["input_ids"])
        X_dict["token_type_ids"] = torch.from_numpy(mmap_file["token_type_ids"])
        X_dict["attention_mask"] = torch.from_numpy(mmap_file["attention_mask"])
        X_dict["word_qid_cnt_mask_score"] = torch.from_numpy(
            mmap_file["word_qid_cnt_mask_score"]
        )
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
    def build_data_entity_dicts(cls, save_dataset_name, X_storage):
        """Returns the X_dict for the entity data.

        Args:
            save_dataset_name: memmap file name with entity data
            X_storage: memmap storage type

        Returns: Dict of labels
        """
        X_dict = {
            "entity_input_ids": [],
            "entity_token_type_ids": [],
            "entity_attention_mask": [],
            "entity_to_mask": [],
        }
        mmap_label_file = np.memmap(save_dataset_name, dtype=X_storage, mode="r")
        X_dict["entity_input_ids"] = torch.from_numpy(
            mmap_label_file["entity_input_ids"]
        )
        X_dict["entity_token_type_ids"] = torch.from_numpy(
            mmap_label_file["entity_token_type_ids"]
        )
        X_dict["entity_attention_mask"] = torch.from_numpy(
            mmap_label_file["entity_attention_mask"]
        )
        X_dict["entity_to_mask"] = torch.from_numpy(mmap_label_file["entity_to_mask"])
        return X_dict

    def __getitem__(self, index):
        r"""Get item by index.

        Args:
          index(index): The index of the item.
        Returns:
          Tuple[Dict[str, Any], Dict[str, Tensor]]: Tuple of x_dict and y_dict
        """
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}

        # Mask the mention tokens
        if self.split == "train" and self.popularity_mask:
            input_ids = self._mask_input_ids(x_dict)
            x_dict["input_ids"] = input_ids
        # Get the entity_cand_eid
        entity_cand_eid = self.alias2cands_model(x_dict["alias_idx"]).long()
        entity_cand_input_ids = []
        entity_cand_token_type_ids = []
        entity_cand_attention_mask = []
        # Get the entity token ids
        for eid in entity_cand_eid:
            if self.split == "train" and self.popularity_mask:
                entity_input_ids = self._mask_entity_input_ids(x_dict, eid)
            else:
                entity_input_ids = self.X_entity_dict["entity_input_ids"][eid]
            entity_cand_input_ids.append(entity_input_ids)
            entity_cand_token_type_ids.append(
                self.X_entity_dict["entity_token_type_ids"][eid]
            )
            entity_cand_attention_mask.append(
                self.X_entity_dict["entity_attention_mask"][eid]
            )
        # Create M x K x token length
        x_dict["entity_cand_input_ids"] = torch.stack(entity_cand_input_ids, dim=0)
        x_dict["entity_cand_token_type_ids"] = torch.stack(
            entity_cand_token_type_ids, dim=0
        )
        x_dict["entity_cand_attention_mask"] = torch.stack(
            entity_cand_attention_mask, dim=0
        )
        x_dict["entity_cand_eval_mask"] = entity_cand_eid == -1
        # Handles the index errors with -1 indexing into an embedding
        x_dict["entity_cand_eid"] = torch.where(
            entity_cand_eid >= 0,
            entity_cand_eid,
            (
                torch.ones_like(entity_cand_eid, dtype=torch.long)
                * (self.num_entities_with_pad_and_nocand - 1)
            ),
        )
        # Add dummy gold_unq_eid_idx for Emmental init - this gets overwritten in the collator in data.py
        y_dict["gold_unq_eid_idx"] = y_dict["gold_cand_K_idx"]
        return x_dict, y_dict

    def _mask_input_ids(self, x_dict):
        """Mask the entity mention with high probability, especially if rare. Further mask tokens 10% of the time"""
        # Get core dump if you don't do this
        input_ids = torch.clone(x_dict["input_ids"])
        cnt_ratio = x_dict["word_qid_cnt_mask_score"]
        probability_matrix = torch.full(cnt_ratio.shape, 0.0)
        fill_v = 0.0
        if torch.any((0.0 <= cnt_ratio) & (cnt_ratio < 0.5)):
            fill_v = 0.95
        elif torch.any((0.5 <= cnt_ratio) & (cnt_ratio < 0.65)):
            fill_v = 0.84
        elif torch.any((0.65 <= cnt_ratio) & (cnt_ratio < 0.8)):
            fill_v = 0.73
        elif torch.any((0.8 <= cnt_ratio) & (cnt_ratio < 0.95)):
            fill_v = 0.62
        elif torch.any(0.95 <= cnt_ratio):
            fill_v = 0.5
        probability_matrix.masked_fill_(cnt_ratio >= 0.0, value=fill_v)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        input_ids.masked_fill_(
            masked_indices,
            value=self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token),
        )

        # Mask all tokens by context_mask_perc
        if self.context_mask_perc > 0.0:
            input_ids_clone = input_ids.clone()
            # We sample a few tokens in each sequence
            probability_matrix = torch.full(
                input_ids_clone.shape, self.context_mask_perc
            )
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            )
            probability_matrix.masked_fill_(
                torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
            )
            if self.tokenizer._pad_token is not None:
                padding_mask = input_ids.eq(self.tokenizer.pad_token_id)
                probability_matrix.masked_fill_(padding_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            input_ids_clone[masked_indices] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.mask_token
            )
            input_ids = input_ids_clone
        return input_ids

    def _mask_entity_input_ids(self, x_dict, eid):
        """Mask the entity to_mask index with high probability, especially if mention is rare."""
        # Get core dump if you don't do this
        entity_input_ids = torch.clone(self.X_entity_dict["entity_input_ids"][eid])
        cnt_ratio = x_dict["word_qid_cnt_mask_score"]
        probability_matrix = torch.tensor(
            self.X_entity_dict["entity_to_mask"][eid]
        ).float()
        fill_v = 0.0
        if torch.any((0.0 <= cnt_ratio) & (cnt_ratio < 0.5)):
            fill_v = 0.95
        elif torch.any((0.5 <= cnt_ratio) & (cnt_ratio < 0.65)):
            fill_v = 0.84
        elif torch.any((0.65 <= cnt_ratio) & (cnt_ratio < 0.8)):
            fill_v = 0.73
        elif torch.any((0.8 <= cnt_ratio) & (cnt_ratio < 0.95)):
            fill_v = 0.62
        elif torch.any(0.95 <= cnt_ratio):
            fill_v = 0.5
        probability_matrix.masked_fill_(probability_matrix > 0.0, value=fill_v)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        entity_input_ids.masked_fill_(
            masked_indices, value=self.tokenizer.convert_tokens_to_ids("[MASK]")
        )
        return entity_input_ids

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["X_dict"]
        del state["Y_dict"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.X_dict, self.Y_dict = self.build_data_dicts(
            self.save_dataset_name,
            self.save_labels_name,
            self.X_storage,
            self.Y_storage,
        )
        return state

    def __repr__(self):
        return (
            f"Bootleg Dataset. Data at {self.save_dataset_name}. "
            f"Labels at {self.save_labels_name}. "
        )
