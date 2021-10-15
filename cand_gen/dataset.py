import logging
import multiprocessing
import os
import re
import shutil
import tempfile
import time
import traceback
import warnings

import numpy as np
import torch
import ujson
from tqdm import tqdm

from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.dataset import convert_examples_to_features_and_save, create_examples
from bootleg.layers.alias_to_ent_encoder import AliasEntityTable
from bootleg.symbols.constants import STOP_WORDS
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.utils import data_utils, utils
from bootleg.utils.classes.emmental_data import RangedEmmentalDataset
from bootleg.utils.data_utils import read_in_akas

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


def get_entity_string(
    qid,
    constants,
    entity_symbols,
    qid2alternatenames,
):
    """
    For each entity, generates a string that is fed into a language model to generate an entity embedding. Returns
    all tokens that are the title of the entity (even if in the description)

    Args:
        qid: QID
        constants: Dict of constants
        entity_symbols: entity symbols
        qid2alternatenames: Dict of QID to list of alternate

    Returns: entity strings

    """
    desc_str = (
        "[ent_desc] " + entity_symbols.get_desc(qid) if constants["use_desc"] else ""
    )
    title_str = entity_symbols.get_title(qid) if entity_symbols.qid_exists(qid) else ""

    # To encourage mention similarity, we remove the (<type>) from titles
    title_str = re.sub(r"(\(.*\))", r"", title_str).strip()

    if constants["use_akas"]:
        # Use a type token sep from Bootleg models to allow for easier sharing of tokenizers
        alternate_names = " [ent_type] " + " [ent_type] ".join(
            qid2alternatenames.get(qid, [])
        )
        alternate_names = " ".join(
            alternate_names.split()[: constants["max_ent_aka_len"]]
        )
        desc_str = " ".join([alternate_names, desc_str])

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
    return ent_str, title_spans


def build_and_save_entity_inputs_initializer(
    constants,
    data_config,
    save_entity_dataset_name,
    X_entity_storage,
    qid2alternatenames_file,
    tokenizer,
):
    global qid2alternatenames_global
    qid2alternatenames_global = ujson.load(open(qid2alternatenames_file))
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
    add_entity_akas = data_config.use_entity_akas
    qid2alternatenames = {}
    if add_entity_akas:
        qid2alternatenames = read_in_akas(entity_symbols)

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
        "max_ent_aka_len": data_config.max_ent_aka_len,
        "use_akas": add_entity_akas,
        "use_desc": data_config.use_entity_desc,
        "print_examples_prep": data_config.print_examples_prep,
    }
    if num_processes == 1:
        input_qids = list(entity_symbols.get_all_qids())
        num_qids, overflowed = build_and_save_entity_inputs_single(
            input_qids,
            constants,
            memfile,
            qid2alternatenames,
            tokenizer,
            entity_symbols,
        )
    else:
        qid2alternatenames_file = tempfile.NamedTemporaryFile()
        with open(qid2alternatenames_file.name, "w") as out_f:
            ujson.dump(qid2alternatenames, out_f)

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
                qid2alternatenames_file.name,
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
        qid2alternatenames_file.close()

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
        qid2alternatenames_global,
        tokenizer_global,
        entitysymbols_global,
    )


def build_and_save_entity_inputs_single(
    input_qids,
    constants,
    memfile,
    qid2alternatenames,
    tokenizer,
    entity_symbols,
):
    printed = 0
    num_overflow = 0
    for qid in tqdm(input_qids, desc="Processing entities"):
        ent_str, title_spans = get_entity_string(
            qid,
            constants,
            entity_symbols,
            qid2alternatenames,
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


class CandGenDataset(RangedEmmentalDataset):
    """CandGen Dataset class to be used in dataloader.

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
                    "for_dump_gold_eid",
                    "i8",
                    1,
                ),  # What the eid of the gold entity is for all aliases
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
        save_dataset_folder = data_utils.get_save_data_folder_candgen(
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
            f"_aka{int(data_config.use_entity_akas)}"
            f"_desc{int(data_config.use_entity_desc)}.bin",
        )
        log_rank_0_debug(logger, f"Seeing if {self.save_entity_dataset_name} exists")
        if data_config.overwrite_preprocessed_data or (
            not os.path.exists(self.save_entity_dataset_name)
        ):
            st_time = time.time()
            log_rank_0_info(logger, "Building entity data from scatch.")
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

        log_rank_0_debug(logger, "Removing temporary output files")
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
                "for_dump_gold_eid": [],  # List of gold entity eids
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
        X_dict["for_dump_gold_eid"] = torch.from_numpy(mmap_file["for_dump_gold_eid"])
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

        # Get the entity_cand_eid
        entity_cand_eid = self.alias2cands_model(x_dict["alias_idx"]).long()
        entity_cand_input_ids = []
        entity_cand_token_type_ids = []
        entity_cand_attention_mask = []
        # Get the entity token ids
        for eid in entity_cand_eid:
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


class CandGenEntityDataset(RangedEmmentalDataset):
    """Bootleg Dataset class for generating entity embeddings.

    Args:
        main_args: input config
        name: internal dataset name
        dataset: dataset file
        tokenizer: sentence tokenizer
        entity_symbols: entity database class
        dataset_threads: number of threads to use
        split: data split

    Returns:
    """

    def __init__(
        self,
        main_args,
        name,
        dataset,
        tokenizer,
        entity_symbols,
        dataset_threads,
        split="test",
    ):
        assert split == "test", "Split must be test split for EntityDataset"
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

        # Storage for saving the data.
        self.X_entity_storage = [
            ("entity_input_ids", "i8", (data_config.max_ent_len)),
            ("entity_token_type_ids", "i8", (data_config.max_ent_len)),
            ("entity_attention_mask", "i8", (data_config.max_ent_len)),
            ("entity_to_mask", "i8", (data_config.max_ent_len)),
        ]
        self.split = split
        self.tokenizer = tokenizer

        # Table to map from alias_idx to entity_cand_eid used in the __get_item__
        self.alias2cands_model = AliasEntityTable(
            data_config=data_config, entity_symbols=entity_symbols
        )
        # Total number of entities used in the __get_item__
        self.num_entities_with_pad_and_nocand = (
            entity_symbols.num_entities_with_pad_and_nocand
        )

        # Folder for entity mmap saved files
        save_entity_folder = data_utils.get_emb_prep_dir(data_config)
        utils.ensure_dir(save_entity_folder)

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
            f"_aka{int(data_config.use_entity_akas)}"
            f"_desc{int(data_config.use_entity_desc)}.bin",
        )
        log_rank_0_debug(logger, f"Seeing if {self.save_entity_dataset_name} exists")
        if data_config.overwrite_preprocessed_data or (
            not os.path.exists(self.save_entity_dataset_name)
        ):
            st_time = time.time()
            log_rank_0_info(logger, "Building entity data from scatch.")
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
        # Add the unique identified of EID (the embeddings are already in this order)
        X_entity_dict["guids"] = torch.arange(len(X_entity_dict["entity_input_ids"]))
        log_rank_0_info(
            logger,
            f"Final data initialization time for {split} is {time.time() - global_start}s",
        )
        # Set spawn back to original/default, which is "fork" or "spawn".
        # This is needed for the Meta.config to be correctly passed in the collate_fn.
        multiprocessing.set_start_method(orig_spawn, force=True)
        super().__init__(name, X_dict=X_entity_dict, uid="guids")

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
        return x_dict

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["X_dict"]
        del state["Y_dict"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        return state

    def __repr__(self):
        return f"Bootleg Entity Dataset. Data at {self.save_entity_dataset_name}."


class CandGenContextDataset(CandGenDataset):
    """CandGen Dataset class for context encoding.

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
        dataset_range: dataset range for subsetting (used in candidate generation)

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
        split="test",
        is_bert=True,
        dataset_range=None,
    ):
        super(CandGenContextDataset, self).__init__(
            main_args=main_args,
            name=name,
            dataset=dataset,
            use_weak_label=use_weak_label,
            tokenizer=tokenizer,
            entity_symbols=entity_symbols,
            dataset_threads=dataset_threads,
            split=split,
            is_bert=is_bert,
        )
        self.X_entity_dict = None
        self.Y_dict = None
        if dataset_range is not None:
            self.data_range = dataset_range
        else:
            self.data_range = list(range(len(next(iter(self.X_dict.values())))))

    def __getitem__(self, index):
        r"""Get item by index.

        Args:
          index(index): The index of the item.
        Returns:
          Tuple[Dict[str, Any], Dict[str, Tensor]]: Tuple of x_dict and y_dict
        """
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        return x_dict

    def __getstate__(self):
        """Get state method"""
        state = self.__dict__.copy()
        del state["X_dict"]
        return state

    def __setstate__(self, state):
        """Set state method"""
        self.__dict__.update(state)
        self.X_dict, _ = self.build_data_dicts(
            self.save_dataset_name,
            self.save_labels_name,
            self.X_storage,
            self.Y_storage,
        )
        return state

    def __repr__(self):
        """Repr method"""
        return f"Bootleg Context Dataset. Data at {self.save_dataset_name}. "
