"""Bootleg default configuration parameters.

In the json file, everything is a string or number. In this python file,
if the default is a boolean, it will be parsed as such. If the default
is a dictionary, True and False strings will become booleans. Otherwise
they will stay string.
"""
import multiprocessing

config_args = {
    "run_config": {
        "spawn_method": (
            "forkserver",
            "multiprocessing spawn method. forkserver will save memory but have slower startup costs.",
        ),
        "eval_batch_size": (128, "batch size for eval"),
        "dump_preds_accumulation_steps": (
            1000,
            "number of eval steps to accumulate the output tensors for before saving results to file",
        ),
        "dump_preds_num_data_splits": (
            1,
            "number of chunks to split the input file; helps with OOM issues",
        ),
        "overwrite_eval_dumps": (False, "overwrite dumped eval data"),
        "dataloader_threads": (16, "data loader threads to feed gpus"),
        "log_level": ("info", "logging level"),
        "dataset_threads": (
            int(multiprocessing.cpu_count() * 0.9),
            "data set threads for prepping data",
        ),
        "result_label_file": (
            "bootleg_labels.jsonl",
            "file name to save predicted entities in",
        ),
        "result_emb_file": (
            "bootleg_embs.npy",
            "file name to save contextualized embs in",
        ),
        "mp_sharing_strategy": (
            None,
            "setting this to 'file_system' when training on machine with less RAM solves pytorch issues",
        ),
    },
    # Parameters for hyperparameter tuning
    "train_config": {
        "batch_size": (32, "batch size"),
    },
    "model_config": {
        "hidden_size": (300, "hidden dimension for the embeddings before scoring"),
        "normalize": (False, "normalize embeddings before dot product"),
        "temperature": (1.0, "temperature for softmax in loss"),
    },
    "data_config": {
        "eval_slices": ([], "slices for evaluation"),
        "train_in_candidates": (
            True,
            "Train in candidates (if False, this means we include NIL entity)",
        ),
        "data_dir": ("data", "where training, testing, and dev data is stored"),
        "data_prep_dir": (
            "prep",
            "directory where data prep files are saved inside data_dir",
        ),
        "entity_dir": (
            "entity_data",
            "where entity profile information and prepped embedding data is stored",
        ),
        "entity_prep_dir": (
            "prep",
            "directory where prepped embedding data is saved inside entity_dir",
        ),
        "entity_map_dir": (
            "entity_mappings",
            "directory where entity json mappings are saved inside entity_dir",
        ),
        "alias_cand_map": (
            "alias2qids",
            "name of alias candidate map file, should be saved in entity_dir/entity_map_dir",
        ),
        "alias_idx_map": (
            "alias2id",
            "name of alias index map file, should be saved in entity_dir/entity_map_dir",
        ),
        "qid_cnt_map": (
            "qid2cnt.json",
            "name of alias index map file, should be saved in entity_dir/entity_map_dir",
        ),
        "max_seq_len": (128, "max token length sentences"),
        "max_seq_window_len": (64, "max window around an entity"),
        "max_ent_len": (128, "max token length for entire encoded entity"),
        "context_mask_perc": (
            0.0,
            "mask percent for context tokens in addition to tail masking",
        ),
        "popularity_mask": (
            True,
            "whether to use popularity masking for training in the entity and context encoders",
        ),
        "overwrite_preprocessed_data": (False, "overwrite preprocessed data"),
        "print_examples_prep": (True, "whether to print examples during prep or not"),
        "use_entity_desc": (True, "whether to use entity descriptions or not"),
        "entity_type_data": {
            "use_entity_types": (False, "whether to use entity type data"),
            "type_symbols_dir": (
                "type_mappings/wiki",
                "directory to type symbols inside entity_dir",
            ),
            "max_ent_type_len": (20, "max WORD length for type sequence"),
        },
        "entity_kg_data": {
            "use_entity_kg": (False, "whether to use entity type data"),
            "kg_symbols_dir": (
                "kg_mappings",
                "directory to kg symbols inside entity_dir",
            ),
            "max_ent_kg_len": (60, "max WORD length for kg sequence"),
        },
        "train_dataset": {
            "file": ("train.jsonl", ""),
            "use_weak_label": (True, "Use weakly labeled mentions"),
        },
        "dev_dataset": {
            "file": ("dev.jsonl", ""),
            "use_weak_label": (True, "Use weakly labeled mentions"),
        },
        "test_dataset": {
            "file": ("test.jsonl", ""),
            "use_weak_label": (True, "Use weakly labeled mentions"),
        },
        "word_embedding": {
            "bert_model": ("bert-base-uncased", ""),
            "context_layers": (12, ""),
            "entity_layers": (12, ""),
            "cache_dir": (
                "pretrained_bert_models",
                "Directory where word embeddings are cached",
            ),
        },
    },
}
