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
    },
    # Parameters for hyperparameter tuning
    "train_config": {"dropout": (0.1, "dropout"), "batch_size": (32, "batch size")},
    "model_config": {
        "attn_class": ("Bootleg", "AttnNetwork class to use"),
        "hidden_size": (256, "hidden dimension"),
        "num_heads": (8, "number of attention head"),
        "ff_inner_size": (
            512,
            "inner size of the pointwise feedforward layers in attn blocks",
        ),
        "num_model_stages": (1, "number of model stages"),
        "num_fc_layers": (1, "number of fully scoring layers at end"),
        "custom_args": ("{}", "custom arguments for the model file"),
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
            "alias2qids.json",
            "name of alias candidate map, should be saved in entity_dir/entity_map_dir",
        ),
        "emb_dir": ("embs", "where embeddings are stored"),
        "max_seq_len": (100, "max token length sentences"),
        "max_aliases": (10, "max aliases per sentence"),
        "overwrite_preprocessed_data": (False, "overwrite preprocessed data"),
        "type_prediction": {
            "use_type_pred": (False, "whether to add type prediction or not"),
            "file": (
                "types_coarse.json",
                "type file from qid to list of type ids",
            ),
            "num_types": (5, "number of types for prediction"),
            "dim": (128, "type dimension"),
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
            "use_sent_proj": (True, ""),
            "layers": (12, ""),
            "freeze": (False, ""),
            "cache_dir": (
                "pretrained_bert_models",
                "Directory where word embeddings are cached",
            ),
        },
        "ent_embeddings": ([], "entity embeddings"),
    },
}
