"""
Bootleg configuration parameters.

In the json file, everything is a string or number. In this python file, if the default is a boolean, it will be parsed as such. If the default is a
dictionary, True and False strings will become booleans. Otherwise they will stay string.

"""
import multiprocessing

config_args = {
    "run_config": {
        "loglevel": ("info", "logging level"),
        "cpu": (False, "use the cpu"),
        "init_checkpoint": ("", "path to initial model checkpoint"),
        "save_dir": ("runs","directory where run gets saved"),
        "timestamp": ("","timestamp of run; if empty, set to current timestamp if train or latest in save_folder if eval"),
        "model_suffix": ("", "model suffix for saving"),
        "max_epochs": (10, "max epochs"),
        "log_steps": (200, "log every save_steps"),
        "eval_steps": (5000, "eval every eval_steps"),
        "save_every_k_eval": (1, "will save model every K times eval is run; model will always save every epoch"),
        "eval_batch_size": (128, "batch size for eval"),
        "dataloader_threads": (16, "data loader threads to feed gpus"),
        "dataset_threads": (int(multiprocessing.cpu_count()*0.9), "data set threads for prepping data"),
        "perc_eval": (1.0, "Sample percent to use during dev eval during training or test eval. If running on lage data, this is important to keep small."),
        "min_eval_size": (-1, "If the percent eval gives less than min_eval_size examples (for min_eval_size > 0), the min eval size of examples will be evaluated."),
        "eval_slices": ([], "slices for evaluation"),
        "gpu": (0, "which GPU to use if NOT distributed"),
        "topk": (5, "topk to use for eval scoring"),
        "result_label_file": ("bootleg_labels.jsonl", "file name to save predicted entities in"),
        "result_emb_file": ("bootleg_embs.npy", "file name to save contextualized embs in"),
        # parameters specific to distributed code
        "distributed": (False, "use distributed or not"),
        "ngpus_per_node": (1, "number of gpus per node"),
        "nodes": (1, "a node may have multiple GPUs"),
        "nr": (0, "ranking within the nodes (can ignore unless you have multiple machines with GPUs)"),
        "dist_url": ("tcp://127.0.0.1:8001", "url used to set up distributed training")
    },
    # Parameters for hyperparameter tuning
    "train_config": {
        "load_optimizer_from_ckpt": (True, "Whether to load optimizer from checkpoint or not"),
        "dropout": (0.1, "dropout"),
        "weight_decay": (1e-5, "weight decay"),
        "batch_size": (32, "batch size"),
        "lr": (1e-4, "learning rate"),
        "seed": (1234, "random seed for initialization"),
        "softmax_temp": (0.25, "temperature for softmax for slice weighting in SBL"),
        "slice_method": ("Normal", "method for slicing: Normal or SBL"),
        "train_heads": ([], "slices for training"),
        "random_nil": (False, "add nils during training"),
        "random_nil_perc": (0.1, "perc of mentions to sample as NIL, includes masked mentions")
    },
    "model_config": {
        "attn_load_class": ("Bootleg", "AttnNetwork class to use"),
        "base_model_load_class": ("model.Model", "BaseModel class to use"),
        "hidden_size": (300, "hidden dimension"),
        "num_heads": (10, "number of attention head"),
        "ff_inner_size": (500, "inner size of the pointwise feedforward layers in attn blocks"),
        "num_model_stages": (1, "number of model stages"),
        "num_fc_layers": (1, "number of fully scoring layers at end"),
        "custom_args": ('{}', "custom arguments for the model file")
    },
    "data_config": {
        "train_in_candidates": (True, "Train in candidates (if False, this means we include NIL entity)"),
        "data_dir": ("data","where training, testing, and dev data is stored"),
        "data_prep_dir": ("prep", "directory where data prep files are saved inside data_dir"),
        "entity_dir": ("entity_data", "where entity profile information and prepped embedding data is stored"),
        "entity_prep_dir": ("prep", "directory where prepped embedding data is saved inside entity_dir"),
        "entity_map_dir": ("entity_mappings", "directory where entity json mappings are saved inside entity_dir"),
        "alias_cand_map": ("alias2qids.json", "name of alias candidate map, should be saved in entity_dir/entity_map_dir"),
        "emb_dir": ("embs","where embeddings are stored"),
        "max_word_token_len": (100, "max token length sentences"),
        "max_aliases": (10, "max aliases per sentence"),
        "overwrite_preprocessed_data": (False, "overwrite preprocessed data"),
        "type_prediction": {
            "use_type_pred": (False, "whether to add type prediction or not"),
            "file": ("hyena_types_coarse.json", "type file from qid to list of type ids"),
            "num_types": (5, "number of types for prediction"),
            "dim": (128, "type dimension")
        },
        "train_dataset":{
            "file": ("train.jsonl", ""),
            "load_class": ("wiki_dataset.WikiDataset", ""),
            "slice_class": ("wiki_slices.WikiSlices", ""),
            "use_weak_label": (True, "Use weakly labeled mentions")
        },
        "dev_dataset": {
            "file": ("dev.jsonl", ""),
            "load_class": ("wiki_dataset.WikiDataset", ""),
            "slice_class": ("wiki_slices.WikiSlices", ""),
            "use_weak_label": (True, "Use weakly labeled mentions")
        },
        "test_dataset": {
            "file": ("test.jsonl", ""),
            "load_class": ("wiki_dataset.WikiDataset", ""),
            "slice_class": ("wiki_slices.WikiSlices", ""),
            "use_weak_label": (True, "Use weakly labeled mentions")
        },
        "word_embedding": {
            "load_class": ("bert_word_emb.BERTWordEmbedding", ""),
            "word_symbols": ("BERTWordSymbols", ""),
            "sent_class": ("bert_sent_emb.BERTSentEmbedding", ""),
            "custom_vocab_embedding_file": ("", ""),
            "layers": (3, ""),
            "freeze_word_emb": (False, ""),
            "freeze_sent_emb": (False, ""),
            "use_lower_case": (False, ""),
            "cache_dir": ("pretrained_bert_models", "Directory where word embeddings are cached"),
            "custom_proj_size": (-1, "If set to positive number, will project the word embeddings before sending them to the sentence embedding."),
        },
        "ent_embeddings": ([
        ], "entity embeddings")
    },
    "prep_config": {
        "chunk_data": (True, "chunk the data set into multiple text files"),
        "ext_subsent": (True, "extract subsentences"),
        "build_data": (True, "build the data set"),
        "batch_prep_embeddings": (True, "batch_prep features as mmap files"),
        "keep_all": (False, "keep all data chunks or not"),
        "prep_train": (True, ""),
        "prep_dev": (True, ""),
        "prep_test": (True, ""),
        "prep_embs": (True, ""),
        "prep_train_slices": (True, ""),
        "prep_dev_eval_slices": (True, ""),
        "prep_test_eval_slices": (True, ""),
    }
}
