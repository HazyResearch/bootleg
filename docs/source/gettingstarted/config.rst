Configuring Bootleg
====================

By default, Bootleg loads the default config from `bootleg/utils/parser/bootleg_args.py <../apidocs/bootleg.utils.parser.html#module-bootleg.utils.parser.bootleg_args>`_. When running a Bootleg model, the user may pass in a custom JSON or YAML config via::

  python3 bootleg/run.py --config_script <path_to_config>

This will override all default values. Further, if a user wishes to overwrite a param from the command line, they can pass in the value, using the dotted path of the argument. For example, to overwrite the data directory (the param ``data_config.data_dir``, the user can enter::

  python3 bootleg/run.py --config_script <path_to_config> --data_config.data_dir <path_to_data>

Bootleg will save the run config (as well as a fully parsed verison with all defaults) in the log directory.

Finally, when evaluating Bootleg using the annotator, Bootleg processes possible mentions in text with three environment flags: ``BOOTLEG_STRIP``, ``BOOTLEG_LOWER``, ``BOOTLEG_LANG_CODE``. The first sets the language to use for Spacy. The second is if the user wants to strip punctuation on mentions (set to False by default). The third is if the user wants to call ``.lower()`` (set to True by default).

Emmental Config
________________

As Bootleg uses Emmental_, the training parameters (e.g., learning rate) are set and handled by Emmental. We provide all Emmental params, as well as our defaults, at `bootleg/utils/parser/emm_parse_args.py <../apidocs/bootleg.utils.parser.html#module-bootleg.utils.parser.emm_parse_args>`_. All Emmental params are under the ``emmental`` configuration group. For example, to change the learning rate and number of epochs in a config, add

.. code-block::

  emmental:
    lr: 1e-4
    n_epochs: 10
  run_config:
    ...

You can also change Emmental params by the command line with ``--emmental.<emmental_param> <value>``.

Example Training Config
________________________
An example training config is shown below

.. code-block::

    emmental:
      lr: 2e-5
      n_epochs: 3
      evaluation_freq: 0.2
      warmup_percentage: 0.1
      lr_scheduler: linear
      log_path: logs/wiki
      l2: 0.01
      grad_clip: 1.0
      fp16: true
    run_config:
      eval_batch_size: 32
      dataloader_threads: 4
      dataset_threads: 50
    train_config:
      batch_size: 32
    model_config:
      hidden_size: 200
    data_config:
      data_dir: bootleg-data/data/wiki_title_0122
      data_prep_dir: prep
      use_entity_desc: true
      entity_type_data:
        use_entity_types: true
        type_symbols_dir: type_mappings/wiki
      entity_kg_data:
        use_entity_kg: true
        kg_symbols_dir: kg_mappings
      entity_dir: bootleg-data/data/wiki_title_0122/entity_db
      max_seq_len: 128
      max_seq_window_len: 64
      max_ent_len: 128
      overwrite_preprocessed_data: false
      dev_dataset:
        file: dev.jsonl
        use_weak_label: true
      test_dataset:
        file: test.jsonl
        use_weak_label: true
      train_dataset:
        file: train.jsonl
        use_weak_label: true
      train_in_candidates: true
      word_embedding:
        cache_dir: bootleg-data/embs/pretrained_bert_models
        bert_model: bert-base-uncased

Default Config
_______________
The default Bootleg config is shown below

.. literalinclude:: ../../../bootleg/utils/parser/bootleg_args.py


.. _Emmental: https://github.com/SenWu/Emmental
