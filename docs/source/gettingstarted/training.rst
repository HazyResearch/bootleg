Basic Training
==============

We describe how to train a Bootleg model for named entity disambiguation (NED), starting from a new dataset. If you already have a dataset in the Bootleg format, you can skip to `Preparing the Config <#2-preparing-the-config>`_. All commands should be run from the root directory of the repo.

Formatting the Data
----------------------

We assume three components are available for input:

#. `Text datasets <#text-datasets>`_
#. `Entity and alias data <#entities-and-aliases>`_
#. `Type and knowledge graph data <#type-and-knowledge-graph-kg-data>`_

For each component, we first describe the data requirements and then discuss how to convert the data to the expected format. Finally, we discuss the expected directory structure to organize the data components. We provide a small dataset sampled from Wikipedia in the directory ``data`` that we will use throughout this tutorial as an example.

Text Datasets
^^^^^^^^^^^^^

Requirements
~~~~~~~~~~~~


#. Text data for training and dev datasets, and if desired, a test dataset, is available. For simplicity, in this tutorial, we just assume there is a dev dataset available.
#. Known aliases (also known as mentions) and linked entities are available. This information can be obtained for Wikipedia, for instance, by using anchor text on Wikipedia pages as aliases and the linked pages as the entity label.

Each dataset will need to follow the format described below.

We assume that the text dataset is formatted in a `jsonlines <https://jsonlines.org>`_ file (each line is a dictionary) with the following keys:


* ``sentence``: the text of the sentence.
* ``sent_idx_unq``: a unique numeric identifier for each sentence in the dataset.
* ``aliases``: the aliases in the sentence to disambiguate. Aliases serve as lookup keys into an alias candidate map to generate candidates, and may not actually appear in the text. For example, the phrase "Victoria Beckham" in the sentence may be weakly labelled as the alias "Victoria" by a simple heuristic.
* ``spans``: the start and end word indices of the aliases in the text, where the end span is exclusive (like python slicing).
* ``qids``: the id of the true entity for each alias. We use canonical Wikidata QIDs in this training tutorial, but any string indentifier will work. See `Input Data`_ for more information.
* ``gold``: True if the entity label was an anchor link in the source dataset or otherwise known to be "ground truth"; False, if the entity label is from weak labeling techniques. While all provided alias-entity pairs can be used for training, only alias-entity pairs with a gold value of True are used for evaluation.
* (Optional) ``slices``: indicates which alias-entity pairs are part of certain data subsets for evaluating performance on important subsets of the data (see the `Advanced Training Tutorial <../advanced/distributed_training.html>`_ for more details).

Using this format, an example line is:

.. code-block:: JSON

   {
       "sentence": "Heidi and her husband Seal live in Vegas . ",
       "sent_idx_unq": 0,
       "aliases": ["heidi", "seal", "vegas"],
       "spans": [[0,1], [4,5], [7,8]],
       "qids": ["Q60036", "Q218091", "Q23768"],
       "gold": [true, true, true]
   }


We also provide sample `training <https://github.com/HazyResearch/bootleg/tree/master/data/sample_text_data/train.jsonl>`_ and `dev <https://github.com/HazyResearch/bootleg/tree/master/data/sample_text_data/dev.jsonl>`_ datasets as examples of text datasets in the proper format.

Entities and Aliases
^^^^^^^^^^^^^^^^^^^^

Our `Entity Profile`_ page details how to create the correct metadata for the entities and aliases and the structural files. Here we list the requirements of the mappings and inputs.

Requirements
~~~~~~~~~~~~

#. There is a set of entities to consider as candidates for training and evaluation. There are entity ids (i.e. QIDs) and titles available for these entities. For instance, quantifier ids may be Wikidata QIDs or Unified Medical Language System (UMLS) Concept Unique Identifiers (CUI). For this tutorial, we refer to the entitiy ids by QID.
#. There is a candidate mapping from aliases to entity candidates. The candidates must be in the set of entities above. If this is not provided, we apply a simple mining technique to generate this from the provided training dataset.

QID-to-Title Mapping
~~~~~~~~~~~~~~~~~~~~

We assume that the set of entity QIDs and their corresponding titles are stored in a JSON file as a dictionary of QID to title pairs. Again, this is all generated for you in `Entity Profile`_. For example,

.. code-block:: JSON

   {
       "Q60036": "Heidi Klum",
       "Q218091": "Seal (musician)",
       "Q23768": "Las Vegas"
   }


The QID-to-title mapping for the sample Wikipedia dataset in `data/sample_entity_data/entity_mappings/qid2title.json <https://github.com/HazyResearch/bootleg/tree/master/data/sample_entity_data/entity_mappings/qid2title.json>`_.

Candidate Mapping
~~~~~~~~~~~~~~~~~

We assume that the candidate mapping is stored in a JSON file as a dictionary of alias to list of [QID, sort_value] pairs, where the sort_value can be any numeric quantity to sort the candidate lists. The sort_value is necessary to choose the candidates when the number of candidates is greater than the maximum allowed (max candidates is a settable parameter). For example (candidates are cut short to display),

.. code-block:: JSON

   {
       "heidi": [["Q60036", 10286], ["Q66019", 10027], ... ]
       "seal": [["Q218091", 10416], ["Q9458", 4504], ... ]
       "vegas": [["Q23768", 7613], ["Q2624848", 3191], ... ]
   }


We provide an example candidate mapping in `data/sample_entity_data/entity_mappings/alias2qids.json <https://github.com/HazyResearch/bootleg/tree/master/data/sample_entity_data/entity_mappings/alias2qids.json>`_. We assume that all aliases are lowercased.

Entity Mappings
~~~~~~~~~~~~~~~

Bootleg also requires additional mappings to indices in internal Bootleg embeddings. For example, our mapping for entity QID to internal entity index. These are all also generated and explained in `Entity Profile`_.

Type and Knowledge Graph (KG) Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the key insights from Bootleg is that leveraging type and knowledge graph information in a simple attention-based network can improve performance on tail entities. However, to leverage this information, we need to provide type and/or knowledge graph information to the model.

Requirements
~~~~~~~~~~~~


#. Type labels from a type ontology (e.g. Wikidata or HYENA types from YAGO) is available for the candidate entities. While we do not need types assigned to all entities, the higher the coverage, the better.
#. Knowledge graph connectivity information, such as whether two entities are connected in knowledge graph, is available between pairs of entities. Furthermore, similar to the type labels, there is a mapping from entities to the knowledge graph relations they participate in.

Type Information
~~~~~~~~~~~~~~~~

We assume that the type data is provided in a JSON file as a dictionary of pairs of QIDs to a list of type ids. If there are *N* distinct types, the type ids should range from 1 to *N*. As multiple types may be associated with an entity, we store the list of type ids with each QID. The maximum number of types considered per an entity is a settable parameter. These are generated also in `Entity Profile`_.

For instance, if we have a type vocabulary of

.. code-block::

   {
       "place": 1,
       "person": 2,
       "city": 3
   }


then we may have an associated QID-to-type mapping of

.. code-block::

   {
       "Q60036": [2],
       "Q218091": [2],
       "Q23768": [1, 3]
   }


An example of the QID-to-type mapping can be found in `data/sample_entity_data/type_mappings/wiki/qid2typeids.json <https://github.com/HazyResearch/bootleg/tree/master/data/sample_entity_data/type_mappings/wiki/qid2typeids.json>`_ with the associated type vocabulary in `data/sample_entity_data/type_mappings/wiki/type_vocab.json <https://github.com/HazyResearch/bootleg/tree/master/data/sample_entity_data/type_mappings/wiki/type_vocab.json>`_.

KG Information
~~~~~~~~~~~~~~

We assume the KG information, like the type information, is provided in a JSON file representing KG triples such that for each subject entity, we have a dictionary of its relationships and the objects of those relationships. We also have a relaiton vocab mapping to map from the relation in the triples to any human readible output for the entity encoder.

For instance, if we have a relation vocabulary of

.. code-block::

   {
       "head of government": "P6",
       "vessel class": "P289",
       "member of sports team": "P54"
   }


then we may have an associated QID-to-relation mapping of

.. code-block::

   {
       "Q60036": {"P289": ["Q31"]},
       "Q218091": {"P289": ["Q893", "Q923"], "P54": ["Q1245"]},
       "Q23768": {"P6": ["Q64"]}
   }

We also provide KG connectivity information in the entity profile although it is not used in the model at this time. This adjacency information is a simple text file where each line is a tab-separated QID pair, if an edge exists between the two QIDs in a relevant knowledge graph. For instance, Q60036 (Heidi Klum) and Q218091 (Seal) share an edge (spouse), so we would have the line below in the connectivity data.

Check out `data/sample_entity_data/kg_mappings/kg_adj.txt <https://github.com/HazyResearch/bootleg/tree/master/data/sample_entity_data/kg_mappings/kg_adj.txt>`_ as an example of QID connectivity from Wikidata.


Directory Structure
^^^^^^^^^^^^^^^^^^^

We assume the data above is saved in the following directory structure, where the specific directory and filenames can be set in the config discussed in `Preparing the Config <#preparing-the-config>`_. We will also discuss how to generate the ``prep`` directories in `Preprocessing the Data <#preprocessing-the-data>`_. The ``emb_data`` directory can be shared across text datasets and entity sets, and the ``entity_data`` directory can be shared across text datasets (if they use the same set of entities).

.. code-block::

   text_data/
       train.jsonl
       dev.jsonl
       prep/

   entity_db/
        type_mappings/
            wiki/
                type_vocab.json
                qid2typenames.json
                config.json
                qid2typeids.json
            relations/
                qid2typeids.json
                config.json
                type_vocab.json
                qid2typenames.json
        kg_mappings/
            config.json
            qid2relations.json
            relation_vocab.json
            kg_adj.txt
        entity_mappings/
            alias2qids.json
            qid2eid.json
            qid2title.json
            qid2desc.json
            alias2id.json
            config.json

Preparing the Config
---------------------

Once the data has been converted to the correct format, we are ready to prepare the config. We provide a sample config in `configs/tutorial/sample_config.yaml <https://github.com/HazyResearch/bootleg/tree/master/configs/tutorial/sample_config.yaml>`_. The full parameter options and defaults for the config file are explain in `Configuring Bootleg <config.html>`_. If values are not provided in the YAML config, the default values are used. We provide a brief overview of the configuration settings here.

The config parameters are organized into five main groups:

* ``emmental``: Emmental parameters.
* ``run_config``: run time settings that aren't set in Emmental; e.g., eval batch size and number of dataloader threads.
* ``train_config``: training parameters of batch size.
* ``model_config``: model parameters of hidden dimension.
* ``data_config``: paths of text data, embedding data, and entity data to use for training and evaluation, as well as configuration details for the entity embeddings.

We highlight a few parameters in the ``emmental``.


* ``log_dir`` should be set to specify where log output and model checkpoints should be saved. When a new model is trained, Emmental automatically generates a timestamp and saves output to a folder with the timestamp inside the ``log_dir``.
* ``evaluation_freq`` indicates how frequently the evaluation on the dev set should be run. Steps corresponds to epochs by default (but can be configured to batches), such that 0.2 means 0.2 of an epoch has been processed.
* ``checkpoint_freq`` indicates when to save a model checkpoint after performing evaluation. If set to 1, then a model checkpoint will be saved every time dev evaluation is run.

See `Emmental Config <https://emmental.readthedocs.io/en/latest/user/config.html>`_ for more information.

We now focus on the ``data_config`` parameters as these are the most unique to Bootleg. We walk through the key parameters in the ``data_config`` to pay attention to.

Directories
^^^^^^^^^^^

We define the paths to the directories through the ``data_dir``\ , ``entity_dir``\, and ``entity_map_dir`` config keys. The first three correspond to the top-level directories introduced in `Directory Structure <#directory-structure>`_. The ``entity_map_dir`` includes the entity JSON mappings produced in `Entities and Aliases <#entities-and-aliases>`_ and should be inside the ``entity_dir``. For example, to follow the directory structure set up in the ``data`` directory, we would have:

.. code-block::

   "data_dir": "data/sample_text_data",
   "entity_dir": "data/sample_entity_data",
   "entity_map_dir": "entity_mappings"

Entity Encoder
^^^^^^^^^^^^^^^

As described in the _Bootleg Model, Bootleg generates an embedding entity from an Transformer encoder. The resources which go in to the encoder input are defined in the config as shown below.

.. code-block::

    data_config:
        ...
        use_entity_desc: true
        entity_type_data:
          use_entity_types: true
          type_labels: type_mappings/wiki/qid2typeids.json
          type_vocab: type_mappings/wiki/type_vocab.json
          max_ent_type_len: 20
        entity_kg_data:
          use_entity_kg: true
          kg_labels: kg_mappings/qid2relations.json
          kg_vocab: kg_mappings/relation_vocab.json
          max_ent_kg_len: 60
        max_seq_len: 128
        max_seq_window_len: 64
        max_ent_len: 128


In this example, the entity input will have descriptions, types, and relations. You can control the total length of each resource by a ``max_ent_type_len`` and ``max_ent_kg_len`` param and the maximum entity length by ``max_ent_len``.

Entity Masking
^^^^^^^^^^^^^^^
A secret sauce to getting our Bootleg encoder to pay attention to the types and relationships is to apply masking of the mention and entity title. Without masking, the model will rely heavily on mention-title memorization and ignore more subtle structural cues required for the tail. To overcome this, we mask entity titles in the entity encoder and mentions in the context encoder. By default, we mask titles and mentions 50% of the time, with more popular entities being masked up to 95% of the time. To turn this off, in ``data_config``, set ``popularity_mask`` to be ``false``.

If desired, we also support MLM style masking of the context input. By default, we do not use this masking, but you can turn it on by setting ``context_mask_perc`` to be between 0.0 and 1.0 in ``data_config``.

Candidates and Aliases
^^^^^^^^^^^^^^^^^^^^^^

Candidate Not in List
~~~~~~~~~~~~~~~~~~~~~

Bootleg supports two types of candidate lists: (1) assume that the true entity must be in the candidate list, (2) use a NIL or "No Candidate" (NC) as another candidate, and does not require that the true candidate is the candidate list. Not that if using (1), during training, the gold candidate *must* be in the list or preprocessing with fail. The gold candidate does not have to be in the candidate set for evaluation. To switch between these two modes, we provide the ``train_in_candidates`` parameter (where True indicates (1)).

Multiple Candidate Maps
~~~~~~~~~~~~~~~~~~~~~~~

Within the ``entity_map_dir`` there may be multiple candidate maps for the same set of entities. For instance, a benchmark dataset may use a specific candidate mapping. To specify which candidate map to use, we set the ``alias_cand_map`` value in the config.

Datasets
^^^^^^^^

We define the train, dev, and test datasets in ``train_dataset``\ , ``dev_dataset``\ , and ``test_dataset`` respectively. For each dataset, we need to specify the name of the file  with the ``file`` key. We can also specify whether to use weakly labeled alias-entity pairs (pairs that are labeled heurisitcally during preprocessing). For training, if ``use_weak_label`` is True, these alias-entity pairs will contribute to the loss. For evaluation, the weakly labelled alias-entity pairs will only be used as more signal for other alias-entity pairs (e.g. for collective disambiguation), but will not be scored.  As an example of a dataset entry, we may have:

.. code-block::

   train_dataset:
      file: train.jsonl
      use_weak_label: true


Word Embeddings
^^^^^^^^^^^^^^^

Bootleg leverages BERT Transformers to encode the entities and mention context. This type of BERT model and its size is configured in the ``word_embedding`` section of the config. You can change which HuggingFace BERT model by the ``bert_model`` param, change its cached direction by ``cache_dir``, and the number of layers by ``context_layers`` and ``entity_layers``.


Finally, in the ``data_config``\ , we define a maximum word token length through ``max_seq_len`` and that max window length around a mention by ``max_seq_window_len``.

Preprocessing the Data
-------------------------

Prior to training, if the data is not already prepared, we will preprocess or prep the data. This is where we convert the context and entity token data to a memory-mapped format for the dataloader to quickly load during training. If the data does not change, this preprocessing only needs to happen once.

*Warning: errors may occur if the file contents change but the file names stay the same, since the preprocessed data uses the file name as a key and will be loaded based on the stale data. In these cases, we recommend removing the ``prep`` directories or assigning a new prep directory (by setting ``data_prep_dir`` or ``entity_prep_dir`` in the config) and repeating preprocessing.*

Prep Directories
^^^^^^^^^^^^^^^^

As the preprocessed knowledge graph and type embedding data only depends on the entities, we store it in a prep directory in the entity directory to be shared across all datasets that use the same entities and knowledge graph/type data. We store all other preprocessed data in a prep directory inside the data directory.


Training the Model
---------------------

After the data is prepped, we are ready to train the model! As this is just a tiny random sample of Wikipedia sentences with sampled KG information, we do not expect the results to be good  (for instance, we haven't seen most aliases in dev in training and we do not have an adequate number of examples to learn reasoning patterns).  We recommend training on GPUs. To train the model on a single GPU, we run:

.. code-block::

   python3 bootleg/run.py --config_script configs/tutorial/sample_config.yaml


If a GPU is not available, we can also get away with training this tiny dataset on the CPU by adding the flag below to the command. Flags follow the same hierarchy and naming as the config, and the ``cpu`` parameter could also have been set directly in the config file in the ``run_config`` section:

.. code-block::

   python3 bootleg/run.py --config_script configs/tutorial/sample_config.json --emmental.device -1

At each eval step, we see a json save of eval metrics. At the beginning end end of the model training, you should see a print out of the log direction. E.g.,

``Saving metrics to logs/turtorial/2021_03_11/20_31_11/02b0bb73``

Inside the log directory, you'll find all checkpoints, the ``emmental.log`` file, ``train_metrics.txt``, and ``train_disambig_metrics.csv``. The latter two files give final eval scores of the model. For example, after 10 epochs, ``train_disambig_metrics.csv`` shows

.. code-block::

    task,dataset,split,slice,mentions,mentions_notNC,acc_boot,acc_boot_notNC,acc_pop,acc_pop_notNC
    NED,Bootleg,dev,final_loss,70,70,0.8714285714285714,0.8714285714285714,0.8714285714285714,0.8714285714285714
    NED,Bootleg,test,final_loss,70,70,0.8714285714285714,0.8714285714285714,0.8714285714285714,0.8714285714285714

The fields are

* ``task``: the task name (will be NED for disambiguation metrics).
* ``dataset``: dataset (if case of multi-modal training)
* ``slice``: the subset of the dataset evaluated. ``final_loss`` is the slice which includes all mentions in the dataset. If you set ``emmental.online_eval`` to be True in the config, training metrics will also be reported and collected.
* ``mentions``: the number of mentions (aliases) under evaluation.
* ``mentions_notNC``: the number of mentions (aliases) under evaluation where the gold QID is in the candidate list.
* ``acc_boot``: the accuracy of Bootleg.
* ``acc_boot_notNC``: the accuracy of Bootleg for notNC mentions.
* ``acc_boot``: the accuracy of a baseline where the first candidate is always selected as the answer.
* ``acc_boot_notNC``: the accuracy of the baseline for notNC mentions.

As our data was very tiny, our model is not doing great, but the train loss is going down!

Evaluating the Model
---------------------

After the model is trained, we can also run eval to get test scores or to save predictions. To eval the model on a single GPU, we run:

.. code-block::

   python3 bootleg/run.py --config_script configs/tutorial/sample_config.yaml --mode dump_preds --emmental.model_path logs/turtorial/2021_03_11/20_31_11/02b0bb73/last_model.pth

This will generate a label file at ``logs/turtorial/2021_03_11/20_38_09/c5e204dc/dev/last_model/bootleg_labels.jsonl`` (path is printed). This can be read it for evaluation and error analysis. Check out the End-to-End Tutorial on our `Tutorials Page <https://github.com/HazyResearch/bootleg/tree/master/tutorials>`_ for seeing how to do this and for evaluating pretrained Bootleg models.

Advanced Training
-----------------

Bootleg supports distributed training using PyTorch's `Distributed Data Parallel <https://pytorch.org/docs/stable/notes/ddp.html>`_ framework. This is useful for training large datasets as it parallelizes the computation by distributing the batches across multiple GPUs. We explain how to use distributed training in Bootleg to train a model on a large dataset (all of Wikipedia with 50 million sentences) in the `Advanced Training Tutorial <../advanced/distributed_training.html>`_.

.. _Input Data: input_data.html
.. _Bootleg Model: model.html
.. _Entity Profile: entity_profile.html
