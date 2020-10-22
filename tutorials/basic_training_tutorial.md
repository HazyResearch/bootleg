# Basic Training Tutorial

We describe how to train a Bootleg model for named entity disambiguation (NED), starting from a new dataset. If you already have a dataset in the Bootleg format, you can skip to [Preparing the Config](#2-preparing-the-config). All commands should be run from the root directory of the repo.

## 1. Formatting the Data

We assume three components are available for input: (1) [text datasets](#text-datasets), (2) [entity and alias data](#entities-and-aliases), and (3) [type and knowledge graph data](#type-and-knowledge-graph-kg-data). For each component, we first describe the data requirements and then discuss how to convert the data to the expected format. Finally, we discuss the expected [directory structure](#directory-structure) to organize the data components. We provide a small dataset sampled from Wikipedia in the directory `data` that we will use throughout this tutorial as an example.

### Text Datasets

#### Requirements
1. Text data for training and dev datasets, and if desired, a test dataset, is available. For simplicity, in this tutorial, we just assume there is a dev dataset available.
2. Known aliases (also known as mentions) and linked entities are available.<sup>[1](#myfootnote1)</sup> This information can be obtained for Wikipedia, for instance, by using anchor text on Wikipedia pages as aliases and the linked pages as the entity label.

Each dataset will need to follow the format described below.

We assume that the text dataset is formatted in a [jsonlines](https://jsonlines.org) file (each line is a dictionary) with the following keys:
- `sentence`: the text of the sentence.
- `sent_idx_unq`: a unique numeric identifier for each sentence in the dataset.
- `aliases`: the aliases in the sentence to disambiguate. Aliases serve as lookup keys into an alias candidate map to generate candidates, and may not actually appear in the text. For example, the phrase "she" in the sentence may be weakly labelled as the alias "Heidi" by a simple heuristic.
- `spans`: the start and end word indices of the aliases in the text, where the end span is exclusive (like python slicing).
- `qids`: the quantifier id of the true entity for each alias.
- `gold`: True if the entity label was an anchor link in the source dataset or otherwise known to be "ground truth"; False, if the entity label is from weak labeling techniques. While all provided alias-entity pairs can be used for training, only alias-entity pairs with a gold value of True are used for evaluation.
- (Optional) `slices`: indicates which alias-entity pairs are part of certain data subsets for evaluating performance on important subsets of the data and training with slice-based learning (see the [Advanced Training Tutorial](advanced_training_tutorial.md)).

Using this format, an example line is:

    {
        "sentence": "Heidi and her husband Seal live in Vegas . ",
        "sent_idx_unq": 0,
        "aliases": ["heidi", "seal", "vegas"],
        "spans": [[0,1], [4,5], [7,8]],
        "qids": ["Q60036", "Q218091", "Q23768"],
        "gold": [true, true, true]
    }

We also provide sample [training](../data/sample_text_data/train.jsonl) and [dev](../data/sample_text_data/dev.jsonl) datasets as examples of text datasets in the proper format.

### Entities and Aliases

#### Requirements
1. There is a set of entities to consider as candidates for training and evaluation. There are quantifier ids (i.e. QIDs) and titles available for these entities. For instance, quantifier ids may be Wikidata QIDs or Unified Medical Language System (UMLS) Concept Unique Identifiers (CUI).
2. (Recommended) There is a candidate mapping from aliases to entity candidates. The candidates must be in the set of entities above. If this is not provided, we apply a simple mining technique to generate this from the provided training dataset.

#### QID-to-Title Mapping
We assume that the set of entity quantifier ids (QIDs) and their corresponding titles are stored in a JSON file as a dictionary of QID to title pairs. For example,

    {
        "Q60036": "Heidi Klum",
        "Q218091": "Seal (musician)",
        "Q23768": "Las Vegas"
    }

We provide an QID-to-title mapping for the sample Wikipedia dataset in [data/sample_entity_data/entity_mappings/qid2title.json](../data/sample_entity_data/entity_mappings/qid2title.json).

#### Candidate Mapping
If provided, we assume that the candidate mapping is stored in a JSON file as a dictionary of alias to list of [QID, sort_value] pairs, where the sort_value can be any numeric quantity to sort the candidate lists. The sort_value is necessary to choose the candidates when the number of candidates is greater than the maximum allowed (max candidates is a settable parameter). For example (candidates are cut short to display),

    {
        "heidi": [["Q60036", 10286], ["Q66019", 10027], ... ]
        "seal": [["Q218091", 10416], ["Q9458", 4504], ... ]
        "vegas": [["Q23768", 7613], ["Q2624848", 3191], ... ]
    }

We provide an example candidate mapping in [data/sample_entity_data/entity_mappings/alias2qids.json](../data/sample_entity_data/entity_mappings/alias2qids.json). We assume that all aliases are lowercased.

*If a candidate mapping isn't available for the new dataset*, we can run the below command to scrape the training data to generate this mapping. Each alias in the training data is added to the candidate mapping with its true label as a candidate. Note that this technique will perform poorly if the aliases are very different between the training and dev datasets.

    python bootleg/utils/gen_alias_cand_map.py --alias2qids_file data/sample_entity_data/alias2qids.json --train_file data/sample_text_data/train.jsonl

#### Entity Mappings

We use the QID-to-title and candidate mappings to generate additional mappings to indices in internal Bootleg embeddings. We store the full set of mappings in an entity directory. In addition to these mappings, the entity directory will also store preprocessed embedding data associated with the entities (see [Preprocessing the Data](#3-preprocessing-the-data)).

To generate the mappings, run the following command from the root directory of the repo:

    python bootleg/utils/gen_entity_mappings.py --entity_dir data/sample_entity_data/ \
    --qid2title data/sample_entity_data/entity_mappings/qid2title.json \
    --alias2qids data/sample_entity_data/entity_mappings/alias2qids.json

We assume that each alias can have a maximum of 30 candidates. To change the maximum number of candidates, we can add `--max_candidates <max_num>` to the command above. Note that increasing the number of maximum candidates increases the memory required for training and inference.

### Type and Knowledge Graph (KG) Data

One of the key insights from Bootleg is that leveraging type and knowledge graph information in a simple attention-based network can improve performance on tail entities. However, to leverage this information, we need to provide type and/or knowledge graph information to the model.

#### Requirements

1. Type labels from a type ontology (e.g. Wikidata or HYENA types from YAGO) is available for the candidate entities. While we do not need types assigned to all entities, the higher the coverage, the better.
2. Knowledge graph connectivity information, such as whether two entities are connected in knowledge graph, is available between pairs of entities. Furthermore, similar to the type labels, there is a mapping from entities to the knowledge graph relations they participate in.

#### Type Information

We assume that the type data is provided in a JSON file as a dictionary of pairs of QIDs to a list of type ids. If there are *N* distinct types, the type ids should range from 0 to *N-1*. As multiple types may be associated with an entity, we store the list of type ids with each QID. The maximum number of types considered per an entity is a settable parameter.

For instance, if we have a type vocabulary of

    {
        "place": 0,
        "person": 1,
        "city": 2
    }

then we may have an associated QID-to-type mapping of

    {
        "Q60036": [1],
        "Q218091": [1],
        "Q23768": [0, 2]
    }

An example of the QID-to-type mapping can be found in [data/sample_emb_data/qid2types.json](../data/sample_emb_data/qid2types.json) with the associated type vocabulary in [data/sample_emb_data/type_vocab.json](../data/sample_emb_data/type_vocab.json).

#### KG Information

We describe the two components of KG data that we provide to the model---KG connectivity data and KG relation data.

*Connectivity Data*

We assume that the connectivity information is provided in a simple text file where each line is a tab-separated QID pair, if an edge exists between the two QIDs in a relevant knowledge graph. For instance, Q60036 (Heidi Klum) and Q218091 (Seal) share an edge (spouse), so we would have the line below in the connectivity data.

    Q60036  Q218091



Check out [data/sample_emb_data/kg_conn.txt](../data/sample_emb_data/kg_conn.txt) as an example of QID connectivity from Wikidata.

*Relation Data*

We treat relation labels as types and assume the same format as type information. An example of a QID-to-relation mapping can be found in [data/sample_emb_data/qid2relations.json](../data/sample_emb_data/qid2relations.json) with the associated relation vocabulary in [data/sample_emb_data/relation_vocab.json](../data/sample_emb_data/relation_vocab.json).

### Directory Structure

We assume the data above is saved in the following directory structure, where the specific directory and filenames can be set in the config discussed in [Preparing the Config](#preparing-the-config). We will also discuss how to generate the `prep` directories in [Preprocessing the Data](#preprocessing-the-data). The `emb_data` directory can be shared across text datasets and entity sets, and the `entity_data` directory can be shared across text datasets (if they use the same set of entities).

```
text_data/
    train.jsonl
    dev.jsonl
    prep/

emb_data/
    qid2types.json
    kg_conn.txt
    qid2relations.json

entity_data/
    entity_mappings/
        config.json
        qid2title.json
        alias2qids.json
        ...
    prep/
```

## 2. Preparing the Config

Once the data has been converted to the correct format, we are ready to prepare the config. We provide a sample config in [configs/sample_config.json](../configs/sample_config.json). The full parameter options and defaults for the config file are in `bootleg/config.py`. If values are not provided in the JSON config, the default values are used. We provide a brief overview of the configuration settings here.

The config parameters are organized into five main groups:
- `run_config`: run time settings, such as cpu v. gpu, distributed training, and frequency of logging.
- `train_config`: training parameters for hyperparameter tuning, such as dropout and learning rate.
- `model_config`: model parameters, such as number of attention heads or hidden dimension.
- `data_config`: paths of text data, embedding data, and entity data to use for training and evaluation, as well as configuration details for the entity embeddings.
- `prep_config`: flags to control preprocessing or prepping of data prior to training/evaluating.

We highlight a few parameters in the `run_config`.
- `save_dir` should be set to specify where log output and model checkpoints should be saved. When a new model is trained, Bootleg automatically generates a timestamp and saves output to a folder with the timestamp inside the `save_dir`.
- `eval_steps` indicates how frequently the evaluation on the dev set should be run. Steps corresponds to batches, such that 10 steps means 10 batches have been processed.
- `save_every_k_eval` indicates when to save a model checkpoint after performing evaluation. If set to 1, then a model checkpoint will be saved every time dev evaluation is run.

We now focus on the `data_config` parameters as these are the most unique to Bootleg. We walk through the key parameters in the `data_config` to pay attention to.

### Directories

We define the paths to the directories through the `data_dir`, `emb_dir`, `entity_dir`, and `entity_map_dir` config keys. The first three correspond to the top-level directories introduced in [Directory Structure](#directory-structure). The `entity_map_dir` includes the entity JSON mappings produced in [Entities and Aliases](#entities-and-aliases) and should be inside the `entity_dir`. For example, to follow the directory structure set up in the `data` directory, we would have:

```
"data_dir": "data/sample_text_data",
"emb_dir": "data/sample_emb_data",
"entity_dir": "data/sample_entity_data",
"entity_map_dir": "entity_mappings"
````

### Entity Payloads
As described in the `README`, Bootleg takes in a set of embeddings to form an **entity payload** for each candidate. These embeddings are concatenated together and projected down to Bootleg's hidden dimension. The embeddings which form the entity payload are defined in the `ent_embeddings` section of the config. We consider the entry below for `ent_embeddings`.

```
"ent_embeddings": [
            {
                "key": "learned",
                "load_class": "LearnedEntityEmb",
                "args": {
                    "learned_embedding_size": 256,
                    "mask_perc": 0.8
                }
            },
            {
               "key": "learned_type",
               "load_class": "LearnedTypeEmb",
               "args": {
                   "type_labels": "qid2types.json",
                   "max_types": 3,
                   "type_dim": 128,
                   "merge_func": "addattn",
                   "attn_hidden_size": 128
               }
            }]
```
In this example, the entity payload consists of two embeddings, a learned entity embedding and a learned type embedding. Each embedding must have a unique `key` which identifies it, as well as a `load_class` that indicates which embedding class to use (all input embedding classes are defined in `bootleg/embeddings`). Finally, each embedding may have custom args defined in the `args` key. For example, in addition to parameters such as the dimension of the embeddings, we have other embedding-specific parameters:
- For the learned entity embedding, we define a regularization parameter here which performs 2D masking (masks the entire entity embedding) on 80% of the entity embeddings during training, to encourage learning information in the other, more generalizable embeddings.
- For the learned type embedding, we specify the path to the QID-to-type mapping introduced in [Type Information](#type-information), as well as the maximum number of types to include per entity (`max_types`) and the mechanism to combine multiple types across entities (`merge_func`).

The custom args are defined in the embedding class specified by `load_class`. By looking at the corresponding embedding class, we can determine what custom args are available to set and how they are used. For example, by the `load_class` for this type embedding above, we know that the type embedding uses the LearnedTypeEmb class. If we look in [bootleg/embeddings/type_embs.py]([bootleg/embeddings/type_embs.py]), we can find the LearnedTypeEmb class. The `emb_args` parameter in `__init__` corresponds to the `args` dictionary in the config, and we can see how `type_dim` is used to set the dimension of the type embedding. We can repeat this process for each key in the custom args.

The contents of the entity payload can easily be modified by adding more or fewer embeddings to the `ent_embeddings` list. For instance, if we want to define a new knowledge graph embedding, we can add a new class to `bootleg/embeddings/kg_embs` and then add an another entry in the `ent_embeddings` list for the new embedding.

### Candidates and Aliases

#### Candidate Not in List
Bootleg supports two types of candidate lists: (1) assume that the true entity must be in the candidate list, (2) use a NIL or "No Candidate" (NC) as another candidate, and do not require that the true candidate is the candidate list. To switch between these two modes, we provide the `train_in_candidates` parameter (where True indicates (1)). Note that if `train_in_candidates` is True, it must be the case that all true entities are in the candidate lists or preprocessing will fail.

#### Maximum Aliases
We can also specify the maximum number of aliases considered for each training example with `max_aliases`. Similar to the maximum number of candidates (see discussion in [Entity Mappings](#entity-mappings)), increasing this number will increase the memory required for training and inference. However, with more aliases, we may also have more signal to leverage for disambiguation. If we have more than ten aliases in a sentence, we use a windowing technique to generate multiple examples, with the aliases divided across them. This windowing process is done automatically during preprocessing.

#### Multiple Candidate Maps
Within the `entity_map_dir` there may be multiple candidate maps for the same set of entities. For instance, a benchmark dataset may use a specific candidate mapping. To specify which candidate map to use, we set the `alias_cand_map` value in the config.

### Datasets
We define the train, dev, and test datasets in `train_dataset`, `dev_dataset`, and `test_dataset` respectively. For each dataset, we need to specify the name of the file  with the `file` key. We can also specify whether to use weakly labeled alias-entity pairs (see [Text Datasets](#text-datasets)). For training, if `use_weak_label` is True, these alias-entity pairs will contribute to the loss. For evaluation, the weakly labelled alias-entity pairs will only be used as more signal for other alias-entity pairs (e.g. for collective disambiguation), but will not be scored.  As an example of a dataset entry, we may have:
```
"train_dataset":
    {
        "file": "train.jsonl",
        "use_weak_label": true
    }
```

### Word Embeddings
Bootleg leverages existing word embeddings to embed sentence tokens. This is configured in the `word_embedding` section of the config. In particular, we currently support using BERT as the backbone for contextual word embeddings. We store the pretrained BERT word embeddings, encoder, and tokenizer in a directory that is specified by the `cache_dir` key. We also support freezing and finetuning BERT through the `freeze_word_emb` and `freeze_sent_emb` keys.

To download the pretrained BERT (English, cased) word embeddings, encoder, and tokenizer, run:

    bash download_bert.sh

To run the preprocessing and training commands later in the tutorial you will need to download the BERT artifacts above.

Finally, in the `data_config`, we define a maximum word token length through `max_word_token_len`. We typically use a length of 100--increasing this length will increase the memory required for training and inference.

## 3. Preprocessing the Data

Prior to training, we have preprocess or prep the data, where we convert the data to a memory-mapped format for the dataloader to quickly load during training and also create arrays to allow quick lookups into the embedding data. For instance we create a torch tensor to store the contents of qid2types JSON file to get indices into a type embedding. If the data does not change, this preprocessing only needs to happen once.

*Warning: errors may occur if the file contents change but the file names stay the same, since the preprocessed data uses the file name as a key and will be loaded based on the stale data. In these cases, we recommend removing the `prep` directories or assigning a new prep directory (by setting `data_prep_dir` or `entity_prep_dir` in the config) and repeating preprocessing.*

### Prep Directories
As the preprocessed knowledge graph and type embedding data only depends on the entities, we store it in a prep directory in the entity directory to be shared across all datasets that use the same entities and knowledge graph/type data. We store all other preprocessed data in a prep directory inside the data directory.

### Separate Prep Stage

While this preprocessing will happen if you directly train a model with no existing prepped data, we support a separate prep command prior to training. This is highly recommended for larger datasets which take a long time to preprocess (preprocessing only requires CPUs!) and helpful if you want to preprocess the data on one machine and train on multiple machines, for example on a cluster for a hyperparameter search.

To run preprocessing, we can simply run:

    python bootleg/prep.py --config_script configs/sample_config.json

## 4. Training the Model

After the data is prepped, we are ready to train the model! As this is just a tiny random sample of Wikipedia sentences with sampled KG information, we do not expect the results to be good  (for instance, we haven't seen most aliases in dev in training and we do not have an adequate number of examples to learn reasoning patterns).  We recommend training on GPUs (default value in the config). To train the model on a single GPU, we run:

    python bootleg/run.py --config_script configs/sample_config.json

If a GPU is not available, we can also get away with training this tiny dataset on the CPU by adding the flag below to the command. Flags follow the same hierarchy and naming as the config, and the `cpu` parameter could also have been set directly in the config file in the `run_config` section:

    python bootleg/run.py --config_script configs/sample_config.json --run_config.cpu True

We get the following results after the 10th epoch (on a NVIDIA P100 GPU).

```
+------------+------------+-------+--------+-------------+------------+-------+-----------+----------+-------+
| head       | slice      |   men |   crct |   crct_top5 |   crct_pop |    f1 |   f1_top5 |   f1_pop |   stp |
+============+============+=======+========+=============+============+=======+===========+==========+=======+
| final_loss | final_loss |    51 |     32 |          44 |         43 | 0.627 |     0.863 |    0.843 |   100 |
+------------+------------+-------+--------+-------------+------------+-------+-----------+----------+-------+
```
While the test performance is poor as expected (this sample is too small!), we do see the loss in the log decreasing indicating that the model is memorizing the examples. We explain the column headers below:
- `head`: the prediction head. `final_loss` is the name of the final prediction head.
- `slice`: the subset of the dataset evaluated. `final_loss` is the slice which includes all mentions in the dataset.
- `men`: the number of mentions (aliases) under evaluation.
- `crct`: the number of mentions correct.
- `crct_top5`: the number of mentions correct in the top K predictions. In this case, K is 5, though this parameter is settable via the config.
- `crct_pop`: the number of mentions correct by a simple "most popular" baseline which simply chooses the most popular candidate for a mention.
- `f1`: the micro-F1 score of Bootleg for classifying entities.
- `f1_top5`: the micro-F1 score for classifying entities in the top K.
- `f1_pop`: the micro-F1 score of the "most popular" baseline for classifying entities.
- `stp`: the number of steps (batches) completed in training.


## Next Steps

### Evaluation
Check out the [Benchmark Tutorial](benchmark_tutorial.ipynb) and the [End-to-End Tutorial](end2end_ned_tutorial.ipynb) for evaluating pretrained Bootleg models.

### Advanced Training
Bootleg supports distributed training using PyTorch's [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html) framework. This is useful for training large datasets as it parallelizes the computation by distributing the batches across multiple GPUs. We explain how to use distributed training in Bootleg to train a model on a large dataset (all of Wikipedia with 50 million sentences) in the [Advanced Training Tutorial](advanced_training_tutorial.md).

<a name="myfootnote1">1</a>: More advanced techniques that are outside the scope of this tutorial can use alias extraction techniques to detect aliases and weak labeling through distant or weak supervision to assign entity labels to the detected aliases.