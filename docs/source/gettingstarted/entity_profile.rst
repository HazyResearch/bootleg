Entity Profiles
=================
Bootleg uses Wikipedia and Wikidata to collect and generate a entity database of metadata associated with an entity. We support both non-structural data (e.g., the title of an entity) and structural data (e.g., the type or relationship of an entity). We now describe how to generate entity profile data from scratch to be used for training and the structure of the profile data we already provide.

Generating Profiles
--------------------
The database of entity data starts with a simple ``jsonl`` file of data associated with an entity. Specifically, each line is a JSON object

.. code-block:: JSON

    {
        "entity_id": "Q16240866",
        "mentions": [["benin national u20 football team",1],["benin national under20 football team",1]],
        "title": "Forbidden fruit",
        "description": "A fruit that once was considered not to be eaten",
        "types": {"hyena": ["<wordnet_football_team_108080025>"],
                  "wiki": ["national association football team"],
                  "relations":["country for sport","sport"]},
        "relations": [
            {"relation":"P1532","object":"Q962"},
        ],
    }

The ``entity_id`` gives a unique string identifier of the entity. It does *not* have to start with a ``Q``. As we normalize to Wikidata, our entities are referred to as QIDs. The ``mentions`` provides a list of known aliases to the entity and a prior score associated with that mention indicating the strength of association. The score is used to order the candidates. The ``types`` provides the different types and entity is and supports different type systems. In the example above, the two type systems are ``hyena`` and ``wiki``. We also have a ``relations`` type system which treats the relationships an entity participates in as types. The ``relations`` JSON field provides the actual KG relationship triples where ``entity_id`` is the head.

.. note::

    By default, Bootleg assigns the score for each mentions as being the global entity count in Wikipedia. We empirically found this was a better scoring method for incorporating Wikidata "also known as" aliases that did not appear in Wikipedia. This means the scores for the mentions for a single entity will be the same.

We provide a more complete `sample of raw profile data <https://github.com/HazyResearch/bootleg/tree/master/data/sample_raw_entity_data/raw_profile.jsonl>`_ to look at.

Once the data is ready, we provide an `EntityProfile <../apidocs/bootleg.symbols.html#module-bootleg.symbols.entity_profile>`_ API to build and interact with the profile data. To create an entity profile for the model from the raw ``jsonl`` data, run

.. code-block:: python

    from bootleg.symbols.entity_profile import EntityProfile
    path_to_file = "data/sample_raw_entity_data/raw_profile.jsonl"
    # edit_mode means you are allowed to modify the profile
    ep = EntityProfile.load_from_jsonl(path_to_file, edit_mode=True)

.. note::

    By default, we assume that each alias can have a maximum of 30 candidates, 10 types, and 100 connections. You can change these by adding ``max_candidates``, ``max_types``, and ``max_connections`` as keyword arguments to ``load_from_jsonl``. Note that increasing the number of maximum candidates increases the memory required for training and inference.

Profile API
--------------------
Now that the profile is loaded, you can interact with the metadata and change it. For example, to get the title and add a type mapping, you'd run

.. code-block:: python

    ep.get_title("Q16240866")
    # This is adding the type "country" to the "wiki" type system
    ep.add_type("Q16240866", "sports team", "wiki")

Once ready to train or run a model with the profile data, simply save it

.. code-block:: python

    ep.save("data/sample_entity_db")

We have already provided the saved dump at ``data/sample_entity_data``.

See our `entity profile tutorial <https://github.com/HazyResearch/bootleg/tree/master/tutorials/entity_profile_tutorial
.ipynb>`_ for a more complete walkthrough notebook of the API.

Training with a Profile
------------------------
Inside the saved folder for the profile, all the mappings needed to run a Bootleg model are provided. There are three subfolders as described below. Note that we use the word ``alias`` and ``mention`` interchangeably.

* ``entity_mappings``: This folder contains non-structural entity data.
    * ``qid2eid``: This is a folder containing a Trie mapping from entity id (we refer to this as QID) to an entity index used internally to extract embeddings. Note that these entity ids start at 1 (0 index is reserved for a "not in candidate list" entity). We use Wikidata QIDs in our tutorials and documentation but any string identifier will work.
    * ``qid2title.json``: This is a mapping from entity QID to entity Wikipedia title.
    * ``qid2desc.json``: This is a mapping from entity QID to entity Wikipedia description.
    * ``alias2qids``: This is a folder containing a RecordTrie mapping from possible mentions (or aliases) to a list possible candidates. We restrict our candidate lists to be a predefined max length, typically 30. Each item in the list is a pair of [QID, QID score] values. The QID score is used for sorting candidates before filtering to the top 30. The scores are otherwise not used in Bootleg. This mapping is mined from both Wikipedia and Wikidata (reach out with a github issue if you want to know more).
    * ``alias2id``: This is a folder containing a Trie mapping from alias to alias index used internally by the model.
    * ``config.json``: This gives metadata associated with the entity data. Specifically, the maximum number of candidates.
* ``type_mappings``: This folder contains type entity data for each type system subfolder. Inside each subfolder are the following files.
    * ``qid2typenames``: Folder containing a RecordTrie mapping from entity QID to a list of type names.
    * ``config.json``: Contains metadata of the maximum number of types allowed for an entity.
* ``kg_mappings``: This folder contains relationship entity data.
    * ``relation_vocab.json``: Mapping from human-readable relation name to relation ID used in qid2relations. Used to generate entity text input.
    * ``qid2relations.json``: Mapping from head entity QID to a dictionary of relation -> list of tail entities.
    * ``kg_adj.txt``: List of all connected entities separated by a tab. This is an unlabeled adjacency matrix.
    * ``config.json``: Contains metadata of the maximum number of tail connections allowed for a particular head entity and relation.

.. note::

    In Bootleg, we add types from a selected type system and add KG relationship triples to our entity encoder.

.. note::

    In our public ``entity_db`` provided to run Bootleg models, we also provide ``alias2qids_unfiltered.json`` which provides our unfiltered, raw candidate mappings. We filter noisy aliases before running mention extraction.

Given this metadata, you simply need to specify the types, relation mappings and correct folder structures in a Bootleg training `config <config.html>`_. Specifically, these are the config parameters that need to be set to be associated with an entity profile.

.. code-block::

    data_config:
      entity_dir: data/sample_entity_data
      use_entity_desc: true
      entity_type_data:
        use_entity_types: true
        type_symbols_dir: type_mappings/wiki
      entity_kg_data:
        use_entity_kg: true
        kg_labels: kg_mappings/qid2relations.json
        kg_vocab: kg_mappings/relation_vocab.json

See our `example config <https://github.com/HazyResearch/bootleg/tree/master/configs/tutorial/sample_config.yaml>`_
for a full reference, and see our `entity profile tutorial <https://github
.com/HazyResearch/bootleg/tree/master/tutorials/entity_profile_tutorial.ipynb>`_ for some methods to help modify
configs to map to the entity profile correctly.
