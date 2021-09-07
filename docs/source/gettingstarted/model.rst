Model Overview
==============
Given an input sentence, list of mentions to be disambiguated, and list of possible candidates for each mention (described in `Input Data`_), Bootleg outputs the most likely candidate for each mention. Bootleg's model is a biencoder architecture and consists of two components: the entity encoder and context encoder. For each entity candidate, the entity encoder generates an embedding representing this entity from a textual input containing entity information such as the title, description, and types. The context encoder embeds the mention and its surrounded context. The selected candidate is the one with the highest dot product.

We now describe each step in detail and explain how to add/remove different parts of the entity encoder in our `Bootleg Config`_.

Entity Encoder
--------------------------
The entity encoder is a BERT Transformer that takes a textual input for an entity and feeds it through BERT. During training, we take the ``[CLS]`` token as the entity embedding. There are four pieces of information we add to the textual input for an entity:

* ``title``: Entity title. Comes from ``qid2title.json``. This is always used.
* ``description``: Entity description. Comes from ``qid2desc.json``. This is toggled on/off.
* ``type``: Entity type from one of the type systems. Comes from ``qid2typeids.json`` of a type system specified in the config. If the entity has multiple types, we add them to the input as ``<type_1> ; <type_2> ; ...``
* ``KG``: Entity KG triples. Comes from ``qid2relations.json`` specified in the config. We add KG relations to the input as ``<predicate_1> <object_1> ; <predicate_2> <object_2> ; ...`` where the head of each triple is the entity in question.

The final entity input is ``<title> [SEP] <types> [SEP] <relations> [SEP] <description>``.

You control what inputs are added by the following part in the input config. All the relevant entity encoder code is in `bootleg/dataset.py <../apidocs/bootleg.datasets.html>`_.

.. code-block::

    data_config:
        ...
        use_entity_desc: true
        entity_type_data:
          use_entity_types: true
          type_labels: type_mappings/wiki/qid2typeids.json
          type_vocab: type_mappings/wiki/type_vocab.json
        entity_kg_data:
          use_entity_kg: true
          kg_labels: kg_mappings/qid2relations.json
          kg_vocab: kg_mappings/relation_vocab.json
        max_seq_len: 128
        max_seq_window_len: 64
        max_ent_len: 128


Context Encoder
------------------
Like the entity encoder, our context encode takes the context of a mention and feeds it through a BERT Transformer. The ``[CLS]`` token is used as th e relevant mention embedding. To allow BERT to understand where the mention is, we separate it by ``[ENT_START]`` and ``[ENT_END]`` clauses. As shown above, you can specify the maximum sequence length for the context encoder and the maximum window length. All the relevant context encoder code is in `bootleg/dataset.py <../apidocs/bootleg.datasets.html>`_.

.. _Input Data: input_data.html
.. _Bootleg Config: config.html
