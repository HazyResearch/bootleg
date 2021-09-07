Unreleased 1.1.0dev0
---------------------
Changed
^^^^^^^^^
We did an architectural change and switched to a biencoder model. This changes our task flow and dataprep. This new model uses less CPU storage and uses the standard BERT architecture. Our entity encoder now takes a textual input of an entity that contains its title, description, KG relationships, and types.

1.0.5 - 2021-08-20
---------------------
Fixed
^^^^^^^^
* Fixed -1 command line argparse error
* Adjusted requirements

1.0.4 - 2021-07-12
---------------------
Added
^^^^^^
* Tutorial to generate contextualized entity embeddings that perform better downstream

1.0.4 - 2021-07-12
---------------------
Fixed
^^^^^^^^
* Bump version of Pydantic to 1.7.4

1.0.3 - 2021-06-29
---------------------
Fixed
^^^^^^^
* Corrected how custom candidates were handled in the BootlegAnnotator when using ``extracted_examples``
* Fixed memory leak in BooltegAnnotator due to missing ``torch.no_grad()``

1.0.2 - 2021-04-28
---------------------

Added
^^^^^^
* Support for ``min_alias_len`` to ``extract_mentions`` and the ``BootlegAnnotator``.
* ``return_embs`` flag to pass into ``BootlegAnnotator`` that will return the contextualized embeddings of the entity (using key ``embs``) and entity candidates (using key ``cand_embs``).

Changed
^^^^^^^^^
* Removed condition that aliases for eval must appear in candidate lists. We now allow for eval to not have known aliases and always mark these as incorrect. When dumping predictions, these get "-1" candidates and null probabilities.

Fixed
^^^^^^^
* Corrected ``fit_to_profile`` to rebuild the title embeddings for the new entities.

1.0.1 - 2021-03-22
-------------------

.. note::

    If upgrading to 1.0.1 from 1.0.0, you will need to re-download our models given the links in the README.md. We altered what keys were saved in the state dict, but the model weights are unchanged.

Added
^^^^^^^
* ``data_config.print_examples_prep`` flag to toggle data example printing during data prep.
* ``data_config.eval_accumulation_steps`` to support subbatching dumping of predictings. We save outputs to separate files of size approximately ``data_config.eval_accumulation_steps*data_config.eval_batch_size`` and merge into a final file at the end.
* Entity Profile API. See the `docs <https://bootleg.readthedocs.io/en/latest/gettingstarted/entity_profile.html>`_. This allows for modifying entity metadata as well as adding and removing entities. We profile methods for refitting a model with a new profile for immediate inference, no finetuning needed.

Changed
^^^^^^^^
* Support for not using multiprocessing if use sets ``data_config.dataset_threads`` to be 1.
* Added better argument parsing to check for arguments that were misspelled or otherwise wouldn't trigger anything.
* Code is now Flake8 compatible.

Fixed
^^^^^^^
* Fixed readthedocs so the BootlegAnnotator was loaded correctly.
* Fixed logging in BootlegAnnotator.
* Fixed ``use_exact_path`` argument in Emmental.

1.0.0 - 2021-02-15
-------------------
We did a major rewrite of our entire codebase and moved to using `Emmental <https://github.com/SenWu/Emmental>`_ for training. Emmental allows for each multi-task training, FP16, and support for both DataParallel and DistributedDataParallel.

The overall functionality of Bootleg remains unchanged. We still support the use of an annotator and bulk mention extraction and evaluation. The core Bootleg model has remained largely unchanged. Checkout our `documentation <https://bootleg.readthedocs.io/gettingstarted/install.html>`_ for more information on getting started. We have new models trained as described in our `README <https://github.com/HazyResearch/bootleg>`_.

.. note::

    This branch os **not** backwards compatible with our old models or code base.

Some more subtle changes are below

Added
^^^^^
* Support for data parallel and distributed data parallel training (through Emmental)
* FP16 (through Emmental)
* Easy install with ``BootlegAnnotator``

Changed
^^^^^^^^
* Mention extraction code and alias map has been updated
* Models trained on October 2020 save of Wikipedia
* Have uncased and cased models

Removed
^^^^^^^
* Support for slice-based learning
* Support for ``batch prepped`` KG embeddings (only use ``batch on the fly``)


.. _@lorr1: https://github.com/lorr1
