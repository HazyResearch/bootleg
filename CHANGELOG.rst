Unreleased_ 1.0.1dev1
-------------------
Added
^^^^^^^
* ``data_config.print_examples_prep`` flag to toggle data example printing during data prep.
* ``data_config.eval_accumulation_steps`` to support subbatching dumping of predictings. We save outputs to separate files of size approximately ``data_config.eval_accumulation_steps*data_config.eval_batch_size`` and merge into a final file at the end.
* Entity Profile API. See the `docs <https://bootleg.readthedocs.io/en/latest/gettingstarted/entity_profile.html>`_. This allows for modifying entity metadata as well as adding and removing entities. We profile methods for refitting a model with a new profile for immediate inference, no finetuning needed.

Changed
^^^^^^^^
* Support for not using multiprocessing if use sets ``data_config.dataset_threads`` to be 1.
* Added better argument parsing to check for arguments that were misspelled or otherwise wouldn't trigger anything.

Fixed
^^^^^^^
* Fixed readthedocs so the BootlegAnnotator was loaded correctly.
* Fixed logging in BootlegAnnotator.
* Fixed ``use_exact_path`` argument in Emmental.

1.0.0 - 2020-02-15
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