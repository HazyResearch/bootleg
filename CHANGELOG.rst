Unreleased_
-------------------
 
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
* Models trained on October 2020 dump of Wikipedia
* Have uncased and cased models

Removed
^^^^^^^
* Support for slice-based learning
* Support for ``batch prepped`` KG embeddings (only use ``batch on the fly``)


.. _@lorr1: https://github.com/lorr1