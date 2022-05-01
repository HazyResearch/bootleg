Quickstart
=============

Getting started is easy. Run the following. This will download our default model.

.. note::

    You will need at least 40 GB of disk space, 12 GB of GPU memory, and 35 GB of CPU memory to run our model. When running for the first time, it will take 10 plus minutes for everything to download and load correctly, depending on network speeds.

.. code-block::

    from bootleg.end2end.bootleg_annotator import BootlegAnnotator
    ann = BootlegAnnotator()
    ann.label_mentions("How many people are in Lincoln")["titles"]

You can also pass in multiple sentences::

    ann.label_mentions(["I am in Lincoln", "I am Lincoln", "I am driving a Lincoln"])["titles"]

Or, you can decide to use a different model (the choices are bootleg_cased, bootleg_uncased, bootleg_cased_mini, and bootleg_uncased_mini - default is bootleg_uncased)::

    ann = BootlegAnnotator(model_name="bootleg_uncased")
    ann.label_mentions("How many people are in Lincoln")["titles"]

Other initialization parameters are at `bootleg/end2end/bootleg_annotator.py <../apidocs/bootleg.end2end.html#module-bootleg.end2end.bootleg_annotator>`_.

Check out our `tutorials <https://github.com/HazyResearch/bootleg/tree/master/tutorials>`_ for more help getting started.

Faster Inference
--------------------
For improved speed, you can pass in a static matrix of all entity embeddings downloaded from `here <https://bootleg-ned-data.s3-us-west-1.amazonaws.com/models/latest/bootleg_uncased_entity_embs.npy.tar.gz>`_.

Then, our annotator can be run as::

    ann = BootlegAnnotator(entity_embs_path=<PATH TO UNTARRED EMBEDDING FILE>)
    ann.label_mentions("How many people are in Lincoln")["titles"]


.. tip::

    If you have a larger amount of data to disambiguate, checkout out our `end-to-end tutorial <https://github.com/HazyResearch/bootleg/tree/master/tutorials/end2end_ned_tutorial.ipynb>`_ showing a more optimized end-to-end pipeline.
