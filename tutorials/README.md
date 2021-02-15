# Tutorials
We provide several tutorials to help users get familiar with Bootleg.

## Introduction
### End to End
In this [tutorial](end2end_ned_tutorial.ipynb), learn how to use Bootleg for end-to-end inference. We start from text data and show how to detect mentions and then link them to entities. We also show how to use Bootleg for "on-the-fly" disambiguation of individual sentences.

## Using Bootleg Representations
### Embeddings Extraction
In this [tutorial](entity_embedding_tutorial.ipynb), we will introduce you to how to take a pretrained Bootleg model and generate entity representations. The next tutorial shows you how to use them in a downstream model.

### Bootleg-Enhanced TACRED
In this [tutorial](downstream_tutorial/), we show you how to integrate Bootleg embeddings into a downstream LSTM model and SPAN-BERT model.

## Training
### Basic Training
In this [tutorial](https://bootleg.readthedocs.io/en/latest/gettingstarted/training.html), learn how to train a Bootleg model on a small dataset. This will cover input data formatting, data preprocessing, and training.

### Advanced Training
In this [tutorial](https://bootleg.readthedocs.io/en/latest/advanced/distributed_training.html), learn how to use distributed training to train a Bootleg model on the full English Wikipedia dump (over 50 million sentences!). You will need access to GPUs to train this model.

