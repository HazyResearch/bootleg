# Tutorials
We provide several tutorials to help users get familiar with [Bootleg](/).

## Introduction
### Benchmark
In this [tutorial](tutorials/benchmark_tutorial.ipynb), learn how to use a Bootleg model pretrained on Wikipedia to run inference on the standard RSS500 NED benchmark, where Bootleg must disambiguate pre-detected mentions in sentences. This will help you get familiar with loading one of our models.

### End to End
In this [tutorial](tutorials/end2end_ned_tutorial.ipynb), learn how to use Bootleg for end-to-end inference. We start from text data and show how to detect mentions and then link them to entities. We also show how to use Bootleg for "on-the-fly" disambiguation of individual sentences.

## Using Bootleg Representations
### Embeddings Extraction
In this [tutorial](tutorials/entity_embedding_tutorial.ipynb), we will introduce you to how to take a pretrained Bootleg model and generate entity representations. The next tutorial shows you how to use them in a downstream model.

### Bootleg-Enhanced TACRED
In this [tutorial](tutorials/downstream_tutorial/), we show you how to integrate Bootleg embeddings into a downstream LSTM model and SPAN-BERT model.

## Training
### Basic Training
In this [tutorial](tutorials/basic_training_tutorial.md), learn how to train a Bootleg model on a small dataset. This will cover input data formatting, data preprocessing, and training.

### Advanced Training
In this [tutorial](tutorials/advanced_training_tutorial.md), learn how to use distributed training to train a Bootleg model on the full English Wikipedia dump (over 50 million sentences!). You will need access to GPUs to train this model.

