# Advanced Training Tutorial

We discuss how to use distributed training to train a Bootleg model on the full Wikipedia dump. This tutorial assumes you have already completed the [Basic Training Tutorial](basic_training_tutorial.md).

As Wikipedia has over 5 million entities and over 50 million sentences, training on the full Wikipedia dump is computationally expensive. We recommend using a [p3dn.24xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance on AWS to train on Wikipedia.


We provide a config for training Wikipedia [here](../configs/wiki_config.json). Note this config is a simplified version of the config used to train the pretrained model provided in the [Benchmark Tutorial](benchmark_tutorial.ipynb) and the [End-to-End Tutorial](end2end_ned_tutorial.ipynb).


## 1. Downloading the Data

We provide scripts to download:
1. Prepped Wikipedia data (training and dev datasets)
2. Pretrained BERT model (used for word embedding backbone)
3. Wikipedia entity data and embedding data

To download the Wikipedia data, run the command below with the directory to download the data to. Note that the prepped Wikipedia data will require ~200GB of disk space and will take some time to download and decompress the prepped Wikipedia data (16GB compressed, ~150GB uncompressed).
```
bash download_wiki.sh <DOWNLOAD_DIRECTORY>
```

To download (2) and (3) above to the `pretrained_bert_models` and `data` directories, respectively, run the commands below:
```
bash download_bert.sh
bash download_data.sh
```

## 2. Setting up Distributed Training

Bootleg supports distributed training using PyTorch's [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html) framework. To use distributed training, there need to be several GPUs available to distribute batches across. We then need to make the following updates to the  `run_config` section of the config file:
```
"distributed": true,
"ngpus_per_node": <# GPUs available>
"eval_batch_size": <total evaluation batch size across GPUs>
```
We also want to update the `batch_size` in the `train_config` section to be the total batch size across GPUs. For instance, with 8 GPUs, if we set the batch size to be 512, each GPU will receive a batch size of 512/8 = 64. We recommend using the largest batch size that will fit in the GPU memory.

From the [Basic Training Tutorial](basic_training_tutorial.md), recall that the directory paths should be set to where we want to save our models and read the data, including:
  - `save_dir` in `run_config`
  - `cache_dir` in `data_config.word_embedding`
  - `data_dir`, `entity_dir`, and `emb_dir` in `data_config`.

We have already set these directories in the [provided Wikipedia config](../configs/wiki_config.json), but you will need to update `data_dir`, `entity_dir`, and `emb_dir` to where you downloaded the data in [Downloading the Data](#1-downloading-the-data) and may want to update `save_dir` to where you want to save the model checkpoints and logs.

## 3. Training the Model

As we provide the Wikipedia data already prepped, we can jump immediately to training. To train the model, we simply run:

    python bootleg/run.py --config_script configs/wiki_config.json

Once the training begins, we should see all GPUs being utilized and should see a log created for each GPU in the save directory.

If we want to change the config (e.g. change the maximum number of aliases or the maximum word token len), we would need to re-prep the data and would run the command below. Note it takes several hours to perform Wikipedia pre-processing on a 56-core machine:

    python bootleg/prep.py --config_script configs/wiki_config.json

## 4. Evaluating with Slices

We use evaluation slices to understand the performance of Bootleg on important subsets of the dataset. To use evaluation slices, alias-entity pairs are labelled as belonging to specific slices in the `slices` key of the dataset.

In the Wikipedia data in this tutorial, we provide three "slices" of the dev dataset in addition to the "final_loss" (all examples) slice. For each of these three slices, the alias being scored must have more than one candidate. This filters trivial examples all models get correct.
- `unif_NS_TS`: The gold entity does not occur in the training dataset.
- `unif_NS_TL`: The gold entity occurs globally 10 or fewer times in the training dataset.
- `unif_NS_TO`: The gold entity occurs globally between 11-1000 times in the training dataset.

To use the slices for evaluation, they must also be specified in the `eval_slices` section of the `run_config` (see the [Wikipedia config](../configs/wiki_config.json) as an example).

When the dev evaluation occurs during training, we should see the performance on each of the slices that are specified in `eval_slices`. These slices help us understand how well Bootleg performs on more challenging subsets. The frequency of dev evaluation can be specified by the `eval_steps` parameter in the `run_config`.
