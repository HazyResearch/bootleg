<img src="docs/images/full_logo.png" width="150"/>


[![Build Status](https://travis-ci.com/HazyResearch/bootleg.svg?branch=master)](https://travis-ci.com/HazyResearch/bootleg)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

***Self-Supervision for Named Entity Disambiguation at the Tail***

Bootleg is a self-supervised named entity disambiguation (NED) system built to improve disambiguation of tail entities using a simple transformer-based architecture (see the [overview](#bootleg-overview) below)

Note that Bootleg is *actively under development* and feedback is welcome. Submit bugs on the Issues page or feel free to submit your contributions as a pull request.

# Getting Started

## Installation
Bootleg requires Python 3.6 or later. We recommend using `pip` or `conda` to install.

If using `pip`:

```
pip install -r requirements.txt
python setup.py develop
```

If using `conda`:

```
conda env create --name <env_name> --file conda_requirements.yml
python setup.py develop
```

Note that the requirements assume CUDA 10.2. To use CUDA 10.1, you will need to run:
```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

or

```
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```

## Tutorials

We provide five tutorials to help users get familiar with Bootleg:
### Training
- [basic training](tutorials/basic_training_tutorial.md): learn how to train a Bootleg model on new data, including formatting data, preprocessing data, and training with a sample dataset.
- [advanced training](tutorials/advanced_training_tutorial.md): learn how to use distributed training to train a Bootleg model on the full English Wikipedia dump (over 50 million sentences!).
### Inference
- [benchmark](tutorials/benchmark_tutorial.ipynb): learn how to use a Bootleg model pretrained on Wikipedia to run inference on the standard RSS500 NED benchmark, where Bootleg must disambiguate pre-detected mentions in sentences.
- [end-to-end](tutorials/end2end_ned_tutorial.ipynb): learn how to use Bootleg for end-to-end inference, starting from text data to detecting and labeling mentions. We also show how to use Bootleg for "on-the-fly" inference of individual sentences.
### Embeddings for Downstream Tasks
- [entity embeddings](tutorials/entity_embedding_tutorial.ipynb): learn how to use pretrained Bootleg models to generate contextual and static entity embeddings for use in downstream tasks.

# Disambiguating at the Tail
The challenge in disambiguation is how to handle tail entities---entities that occur infrequently, or not at all, in training data. Models that rely on textual co-occurrence patterns struggle to disambiguate the tail as tail entities are not seen enough times during training to learn associated textual cues. The key to disambiguating the tail is to leverage structural sources of information (e.g., type or knowledge graph (KG)) that is readily available for the tail (see left panel below for example). In fact, 75% of all entities in Wikidata that are *not* in Wikipedia have type or KG information. Bootleg incorporates both textual and structural information to perform disambiguation, as described next.

# Bootleg Overview
Given an input sentence, Bootleg takes the sentence and outputs a predicted entity for each detected mention. Bootleg first extracts mentions in the
sentence by querying our the mentions in a pre-mined candidate mapping (see [extract_mentions.py](bootleg/extract_mentions.py)). For each mention, we extract its set of possible candidate entities (done in [prep.py](bootleg/prep.py))
and any structural information about that entity, e.g., type information or knowledge graph (KG) information. The structural information is stored as embeddings in their
associated embedding classes. Bootleg leverages these embeddings as *entity payloads* along with the sentence information as *word embeddings* to predict which entity (possibly the NIL entity)
is associated with each mention.

![Dataflow](docs/images/bootleg_dataflow.png "Bootleg Dataflow")

## Entity Payload
We use three embeddings for the entity payloads. Each entity gets the following embeddings:
* Entity: learned embedding
* Type: learned embedding for each of its types
* Relation: learned embedding for each relation it participates in on Wikidata

We also allow the use of other entity-based features. In this model, we use a title embedding and a Wikipedia page co-occurrence statistical feature.

These embeddings are concatenated and projected to form an entity payload.

## Architecture
* Input: contextualized word embeddings (e.g. BERT) and entity payloads
* Network: uses [transformer](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) modules to learn patterns over phrases and entities
    * Phrase Module: attention over the sentence and entity payloads
    * Co-Occurrence Module: self-attention over the entity payloads
    * KG Module: takes the sum of the output of the phrase and co-occurrence modules and leverages KG connectivity among candidates as weights in an attention
* Score: uses MLP softmax layers to score each mention and candidate independently, selecting the most likely candidate per mention


<p align="center">
    <img src="docs/images/bootleg_architecture.png " width="500" class="center"/>
</p>

In the figure above, *M* represents the maximum number of mentions (or aliases) in the sentence, *K* represents the maximum number of candidates considered per mention, and *N* represents the maximum number of sub-words in the sentence. Typically, we use *M*=10, *K*=30, and *N*=100. Additionally, *H* is the hidden dimension used throughout the backbone, *E* is the dimension of the learned entity embedding, *R* the dimension of the learned relation embedding, and *T* the dimension of the learned type embedding. We further select an entity's 3 most popular types and 50 most unique Wikidata relations. These are all tunable parameters in Bootleg.


## Inference
Given a pretrained model, we support three types of inference: `--mode eval`, `--mode dump_preds`, and `--mode dump_embs`. `Eval` mode is the fastest option and will run the test files through the model and output aggregated quality metrics to the log. `Dump_preds` mode will write the individual predictions and corresponding probabilities to a jsonlines file. This is useful for error analysis. `Dump_embs` mode is the same as `dump_preds`, but will additionally  output contextual entity embeddings. These can then be read and processed in a downstream system.
 <!-- This dump will also include information for querying the other structural embeddings (e.g., -->
 <!-- type or KG embeddings) if desired. -->

## Training
We recommend using GPUs for training Bootleg models. For large datasets, we support distributed training with Pytorch's Distributed DataParallel framework to distribute batches across multiple GPUs. Check out the [Basic Training](tutorials/basic_training_tutorial.md) and [Advanced Training](tutorials/advanced_training_tutorial.md) tutorials for more information and sample data!
