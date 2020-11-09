Position-aware Attention RNN Model for Relation Extraction with Bootleg Embeddings
=========================

This repo contains the *PyTorch* code for paper [Position-aware Attention and Supervised Data Improve Slot Filling](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf), and incorporates Bootleg embeddings as a feature. 

**The TACRED dataset**: Details on the TAC Relation Extraction Dataset can be found on [this dataset website](https://nlp.stanford.edu/projects/tacred/).


## Preparation

In addition to the Bootleg preparation steps described in the introductory README file, we have these prep steps:

First, download and unzip GloVe vectors from the Stanford website, with:
```
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py ../dataset/tacred ../dataset/vocab --glove_dir ../dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

These downloads are approximately 130 MB. 


## Training

#### Execution
We provided a run.sh script, though the commands for launching training are provided again here. 


Train a position-aware attention RNN model with:
```
python train.py --data_dir dataset/tacred --vocab_dir dataset/vocab --id 00 --info "Position-aware attention model"
```

Use `--topn N` to finetune the top N word vectors only. The script will do the preprocessing automatically (word dropout, entity masking, etc.).

Train an LSTM model with:
```
python train.py --data_dir dataset/tacred --vocab_dir dataset/vocab --no-attn --id 01 --info "LSTM model" 
```

To train with Bootleg embeddings, add the flag `--use_ctx_ent` as shown below: 
```
python train.py --data_dir dataset/tacred --vocab_dir dataset/vocab --no-attn --id 01 --info "LSTM model" --use_ctx_ent 
```

Additionally, prior works have used a feature for the first token in a span -- for example if an entity is Barack Obama, and I only attach the entity representation at the position Barack, rather than at both Barack and Obama -- the above Bootleg preparation includes a feature to only use the first token and the flag to signal this during training is ```--use_first_ent_span_tok```

Model checkpoints and logs will be saved to `./saved_models/00`.

#### Details
To incorporate Bootleg embeddings in this model, we add a feature to the dataloader (see `data/loader.py`) for the Bootleg entity ids, and we concatenate the Bootleg contextual embedding corresponding to this id to the RNN input (see ```model/rnn.py```).


## Evaluation

Run evaluation on the test set with:
```
python eval.py saved_models/00 --dataset test
```

Again the flags to use Bootleg are: 
```
python eval.py --model_dir ../saved_models/00 --out ../saved_models/00 --use_ctx_ent
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file. Add `--out saved_models/out/test1.pkl` to write model probability output to files (for ensemble, etc.).

We added support to write out raw predictions during eval. They will save to the model_dir/timestamp_dataset.csv location by default (e.g. `../saved_models/00/11082020-123059_test_ent.csv`).

## Ensemble

Please see the example script `ensemble.sh`.

## License

All work contained in this package is licensed under the Apache License, Version 2.0. See the included LICENSE file.
