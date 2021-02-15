# Bootleg Downstream Tutorial
NED is critical for numerous downstream applications, such as search, personal assistants, and Q&A, and in these tutorials, we show how to leverage [Bootleg's](https://github.com/HazyResearch/bootleg) output entity representations for one such downstream task that benefits from entity-based knowledge: TACRED. The goal in TACRED is to extract the relationship between a subject and object pair in text. Given the task data, we must
    - (1) identify the entities in the examples,
    - (2) obtain contextual Bootleg embeddings for these entities,
    - (3) add the identified named entity as features in the TACRED data for each corresponding example, and
    - (4) incorporate the Bootleg embeddings corresponding to these id's in the downstream task model.
We show how to incorporate Bootleg embeddings in both an LSTM model (used in the original TACRED paper) and in a SpanBERT model. Using Bootleg, we set a new SotA result on TACRED revised test data at 80.3 F1 points; this improves upon SpanBERT without Bootleg by 2.3 F1 points and the prior SotA KnowBERT by 1.0 F1 points. 

First we will describe how to setup and then how to execute steps 1-4 from above.

## Setup

#### Environment
Dependencies: For the Python dependencies, we recommend using a virtualenv. Once you have cloned the repository, change directories to the root of the repository and run

```virtualenv -p python3.6 .venv```

Once the virtual environment is created, activate it by running:

```source .venv/bin/activate```
Any Python libraries installed will now be contained within this virtual environment. To deactivate the environment, simply run:

```deactivate```

We provide a requirements.txt file that includes requirements for Bootleg and the downstream tasks. Then, install any python dependencies by running:

```pip install -r requirements.txt```


#### Bootleg Setup
This tutorial assumes you have gone through the setup for installing Bootleg and downloaded the pretrained Bootleg model and configs, Wiki entity data and embedding data, and pretrained BERT model using the commands: 

```
bash download_model.sh
bash download_data.sh
bash download_bert.sh
```

In total, these downloads require 12-13 GB of memory. 

#### Memory: 
For this tutorial, we used 16 GB GPU memory for training the SpanBERT-large model with Bootleg embeddings (batch size 32, gradient accummulation 1). The LSTM model can be trained within 8 GB GPU memory (50 batch size). The size of the contextual entity embedding matrix is approximately 3 GB for the TACRED datasets. 


## Now we explain how to perform steps 1-4:
#### Preliminaries: 
We start with definitions of terms in the NED process, which are described in further detail [here](https://github.com/HazyResearch/bootleg/blob/master/tutorials/basic_training_tutorial.md). NED is the task of mapping "strings" to "things" in a knowledge base. For example, consider the text, "When was Lincoln the president?". Given the input text, a `mention extractor` identifies words, such as "Lincoln", that may potentially refer to named entities. A `candidate map` maps the string mention to the entities the string might be referring to: Abraham Lincoln, Lincoln (Nebraska), Lincoln (Car company), etc. The Bootleg `NED model` is trained to disambiguate one entity amongst the potential candidates: Abraham Lincoln.


#### Step 1: Identify entities in the TACRED data. 
Here we provide our input text (i.e. the TACRED data) as the `infile` (jsonl format); the candidate map, `cand_map` (json format), which maps textual mentions to a list of candidate entities the mention could be referring to; and the `outfile` (jsonl format), which will include the original `infile` plus candidates extracted from the input text. The lines of code executing this step are in  `bootleg_utilities/tacred_e2ebootleg_ned.py` as follows:

```
from bootleg.end2end.extract_mentions import extract_mentions
extract_mentions(in_filepath=infile, out_filepath=outfile, cand_map_file=cand_map)
```  

Note that the mention extractor code requires the `infile` to be in `jsonl` format, where an example is as follows (including the `id` is optional):

```
{
    "sentence": "The Academy of Motion Picture Arts and Sciences awarded Edwards -- who was married to actress Julie Andrews -- an honorary lifetime achievement Oscar in 2004 .",
    "id": "e779865fb91dc9cef1f3
}
```


We provide a script `convert_to_jsonl.py` that accepts the train, test, and dev data and outputs a single file in the required format (preserving the TACRED example ids). We consolidate the examples to one file and split it out at the end so that we only have to run steps (1) and (2) for one file, rather than 3 separate times!

If you have predefined mentions for your downstream task, you can skip the step of running mention extraction, so long as your data matches the format of `outfile` and all mentions are in the candidate map. An example in `outfile` is:

```
{
    "sentence": "The Academy of Motion Picture Arts and Sciences awarded Edwards -- who was married to actress Julie Andrews -- an honorary lifetime achievement Oscar in 2004 .",
    "id": "e779865fb91dc9cef1f3",
    "aliases": ["academy of motion picture arts and sciences", "edwards", "julie andrews", "lifetime", "oscar"],
    "spans": [[1, 8], [9, 10], [16, 18], [21, 22], [23, 24]],
    "qids": ["Q212329", "Q704008", "Q161819", "Q1319610", "Q19020"],
    "gold": [true, true, true, true, true],
    "sent_idx_unq": 97180
}
```

#### Step 2: Obtain contextual Bootleg embeddings for these entities. 
Using the pretrained Bootleg model, we run inference over the `outfile` contents to disambiguate between the candidates for each example. Code for this step is in: `bootleg_utilities/tacred_e2ebootleg_ned.py` as follows:

```
bootleg_label_file, bootleg_emb_file = run.model_eval(args=config_args, mode="dump_embs", logger=logger, is_writer=True)
contextual_entity_embs = np.load(bootleg_emb_file)
```

The outputs from this step are `bootleg_label_file` (jsonl format), which stores the disambiguated entities for each TACRED example and `bootleg_emb_file`(npy format), which contains a matrix of the contextual entity embedding associated with each entity. Take note of where these files are written out. The last two lines logged by this command will look as follows (saved to a results folder in `/dataset/tacred/`):

```
Saving contextual entity embeddings to ../dataset/tacred/results/20200914_104853/all_tacred_bootoutput/eval/bootleg_model/bootleg_embs.npy
Wrote predictions to ../dataset/tacred/results/20200914_104853/all_tacred_bootoutput/eval/bootleg_model/bootleg_labels.jsonl
```

#### Step 3: Add Bootleg features to the TACRED data. 
We add the ids for disambiguated entities as features in the TACRED data so that we can extract the corresponding Bootleg emmbeddings during training. Code for this step is in: `bootleg_utilities/add_bootleg_feature.py`. Run the command where the `--bootleg_directory` flag contains your directory of the `bootleg_labels.jsonl`:

```
python add_bootleg_feature.py --bootleg_directory ../dataset/tacred/results/20200914_104853/all_tacred_bootoutput/eval/bootleg_model/
```

In the file `dataset/tacred/dev_samples_w_bootleg_features.json` we have provided examples of the output data with Bootleg features (based on the original dev set samples publicly released from TACRED). 

We incorporate the Bootleg embeddings in the downstream task model and train the model to observe the result of using Bootleg. To prepare the Bootleg embeddings to be used in the model, we need to add embeddings for the padding and unknown OOV special tokens. To do this, run the following where the `--embfile` flag contains your directory of the `bootleg_embs.npy`:

```
python prepare_entity_vocab_bootleg.py --embfile ../dataset/tacred/results/20200914_104853/all_tacred_bootoutput/eval/bootleg_model/bootleg_embs.npy
```

The output of this preparation step will be a file called `ent_embeddings.npy`, which will be used by the downstream models.

#### Step 4: Downstream model.
To launch a model, run `bash run.sh`. 

You should first begin with "lstm_model", which integrates Bootleg in an LSTM downstream model; be sure to follow the preparation step for downloading GloVe embeddings described in the lstm_model/ directory README. The command to run evaluation on the TACRED test data is as follows (you should replace the model_dir and out flags as desired):

```
python lstm_model/eval.py --model_dir saved_models/00/ --out saved_models/00/ --use_ctx_ent
```

We added support to write out raw predictions during eval. They will save to the model_dir/timestamp_dataset.csv location by default (e.g. `../saved_models/00/11082020-123059_test_ent.csv`).

Next, we provide the "spanbert_model", which integrates Bootleg in a SpanBERT downstream model. Implementation details for how we integrated the Bootleg embeddings are described in each model directory's README. 


## Additional resources
Tutorials diving into steps (1) and (2) in more depth are provided [here](https://github.com/HazyResearch/bootleg/tree/master/tutorials). 

Results and additional details from using Bootleg for the TACRED task are described in our [paper](https://arxiv.org/abs/2010.10363). For our SotA result, which is built over the SpanBERT model, we found that fine-tuning the Bootleg model gave us better performance (the released Bootleg model does not fine-tune). For efficiency, we took a smaller dataset to fine-tune a model: we took Wikipedia sentences corresponding to candidate entities (whether the correct disambiguated entity or incorrect candidate) for a random sample of TACRED examples. We fine-tuned using this data and used the resulting Bootleg model to obtain our Bootleg embeddings. We report the test F1 corresponding to the best dev F1 checkpoint. 


