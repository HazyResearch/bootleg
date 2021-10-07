"""
Run evaluation with saved models.
"""

import argparse
import csv
import datetime
import json
import pickle
import random

import torch
from data.loader import DataLoader
from model.rnn import RelationModel
from utils import constant, helper, scorer, torch_utils
from utils.vocab import Vocab

now = datetime.datetime.now()
timestamp = now.strftime("%m%d%Y-%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    type=str,
    default="logs/tacred_lstm_ent/00",
    help="Directory of the model.",
)
parser.add_argument(
    "--out",
    default="logs/tacred_lstm_evals",
    type=str,
    help="Save model predictions to this dir.",
)
parser.add_argument(
    "--bootleg_qid_name",
    type=str,
    default="/dfs/scratch0/lorr1/projects/bootleg-data/data/wiki_title_0122/entity_db/entity_mappings/qid2title.json",
    help="Path to the qid2title.json file in bootleg_emmental downloads",
)
parser.add_argument(
    "--outfile",
    type=str,
    default="results.json",
    help="Save raw predictions to this file.",
)
parser.add_argument(
    "--model", type=str, default="best_model.pt", help="Name of the model file."
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="/dfs/scratch0/lorr1/projects/bootleg-data/downstream/tacred/",
)
parser.add_argument(
    "--dataset", type=str, default="test_ent", help="Evaluate on dev or test."
)
parser.add_argument(
    "--use_ctx_ent",
    action="store_true",
    help="whether to use contextual entity embeddings",
)
parser.add_argument(
    "--use_first_ent_span_tok",
    action="store_true",
    help="whether to just use the first token in the entity span",
)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + "/" + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = RelationModel(opt)
model.load(model_file)

# load vocab
vocab_file = args.model_dir + "/vocab.pkl"
vocab = Vocab(vocab_file, load=True)
assert opt["vocab_size"] == vocab.size, "Vocab size must match that in the saved model."

# load entity vocab
if args.use_ctx_ent:
    ent_vocab_file = args.model_dir + "/ent_vocab.pkl"
    ent_vocab = Vocab(ent_vocab_file, load=True)
    assert opt["ent_vocab_size"] == ent_vocab.size
else:
    ent_vocab_file = ""
    ent_vocab = None
    ent_emb_file = ""
    ent_emb_matrix = None

# load data
data_file = args.data_dir + "/{}.json".format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt["batch_size"]))
batch = DataLoader(
    data_file,
    opt["batch_size"],
    opt,
    vocab,
    ent_vocab,
    first_ent_span_token=args.use_first_ent_span_tok,
    evaluation=True,
)

helper.print_config(opt)
id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
for i, b in enumerate(batch):
    preds, probs, _ = model.predict(b)
    predictions += preds
    all_probs += probs
predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

############################################################################################################

# for saving the raw predictions
bootleg_qid_name = args.bootleg_qid_name
with open(bootleg_qid_name) as qid_file:
    qid_dict = json.load(qid_file)
qid_dict["UNK"] = "UNK"

# Collect the data that I want for error analysis
errors = {}
pred_file = args.out + args.outfile
if len(pred_file) > 0:
    with open(data_file) as infile:
        data = json.load(infile)
    for i in range(len(data)):
        data[i]["pred"] = predictions[i]
        data[i]["prob"] = all_probs[i]
        if predictions[i] != data[i]["relation"] or 1:  # just save it all
            errors[i] = {}

            tokens = data[i]["token"]
            tokens = [t.lower() for t in tokens]
            errors[i]["id"] = data[i]["id"]
            errors[i]["example"] = " ".join(tokens)
            errors[i]["relation"] = data[i]["relation"]
            errors[i]["prediction"] = data[i]["pred"]
            errors[i]["stanford_ner"] = data[i]["stanford_ner"]

            subj_s = int(data[i]["subj_start"])
            subj_e = int(data[i]["subj_end"]) + 1
            obj_s = int(data[i]["obj_start"])
            obj_e = int(data[i]["obj_end"]) + 1

            errors[i]["subj_type"] = data[i]["subj_type"]
            errors[i]["subj"] = tokens[subj_s:subj_e]
            errors[i]["subj_pos"] = data[i]["stanford_pos"][subj_s:subj_e]
            errors[i]["subj_ner"] = data[i]["stanford_ner"][subj_s:subj_e]

            errors[i]["obj_type"] = data[i]["obj_type"]
            errors[i]["obj"] = tokens[obj_s:obj_e]
            errors[i]["obj_pos"] = data[i]["stanford_pos"][obj_s:obj_e]
            errors[i]["obj_ner"] = data[i]["stanford_ner"][obj_s:obj_e]

            errors[i]["subj_leng"] = subj_e - subj_s
            errors[i]["obj_leng"] = obj_e - obj_e
            errors[i]["separation_dist"] = obj_s - subj_e
            errors[i]["num_ner"] = len(
                [ner for ner in data[i]["stanford_ner"] if ner != "O"]
            )
            errors[i]["prop_ner"] = errors[i]["num_ner"] / len(
                errors[i]["stanford_ner"]
            )

            qids = data[i].get("ent_id", [])
            ex_qids = []
            for qid in qids:
                if qid in qid_dict:
                    ex_qids.append(qid_dict[qid])
                else:
                    ex_qids.append("NF")
            errors[i]["mentions"] = ex_qids
            errors[i]["qids"] = qids
            errors[i]["subj_mentions"] = ex_qids[subj_s:subj_e]
            errors[i]["subj_qids"] = qids[subj_s:subj_e]
            errors[i]["obj_mentions"] = ex_qids[obj_s:obj_e]
            errors[i]["obj_qids"] = qids[obj_s:obj_e]
            errors[i]["real_mentions"] = len([qid for qid in ex_qids if qid != "UNK"])
            errors[i]["prop_mentions"] = (
                errors[i]["real_mentions"] / len(errors[i]["mentions"])
                if len(errors[i]["mentions"]) > 0
                else 1
            )


# Convert the bootleg_emmental entity QIDs to the wikidata mentions
def save_csv(obj, name):
    cols = obj[0].keys()
    print(cols)
    csv_columns = cols
    f = open(name + ".csv", "w")
    w = csv.DictWriter(f, fieldnames=csv_columns)
    w.writeheader()
    for k, v in obj.items():
        w.writerow(v)
    print("Wrote to file!")


# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(args.out)
    save_csv(errors, "{}/{}_{}".format(args.out, timestamp, args.dataset))
    filename_probs = "{}/{}_{}".format(args.out, timestamp, "probs.pkl")
    with open(filename_probs, "wb") as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")
