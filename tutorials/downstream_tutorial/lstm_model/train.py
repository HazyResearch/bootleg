"""
Train a model on TACRED.
"""

import argparse
import os
import random
import time
from datetime import datetime
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.rnn import RelationModel
from utils import constant, helper, scorer
from utils.vocab import Vocab

from data.loader import DataLoader

parser = argparse.ArgumentParser()

# word embeddings
parser.add_argument(
    "--data_dir",
    type=str,
    default="/dfs/scratch0/lorr1/projects/bootleg-data/downstream/tacred/",
    help="directory containing the tacred datasets",
)
parser.add_argument(
    "--vocab_dir",
    type=str,
    default="/dfs/scratch0/lorr1/projects/bootleg-data/downstream/tacred/vocab",
    help="directory containing the word emb vocab",
)
parser.add_argument(
    "--emb_dim", type=int, default=300, help="Word embedding dimension."
)
parser.add_argument(
    "--train_file_name",
    type=str,
    default="train_ent.json",
    help="train data with bootleg feature added",
)
parser.add_argument(
    "--dev_file_name",
    type=str,
    default="dev_ent.json",
    help="dev data with bootleg feature added",
)

# bootleg embeddings
parser.add_argument(
    "--ent_vocab_dir",
    type=str,
    default="/dfs/scratch0/lorr1/projects/bootleg-data/downstream/tacred/",
    help="directory containing the bootleg_emmental emb vocab (ent_embedding.npy and ent_vocab.pkl)",
)
parser.add_argument(
    "--ent_emb_dim", type=int, default=256, help="Entity embedding dimension."
)
parser.add_argument(
    "--use_ctx_ent",
    action="store_true",
    help="whether to use contextual entity embeddings",
)
parser.add_argument(
    "--use_first_ent_span_tok",
    action="store_true",
    help="whether to just use first token in entity span",
)

parser.add_argument("--ner_dim", type=int, default=30, help="NER embedding dimension.")
parser.add_argument("--pos_dim", type=int, default=30, help="POS embedding dimension.")
parser.add_argument(
    "--hidden_dim", type=int, default=200, help="RNN hidden state size."
)
parser.add_argument("--num_layers", type=int, default=2, help="Num of RNN layers.")
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Input and RNN dropout rate."
)
parser.add_argument(
    "--word_dropout",
    type=float,
    default=0.04,
    help="The rate at which randomly set a word to UNK.",
)
parser.add_argument(
    "--topn", type=int, default=1e10, help="Only finetune top N embeddings."
)
parser.add_argument(
    "--lower", dest="lower", action="store_true", help="Lowercase all words."
)
parser.add_argument("--no-lower", dest="lower", action="store_false")
parser.set_defaults(lower=False)

parser.add_argument(
    "--attn", dest="attn", action="store_true", help="Use attention layer."
)
parser.add_argument("--no-attn", dest="attn", action="store_false")
parser.set_defaults(attn=True)
parser.add_argument("--attn_dim", type=int, default=200, help="Attention size.")
parser.add_argument(
    "--pe_dim", type=int, default=30, help="Position encoding dimension."
)

parser.add_argument("--lr", type=float, default=1.0, help="Applies to SGD and Adagrad.")
parser.add_argument("--lr_decay", type=float, default=0.9)
parser.add_argument(
    "--optim", type=str, default="sgd", help="sgd, adagrad, adam or adamax."
)
parser.add_argument("--num_epoch", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument(
    "--max_grad_norm", type=float, default=5.0, help="Gradient clipping."
)
parser.add_argument("--log_step", type=int, default=20, help="Print log every k steps.")
parser.add_argument(
    "--log", type=str, default="logs.txt", help="Write training log to file."
)
parser.add_argument(
    "--save_epoch", type=int, default=5, help="Save model checkpoints every k epochs."
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="logs/tacred_lstm_ent",
    help="Root dir for saving models.",
)
parser.add_argument(
    "--id", type=str, default="00", help="Model ID under which to save models."
)
parser.add_argument(
    "--info", type=str, default="", help="Optional info for the experiment."
)

parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
parser.add_argument("--cpu", action="store_true", help="Ignore CUDA.")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)
opt["num_class"] = len(constant.LABEL_TO_ID)

# load vocab
vocab_file = opt["vocab_dir"] + "/vocab.pkl"
vocab = Vocab(vocab_file, load=True)
opt["vocab_size"] = vocab.size
emb_file = opt["vocab_dir"] + "/embedding.npy"
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt["emb_dim"]

# load contextual entity vocab
if args.use_ctx_ent:
    ent_vocab_file = opt["ent_vocab_dir"] + "/ent_vocab.pkl"
    ent_vocab = Vocab(ent_vocab_file, load=True)
    opt["ent_vocab_size"] = ent_vocab.size
    ent_emb_file = opt["ent_vocab_dir"] + "/ent_embedding.npy"
    ent_emb_matrix = np.load(ent_emb_file)
    assert (
        ent_emb_matrix.shape[0] == ent_vocab.size
    ), f"{ent_emb_matrix.shape[0]} vs {ent_vocab.size}"
    assert (
        ent_emb_matrix.shape[1] == opt["ent_emb_dim"]
    ), f"{ent_emb_matrix.shape[0]} vs {opt['ent_emb_dim']}"
else:
    ent_vocab_file = ""
    ent_vocab = None
    ent_emb_file = ""
    ent_emb_matrix = None
    opt["ent_emb_dim"] = 0
    opt["ent_vocab_size"] = 0

# load data
print(
    "Loading data from {} with batch size {}...".format(
        opt["data_dir"], opt["batch_size"]
    )
)
train_batch = DataLoader(
    opt["data_dir"] + opt["train_file_name"],
    opt["batch_size"],
    opt,
    vocab,
    ent_vocab,
    first_ent_span_token=args.use_first_ent_span_tok,
    evaluation=False,
)
dev_batch = DataLoader(
    opt["data_dir"] + opt["dev_file_name"],
    opt["batch_size"],
    opt,
    vocab,
    ent_vocab,
    first_ent_span_token=args.use_first_ent_span_tok,
    evaluation=True,
)

model_id = opt["id"] if len(opt["id"]) > 1 else "0" + opt["id"]
model_save_dir = opt["save_dir"] + "/" + model_id
opt["model_save_dir"] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + "/config.json", verbose=True)
vocab.save(model_save_dir + "/vocab.pkl")
if args.use_ctx_ent:
    ent_vocab.save(model_save_dir + "/ent_vocab.pkl")
file_logger = helper.FileLogger(
    model_save_dir + "/" + opt["log"], header="# epoch\ttrain_loss\tdev_loss\tdev_f1"
)

# print model info
helper.print_config(opt)

# model
model = RelationModel(opt, emb_matrix=emb_matrix, ent_emb_matrix=ent_emb_matrix)

id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt["lr"]

global_step = 0
global_start_time = time.time()
format_str = (
    "{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}"
)
max_steps = len(train_batch) * opt["num_epoch"]

# start training
for epoch in range(1, opt["num_epoch"] + 1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = model.update(batch)
        train_loss += loss
        if global_step % opt["log_step"] == 0:
            duration = time.time() - start_time
            print(
                format_str.format(
                    datetime.now(),
                    global_step,
                    max_steps,
                    epoch,
                    opt["num_epoch"],
                    loss,
                    duration,
                    current_lr,
                )
            )

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, _, loss = model.predict(batch)
        predictions += preds
        dev_loss += loss
    predictions = [id2label[p] for p in predictions]
    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)

    train_loss = (
        train_loss / train_batch.num_examples * opt["batch_size"]
    )  # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt["batch_size"]
    print(
        "epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(
            epoch, train_loss, dev_loss, dev_f1
        )
    )
    file_logger.log(
        "{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1)
    )

    # save
    model_file = model_save_dir + "/checkpoint_epoch_{}.pt".format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + "/best_model.pt")
        print("new best model saved.")
    if epoch % opt["save_epoch"] != 0:
        os.remove(model_file)

    # lr schedule
    if (
        len(dev_f1_history) > 10
        and dev_f1 <= dev_f1_history[-1]
        and opt["optim"] in ["sgd", "adagrad"]
    ):
        current_lr *= opt["lr_decay"]
        model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))
