import argparse
import os
import pickle
import sys

import numpy as np


def main(dir, outdir):
    embs_bootleg = {
        "ent": {
            "inname": "bootleg_embs.npy",
            "outname": "ent_embedding.npy",
            "outvocab": "ent_vocab.pkl",
        }
    }

    for k, names_lst in embs_bootleg.items():
        datafile = dir + "/" + names_lst["inname"]

        if os.path.isfile(datafile):
            emb_data = np.load(datafile)

            new_embs = np.zeros(
                (emb_data.shape[0] + 2, emb_data.shape[1]), dtype="float"
            )
            new_embs[2:] = emb_data
            np.save(
                "{DIR}/{out}".format(DIR=outdir, out=names_lst["outname"]), new_embs
            )

            v = [i - 2 for i in range(new_embs.shape[0])]
            vocab_file = "{DIR}/{outvoc}".format(
                DIR=outdir, outvoc=names_lst["outvocab"]
            )

            with open(vocab_file, "wb") as outfile:
                pickle.dump(v, outfile)
            print("Saved {} at {}".format(k, vocab_file))
        else:
            print("Check path, could not find {}".format(datafile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embfile",
        type=str,
        help="directory where bootleg_emmental embedding npy matrix from running inference is located",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../dataset/tacred/",
        help="directory where we want to save the prepared entity vocab files",
    )
    args = parser.parse_args()
    dir = args.embfile
    outdir = args.outdir
    main(dir, outdir)
