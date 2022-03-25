"""
Compute QID counts.

Helper function that computes a dictionary of QID -> count in training data.

If a QID is not in this dictionary, it has a count of zero.
"""

import argparse
import multiprocessing
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import ujson
from tqdm.auto import tqdm


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="train.jsonl",
    )

    args = parser.parse_args()
    return args


def get_char_spans(spans, text):
    """
    Get character spans instead of default word spans.

    Args:
        spans: word spans
        text: text

    Returns: character spans
    """
    word_i = 0
    prev_is_space = True
    char2word = {}
    word2char = defaultdict(list)
    for char_i, c in enumerate(text):
        if c.isspace():
            if not prev_is_space:
                word_i += 1
                prev_is_space = True
        else:
            prev_is_space = False
            char2word[char_i] = word_i
            word2char[word_i].append(char_i)
    char_spans = []
    for span in spans:
        char_l = min(word2char[span[0]])
        char_r = max(word2char[span[1] - 1]) + 1
        char_spans.append([char_l, char_r])
    return char_spans


def convert_char_spans(num_processes, file):
    """Add char spans to jsonl file."""
    pool = multiprocessing.Pool(processes=num_processes)
    num_lines = sum([1 for _ in open(file)])
    temp_file = Path(tempfile.gettempdir()) / "_convert_char_spans.jsonl"
    with open(file) as in_f, open(temp_file, "wb") as out_f:
        for res in tqdm(
            pool.imap_unordered(convert_char_spans_helper, in_f, chunksize=100),
            total=num_lines,
            desc="Adding char spans",
        ):
            out_f.write(bytes(res, encoding="utf-8"))
        out_f.seek(0)
    pool.close()
    pool.join()
    shutil.copy(temp_file, file)
    os.remove(temp_file)
    return


def convert_char_spans_helper(line):
    """Get char spans helper.

    Parses line, adds char spans, and dumps it back again
    """
    line = ujson.loads(line)
    line["char_spans"] = get_char_spans(line["spans"], line["sentence"])
    to_write = ujson.dumps(line) + "\n"
    return to_write


def main():
    """Run."""
    args = parse_args()
    print(ujson.dumps(vars(args), indent=4))
    num_processes = int(0.8 * multiprocessing.cpu_count())
    print(f"Getting slice counts from {args.file}")
    convert_char_spans(num_processes, args.file)


if __name__ == "__main__":
    main()
