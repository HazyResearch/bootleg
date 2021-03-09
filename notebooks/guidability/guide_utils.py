import os
import sys
import time
from pathlib import Path

import feather
import pandas as pd
import ujson
from tqdm import tqdm

sys.path.append("/dfs/scratch0/lorr1/projects/bootleg/tutorials")
import string

from metadata import WLMetadata
from nltk.stem import PorterStemmer

from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.symbols.kg_symbols import KGSymbols
from bootleg.symbols.type_symbols import TypeSymbols

ps = PorterStemmer()
PUNC = [c for c in string.punctuation]

KG_KEY = "kg_rels"
TYPE_WD = "type_wd"
TYPE_HY = "type_hy"
TYPE_REL = "type_rel"


def regiater_funcs():
    """
    LF function decorator
    """
    all_funcs = {}

    def registrar(func):
        all_funcs[func.__name__] = func
        # normally a decorator returns a wrapped function, but here we return func unmodified, after registering it
        return func

    registrar.all = all_funcs
    return registrar


def load_train_data(
    train_file, title_map, cands_map=None, type_symbols=None, kg_symbols=None
):
    """Loads a jsonl file and creates a pandas DataFrame. Adds candidates, types, and KGs if available."""
    if cands_map is None:
        cands_map = {}
    if type_symbols is None:
        type_symbols = []
    if kg_symbols is None:
        kg_symbols = []
    num_lines = sum(1 for _ in open(train_file))
    rows = []
    with open(train_file) as f:
        for line in tqdm(f, total=num_lines):
            line = ujson.loads(line)
            # for each alias, append a row in the merged result table
            res = {
                "sentence": line["sentence"],
                "sentence_split": line["sentence"].split(),
                "sent_idx": line["sent_idx_unq"],
                "qids": line["qids"],
                "aliases": line["aliases"],
                "spans": line["spans"],
                "sources": line.get("sources", ["gold"] * len(line["aliases"])),
                # "slices": line.get("slices", {}),
                "gold": line["gold"],
                "titles": [title_map[q] if q != "Q-1" else "Q-1" for q in line["qids"]],
            }
            if len(cands_map) > 0:
                res["cands_qids"] = [
                    [q[0] for q in cands_map[al]] for al in line["aliases"]
                ]
                res["cands_names"] = [
                    [title_map[q[0]] for q in cands_map[al]] for al in line["aliases"]
                ]
                res["num_cands"] = [len(cands) for cands in res["cands_qids"]]
            for type_sym in type_symbols:
                type_nm = os.path.basename(os.path.splitext(type_sym.type_file)[0])
                res[f"{type_nm}_gld"] = [type_sym.get_types(q) for q in line["qids"]]
            for kg_sym in kg_symbols:
                kg_nm = os.path.basename(os.path.splitext(kg_sym.kg_adj_file)[0])
                all_connected_pairs = []
                for qid1 in line["qids"]:
                    connected_pairs_gld = []
                    for qid2 in line["qids"]:
                        if qid1 != qid2 and kg_sym.is_connected(qid1, qid2):
                            connected_pairs_gld.append(qid2)
                    all_connected_pairs.append(connected_pairs_gld)
                res[f"{kg_nm}_gld"] = all_connected_pairs
            rows.append(res)
    return pd.DataFrame(rows)


def load_metadata(input_dir, emb_dir):
    entity_dump = EntitySymbols(load_dir=input_dir / "entity_db/entity_mappings")
    a2q = entity_dump.get_alias2qids()
    q2title = entity_dump.get_qid2title()
    types_hy = TypeSymbols(
        entity_dump,
        emb_dir,
        max_types=3,
        type_vocab_file="hyena_vocab.json",
        type_file="hyena_types_1229.json",
    )
    types_wd = TypeSymbols(
        entity_dump,
        emb_dir,
        max_types=3,
        type_vocab_file="wikidatatitle_to_typeid_1229.json",
        type_file="wikidata_types_1229.json",
    )
    types_rel = TypeSymbols(
        entity_dump,
        emb_dir,
        max_types=50,
        type_vocab_file="relation_to_typeid_1229.json",
        type_file="kg_relation_types_1229.json",
    )
    kg_syms = KGSymbols(entity_dump, emb_dir, "kg_adj_1229.txt")
    return a2q, entity_dump, types_hy, types_wd, types_rel, kg_syms, q2title


def load_train_df(train_file, emb_dir):
    input_dir = Path(train_file).parent
    print("Loading metadata")
    a2q, entity_dump, types_hy, types_wd, types_rel, kg_syms, q2title = load_metadata(
        input_dir, emb_dir
    )
    print("Loading training data")
    train_df = load_train_data(
        train_file,
        q2title,
        a2q,
        type_symbols=[types_wd, types_hy, types_rel],
        kg_symbols=[kg_syms],
    )
    return train_df, entity_dump, types_hy, types_wd, types_rel, kg_syms


def prep_for_run(
    cache_file, cache_dir, cache_metadata_dir, input_dir, emb_dir, input_file, overwrite
):
    entity_dump, types_hy, types_wd, types_rel, kg_syms = None, None, None, None, None
    if os.path.exists(cache_file) and not overwrite:
        print(f"Cache file {cache_file} exist. Loading...")
        st = time.time()
        train_df = feather.read_dataframe(cache_file)
        print(f"Loaded in {time.time() - st}")
    else:
        print(
            f"Cache file {cache_file} does not exist or overwrite is True. Recreating..."
        )
        st = time.time()
        train_df, entity_dump, types_hy, types_wd, types_rel, kg_syms = load_train_df(
            input_file, emb_dir
        )
        print(f"Saving dataset to {cache_file}")
        os.makedirs(cache_dir, exist_ok=True)
        feather.write_dataframe(train_df, cache_file)
        print(f"Created in {time.time() - st}")
    if not os.path.exists(cache_metadata_dir) or overwrite:
        print(
            f"Metadata cache dir {cache_metadata_dir} does not exist or overwrite is True. Recreating..."
        )
        st = time.time()
        if entity_dump is None:
            _, entity_dump, types_hy, types_wd, types_rel, kg_syms, _ = load_metadata(
                input_dir, emb_dir
            )
        kg_dict = {KG_KEY: kg_syms}
        type_dict = {TYPE_WD: types_wd, TYPE_HY: types_hy, TYPE_REL: types_rel}
        wl_metadata = WLMetadata(
            entity_dump=entity_dump, dict_type_syms=type_dict, dict_kg_syms=kg_dict
        )
        os.makedirs(cache_metadata_dir, exist_ok=True)
        wl_metadata.save(cache_metadata_dir)
        print(f"Created metadata in {time.time() - st}")
    return train_df


def write_df_to_jsonl(df, file):
    with open(file, "w") as out_f:
        for i, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
            # Output sentence
            out_d = {
                "sentence": str(row["sentence"]),
                "sent_idx_unq": int(row["sent_idx"]),
                "aliases": list(row["aliases"]),
                "sources": list(row["sources"]),
                "spans": [list(map(int, sp)) for sp in row["spans"]],
                "qids": list(row["qids"]),
                "gold": list(map(bool, row["gold"])),
            }
            out_f.write(ujson.dumps(out_d) + "\n")
    return
