import argparse
import os

import ujson

from bootleg.symbols.entity_symbols import EntitySymbols

# generates entity mappings for the entity directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qid2title", type=str, required=True, help="Path for qid2title file"
    )
    parser.add_argument(
        "--alias2qids", type=str, required=True, help="Path for alias2qids file"
    )
    parser.add_argument(
        "--alias_cand_map_file",
        type=str,
        default="alias2qids.json",
        help="Name of file to write the alias2qids in the entity_dir",
    )
    parser.add_argument(
        "--entity_dir", type=str, required=True, help="Directory to write entity_db"
    )
    parser.add_argument(
        "--entity_map_dir",
        type=str,
        default="entity_mappings",
        help="Directory to write entity_mappings inside entity_dir",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=30,
        help="Maximum number of candidates per alias",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # compute the max alias len in alias2qids
    with open(args.alias2qids) as f:
        alias2qids = ujson.load(f)

    with open(args.qid2title) as f:
        qid2title = ujson.load(f)

    for alias in alias2qids:
        assert (
            alias.lower() == alias
        ), f"bootleg assumes lowercase aliases in alias candidate maps: {alias}"
        # ensure only max_candidates per alias
        qids = sorted(alias2qids[alias], key=lambda x: (x[1], x[0]), reverse=True)
        alias2qids[alias] = qids[: args.max_candidates]

    entity_mappings = EntitySymbols(
        max_candidates=args.max_candidates,
        alias2qids=alias2qids,
        qid2title=qid2title,
        alias_cand_map_file=args.alias_cand_map_file,
    )

    entity_mappings.save(os.path.join(args.entity_dir, args.entity_map_dir))
    print("entity mappings exported.")


if __name__ == "__main__":
    main()
