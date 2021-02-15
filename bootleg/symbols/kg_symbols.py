"""KG symbols class."""

import os

import ujson as json
from tqdm import tqdm

from bootleg.utils import utils


class KGSymbols:
    """KG symbols class.

    Args:
        entity_symbols: entity symbols
        emb_dir: embedding directory
        kg_adj_file: kg adjacency file
    """

    def __init__(self, entity_symbols, emb_dir, kg_adj_file):
        self.kg_adj_file = os.path.join(emb_dir, kg_adj_file)
        self.qid2connections = self.load_kg_adj_file(entity_symbols)

    def load_kg_adj_file(self, entity_symbols):
        """Loads kg adjacency file where each row is tab separated connected
        pair.

        Args:
            entity_symbols: entity symbols

        Returns: Dict of string QID to set of connected QIDs
        """
        assert self.kg_adj_file != "", f"You need to provide a kg_adj_file file."
        print(f"Loading kg adj from {self.kg_adj_file}")
        num_lines = sum(1 for _ in open(self.kg_adj_file))
        rel_mapping = {}
        with open(self.kg_adj_file, "r") as f:
            for line in tqdm(f, total=num_lines):
                head, tail = line.strip().split()
                if (
                    head not in entity_symbols.get_all_qids()
                    or tail not in entity_symbols.get_all_qids()
                ):
                    continue
                # add heads and tails
                if head in rel_mapping:
                    rel_mapping[head].add(tail)
                else:
                    rel_mapping[head] = {tail}
                if tail in rel_mapping:
                    rel_mapping[tail].add(head)
                else:
                    rel_mapping[tail] = {head}
            return rel_mapping

    def is_connected(self, qid1, qid2):
        """Checks if two QIDs are connected in KG.

        Args:
            qid1: QID one
            qid2: QID two

        Returns: boolean
        """
        return qid2 in self.qid2connections.get(qid1, {})
