"""KG symbols class"""

import ujson as json
from tqdm import tqdm
import os


from bootleg.utils import utils


class KGSymbols:
    """
    KG symbols class.

    Attributes:
        kg_adj: kg adjacency matrix
    """
    def __init__(self, entity_symbols, emb_dir, kg_adj_file):
        self.kg_adj_file = os.path.join(emb_dir, kg_adj_file)
        self.qid2connections = self.load_kg_adj_file(entity_symbols, emb_dir)


    def load_kg_adj_file(self, entity_symbols, emb_dir):
        """Loads kg adjacency file where each row is tab separated connected pair"""
        assert self.kg_adj_file != "", f"You need to provide a kg_adj_file file."
        print(f"Loading kg adj from {self.kg_adj_file}")
        num_lines = sum(1 for line in open(self.kg_adj_file))
        rel_mapping = {}
        with open(self.kg_adj_file, "r") as f:
            total = 0.
            count = 0.
            for line in tqdm(f, total=num_lines):
                head, tail = line.strip().split()
                if head not in entity_symbols.get_all_qids() or tail not in entity_symbols.get_all_qids():
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
        return qid2 in self.qid2connections.get(qid1, {})