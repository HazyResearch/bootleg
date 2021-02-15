"""Type symbols class."""

import os

import ujson as json
from tqdm import tqdm

from bootleg.utils import utils


class TypeSymbols:
    """Type symbols class.

    Args:
        entity_symbols: entity symbols
        emb_dir: embedding directory
        max_types: maximum number of types per entity
        type_vocab_file: json vocab file of type id to type names
        type_file: json file for QID to type id
    """

    def __init__(
        self,
        entity_symbols,
        emb_dir,
        max_types,
        type_vocab_file,
        type_file,
    ):
        self.type_vocab_file = type_vocab_file
        self.type_file = os.path.join(emb_dir, type_file)
        self.qid2typenames, self.qid2typeid, self.typeid2typename = self.load_types(
            entity_symbols, emb_dir, max_types
        )

    def load_type_file(
        self,
        type_file,
        entity_symbols,
        max_types,
        qid2typeid,
        qid2typenames,
        typeid2typename,
    ):
        """Loads type file and generates QID to type id mappings.

        Args:
            type_file: json QID to list of type id file
            entity_symbols: entity symbols
            max_types: maximum number of types per entity
            qid2typeid: QID to typeid dict to add types to
            qid2typenames: QID to type names dict to add types to
            typeid2typename: Typeid to type names dict to add types to

        Returns: qid2typeid dict, qid2typenames dict
        """
        with open(type_file, "r") as f:
            total = 0.0
            count = 0.0
            qid2typeid_raw = json.load(f)
            for qid in tqdm(qid2typeid_raw, desc=f"Reading {type_file}"):
                typeids = qid2typeid_raw[qid][:max_types]
                total += len(typeids)
                # Use identity map if type is not in vocab
                for t in typeids:
                    if t not in typeid2typename:
                        typeid2typename[t] = str(t)
                qidtypenames = [typeid2typename[t] for t in typeids]
                count += 1
                if entity_symbols.qid_exists(qid):
                    qid2typeid[qid] = typeids
                    qid2typenames[qid] = qidtypenames
            return qid2typeid, qid2typenames

    def load_types(self, entity_symbols, emb_dir, max_types):
        """Loads all type information.

        Args:
            entity_symbols: entity symbols
            emb_dir: embedding directory
            max_types: maximum number of types per entity

        Returns: qid2typenames dict, qid2typeid dict, typeid2typename dict
        """
        # load type vocab
        if self.type_vocab_file == "":
            print(
                "You did not give a type vocab file (from type name to typeid). We will use identity mapping"
            )
            typeid2typename = {}
        else:
            extension = os.path.splitext(self.type_vocab_file)[-1]
            if extension == ".json":
                type_vocab = utils.load_json_file(
                    os.path.join(emb_dir, self.type_vocab_file)
                )
            else:
                print(
                    f"We only support loading json files for TypeSymbol. You have a file ending in {extension}"
                )
                return {}, {}, {}
            typeid2typename = {i: v for v, i in type_vocab.items()}
        # load mapping of entities to type ids
        qid2typenames = {qid: [] for qid in entity_symbols.get_all_qids()}
        qid2typeid = {qid: [] for qid in entity_symbols.get_all_qids()}
        print(f"Loading types from {self.type_file}")
        qid2typeid, qid2typenames = self.load_type_file(
            type_file=self.type_file,
            max_types=max_types,
            entity_symbols=entity_symbols,
            qid2typeid=qid2typeid,
            qid2typenames=qid2typenames,
            typeid2typename=typeid2typename,
        )
        return qid2typenames, qid2typeid, typeid2typename

    def get_types(self, qid):
        """Gets the type names associated with the given QID.

        Args:
            qid: QID

        Returns: list of typename strings
        """
        types = self.qid2typenames.get(qid, [])
        return types

    def get_typeids(self, qid):
        """Gets the type ids associated with the given QID.

        Args:
            qid: QID

        Returns: list of type id ints
        """
        return self.qid2typeid.get(qid, [])
