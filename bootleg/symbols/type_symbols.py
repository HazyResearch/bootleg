"""Type symbols class."""

import os
from typing import Dict, List, Optional

from tqdm import tqdm

from bootleg.symbols.constants import edit_op
from bootleg.utils import utils


class TypeSymbols:
    """Type Symbols class for managing type metadata."""

    def __init__(
        self,
        qid2typenames: Dict[str, List[str]],
        qid2typeid: Optional[Dict[str, List[int]]] = None,
        type_vocab: Optional[Dict[str, int]] = None,
        max_types: Optional[int] = 10,
        edit_mode: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        if max_types <= 0:
            raise ValueError("max_types must be greater than 0")
        self.max_types = max_types
        self.edit_mode = edit_mode
        self.verbose = verbose
        self._qid2typenames: Dict[str, List[str]] = {}

        if type_vocab is None:
            all_typenames = set(
                [t for typeset in qid2typenames.values() for t in typeset]
            )
            # +1 to save space for the UNK type
            self._type_vocab: Dict[str, int] = {
                v: i + 1 for i, v in enumerate(sorted(all_typenames))
            }
        else:
            self._type_vocab: Dict[str, int] = type_vocab
            assert (
                0 not in self._type_vocab.values()
            ), "You can't have a type id that is 0. That is reserved for UNK"

        for qid in qid2typenames:
            self._qid2typenames[qid] = qid2typenames.get(qid, [])[: self.max_types]

        if qid2typeid is None:
            self._qid2typeid: Dict[str, List[int]] = {
                qid: list(map(lambda x: self._type_vocab[x], typnames))
                for qid, typnames in self._qid2typenames.items()
            }
        else:
            self._qid2typeid: Dict[str, List[int]] = qid2typeid

        for typeids in self._qid2typeid.values():
            assert 0 not in typeids, "Typeids can't be 0. This is reserved for UNK type"

        self._typename2qids = None
        if edit_mode:
            self._typename2qids = {}
            for qid in tqdm(
                self._qid2typenames,
                total=len(self._qid2typenames),
                desc="Building edit mode objs",
                disable=not verbose,
            ):
                for typname in self._qid2typenames[qid]:
                    if typname not in self._typename2qids:
                        self._typename2qids[typname] = set()
                    self._typename2qids[typname].add(qid)
            # In case extra types in vocab without qids
            for typname in self._type_vocab:
                if typname not in self._typename2qids:
                    self._typename2qids[typname] = set()

    def save(self, save_dir, prefix=""):
        """Dumps the type symbols.

        Args:
            save_dir: directory string to save
            prefix: prefix to add to beginning to file

        Returns:
        """
        utils.ensure_dir(str(save_dir))
        utils.dump_json_file(
            filename=os.path.join(save_dir, "config.json"),
            contents={
                "max_types": self.max_types,
            },
        )
        utils.dump_json_file(
            filename=os.path.join(save_dir, f"{prefix}qid2typenames.json"),
            contents=self._qid2typenames,
        )
        utils.dump_json_file(
            filename=os.path.join(save_dir, f"{prefix}qid2typeids.json"),
            contents=self._qid2typeid,
        )
        utils.dump_json_file(
            filename=os.path.join(save_dir, f"{prefix}type_vocab.json"),
            contents=self._type_vocab,
        )

    @classmethod
    def load_from_cache(cls, load_dir, prefix="", edit_mode=False, verbose=False):
        """Loads type symbols from load_dir.

        Args:
            load_dir: directory to load from
            prefix: prefix to add to beginning to file
            edit_mode: edit mode flag
            verbose: verbose flag

        Returns: TypeSymbols
        """
        config = utils.load_json_file(filename=os.path.join(load_dir, "config.json"))
        max_types = config["max_types"]
        qid2typenames: Dict[str, List[str]] = utils.load_json_file(
            filename=os.path.join(load_dir, f"{prefix}qid2typenames.json")
        )
        qid2typeid: Dict[str, List[int]] = utils.load_json_file(
            filename=os.path.join(load_dir, f"{prefix}qid2typeids.json")
        )
        type_vocab: Dict[str, int] = utils.load_json_file(
            filename=os.path.join(load_dir, f"{prefix}type_vocab.json")
        )
        return cls(qid2typenames, qid2typeid, type_vocab, max_types, edit_mode, verbose)

    def get_all_types(self):
        """Returns all typenames.

        Returns:
        """
        return list(self._type_vocab.keys())

    def get_types(self, qid):
        """Gets the type names associated with the given QID.

        Args:
            qid: QID

        Returns: list of typename strings
        """
        types = self._qid2typenames.get(qid, [])
        return types

    def get_typeids(self, qid):
        """Gets the type ids associated with the given QID.

        Args:
            qid: QID

        Returns: list of type id ints
        """
        return self._qid2typeid.get(qid, [])

    # ============================================================
    # EDIT MODE OPERATIONS
    # ============================================================
    @edit_op
    def get_entities_of_type(self, typename):
        """Get all entity QIDs of type ``typename``.

        Args:
            typename: typename

        Returns: List
        """
        if typename not in self._type_vocab:
            raise ValueError(f"{typename} is not a type in the typesystem")
        # This will not be None as we are in edit mode
        return self._typename2qids.get(typename, [])

    @edit_op
    def add_type(self, qid, typename):
        """Adds the type to the QID. If the QID already has maximum types, the
        last type is removed and replaced by ``typename``.

        Args:
            qid: QID
            typename: type name

        Returns:
        """
        if typename not in self._type_vocab:
            max_type_id = max(self._type_vocab.values()) + 1
            self._type_vocab[typename] = max_type_id
            self._typename2qids[typename] = set()
        typeid = self._type_vocab[typename]
        # Update qid->type mappings
        if typename not in self._qid2typenames[qid]:
            assert (
                typeid not in self._qid2typeid[qid]
            ), f"Invalid state a typeid is in self._qid2typeid for {typename} and {qid}"
            # Remove last type if too many types
            if len(self._qid2typenames[qid]) >= self.max_types:
                type_to_remove = self._qid2typenames[qid][-1]
                self.remove_type(qid, type_to_remove)
            self._qid2typenames[qid].append(typename)
            self._qid2typeid[qid].append(typeid)
            # As we are in edit mode, self._typename2qids will not be None
            self._typename2qids[typename].add(qid)
        return

    @edit_op
    def remove_type(self, qid, typename):
        """Remove the type from the QID.

        Args:
            qid: QID
            typename: type name to remove

        Returns:
        """
        if typename not in self._type_vocab:
            raise ValueError(
                f"The type {typename} is not in our vocab. We only support adding types in our vocab."
            )
        if typename not in self._qid2typenames[qid]:
            return
        assert (
            self._type_vocab[typename] in self._qid2typeid[qid]
        ), f"Invalid state a typeid is in self._qid2typeid for {typename} and {qid}"
        assert (
            typename in self._typename2qids
        ), f"Invalid state a typename is in self._typename2qids for {typename} and {qid}"
        self._qid2typenames[qid].remove(typename)
        self._qid2typeid[qid].remove(self._type_vocab[typename])
        # As we are in edit mode, self._typename2qids will not be None
        # Further, we want to keep the typename even if list is empty as our type system doesn't change
        self._typename2qids[typename].remove(qid)
        return

    @edit_op
    def add_entity(self, qid, types):
        """
        Add an entity QID with its types to our mappings
        Args:
            qid: QID
            types: list of type names

        Returns:

        """
        for typename in types:
            if typename not in self._type_vocab:
                max_type_id = max(self._type_vocab.values()) + 1
                self._type_vocab[typename] = max_type_id
                self._typename2qids[typename] = set()
        # Add the qid to the qid dicts so we can call the add/remove functions
        self._qid2typenames[qid] = []
        self._qid2typeid[qid] = []
        for typename in types:
            self._qid2typenames[qid].append(typename)
            self._qid2typeid[qid].append(self._type_vocab[typename])
        # Cutdown to max types
        self._qid2typenames[qid] = self._qid2typenames[qid][: self.max_types]
        self._qid2typeid[qid] = self._qid2typeid[qid][: self.max_types]
        # Add to typenames to qids
        for typename in self._qid2typenames[qid]:
            self._typename2qids[typename].add(qid)
        return

    @edit_op
    def reidentify_entity(self, old_qid, new_qid):
        """Rename ``old_qid`` to ``new_qid``.

        Args:
            old_qid: old QID
            new_qid: new QID

        Returns:
        """
        assert (
            old_qid in self._qid2typenames and new_qid not in self._qid2typenames
        ), f"Internal Error: checks on existing versus new qid for {old_qid} and {new_qid} failed"
        # Update qid2typenames
        self._qid2typenames[new_qid] = self._qid2typenames[old_qid]
        del self._qid2typenames[old_qid]
        # Update qid2typeid
        self._qid2typeid[new_qid] = self._qid2typeid[old_qid]
        del self._qid2typeid[old_qid]
        # Update qid2typenames
        for typename in self._qid2typenames[new_qid]:
            self._typename2qids[typename].remove(old_qid)
            self._typename2qids[typename].add(new_qid)

    @edit_op
    def prune_to_entities(self, entities_to_keep):
        """Remove all entities except those in ``entities_to_keep``.

        Args:
            entities_to_keep: Set of entities to keep

        Returns:
        """
        # Update qid2typenames
        self._qid2typenames = {
            k: v for k, v in self._qid2typenames.items() if k in entities_to_keep
        }
        # Update qid2typeid
        self._qid2typeid = {
            k: v for k, v in self._qid2typeid.items() if k in entities_to_keep
        }
        # Update qid2typenames, keeping the typenames even if empty lists
        for typename in self._typename2qids:
            self._typename2qids[typename] = self._typename2qids[typename].intersection(
                entities_to_keep
            )
