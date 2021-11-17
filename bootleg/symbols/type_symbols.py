"""Type symbols class."""

import copy
import os
from typing import Dict, List, Optional, Set, Union

from tqdm import tqdm

from bootleg.symbols.constants import edit_op
from bootleg.utils import utils
from bootleg.utils.classes.nested_vocab_tries import TwoLayerVocabularyScoreTrie


def _convert_to_trie(qid2typenames, max_types):
    all_typenames = set()
    qid2typenames_filt = {}
    for q, typs in qid2typenames.items():
        all_typenames.update(set(typs))
        qid2typenames_filt[q] = typs[:max_types]
    qid2typenames_trie = TwoLayerVocabularyScoreTrie(
        input_dict=qid2typenames_filt,
        vocabulary=all_typenames,
        max_value=max_types,
    )
    return qid2typenames_trie


class TypeSymbols:
    """Type Symbols class for managing type metadata."""

    def __init__(
        self,
        qid2typenames: Union[Dict[str, List[str]], TwoLayerVocabularyScoreTrie],
        max_types: Optional[int] = 10,
        edit_mode: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """Type Symbols initializer."""
        if max_types <= 0:
            raise ValueError("max_types must be greater than 0")
        self.max_types = max_types
        self.edit_mode = edit_mode
        self.verbose = verbose
        if self.edit_mode:
            self._load_edit_mode(
                qid2typenames,
            )
        else:
            self._load_non_edit_mode(
                qid2typenames,
            )

    def _load_edit_mode(
        self, qid2typenames: Union[Dict[str, List[str]], TwoLayerVocabularyScoreTrie]
    ):
        """Load qid to type mappings in edit mode."""
        if isinstance(qid2typenames, TwoLayerVocabularyScoreTrie):
            self._qid2typenames: Union[
                Dict[str, List[str]], TwoLayerVocabularyScoreTrie
            ] = qid2typenames.to_dict(keep_score=False)
        else:
            self._qid2typenames: Union[
                Dict[str, List[str]], TwoLayerVocabularyScoreTrie
            ] = {q: typs[: self.max_types] for q, typs in qid2typenames.items()}
        self._all_typenames: Union[Set[str], None] = set(
            [t for typeset in self._qid2typenames.values() for t in typeset]
        )
        self._typename2qids: Union[Dict[str, set], None] = {}
        for qid in tqdm(
            self._qid2typenames,
            total=len(self._qid2typenames),
            desc="Building edit mode objs",
            disable=not self.verbose,
        ):
            for typname in self._qid2typenames[qid]:
                if typname not in self._typename2qids:
                    self._typename2qids[typname] = set()
                self._typename2qids[typname].add(qid)
        # In case extra types in vocab without qids
        for typname in self._all_typenames:
            if typname not in self._typename2qids:
                self._typename2qids[typname] = set()

    def _load_non_edit_mode(
        self, qid2typenames: Union[Dict[str, List[str]], TwoLayerVocabularyScoreTrie]
    ):
        """Load qid to type mappings in non edit mode (read only mode)."""
        if isinstance(qid2typenames, dict):
            self._qid2typenames: Union[
                Dict[str, List[str]], TwoLayerVocabularyScoreTrie
            ] = _convert_to_trie(qid2typenames, self.max_types)
        else:
            self._qid2typenames: Union[
                Dict[str, List[str]], TwoLayerVocabularyScoreTrie
            ] = qid2typenames

        self._all_typenames: Union[Set[str], None] = None
        self._typename2qids: Union[Dict[str, set], None] = None

    def save(self, save_dir, prefix=""):
        """Dump the type symbols.

        Args:
            save_dir: directory string to save
            prefix: prefix to add to beginning to file
        """
        utils.ensure_dir(str(save_dir))
        utils.dump_json_file(
            filename=os.path.join(save_dir, "config.json"),
            contents={
                "max_types": self.max_types,
            },
        )
        if isinstance(self._qid2typenames, dict):
            qid2typenames = _convert_to_trie(self._qid2typenames, self.max_types)
            qid2typenames.dump(os.path.join(save_dir, f"{prefix}qid2typenames"))
        else:
            self._qid2typenames.dump(os.path.join(save_dir, f"{prefix}qid2typenames"))

    @classmethod
    def load_from_cache(cls, load_dir, prefix="", edit_mode=False, verbose=False):
        """Load type symbols from load_dir.

        Args:
            load_dir: directory to load from
            prefix: prefix to add to beginning to file
            edit_mode: edit mode flag
            verbose: verbose flag

        Returns: TypeSymbols
        """
        config = utils.load_json_file(filename=os.path.join(load_dir, "config.json"))
        max_types = config["max_types"]
        # For backwards compatibility, check if trie directory exists, otherwise load from json
        type_load_dir = os.path.join(load_dir, f"{prefix}qid2typenames")
        if not os.path.exists(type_load_dir):
            qid2typenames: Union[
                Dict[str, List[str]], TwoLayerVocabularyScoreTrie
            ] = utils.load_json_file(
                filename=os.path.join(load_dir, f"{prefix}qid2typenames.json")
            )
        else:
            qid2typenames: Union[
                Dict[str, List[str]], TwoLayerVocabularyScoreTrie
            ] = TwoLayerVocabularyScoreTrie(load_dir=type_load_dir, max_value=max_types)
        return cls(qid2typenames, max_types, edit_mode, verbose)

    def get_all_types(self):
        """Return all typenames."""
        if isinstance(self._qid2typenames, dict):
            return self._all_typenames
        else:
            return set(self._qid2typenames.vocab_keys())

    def get_types(self, qid):
        """Get the type names associated with the given QID.

        Args:
            qid: QID

        Returns: list of typename strings
        """
        if isinstance(self._qid2typenames, dict):
            types = self._qid2typenames.get(qid, [])
        else:
            if self._qid2typenames.is_key_in_trie(qid):
                # TwoLayerVocabularyScoreTrie assumes values are list of pairs - we only want type name which is first
                types = self._qid2typenames.get_value(qid, keep_score=False)
            else:
                types = []
        return types

    def get_qid2typename_dict(self):
        """Return dictionary of qid to typenames.

        Returns: Dict of QID to list of typenames.
        """
        if isinstance(self._qid2typenames, dict):
            return copy.deepcopy(self._qid2typenames)
        else:
            return self._qid2typenames.to_dict(keep_score=False)

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
        if typename not in self._all_typenames:
            raise ValueError(f"{typename} is not a type in the typesystem")
        # This will not be None as we are in edit mode
        return self._typename2qids.get(typename, [])

    @edit_op
    def add_type(self, qid, typename):
        """Add the type to the QID.

        If the QID already has maximum types, the
        last type is removed and replaced by ``typename``.

        Args:
            qid: QID
            typename: type name
        """
        if typename not in self._all_typenames:
            self._all_typenames.add(typename)
            self._typename2qids[typename] = set()
        # Update qid->type mappings
        if typename not in self._qid2typenames[qid]:
            # Remove last type if too many types
            if len(self._qid2typenames[qid]) >= self.max_types:
                type_to_remove = self._qid2typenames[qid][-1]
                self.remove_type(qid, type_to_remove)
            self._qid2typenames[qid].append(typename)
            # As we are in edit mode, self._typename2qids will not be None
            self._typename2qids[typename].add(qid)
        return

    @edit_op
    def remove_type(self, qid, typename):
        """Remove the type from the QID.

        Args:
            qid: QID
            typename: type name to remove
        """
        if typename not in self._all_typenames:
            raise ValueError(
                f"The type {typename} is not in our vocab. We only support adding types in our vocab."
            )
        if typename not in self._qid2typenames[qid]:
            return
        assert (
            typename in self._typename2qids
        ), f"Invalid state a typename is in self._typename2qids for {typename} and {qid}"
        self._qid2typenames[qid].remove(typename)
        # As we are in edit mode, self._typename2qids will not be None
        # Further, we want to keep the typename even if list is empty as our type system doesn't change
        self._typename2qids[typename].remove(qid)
        return

    @edit_op
    def add_entity(self, qid, types):
        """
        Add an entity QID with its types to our mappings.

        Args:
            qid: QID
            types: list of type names
        """
        for typename in types:
            if typename not in self._all_typenames:
                self._all_typenames.add(typename)
                self._typename2qids[typename] = set()
        # Add the qid to the qid dicts so we can call the add/remove functions
        self._qid2typenames[qid] = []
        for typename in types:
            self._qid2typenames[qid].append(typename)
        # Cutdown to max types
        self._qid2typenames[qid] = self._qid2typenames[qid][: self.max_types]
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
        """
        assert (
            old_qid in self._qid2typenames and new_qid not in self._qid2typenames
        ), f"Internal Error: checks on existing versus new qid for {old_qid} and {new_qid} failed"
        # Update qid2typenames
        self._qid2typenames[new_qid] = self._qid2typenames[old_qid]
        del self._qid2typenames[old_qid]
        # Update qid2typenames
        for typename in self._qid2typenames[new_qid]:
            self._typename2qids[typename].remove(old_qid)
            self._typename2qids[typename].add(new_qid)

    @edit_op
    def prune_to_entities(self, entities_to_keep):
        """Remove all entities except those in ``entities_to_keep``.

        Args:
            entities_to_keep: Set of entities to keep
        """
        # Update qid2typenames
        self._qid2typenames = {
            k: v for k, v in self._qid2typenames.items() if k in entities_to_keep
        }
        # Update qid2typenames, keeping the typenames even if empty lists
        for typename in self._typename2qids:
            self._typename2qids[typename] = self._typename2qids[typename].intersection(
                entities_to_keep
            )
