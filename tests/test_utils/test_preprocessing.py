"""Test preprocessing utils."""
import os
import tempfile
import unittest
from pathlib import Path

import ujson

from bootleg.symbols.entity_symbols import EntitySymbols


class PreprocessingUtils(unittest.TestCase):
    """Preprocessing utils test."""

    def setUp(self) -> None:
        """Set up."""
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Tear down."""
        self.test_dir.cleanup()

    def write_data(self, file, data):
        """Write data."""
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as out_f:
            for line in data:
                out_f.write(ujson.dumps(line) + "\n")

    def test_get_train_qid_counts(self):
        """Test get train qid counts."""
        in_file = Path(self.test_dir.name) / "train.jsonl"
        out_file = Path(self.test_dir.name) / "train_counts_out.json"

        data = [{"qids": [f"Q{i}" for i in range(5)]}] * 100

        self.write_data(in_file, data)

        os.system(
            f"python3 bootleg/utils/preprocessing/get_train_qid_counts.py "
            f"--train_file {in_file} "
            f"--out_file {out_file}"
        )

        res = ujson.load(open(out_file, "r"))

        assert len(res) == 5
        for k in res:
            assert res[k] == 100

    def test_compute_statistics(self):
        """Test compute statistics."""
        in_file = Path(self.test_dir.name) / "train.jsonl"
        entity_db = Path(self.test_dir.name) / "entity_db" / "entity_mappings"

        alias2qids = {
            "happy": [["Q1", 1.0], ["Q2", 1.0], ["Q3", 1.0]],
            "cow": [["Q4", 1.0], ["Q5", 1.0], ["Q6", 1.0]],
            "batman": [["Q7", 1.0], ["Q8", 1.0]],
        }

        qid2title = {
            "Q1": "aack",
            "Q2": "back",
            "Q3": "cack",
            "Q4": "dack",
            "Q5": "eack",
            "Q6": "fack",
            "Q7": "gack",
            "Q8": "hack",
        }

        mock_entity_db = EntitySymbols(alias2qids, qid2title)

        mock_entity_db.save(entity_db)

        data = [
            {
                "qids": ["Q1", "Q4", "Q7"],
                "unswap_aliases": ["happy", "cow", "batman"],
                "sentence": "happy cow batman",
            }
        ] * 100

        self.write_data(in_file, data)
        os.system(
            f"python3 bootleg/utils/preprocessing/compute_statistics.py "
            f"--data_dir {self.test_dir.name} "
            f"--save_dir {self.test_dir.name}"
        )

        out_dir = Path(self.test_dir.name) / "stats"
        assert out_dir.exists()
        alias_cnts = ujson.load(open(out_dir / "alias_counts.json"))
        assert len(alias_cnts) == 3
        assert all(v == 100 for v in alias_cnts.values())

    def test_sample_eval_data(self):
        """Test sample eval data."""
        in_file = Path(self.test_dir.name) / "train.jsonl"
        data = [
            {
                "qids": ["Q1", "Q4", "Q7"],
                "sent_idx_unq": i,
                "aliases": ["happy", "cow", "batman"],
                "gold": [True, True, False],
                "slices": {"slice_1": {"0": 1.0, "1": 1.0, "2": 1.0}},
                "sentence": "happy cow batman",
            }
            for i in range(100)
        ]
        self.write_data(in_file, data)

        os.system(
            f"python3 bootleg/utils/preprocessing/sample_eval_data.py "
            f"--data_dir {self.test_dir.name} "
            f"--slice slice_1 --file train.jsonl --out_file_name train_out.jsonl --min_sample_size 10"
        )

        out_file = Path(self.test_dir.name) / "train_out.jsonl"
        assert out_file.exists()
        alias_out = [ln for ln in open(out_file)]
        assert len(alias_out) == 10
