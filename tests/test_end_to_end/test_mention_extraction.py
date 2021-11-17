"""Test mention extraction."""
import os
import tempfile
import unittest
from pathlib import Path

import ujson

from bootleg.symbols.entity_symbols import EntitySymbols


class MentionExtractionTest(unittest.TestCase):
    """Mention extraction test."""

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

    def test_mention_extraction(self):
        """Test that mention extraction runs without crashing."""
        in_file = Path(self.test_dir.name) / "train.jsonl"
        out_file = Path(self.test_dir.name) / "train_out.jsonl"
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
                "sentence": "happy cow batman",
            }
        ] * 100

        self.write_data(in_file, data)
        os.system(
            f"python3 bootleg/end2end/extract_mentions.py "
            f"--in_file {str(in_file)} "
            f"--out_file {str(out_file)} "
            f"--entity_db {str(entity_db)} "
            f"--num_workers 1 "
            f"--num_chunks 10"
        )

        assert out_file.exists()
        out_data = [ln for ln in open(out_file)]
        assert len(out_data) == 100

        os.system(
            f"python3 bootleg/end2end/extract_mentions.py "
            f"--in_file {str(in_file)} "
            f"--out_file {str(out_file)} "
            f"--entity_db {str(entity_db)} "
            f"--num_workers 2 "
            f"--num_chunks 10"
        )

        assert out_file.exists()
        out_data = [ln for ln in open(out_file)]
        assert len(out_data) == 100
