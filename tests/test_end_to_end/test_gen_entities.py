"""Test generate entities."""
import os
import shutil
import unittest

import emmental
import numpy as np
import torch
import ujson

import bootleg.extract_all_entities as extract_all_entities
import bootleg.run as run
from bootleg.utils import utils
from bootleg.utils.parser import parser_utils


class TestGenEntities(unittest.TestCase):
    """Test generate entites."""

    def setUp(self) -> None:
        """Set up."""
        self.args = parser_utils.parse_boot_and_emm_args(
            "tests/run_args/test_end2end.json"
        )
        # This _MUST_ get passed the args so it gets a random seed set
        emmental.init(log_dir="tests/temp_log", config=self.args)
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)

    def tearDown(self) -> None:
        """Tear down."""
        dir = os.path.join(
            self.args.data_config.data_dir, self.args.data_config.data_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir, ignore_errors=True)
        dir = os.path.join(
            self.args.data_config.entity_dir, self.args.data_config.entity_prep_dir
        )
        if utils.exists_dir(dir):
            shutil.rmtree(dir, ignore_errors=True)
        dir = os.path.join("tests/temp_log")
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def test_end2end(self):
        """Test end to end."""
        # For the collate and dataloaders to play nicely, the spawn must be fork (this is set in run.py)
        torch.multiprocessing.set_start_method("fork", force=True)

        # Train and save model
        run.run_model(mode="train", config=self.args)

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        out_emb_file = extract_all_entities.run_model(config=self.args)
        assert os.path.exists(out_emb_file)
        embs = np.load(out_emb_file)
        assert list(embs.shape) == [6, 32]

        final_result_file, final_out_emb_file = run.run_model(
            mode="dump_embs", config=self.args, entity_emb_file=out_emb_file
        )

        lines = [ujson.loads(ln) for ln in open(final_result_file)]
        embs = np.load(final_out_emb_file)

        final_result_file, final_out_emb_file = run.run_model(
            mode="dump_embs", config=self.args, entity_emb_file=None
        )
        lines_no_emb_file = [ujson.loads(ln) for ln in open(final_result_file)]
        embs_no_emb_file = np.load(final_out_emb_file)
        assert len(lines) == len(lines_no_emb_file)
        assert embs.shape == embs_no_emb_file.shape


if __name__ == "__main__":
    unittest.main()
