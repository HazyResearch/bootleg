import os
import shutil
import unittest

import torch

import cand_gen.eval as eval
import cand_gen.train as train
import emmental
from bootleg.utils import utils
from cand_gen.utils.parser import parser_utils


class TestGenEntities(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_candgen.json"
        )
        # This _MUST_ get passed the args so it gets a random seed set
        emmental.init(log_dir="test/temp_log", config=self.args)
        if not os.path.exists(emmental.Meta.log_path):
            os.makedirs(emmental.Meta.log_path)

    def tearDown(self) -> None:
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
        dir = os.path.join("test/temp_log")
        if os.path.exists(dir):
            shutil.rmtree(dir, ignore_errors=True)

    def test_end2end(self):
        # For the collate and dataloaders to play nicely, the spawn must be fork (this is set in run.py)
        torch.multiprocessing.set_start_method("fork", force=True)

        # Train and save model
        train.run_model(config=self.args)

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        candidates_file, metrics_file = eval.run_model(config=self.args)
        assert os.path.exists(candidates_file)
        assert os.path.exists(candidates_file)
        num_sents = len([_ for _ in open(candidates_file)])
        assert num_sents == 17


if __name__ == "__main__":
    unittest.main()
