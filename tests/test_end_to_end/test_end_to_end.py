"""End2end test."""
import os
import shutil
import unittest

import emmental
import ujson

from bootleg.run import run_model
from bootleg.utils import utils
from bootleg.utils.parser import parser_utils


class TestEnd2End(unittest.TestCase):
    """Test end to end."""

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
        """End2end base test."""
        # Just setting this for testing pipelines
        scores = run_model(mode="train", config=self.args)
        assert type(scores) is dict
        assert len(scores) > 0
        assert scores["model/all/dev/loss"] < 1.1

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file = run_model(mode="dump_preds", config=self.args)
        assert os.path.exists(result_file)
        results = [ujson.loads(li) for li in open(result_file)]
        assert 19 == len(results)  # 18 total sentences
        assert len([f for li in results for f in li["entity_ids"]]) == 52

    # Doubling up a test here to also test accumulation steps
    def test_end2end_accstep(self):
        """Test end2end with accumulation steps."""
        # Just setting this for testing pipelines
        self.args.data_config.dump_preds_accumulation_steps = 2
        self.args.run_config.dataset_threads = 2
        scores = run_model(mode="train", config=self.args)
        assert type(scores) is dict
        assert len(scores) > 0
        assert scores["model/all/dev/loss"] < 1.1

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file = run_model(mode="dump_preds", config=self.args)
        assert os.path.exists(result_file)
        results = [ujson.loads(li) for li in open(result_file)]
        assert 19 == len(results)  # 18 total sentences
        assert len([f for li in results for f in li["entity_ids"]]) == 52

    # Doubling up a test here to also test greater than 1 eval batch size
    def test_end2end_evalbatch(self):
        """Test end2end with eval batch size."""
        self.args.data_config.dump_preds_accumulation_steps = 2
        self.args.run_config.dataset_threads = 2
        self.args.run_config.eval_batch_size = 2

        scores = run_model(mode="train", config=self.args)
        assert type(scores) is dict
        assert len(scores) > 0
        assert scores["model/all/dev/loss"] < 1.1

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file = run_model(mode="dump_preds", config=self.args)
        assert os.path.exists(result_file)
        results = [ujson.loads(li) for li in open(result_file)]
        assert 19 == len(results)  # 18 total sentences
        assert len([f for li in results for f in li["entity_ids"]]) == 52

        shutil.rmtree("tests/temp", ignore_errors=True)

    # Doubling up a test here to also test long context
    def test_end2end_bert_long_context(self):
        """Test end2end with longer sentence context."""
        self.args.data_config.max_seq_len = 256
        self.args.run_config.dump_preds_num_data_splits = 4
        scores = run_model(mode="train", config=self.args)
        assert type(scores) is dict
        assert len(scores) > 0
        assert scores["model/all/dev/loss"] < 1.1

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file = run_model(mode="dump_preds", config=self.args)
        assert os.path.exists(result_file)
        results = [ujson.loads(li) for li in open(result_file)]
        assert 19 == len(results)  # 18 total sentences
        assert len([f for li in results for f in li["entity_ids"]]) == 52

        shutil.rmtree("tests/temp", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
