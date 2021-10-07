import os
import shutil
import unittest

import ujson

import emmental
from bootleg.run import run_model
from bootleg.utils import utils
from bootleg.utils.parser import parser_utils


class TestEnd2End(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parser_utils.parse_boot_and_emm_args(
            "test/run_args/test_end2end.json"
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

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        results = [ujson.loads(li) for li in open(result_file)]
        assert 19 == len(results)  # 18 total sentences
        assert set([f for li in results for f in li["ctx_emb_ids"]]) == set(
            range(52)
        )  # 38 total mentions
        assert os.path.exists(out_emb_file)

    # Doubling up a test here to also test accumulation steps
    def test_end2end_accstep(self):
        # Just setting this for testing pipelines
        self.args.data_config.eval_accumulation_steps = 2
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

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        results = [ujson.loads(li) for li in open(result_file)]
        assert 19 == len(results)  # 18 total sentences
        assert set([f for li in results for f in li["ctx_emb_ids"]]) == set(
            range(52)
        )  # 38 total mentions
        assert os.path.exists(out_emb_file)

    # Doubling up a test here to also test greater than 1 eval batch size
    def test_end2end_evalbatch(self):
        self.args.data_config.eval_accumulation_steps = 2
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

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        results = [ujson.loads(li) for li in open(result_file)]
        assert 19 == len(results)  # 18 total sentences
        assert set([f for li in results for f in li["ctx_emb_ids"]]) == set(
            range(52)
        )  # 38 total mentions
        assert os.path.exists(out_emb_file)

        shutil.rmtree("test/temp", ignore_errors=True)

    # Doubling up a test here to also test long context
    def test_end2end_bert_long_context(self):
        self.args.data_config.max_seq_len = 256
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

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        results = [ujson.loads(li) for li in open(result_file)]
        assert 19 == len(results)  # 18 total sentences
        assert set([f for li in results for f in li["ctx_emb_ids"]]) == set(
            range(52)
        )  # 38 total mentions
        assert os.path.exists(out_emb_file)

        shutil.rmtree("test/temp", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
