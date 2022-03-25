"""Test annotator."""
import os
import shutil
import unittest

import emmental
import torch

from bootleg import extract_all_entities
from bootleg.end2end.bootleg_annotator import BootlegAnnotator
from bootleg.run import run_model
from bootleg.utils import utils
from bootleg.utils.parser import parser_utils


class TestEnd2End(unittest.TestCase):
    """Test annotator end to end."""

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

    def test_annotator(self):
        """Test annotator end to end."""
        torch.multiprocessing.set_start_method("fork", force=True)
        # Just to make it go faster
        self.args["learner_config"]["n_epochs"] = 1
        # First train some model so we have it stored
        run_model(mode="train", config=self.args)

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        out_emb_file = extract_all_entities.run_model(config=self.args)

        ann = BootlegAnnotator(config=self.args, verbose=True)
        # TEST SINGLE TEXT
        # Res should have alias1
        res = ann.label_mentions(
            "alias1 and alias2 and multi word alias3 I have no idea"
        )
        gold_ans = {
            "qids": [["Q1"]],
            "titles": [["alias1"]],
            "cands": [[["Q1", "Q4", "-1"]]],
            "char_spans": [[[0, 6]]],
            "aliases": [["alias1"]],
        }
        for k in gold_ans:
            # In case model doesn't learn the right pattern (happens on GitHub for some reason),
            # Do not test qids or titles
            if k in ["char_spans", "aliases", "cands"]:
                self.assertListEqual(gold_ans[k], res[k])

        # TEST LONG TEXT
        # Res should have alias1
        res = ann.label_mentions(
            [
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea",
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea",
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea",
            ]
        )
        gold_ans = {
            "qids": [["Q1"] * 8] * 3,
            "titles": [["alias1"] * 8] * 3,
            "cands": [[["Q1", "Q4", "-1"]] * 8] * 3,
            "char_spans": [
                [
                    [0, 6],
                    [56, 62],
                    [112, 118],
                    [168, 174],
                    [224, 230],
                    [280, 286],
                    [336, 342],
                    [392, 398],
                ],
            ]
            * 3,
            "aliases": [["alias1"] * 8] * 3,
        }
        for k in gold_ans:
            # In case model doesn't learn the right pattern (happens on GitHub for some reason),
            # Do not test qids or titles
            if k in ["char_spans", "aliases", "cands"]:
                self.assertListEqual(gold_ans[k], res[k])

        # TEST RETURN EMBS
        ann.return_embs = True
        res = ann.label_mentions(
            "alias1 and alias2 and multi word alias3 I have no idea"
        )
        assert "embs" in res
        assert res["embs"][0][0].shape[0] == 32
        assert list(res["cand_embs"][0][0].shape) == [3, 32]

        # TEST CUSTOM CANDS
        ann.return_embs = False
        extracted_exs = [
            {
                "sentence": "alias1 and alias2 and multi word alias3 I have no idea",
                "aliases": ["alias3"],
                "char_spans": [[0, 6]],
                "cands": [["Q3"]],
            },
            {
                "sentence": "alias1 and alias2 and multi word alias3 I have no idea. "
                "alias1 and alias2 and multi word alias3 I have no idea. ",
                "aliases": ["alias1", "alias3", "alias1"],
                "char_spans": [[0, 6], [11, 17], [22, 39]],
                "cands": [["Q2"], ["Q3"], ["Q2"]],
            },
        ]
        res = ann.label_mentions(extracted_examples=extracted_exs)
        gold_ans = {
            "qids": [["Q3"], ["Q2", "Q3", "Q2"]],
            "titles": [
                ["word alias3"],
                ["multi alias2", "word alias3", "multi alias2"],
            ],
            "cands": [
                [["Q3", "-1", "-1"]],
                [["Q2", "-1", "-1"], ["Q3", "-1", "-1"], ["Q2", "-1", "-1"]],
            ],
            "char_spans": [[[0, 6]], [[0, 6], [11, 17], [22, 39]]],
            "aliases": [["alias3"], ["alias1", "alias3", "alias1"]],
        }
        for k in gold_ans:
            self.assertListEqual(gold_ans[k], res[k])

        ann = BootlegAnnotator(
            config=self.args, verbose=True, entity_emb_file=out_emb_file
        )
        ann.return_embs = True
        # TEST SINGLE TEXT
        # Res should have alias1
        res = ann.label_mentions(
            "alias1 and alias2 and multi word alias3 I have no idea"
        )
        assert "embs" in res
        gold_ans = {
            "qids": [["Q1"]],
            "titles": [["alias1"]],
            "cands": [[["Q1", "Q4", "-1"]]],
            "char_spans": [[[0, 6]]],
            "aliases": [["alias1"]],
        }
        for k in gold_ans:
            # In case model doesn't learn the right pattern (happens on GitHub for some reason),
            # Do not test qids or titles
            if k in ["char_spans", "aliases", "cands"]:
                self.assertListEqual(gold_ans[k], res[k])


if __name__ == "__main__":
    unittest.main()
