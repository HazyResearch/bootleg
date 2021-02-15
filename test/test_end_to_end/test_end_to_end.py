import os
import shutil
import unittest

import torch

import emmental
from bootleg.run import run_model
from bootleg.utils import utils
from bootleg.utils.classes.dotted_dict import DottedDict
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

    def test_end2end_withkg(self):
        # For the collate and dataloaders to play nicely, the spawn must be fork (this is set in run.py)
        torch.multiprocessing.set_start_method("fork", force=True)

        scores = run_model(mode="train", config=self.args)

        assert type(scores) is dict
        assert len(scores) > 0
        assert scores["model/all/train/loss"] < 0.08

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        assert os.path.exists(out_emb_file)

    def test_end2end_withoutkg(self):
        # KG IS LAST EMBEDDING SO WE REMOVE IT
        self.args.data_config.ent_embeddings = self.args.data_config.ent_embeddings[:-1]

        scores = run_model(mode="train", config=self.args)

        assert type(scores) is dict
        assert len(scores) > 0
        assert scores["model/all/train/loss"] < 0.05

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        assert os.path.exists(out_emb_file)

    def test_end2end_withtype(self):
        self.args.data_config.type_prediction.use_type_pred = True
        self.args.model_config.hidden_size = 20

        # unfreezing the word embedding helps the type prediction task
        self.args.data_config.word_embedding.freeze = False

        scores = run_model(mode="train", config=self.args)

        assert type(scores) is dict
        assert len(scores) > 0
        # losses from two tasks contribute to this
        assert scores["model/all/train/loss"] < 0.08

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        assert os.path.exists(out_emb_file)

    def test_end2end_withtitle(self):
        self.args.data_config.ent_embeddings.append(
            DottedDict(
                {
                    "key": "title1",
                    "load_class": "TitleEmb",
                    "send_through_bert": True,
                    "through_bert_metadata_keys": [4, 5],
                    "args": {"proj": 6},
                }
            )
        )
        scores = run_model(mode="train", config=self.args)

        assert type(scores) is dict
        assert len(scores) > 0
        assert scores["model/all/train/loss"] < 0.08

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        assert os.path.exists(out_emb_file)

    def test_end2end_withreg(self):
        reg_file = "test/temp/reg_file.csv"
        utils.ensure_dir("test/temp")
        reg_data = [
            ["qid", "regularization"],
            ["Q1", "0.5"],
            ["Q2", "0.3"],
            ["Q3", "0.2"],
            ["Q4", "0.9"],
        ]
        with open(reg_file, "w") as out_f:
            for item in reg_data:
                out_f.write(",".join(item) + "\n")

        self.args.data_config.ent_embeddings[0]["args"]["regularize_mapping"] = reg_file

        scores = run_model(mode="train", config=self.args)

        assert type(scores) is dict
        assert len(scores) > 0
        assert scores["model/all/train/loss"] < 0.05

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        assert os.path.exists(out_emb_file)

        shutil.rmtree("test/temp", ignore_errors=True)

    def test_end2end_bert(self):
        self.args.model_config.attn_class = "BERTNED"
        # Only take the learned entity embeddings for BERTNED
        self.args.data_config.ent_embeddings = self.args.data_config.ent_embeddings[:1]
        # Set the learned embedding to hidden size for BERTNED
        self.args.data_config.ent_embeddings[0].args.learned_embedding_size = 20
        self.args.data_config.word_embedding.use_sent_proj = False

        scores = run_model(mode="train", config=self.args)

        assert type(scores) is dict
        assert len(scores) > 0
        assert scores["model/all/train/loss"] < 0.5

        self.args["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"
        emmental.Meta.config["model_config"][
            "model_path"
        ] = f"{emmental.Meta.log_path}/last_model.pth"

        result_file, out_emb_file = run_model(mode="dump_embs", config=self.args)
        assert os.path.exists(result_file)
        assert os.path.exists(out_emb_file)

        shutil.rmtree("test/temp", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
