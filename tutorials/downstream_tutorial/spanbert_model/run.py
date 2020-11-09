import argparse
import logging
import sys

import emmental
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.utils.parse_args import parse_args, parse_args_to_config
from emmental.utils.utils import nullable_int, nullable_string, str2bool

from data import get_dataloaders
from task import create_task
from task_config import ID_TO_LABEL
from utils import write_to_file, write_to_json_file

logger = logging.getLogger(__name__)


def add_application_args(parser):

    # Application configuration
    application_config = parser.add_argument_group("Application configuration")

    application_config.add_argument(
        "--data_dir", type=str, default="data", help="The path to TACRED dataset"
    )

    application_config.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )

    application_config.add_argument(
        "--valid_batch_size",
        type=nullable_int,
        default=None,
        help="Validation batch size",
    )

    application_config.add_argument(
        "--train", type=str2bool, default=True, help="Whether training or not"
    )

    application_config.add_argument(
        "--bert_model",
        type=str,
        default="spanbert-base-cased",
        choices=["spanbert-base-cased", "spanbert-large-cased"],
        help="Which bert pretrained model to use",
    )

    application_config.add_argument(
        "--max_data_samples", type=int, default=None, help="Maximum data samples to use"
    )

    application_config.add_argument(
        "--max_seq_length", type=int, default=200, help="Maximum sentence length"
    )

    application_config.add_argument(
        "--feature_mode",
        type=str,
        default="ner",
        choices=["text", "ner", "text_ner", "ner_text"],
    )

    application_config.add_argument(
        "--ent_emb_file",
        type=nullable_string,
        default=None,
        help="Entity embedding file",
    )
    
    application_config.add_argument(
        "--static_ent_emb_file",
        type=nullable_string,
        default=None,
        help="Static entity embedding file",
    )

    application_config.add_argument(
        "--type_emb_file",
        type=nullable_string,
        default=None,
        help="Type embedding file",
    )

    application_config.add_argument(
        "--rel_emb_file",
        type=nullable_string,
        default=None,
        help="Relation embedding file",
    )

    application_config.add_argument(
        "--encode_first",
        type=str2bool,
        default=False,
        help="Whether only encodes the first token or not",
    )

    application_config.add_argument(
        "--kg_encoder_layer",
        type=int,
        default=4,
        help="Number of kg encoder layer",
    )

    application_config.add_argument(
        "--tanh",
        type=str2bool,
        default=False,
        help="Whether uses tanh in proj",
    )

    application_config.add_argument(
        "--norm",
        type=str2bool,
        default=False,
        help="Whether uses norm in proj",
    )


if __name__ == "__main__":
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        description="Commandline interface for TACRED application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser = parse_args(parser=parser)
    add_application_args(parser)

    args = parser.parse_args()

    # Initialize Emmental
    config = parse_args_to_config(args)
    emmental.init(log_dir=config["meta_config"]["log_path"], config=config)

    # Log configuration into files
    cmd_msg = " ".join(sys.argv)
    logger.info(f"COMMAND: {cmd_msg}")
    write_to_file(f"{emmental.Meta.log_path}/cmd.txt", cmd_msg)

    logger.info(f"Config: {emmental.Meta.config}")
    write_to_file(f"{emmental.Meta.log_path}/config.txt", emmental.Meta.config)

    # Create dataloaders
    dataloaders = get_dataloaders(args)

    # Specify parameter group for Adam BERT
    def grouped_parameters(model):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        return [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": emmental.Meta.config["learner_config"][
                    "optimizer_config"
                ]["l2"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    emmental.Meta.config["learner_config"]["optimizer_config"][
        "parameters"
    ] = grouped_parameters

    # Create tasks
    model = EmmentalModel(name="TACRED_task")
    model.add_task(create_task(args))

    # Load the best model from the pretrained model
    if config["model_config"]["model_path"] is not None:
        model.load(config["model_config"]["model_path"])

    if args.train:
        emmental_learner = EmmentalLearner()
        emmental_learner.learn(model, dataloaders)

    # Remove all extra augmentation policy
    for idx in range(len(dataloaders)):
        dataloaders[idx].dataset.transform_cls = None

    scores = model.score(dataloaders)

    # Save metrics and models
    logger.info(f"Metrics: {scores}")
    scores["log_path"] = emmental.Meta.log_path
    write_to_json_file(f"{emmental.Meta.log_path}/metrics.txt", scores)
    model.save(f"{emmental.Meta.log_path}/last_model.pth")

    for dataloader in dataloaders:
        if dataloader.split == "train":
            continue
        preds = model.predict(dataloader, return_preds=True)
        res = ""
        for uid, pred in zip(preds["uids"]["TACRED"], preds["preds"]["TACRED"]):
            res += f"{uid}\t{ID_TO_LABEL[pred]}\n"
        write_to_file(
            f"{emmental.Meta.log_path}/{dataloader.split}_predictions.txt", res.strip()
        )

        res = ""
        for uid, pred in zip(preds["uids"]["TACRED"], preds["golds"]["TACRED"]):
            res += f"{uid}\t{ID_TO_LABEL[pred]}\n"
        write_to_file(
            f"{emmental.Meta.log_path}/{dataloader.split}_golds.txt", res.strip()
        )
