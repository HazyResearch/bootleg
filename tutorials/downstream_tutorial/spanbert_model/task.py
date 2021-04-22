import logging
from functools import partial

import torch.nn.functional as F
from ent_bert_encoder import EntBertEncoder
from pytorch_pretrained_bert.modeling import BertConfig, BertModel
from scorer import tacred_scorer
from task_config import LABEL_TO_ID
from torch import nn

from emmental.scorer import Scorer
from emmental.task import EmmentalTask

logger = logging.getLogger(__name__)


def ce_loss(module_name, intermediate_output_dict, Y, active):
    return F.cross_entropy(
        intermediate_output_dict[module_name][0][active], Y.view(-1)[active]
    )


def output(module_name, intermediate_output_dict):
    return F.softmax(intermediate_output_dict[module_name][0], dim=1)


ENT_BERT_ENCODER_CONFIG = {
    "attention_probs_dropout_prob": 0.1,
    "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 512,
    "num_attention_heads": 16,
    "num_hidden_layers": 4,
}


def create_task(args):
    task_name = "TACRED"

    bert_model = BertModel.from_pretrained(args.bert_model, cache_dir="./cache/")

    bert_output_dim = 768 if "base" in args.bert_model else 1024

    config = ENT_BERT_ENCODER_CONFIG
    if (
        args.ent_emb_file is not None
        or args.static_ent_emb_file is not None
        or args.type_emb_file is not None
        or args.rel_emb_file is not None
    ):
        config["num_hidden_layers"] = args.kg_encoder_layer
        output_size = ENT_BERT_ENCODER_CONFIG["hidden_size"]
    else:
        output_size = bert_output_dim
        ENT_BERT_ENCODER_CONFIG["hidden_size"] = output_size
    config = BertConfig.from_dict(config)
    logger.info(config)
    encoder = EntBertEncoder(
        config,
        bert_output_dim,
        output_size,
        args.ent_emb_file,
        args.static_ent_emb_file,
        args.type_emb_file,
        args.rel_emb_file,
        tanh=args.tanh,
        norm=args.norm,
    )

    task = EmmentalTask(
        name=task_name,
        module_pool=nn.ModuleDict(
            {
                "bert": bert_model,
                "encoder": encoder,
                f"{task_name}_pred_head": nn.Linear(
                    output_size, len(LABEL_TO_ID.keys())
                ),
            }
        ),
        task_flow=[
            {
                "name": "bert",
                "module": "bert",
                "inputs": [
                    ("_input_", "token_ids"),
                    ("_input_", "token_segments"),
                    ("_input_", "token_masks"),
                ],
            },
            {
                "name": "encoder",
                "module": "encoder",
                "inputs": [
                    ("bert", 0),
                    ("_input_", "token_ent_ids"),
                    ("_input_", "token_static_ent_ids"),
                    ("_input_", "token_type_ent_ids"),
                    ("_input_", "token_rel_ent_ids"),
                    ("_input_", "token_masks"),
                ],
            },
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [("encoder", 1)],
            },
        ],
        loss_func=partial(ce_loss, f"{task_name}_pred_head"),
        output_func=partial(output, f"{task_name}_pred_head"),
        scorer=Scorer(customize_metric_funcs={"tacred_scorer": tacred_scorer}),
    )

    return task
