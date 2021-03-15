import logging
import os
from functools import partial

import torch.nn.functional as F
from torch import nn

from bootleg.layers.bert_encoder import BertEncoder
from bootleg.layers.mention_type_prediction import TypePred
from bootleg.scorer import BootlegSlicedScorer
from bootleg.symbols.constants import BERT_MODEL_NAME
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.task_config import NED_TASK, TYPE_PRED_TASK
from bootleg.utils import embedding_utils
from emmental.scorer import Scorer
from emmental.task import EmmentalTask


def type_output(module_name, intermediate_output_dict):
    """Returns the probabilities output for type prediction.

    Args:
        module_name: module name (will be the type predcition task)
        intermediate_output_dict: intermediate Emmental task flow outputs

    Returns: probabilities for each type (B x M x num_types)
    """
    out = intermediate_output_dict[module_name][1]
    return F.softmax(out, dim=2)


def type_loss(module_name, intermediate_output_dict, Y, active):
    """Returns the type prediction loss.

    Args:
        module_name: module name (will be the type predcition task)
        intermediate_output_dict: intermediate Emmental task flow outputs
        Y: gold labels
        active: which examples are "active" (used in Emmental slicing)

    Returns: loss
    """
    out = intermediate_output_dict[module_name][1][active]
    batch_size, M, num_types = out.shape
    # just take the positive label
    labels = Y[active].reshape(batch_size * M)
    input = out.reshape(batch_size * M, num_types)
    temp = F.cross_entropy(input, labels, ignore_index=-1)
    return temp


def update_ned_task(model):
    """Updates the NED task in the Emmental model to add the type prediction
    task dependency. In particular, the NED task takes in the mention type
    embedding from the type prediction task.

    Args:
        Emmental model with NED_task
    """
    assert NED_TASK in model.task_names

    type_pred_step = {
        "name": "type_prediction",
        "module": "type_prediction",  # output: embedding_dict, batch_type_pred
        "inputs": [
            (BERT_MODEL_NAME, 0),  # sentence embedding
            ("_input_", "start_span_idx"),
        ],
    }

    ned_task_flow = model.task_flows[NED_TASK]

    for step_idx, step in enumerate(ned_task_flow):
        if step["name"] == "embedding_payload":
            payload_step_idx = step_idx

            # Update the inputs to the embedding payload to include the type prediction output
            step["inputs"].append(("type_prediction", 0))

    # Add the type prediction step prior to the embedding payload
    ned_task_flow.insert(payload_step_idx, type_pred_step)

    model.task_flows[NED_TASK] = ned_task_flow


def create_task(args, entity_symbols=None, slice_datasets=None):
    """Creates a type prediction task.

    Args:
        args: args
        entity_symbols: entity symbols
        slice_datasets: slice datasets used in scorer (default None)

    Returns: EmmentalTask for type prediction
    """
    if entity_symbols is None:
        entity_symbols = EntitySymbols.load_from_cache(
            load_dir=os.path.join(
                args.data_config.entity_dir, args.data_config.entity_map_dir
            ),
            alias_cand_map_file=args.data_config.alias_cand_map,
            alias_idx_file=args.data_config.alias_idx_map,
        )

    # Create sentence encoder
    bert_model = BertEncoder(
        args.data_config.word_embedding, output_size=args.model_config.hidden_size
    )

    # Create type prediction module
    # Add 1 for pad type
    type_prediction = TypePred(
        args.model_config.hidden_size,
        args.data_config.type_prediction.dim,
        args.data_config.type_prediction.num_types + 1,
        embedding_utils.get_max_candidates(entity_symbols, args.data_config),
    )

    # Create scorer
    sliced_scorer = BootlegSlicedScorer(
        args.data_config.train_in_candidates, slice_datasets
    )

    # Create module pool
    # BERT model will be shared across tasks as long as the name matches
    module_pool = nn.ModuleDict(
        {BERT_MODEL_NAME: bert_model, "type_prediction": type_prediction}
    )

    # Create task flow
    task_flow = [
        {
            "name": BERT_MODEL_NAME,
            "module": BERT_MODEL_NAME,
            "inputs": [
                ("_input_", "entity_cand_eid"),
                ("_input_", "token_ids"),
            ],  # We pass the entity_cand_eids to BERT in case of embeddings that require word information
        },
        {
            "name": "type_prediction",
            "module": "type_prediction",  # output: embedding_dict, batch_type_pred
            "inputs": [
                (BERT_MODEL_NAME, 0),  # sentence embedding
                ("_input_", "start_span_idx"),
            ],
        },
    ]

    return EmmentalTask(
        name=TYPE_PRED_TASK,
        module_pool=module_pool,
        task_flow=task_flow,
        loss_func=partial(type_loss, "type_prediction"),
        output_func=partial(type_output, "type_prediction"),
        require_prob_for_eval=False,
        require_pred_for_eval=True,
        scorer=Scorer(
            customize_metric_funcs={
                f"{TYPE_PRED_TASK}_scorer": sliced_scorer.type_pred_score
            }
        ),
    )
