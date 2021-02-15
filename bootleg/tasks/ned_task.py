import logging
import os

import torch
from torch import nn

from bootleg.layers.attn_networks import BERTNED, Bootleg, BootlegM2E
from bootleg.layers.bert_encoder import BertEncoder
from bootleg.layers.embedding_payload import EmbeddingPayload, EmbeddingPayloadBase
from bootleg.layers.prediction_layer import NoopPredictionLayer, PredictionLayer
from bootleg.scorer import BootlegSlicedScorer
from bootleg.symbols.constants import BERT_MODEL_NAME, DISAMBIG, FINAL_LOSS, PRED_LAYER
from bootleg.symbols.entity_symbols import EntitySymbols
from bootleg.task_config import NED_TASK
from bootleg.tasks.task_getters import get_embedding_tasks
from bootleg.utils import eval_utils
from emmental.scorer import Scorer
from emmental.task import EmmentalTask


def disambig_output(intermediate_output_dict):
    """Function to return the probs for a task in Emmental.

    Args:
        intermediate_output_dict: output dict from Emmental task flow

    Returns: NED probabilities for candidates (B x M x K)
    """
    out = intermediate_output_dict[PRED_LAYER]["final_scores"][DISAMBIG][FINAL_LOSS]
    mask = intermediate_output_dict["_input_"]["entity_cand_eid_mask"]
    return eval_utils.masked_class_logsoftmax(pred=out, mask=~mask).exp()


def disambig_loss(intermediate_output_dict, Y, active):
    """Returns the entity disambiguation loss on prediction heads.

    Args:
        intermediate_output_dict: output dict from the Emmental task flor
        Y: gold labels
        active: whether examples are "active" or not (used in Emmental slicing)

    Returns: loss
    """
    # Grab the first value of training (when doing distributed training, we will have one per process)
    training = intermediate_output_dict[PRED_LAYER]["training"][0].item()
    assert type(training) is bool
    outs = intermediate_output_dict[PRED_LAYER]["final_scores"][DISAMBIG]
    mask = intermediate_output_dict["_input_"]["entity_cand_eid_mask"][active]
    labels = Y[active]
    # During eval, even if our model does not predict a NIC candidate, we allow for a NIC gold QID
    # This qid gets assigned the label of -2 and is always incorrect
    # As NLLLoss assumes classes of 0 to #classes-1 except for pad idx, we manually mask
    # the -2 labels for the loss computation only. As this is just for eval, it won't matter.
    masked_labels = labels
    if not training:
        label_mask = labels == -2
        masked_labels = torch.where(~label_mask, labels, torch.ones_like(labels) * -1)

    temp = 0
    for out in outs.values():
        # batch x M x K -> transpose -> swap K classes with M spans for "k-dimensional" NLLloss
        log_probs = eval_utils.masked_class_logsoftmax(
            pred=out[active], mask=~mask
        ).transpose(1, 2)
        temp += nn.NLLLoss(ignore_index=-1)(log_probs, masked_labels.long())

    return temp


def create_task(args, entity_symbols=None, slice_datasets=None):
    """Returns an EmmentalTask for named entity disambiguation (NED).

    Args:
        args: args
        entity_symbols: entity symbols (default None)
        slice_datasets: slice datasets used in scorer (default None)

    Returns: EmmentalTask for NED
    """

    if entity_symbols is None:
        entity_symbols = EntitySymbols(
            load_dir=os.path.join(
                args.data_config.entity_dir, args.data_config.entity_map_dir
            ),
            alias_cand_map_file=args.data_config.alias_cand_map,
        )

    # Create sentence encoder
    bert_model = BertEncoder(
        args.data_config.word_embedding, output_size=args.model_config.hidden_size
    )

    # Gets the tasks that query for the individual embeddings (e.g., word, entity, type, kg)
    # The device dict will store which embedding modules we want on the cpu
    (
        embedding_task_flows,  # task flows for standard embeddings (e.g., kg, type, entity)
        embedding_module_pool,  # module for standard embeddings
        embedding_module_device_dict,  # module device dict for standard embeddings
        extra_bert_embedding_layers,  # some embeddings output indices for BERT so we handle these embeddings in our BERT layer (see comments in get_through_bert_embedding_tasks)
        embedding_payload_inputs,  # the layers that are fed into the payload
        embedding_total_sizes,  # total size of all embeddings
    ) = get_embedding_tasks(args, entity_symbols)

    # Add the extra embedding layers to BERT module
    for emb_obj in extra_bert_embedding_layers:
        bert_model.add_embedding(emb_obj)

    # Create the embedding payload, attention network, and prediction layer modules
    if args.model_config.attn_class == "BootlegM2E":
        embedding_payload = EmbeddingPayload(
            args, entity_symbols, embedding_total_sizes
        )
        attn_network = BootlegM2E(args, entity_symbols)
        pred_layer = PredictionLayer(args)

    elif args.model_config.attn_class == "Bootleg":
        embedding_payload = EmbeddingPayload(
            args, entity_symbols, embedding_total_sizes
        )
        attn_network = Bootleg(args, entity_symbols)
        pred_layer = PredictionLayer(args)

    elif args.model_config.attn_class == "BERTNED":
        # Baseline model
        embedding_payload = EmbeddingPayloadBase(
            args, entity_symbols, embedding_total_sizes
        )
        attn_network = BERTNED(args, entity_symbols)
        pred_layer = NoopPredictionLayer(args)

    else:
        raise ValueError(f"{args.model_config.attn_class} is not supported.")

    sliced_scorer = BootlegSlicedScorer(
        args.data_config.train_in_candidates, slice_datasets
    )

    # Create module pool and combine with embedding module pool
    module_pool = nn.ModuleDict(
        {
            BERT_MODEL_NAME: bert_model,
            "embedding_payload": embedding_payload,
            "attn_network": attn_network,
            PRED_LAYER: pred_layer,
        }
    )
    module_pool.update(embedding_module_pool)

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
        *embedding_task_flows,  # Add task flows to create embedding inputs
        {
            "name": "embedding_payload",
            "module": "embedding_payload",  # outputs: embedding_tensor
            "inputs": [
                ("_input_", "start_span_idx"),
                ("_input_", "end_span_idx"),
                *embedding_payload_inputs,  # all embeddings
            ],
        },
        {
            "name": "attn_network",
            "module": "attn_network",  # output: predictions from layers, output entity embeddings
            "inputs": [
                (BERT_MODEL_NAME, 0),  # sentence embedding
                (BERT_MODEL_NAME, 1),  # sentence embedding mask
                ("embedding_payload", 0),
                ("_input_", "entity_cand_eid_mask"),
                ("_input_", "start_span_idx"),
                ("_input_", "end_span_idx"),
                (
                    "_input_",
                    "batch_on_the_fly_kg_adj",
                ),  # special kg adjacency embedding prepped in dataloader
            ],
        },
        {
            "name": PRED_LAYER,
            "module": PRED_LAYER,
            "inputs": [
                (
                    "attn_network",
                    "intermed_scores",
                ),  # output predictions from intermediate layers from the model
                (
                    "attn_network",
                    "ent_embs",
                ),  # output entity embeddings (from all KG modules)
                (
                    "attn_network",
                    "final_scores",
                ),  # score (empty except for baseline model)
            ],
        },
    ]

    return EmmentalTask(
        name=NED_TASK,
        module_pool=module_pool,
        task_flow=task_flow,
        loss_func=disambig_loss,
        output_func=disambig_output,
        require_prob_for_eval=False,
        require_pred_for_eval=True,
        # action_outputs are used to stitch together sentence fragments
        action_outputs=[
            ("_input_", "sent_idx"),
            ("_input_", "subsent_idx"),
            ("_input_", "alias_orig_list_pos"),
            ("_input_", "for_dump_gold_cand_K_idx_train"),
            (PRED_LAYER, "ent_embs"),  # entity embeddings
        ],
        scorer=Scorer(
            customize_metric_funcs={f"{NED_TASK}_scorer": sliced_scorer.bootleg_score}
        ),
        module_device=embedding_module_device_dict,
    )
