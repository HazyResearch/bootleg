import logging

import ujson
from torch import nn

from bootleg import log_rank_0_debug, log_rank_0_info
from bootleg.symbols.constants import (
    BERT_MODEL_NAME,
    DROPOUT_1D,
    DROPOUT_2D,
    FREEZE,
    NORMALIZE,
    SEND_THROUGH_BERT,
)
from bootleg.utils import embedding_utils
from bootleg.utils.utils import import_class

logger = logging.getLogger(__name__)


def get_embedding_tasks(args, entity_symbols):
    """Returns the embedding query task flows for each embedding in
    data_config.ent_embeddings.

    Args:
        args: args
        entity_symbols: entity symbols

    Returns: Emmental task flow,
             Emmental module pool,
             Emmental module device dict,
             embeddings that get added to BERT encoder,
             task flow inputs to embedding payload,
             total embedding sizes,
    """
    task_flow = []
    embedding_payload_inputs = []
    # These will be added to our bert module so it can handle their forward() and call to BERT
    extra_bert_embedding_layers = []
    num_through_bert_embeddings = 0
    module_pool = nn.ModuleDict()
    module_device_dict = {}
    total_sizes = {}

    # Entity Embedding
    log_rank_0_info(logger, "Loading embeddings...")
    for emb in args.data_config.ent_embeddings:
        (
            cpu,
            dropout1d_perc,
            dropout2d_perc,
            emb_args,
            freeze,
            normalize,
            through_bert,
        ) = embedding_utils.get_embedding_args(emb)
        # print out values like they do for BERT configs
        to_print = {
            "load_class": emb.load_class,
            "key": emb.key,
            "cpu": cpu,
            FREEZE: freeze,
            DROPOUT_1D: dropout1d_perc,
            DROPOUT_2D: dropout2d_perc,
            NORMALIZE: normalize,
            # Whether the output of this embedding is an index that needs a BERT forward pass (e.g., title embedding)
            SEND_THROUGH_BERT: through_bert,
        }
        log_rank_0_debug(
            logger,
            f'Embedding "{emb.key}" has params (can be changed in the config)\n{ujson.dumps(to_print, indent=4)}',
        )

        mod, load_class = import_class("bootleg.embeddings", emb.load_class)

        # Create object for each embedding specified in the config
        emb_obj = getattr(mod, load_class)(
            main_args=args,
            emb_args=emb_args,
            entity_symbols=entity_symbols,
            key=emb.key,
            cpu=cpu,
            normalize=normalize,
            dropout1d_perc=dropout1d_perc,
            dropout2d_perc=dropout2d_perc,
        )
        if freeze:
            log_rank_0_info(logger, f"FREEZING {emb.key}")
            emb_obj.freeze_params()

        assert emb.key not in total_sizes, f"You have {emb.key} used more than once"
        total_sizes[emb.key] = emb_obj.get_dim()

        # Add in the embeddings that output indices for BERT. Due to how DDP handles multiple calls through one module,
        # we must wrap all forward passes through BERT inside one module. Therefore, we do not add these embeddings
        # to the standard module_pool and task_flow but instead will add them to our BertEncoder.
        # This encoder will then call the forward() for the emb_obj and call postprocess_embedding for
        # any postprocessing.
        if through_bert:
            # The BertEncoder will output the through bert embeddings after it's standard 2 outputs
            # of sentence embedding and sentence mask
            bert_output_for_payload = (BERT_MODEL_NAME, 2 + num_through_bert_embeddings)
            num_through_bert_embeddings += 1
            embedding_payload_inputs.append(bert_output_for_payload)
            extra_bert_embedding_layers.append(emb_obj)
        else:
            if cpu:
                module_device_dict[emb.key] = -1
            module_pool[emb.key] = emb_obj
            task_flow.append(
                {
                    "name": f"embedding_{emb.key}",
                    "module": emb.key,
                    "inputs": [
                        ("_input_", "entity_cand_eid"),
                        (
                            "_input_",
                            "batch_on_the_fly_kg_adj",
                        ),  # special kg adjacency embedding prepped in dataloader
                    ],
                }
            )
            embedding_payload_inputs.append((task_flow[-1]["name"], 0))

    return (
        task_flow,
        module_pool,
        module_device_dict,
        extra_bert_embedding_layers,
        embedding_payload_inputs,
        total_sizes,
    )
