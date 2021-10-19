import torch.nn.functional as F
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from torch import nn
from transformers import AutoModel

from bootleg.layers.bert_encoder import Encoder
from cand_gen.task_config import CANDGEN_TASK


class EntityGenOutput:
    """Entity gen for output."""

    def __init__(self, normalize):
        """Entity gen for output initializer."""
        self.normalize = normalize

    def entity_output_func(self, intermediate_output_dict):
        """Entity output func."""
        ent_out = intermediate_output_dict["entity_encoder"][0]
        if self.normalize:
            ent_out = F.normalize(ent_out, p=2, dim=-1)
        return ent_out


def create_task(args, len_context_tok):
    """Returns an EmmentalTask for a forward pass through the entity encoder only.

    Args:
        args: args
        len_context_tok: number of tokens in the tokenizer

    Returns: EmmentalTask for entity embedding extraction
    """

    entity_model = AutoModel.from_pretrained(args.data_config.word_embedding.bert_model)
    entity_model.encoder.layer = entity_model.encoder.layer[
        : args.data_config.word_embedding.entity_layers
    ]
    entity_model.resize_token_embeddings(len_context_tok)
    entity_model = Encoder(entity_model, args.model_config.hidden_size)

    # Create module pool and combine with embedding module pool
    module_pool = nn.ModuleDict(
        {
            "entity_encoder": entity_model,
        }
    )

    # Create task flow
    task_flow = [
        {
            "name": "entity_encoder",
            "module": "entity_encoder",
            "inputs": [
                ("_input_", "entity_input_ids"),
                ("_input_", "entity_attention_mask"),
                ("_input_", "entity_token_type_ids"),
            ],
        }
    ]

    return EmmentalTask(
        name=CANDGEN_TASK,
        module_pool=module_pool,
        task_flow=task_flow,
        loss_func=None,
        output_func=EntityGenOutput(args.model_config.normalize).entity_output_func,
        require_prob_for_eval=False,
        require_pred_for_eval=True,
        scorer=Scorer(),
    )
