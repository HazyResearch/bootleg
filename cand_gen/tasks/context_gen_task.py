from torch import nn
from transformers import AutoModel

from bootleg.layers.bert_encoder import Encoder
from cand_gen.task_config import CANDGEN_TASK
from emmental.scorer import Scorer
from emmental.task import EmmentalTask


def context_output_func(intermediate_output_dict):
    return intermediate_output_dict["context_encoder"][0]


def create_task(args, len_context_tok):
    """Returns an EmmentalTask for a forward pass through the entity encoder only.

    Args:
        args: args
        len_context_tok: number of tokens in the tokenizer

    Returns: EmmentalTask for entity embedding extraction
    """

    # Create sentence encoder
    context_model = AutoModel.from_pretrained(
        args.data_config.word_embedding.bert_model
    )
    context_model.encoder.layer = context_model.encoder.layer[
        : args.data_config.word_embedding.context_layers
    ]
    context_model.resize_token_embeddings(len_context_tok)
    context_model = Encoder(context_model, args.model_config.hidden_size)

    # Create module pool and combine with embedding module pool
    module_pool = nn.ModuleDict(
        {
            "context_encoder": context_model,
        }
    )

    # Create task flow
    task_flow = [
        {
            "name": "context_encoder",
            "module": "context_encoder",
            "inputs": [
                ("_input_", "input_ids"),
                ("_input_", "token_type_ids"),
                ("_input_", "attention_mask"),
            ],
        },
    ]

    return EmmentalTask(
        name=CANDGEN_TASK,
        module_pool=module_pool,
        task_flow=task_flow,
        loss_func=None,
        output_func=context_output_func,
        require_prob_for_eval=False,
        require_pred_for_eval=True,
        scorer=Scorer(),
    )
