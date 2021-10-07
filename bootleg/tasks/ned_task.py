import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from bootleg.layers.bert_encoder import Encoder
from bootleg.scorer import BootlegSlicedScorer
from bootleg.task_config import NED_TASK
from bootleg.utils import eval_utils
from emmental.scorer import Scorer
from emmental.task import EmmentalTask


class DisambigLoss:
    def __init__(self, normalize, temperature):
        self.normalize = normalize
        self.temperature = temperature

    def disambig_output(self, intermediate_output_dict):
        """Function to return the probs for a task in Emmental.

        Args:
            intermediate_output_dict: output dict from Emmental task flow

        Returns: NED probabilities for candidates (B x M x K)
        """
        mask = intermediate_output_dict["_input_"]["entity_cand_eval_mask"]
        out = intermediate_output_dict["context_encoder"][0].unsqueeze(1)
        ent_out = intermediate_output_dict["entity_encoder"][0]
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
            ent_out = F.normalize(ent_out, p=2, dim=-1)
        pred = torch.bmm(out, ent_out.transpose(-2, -1))
        mask = mask.reshape(*pred.shape)
        ret = eval_utils.masked_class_logsoftmax(
            pred=pred, mask=~mask, temp=self.temperature
        ).squeeze(
            1
        )  # Squeeze single alias
        return ret.exp()

    def disambig_loss(self, intermediate_output_dict, Y, active):
        """Returns the entity disambiguation loss on prediction heads.

        Args:
            intermediate_output_dict: output dict from the Emmental task flor
            Y: gold labels
            active: whether examples are "active" or not (used in Emmental slicing)

        Returns: loss
        """
        # Grab the first value of training (when doing distributed training, we will have one per process)
        if len(intermediate_output_dict["context_encoder"][1].shape) <= 0:
            training = intermediate_output_dict["context_encoder"][1].item()
        else:
            training = intermediate_output_dict["context_encoder"][1][0].item()
        assert type(training) is bool
        mask = intermediate_output_dict["_input_"]["entity_cand_eval_mask"]
        out = intermediate_output_dict["context_encoder"][0].unsqueeze(1)
        ent_out = intermediate_output_dict["entity_encoder"][0]
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
            ent_out = F.normalize(ent_out, p=2, dim=-1)
        pred = torch.bmm(out, ent_out.transpose(-2, -1))
        mask = mask.reshape(*pred.shape)

        labels = Y
        # During eval, even if our model does not predict a NIC candidate, we allow for a NIC gold QID
        # This qid gets assigned the label of -2 and is always incorrect
        # As NLLLoss assumes classes of 0 to #classes-1 except for pad idx, we manually mask
        # the -2 labels for the loss computation only. As this is just for eval, it won't matter.
        masked_labels = labels
        if not training:
            label_mask = labels == -2
            masked_labels = torch.where(
                ~label_mask, labels, torch.ones_like(labels) * -1
            )
        log_probs = eval_utils.masked_class_logsoftmax(
            pred=pred, mask=~mask, temp=self.temperature
        ).squeeze(
            1
        )  # Squeeze single alias
        loss = nn.NLLLoss(ignore_index=-1)(log_probs, masked_labels.long())
        return loss

    def batch_cands_disambig_output(self, intermediate_output_dict):
        """Function to return the probs for a task in Emmental.

        Args:
            intermediate_output_dict: output dict from Emmental task flow
        Returns: NED probabilities for candidates (B x M x K)
        """
        out = intermediate_output_dict["context_encoder"][0]
        ent_out = intermediate_output_dict["entity_encoder"][0]
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
            ent_out = F.normalize(ent_out, p=2, dim=-1)
        score = torch.mm(out, ent_out.t()) / self.temperature
        return F.softmax(score, dim=-1)

    def batch_cands_disambig_loss(self, intermediate_output_dict, Y, active):
        """Returns the entity disambiguation loss on prediction heads.

        Args:
            intermediate_output_dict: output dict from the Emmental task flor
            Y: gold labels
            active: whether examples are "active" or not (used in Emmental slicing)
        Returns: loss
        """
        # Grab the first value of training (when doing distributed training, we will have one per process)
        training = intermediate_output_dict["context_encoder"][1].item()
        assert type(training) is bool
        out = intermediate_output_dict["context_encoder"][0]
        ent_out = intermediate_output_dict["entity_encoder"][0]
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
            ent_out = F.normalize(ent_out, p=2, dim=-1)
        score = torch.mm(out, ent_out.t()) / self.temperature

        labels = Y
        masked_labels = labels.reshape(out.shape[0])
        if not training:
            label_mask = labels == -2
            masked_labels = torch.where(
                ~label_mask, labels, torch.ones_like(labels) * -1
            )
            masked_labels = masked_labels.reshape(out.shape[0])
        temp = nn.CrossEntropyLoss(ignore_index=-1)(score, masked_labels.long())
        return temp


def create_task(args, use_batch_cands, len_context_tok, slice_datasets=None):
    """Returns an EmmentalTask for named entity disambiguation (NED).

    Args:
        args: args
        entity_symbols: entity symbols (default None)
        slice_datasets: slice datasets used in scorer (default None)

    Returns: EmmentalTask for NED
    """
    disamig_loss = DisambigLoss(
        args.model_config.normalize, args.model_config.temperature
    )
    output_func = disamig_loss.disambig_output
    loss_func = disamig_loss.disambig_loss
    if use_batch_cands:
        loss_func = disamig_loss.batch_cands_disambig_loss
        output_func = disamig_loss.batch_cands_disambig_output

    # Create sentence encoder
    context_model = AutoModel.from_pretrained(
        args.data_config.word_embedding.bert_model
    )
    context_model.encoder.layer = context_model.encoder.layer[
        : args.data_config.word_embedding.context_layers
    ]
    context_model.resize_token_embeddings(len_context_tok)
    context_model = Encoder(context_model, args.model_config.hidden_size)

    entity_model = AutoModel.from_pretrained(args.data_config.word_embedding.bert_model)
    entity_model.encoder.layer = entity_model.encoder.layer[
        : args.data_config.word_embedding.entity_layers
    ]
    entity_model.resize_token_embeddings(len_context_tok)
    entity_model = Encoder(entity_model, args.model_config.hidden_size)

    sliced_scorer = BootlegSlicedScorer(
        args.data_config.train_in_candidates, slice_datasets
    )

    # Create module pool and combine with embedding module pool
    module_pool = nn.ModuleDict(
        {
            "context_encoder": context_model,
            "entity_encoder": entity_model,
        }
    )

    # Create task flow
    task_flow = [
        {
            "name": "entity_encoder",
            "module": "entity_encoder",
            "inputs": [
                ("_input_", "entity_cand_input_ids"),
                ("_input_", "entity_cand_attention_mask"),
                ("_input_", "entity_cand_token_type_ids"),
            ],
        },
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
        name=NED_TASK,
        module_pool=module_pool,
        task_flow=task_flow,
        loss_func=loss_func,
        output_func=output_func,
        require_prob_for_eval=False,
        require_pred_for_eval=True,
        # action_outputs are used to stitch together sentence fragments
        action_outputs=[
            ("_input_", "sent_idx"),
            ("_input_", "subsent_idx"),
            ("_input_", "alias_orig_list_pos"),
            ("_input_", "for_dump_gold_cand_K_idx_train"),
            ("entity_encoder", 0),  # entity embeddings
        ],
        scorer=Scorer(
            customize_metric_funcs={f"{NED_TASK}_scorer": sliced_scorer.bootleg_score}
        ),
    )
