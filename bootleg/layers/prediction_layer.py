"""Prediction heads."""
import logging

import torch

from bootleg.layers.helper_modules import MLP
from bootleg.symbols.constants import DISAMBIG, FINAL_LOSS
from bootleg.utils import model_utils

logger = logging.getLogger(__name__)


class PredictionLayer(torch.nn.Module):
    """Prediction layer.

    Generates batch x M x K matrix of scores for each mention and
    candidate to be passed to scorer.

    Args:
        args: args
    """

    def __init__(self, args):
        super(PredictionLayer, self).__init__()
        self.hidden_size = args.model_config.hidden_size
        self.num_fc_layers = args.model_config.num_fc_layers
        self.dropout = args.train_config.dropout
        self.prediction_head = MLP(
            self.hidden_size, self.hidden_size, 1, 1, self.dropout
        )

    def forward(self, out_dict, context_matrix_dict, final_score=None):
        """Model forward. Must pass the self.training bool forward to loss
        function.

        Args:
            out_dict: Dict of intermediate scores (B x M x K)
            context_matrix_dict: Dict of output embedding matrices, e.g., from KG modules (B x M x K x H)
            final_score: Final output scores (B x M x K) (default None)

        Returns: Dict of Dict with final scores added (B x M x K), final output embedding (B x M x K x H),
                tensor of is_training Bool
        """
        score = model_utils.max_score_context_matrix(
            context_matrix_dict, self.prediction_head
        )
        out_dict[DISAMBIG][FINAL_LOSS] = score
        if "context_matrix_main" not in context_matrix_dict:
            context_matrix_dict[
                "context_matrix_main"
            ] = model_utils.generate_final_context_matrix(
                context_matrix_dict, ending_key_to_exclude="_nokg"
            )
        final_entity_embs = context_matrix_dict["context_matrix_main"]
        # Must make the self.training bool a tensor that is required for the loss to be gatherable for DP
        return {
            "final_scores": out_dict,
            "ent_embs": final_entity_embs,
            "training": (
                torch.tensor([1], device=final_entity_embs.device) * self.training
            ).bool(),
        }


class NoopPredictionLayer(torch.nn.Module):
    """Noop prediciton layer.

    Used for BERT NED base model. Must pass the self.training bool forward to loss function.

    Args:
        args: args
    """

    def __init__(self, args):
        super(NoopPredictionLayer, self).__init__()

    def forward(self, out_dict, context_matrix_dict, final_score):
        """Model forward. Must pass the self.training bool forward to loss
        function.

        Args:
            out_dict: Dict of intermediate scores (B x M x K)
            context_matrix_dict: Dict of output embedding matrices, e.g., from KG modules (B x M x K x H)
            final_score: Final output scores (B x M x K) (default None)

        Returns: Dict of Dict with final scores added (B x M x K), final output embedding (B x M x K x H),
                 tensor of is_training Bool
        """
        assert len(context_matrix_dict.values()) == 1
        final_entity_embs = list(context_matrix_dict.values())[0]
        out_dict[DISAMBIG][FINAL_LOSS] = final_score
        return {
            "final_scores": out_dict,
            "ent_embs": final_entity_embs,
            "training": (
                torch.tensor([1], device=final_entity_embs.device) * self.training
            ).bool(),
        }
