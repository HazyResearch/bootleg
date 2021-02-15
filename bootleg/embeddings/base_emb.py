"""Base embeddings."""
import logging
from typing import Dict, List

import torch
import torch.nn as nn

from bootleg import log_rank_0_info
from bootleg.utils import model_utils

logger = logging.getLogger(__name__)


class EntityEmb(nn.Module):
    """Base embedding that all embedding classes extend.

    If an embedding should be preprocessed in prep phase or in the __get_item__ of the dataset, it must

    - define self.kg_adj_process_func in the init
    - set batch_on_the_fly: True in config to be processed in prep
    - access the prepped embedding in the forward by batch_on_the_fly_data by the embedding key

    Args:
        main_args: main args
        emb_args: specific embedding args
        entity_symbols: entity symbols
        key: unique embedding key
        cpu: bool of if one cpu or not
        normalize: bool if normalize embeddings or not
        dropout1d_perc: 1D dropout percent
        dropout2d_perc: 2D dropout percent
    """

    def __init__(
        self,
        main_args,
        emb_args,
        entity_symbols,
        key,
        cpu,
        normalize,
        dropout1d_perc,
        dropout2d_perc,
    ):
        super(EntityEmb, self).__init__()
        self.key = key
        self.cpu = cpu
        self.normalize = normalize
        self.dropout1d_perc = dropout1d_perc
        self.dropout2d_perc = dropout2d_perc
        assert not (
            self.dropout1d_perc > 0 and self.dropout2d_perc > 0
        ), f"You have both 1D and 2D dropout set to be > 0. You can only have one."
        self.from_pretrained = (
            main_args.model_config.model_path is not None
            and len(main_args.model_config.model_path) > 0
        )

    def forward(
        self,
        entity_cand_eid: torch.LongTensor,
        batch_on_the_fly_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Model forward.

        Args:
            entity_cand_eid:  entity candidate EIDs (B x M x K)
            batch_on_the_fly_data: dict of batch on the fly embeddings

        Returns: B x M x K x dim embedding
        """
        raise ValueError("Not implemented")

    def get_dim(self):
        """Gets the output dimension of the embedding."""
        raise ValueError("Not implemented")

    def get_key(self):
        """Gets the unique key of the embedding."""
        return self.key

    def normalize_and_dropout_emb(self, embedding: torch.Tensor) -> torch.Tensor:
        """Whether to normalize and dropout embedding.

        Args:
            embedding: embedding

        Returns: adjusted embedding
        """
        if self.dropout1d_perc > 0:
            embedding = model_utils.emb_1d_dropout(
                self.training, self.dropout1d_perc, embedding
            )
        elif self.dropout2d_perc > 0:
            embedding = model_utils.emb_2d_dropout(
                self.training, self.dropout2d_perc, embedding
            )
        # We enforce that self.normalize is instantiated inside each subclass
        if self.normalize is True:
            embedding = model_utils.normalize_matrix(embedding, dim=-1)
        return embedding

    def freeze_params(self):
        """Freezes the parameters of the module.

        Returns:
        """
        for name, param in self.named_parameters():
            param.requires_grad = False
            log_rank_0_info(logger, f"Freezing {name}")
        return

    def unfreeze_params(self):
        """Unfreezes the parameters of the module.

        Returns:
        """
        for name, param in self.named_parameters():
            param.requires_grad = True
            log_rank_0_info(logger, f"Unfreezing {name}")
        return
