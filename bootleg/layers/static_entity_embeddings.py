"""Entity embeddings."""
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EntityEmbedding(torch.nn.Module):
    """Static entity embeddings class.

    Args:
        entity_emb_file: numpy file of entity embeddings
    """

    def __init__(self, entity_emb_file):
        super(EntityEmbedding, self).__init__()
        embs = torch.FloatTensor(np.load(entity_emb_file))
        # Add -1 padding row; not required as dump from Bootleg should include PAD entity but as a safety
        embs = torch.cat([embs, torch.zeros(1, embs.shape[-1])], dim=0)
        self.embeddings = torch.nn.Embedding.from_pretrained(embs, padding_idx=-1)

    def forward(self, entity_cand_eid):
        """Model forward.

        Args:
            entity_cand_eid:  entity candidate EIDs (B x M x K)

        Returns: B x M x K x dim tensor of entity embeddings
        """
        return self.embeddings(entity_cand_eid)
