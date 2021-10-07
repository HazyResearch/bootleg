"""Model utils."""
import logging

import torch
import torch.nn.functional as F
from emmental import Meta

from bootleg import log_rank_0_debug

logger = logging.getLogger(__name__)


def move_to_device(model):
    """Move model to specific device. To work with index tensor (that may or
    may not be registered buffers), we return the model.

    Args:
        model: model to move

    Returns: model on the device.
    """
    if Meta.config["model_config"]["device"] != -1:
        if torch.cuda.is_available():
            device = (
                f"cuda:{Meta.config['model_config']['device']}"
                if isinstance(Meta.config["model_config"]["device"], int)
                else Meta.config["model_config"]["device"]
            )
            model = model.to(torch.device(device))
    return model


def count_parameters(model, requires_grad, logger):
    """Counts the number of parameters, printing along the way, with
    param.required_grad == requires_grad.

    Args:
        model: model to count
        requires_grad: whether to look at grad or no grad params
        logger: logger

    Returns:
    """
    for p in [
        p for p in model.named_parameters() if p[1].requires_grad is requires_grad
    ]:
        log_rank_0_debug(
            logger,
            "{:s} {:d} {:.2f} MB".format(
                p[0], p[1].numel(), p[1].numel() * 4 / 1024 ** 2
            ),
        )
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad is requires_grad
    )


def get_lr(optimizer):
    """Returns LR of optimizer.

    Args:
        optimizer: optimizer

    Returns: learning rate value
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def select_alias_word_sent(alias_pos_in_sent, sent_embedding):
    """Given a tensor of positions (batch x M), we subselect those embeddings
    in the sentence embeddings. Requires that max(alias_pos_in_sent) <
    sent_embedding.shape[1]

    Args:
        alias_pos_in_sent: position tensor (batch x M)
        sent_embedding: sentence embedding to extract the position embedding (batch x seq_len x hidden_dim)

    Returns:
        tensor where item [b, i] is a hidden_dim sized vector from sent_embedding[b][alias_pos_in_sent[i]]
    """
    # get alias words from sent embedding
    # batch x seq_len x hidden_size -> batch x M x hidden_size
    batch_size, M = alias_pos_in_sent.shape
    _, seq_len, hidden_size = sent_embedding.shape

    # expand so we can use gather
    sent_tensor = sent_embedding.unsqueeze(1).expand(
        batch_size, M, seq_len, hidden_size
    )
    # gather can't take negative values so we set them to the first word in the sequence
    # we mask these out later
    alias_idx_sent_mask = alias_pos_in_sent == -1
    # copy the alias_pos_in_sent tensor to avoid overwrite errors and set where alias_post_in_sent == -1 to be 0;
    # gather can't handle -1 indices
    alias_pos_in_sent_cpy = torch.where(
        alias_pos_in_sent == -1, torch.zeros_like(alias_pos_in_sent), alias_pos_in_sent
    )
    alias_word_tensor = torch.gather(
        sent_tensor,
        2,
        alias_pos_in_sent_cpy.long()
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(batch_size, M, 1, hidden_size),
    ).squeeze(2)
    # mask embedding values
    alias_word_tensor[alias_idx_sent_mask] = 0
    return alias_word_tensor


# Mask of True means keep
def selective_avg(mask, embeds):
    """
    Averages the embeddings along the second last dimension (dim = -2), respecting the mask.
    Mask of True means KEEP in average.

    Args:
        mask: mask (batch x M)
        embeds: tensor to average (batch x M x H)

    Returns: average tensor (batch x H) where rows s.t. mask was False were excluded from calculation

    """
    num_valid = mask.sum(-1)
    # replace zero by one to avoid divide by zero errors
    num_valid = torch.where(num_valid == 0, torch.ones_like(num_valid), num_valid)
    # mask out the padded values before sum
    embeds_masked = torch.where(mask.unsqueeze(-1), embeds, torch.zeros_like(embeds))
    total_k = embeds_masked.sum(-2)
    average_val = total_k / num_valid.unsqueeze(-1)
    return average_val


def normalize_matrix(mat, dim=3, p=2):
    """Normalize matrix.

    Args:
        mat: matrix
        dim: dimension
        p: p value for norm

    Returns: normalized matrix
    """
    mat_divided = F.normalize(mat, p=p, dim=dim)
    return mat_divided


def get_max_candidates(entity_symbols, data_config):
    """Returns the maximum number of candidates used in the model, taking into
    account train_in_candidates If train_in_canddiates is False, we add a NC
    entity candidate (for null candidate)

    Args:
        entity_symbols: entity symbols
        data_config: data config

    Returns:
    """
    return entity_symbols.max_candidates + int(not data_config.train_in_candidates)
