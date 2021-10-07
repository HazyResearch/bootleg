"""Model utils."""
import logging

import torch
from emmental import Meta

from bootleg import log_rank_0_debug

logger = logging.getLogger(__name__)


def move_to_device(model):
    """Move model to specific device.

    To work with index tensor (that may or
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
    """Count the number of parameters.

    Args:
        model: model to count
        requires_grad: whether to look at grad or no grad params
        logger: logger
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
    """Return LR of optimizer.

    Args:
        optimizer: optimizer

    Returns: learning rate value
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_max_candidates(entity_symbols, data_config):
    """
    Get max candidates.

    Returns the maximum number of candidates used in the model, taking into
    account train_in_candidates If train_in_canddiates is False, we add a NC
    entity candidate (for null candidate)

    Args:
        entity_symbols: entity symbols
        data_config: data config
    """
    return entity_symbols.max_candidates + int(not data_config.train_in_candidates)
