"""Print functions for distributed computation."""
import torch


def log_rank_0_info(logger, message):
    """If distributed is initialized log info only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)


def log_rank_0_debug(logger, message):
    """If distributed is initialized log debug only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.debug(message)
    else:
        logger.debug(message)
