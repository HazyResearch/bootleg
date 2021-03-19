"""Model utils."""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from bootleg import log_rank_0_debug
from bootleg.symbols.constants import FINAL_LOSS
from emmental import Meta

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


def emb_1d_dropout(training, mask_perc, tensor):
    """Standard 1D dropout of tensor.

    Args:
        training: if the model is in train mode or not
        mask_perc: percent to dropout
        tensor: tensor to dropout

    Returns: tensor after 1D dropout
    """
    return nn.functional.dropout(tensor, p=mask_perc, training=training)


# Masks an entire row of the embedding (2d mask). Each row gets masked with probability mask_perc.
def emb_2d_dropout(training, mask_perc, tensor):
    """2D dropout of tensor where entire row gets zero-ed out with prob
    mask_perc.

    Args:
        training: if the model is in train mode or not
        mask_perc: percent to dropout
        tensor: tensor to dropout

    Returns: tensor after 2D dropout
    """
    batch, M, K, dim = tensor.shape
    if training and mask_perc > 0:
        # reshape for masking
        tensor = tensor.contiguous().reshape(batch * M * K, dim)
        # randomly mask each entity embedding
        bern_prob = (torch.ones(batch * M * K, 1) * mask_perc).to(tensor.device)
        zero_mask = torch.bernoulli(bern_prob) > 0
        tensor = tensor.masked_fill(zero_mask, 0)
        tensor = tensor.contiguous().reshape(batch, M, K, dim)
    return tensor


#
# This tensor is the probability of something being masked
def emb_dropout_by_tensor(training, regularization_tensor, tensor):
    """This applied 2D dropout each row of an embedding matrix (type, kg, ...)
    based on the weights in the regularization tensor.

    Args:
        training: if the model is in train mode or not
        mask_perc: tensor of dropout weights with shape (*)
        tensor: tensor to dropout with shape (*, dim)

    Returns: tensor after dropout
    """
    assert list(regularization_tensor.size()) == list(
        tensor.size()[:-1]
    ), f"{regularization_tensor.size()} should be the same size as {tensor.size()[:-1]}"
    if training:
        # randomly mask each entity embedding
        zero_mask = (torch.bernoulli(regularization_tensor) > 0).unsqueeze(-1)
        tensor = tensor.masked_fill(zero_mask, 0)
    return tensor


def max_score_context_matrix(context_matrix_dict, prediction_head):
    """For each context matrix value in a dict of matrices, project to batch x
    M x K and take max as final score.

    Args:
        context_matrix_dict: Dict of batch x M x K x H matrices of embeddings for each candidate
        prediction_head: projection layer that transforms each matrix to batch x M x K for scoring

    Return:
        batch x M x K tensor with the max score along along the -1 dimension for each batch and mention
    """
    batch_size, M, K, H = list(context_matrix_dict.values())[0].shape
    preds_to_cat = []
    for key in context_matrix_dict:
        pred = prediction_head(context_matrix_dict[key])
        pred = pred.squeeze(2).reshape(batch_size, M, K)
        preds_to_cat.append(pred.unsqueeze(3))
    score = torch.max(torch.cat(preds_to_cat, dim=-1), dim=-1)[0]
    return score


def generate_final_context_matrix(context_matrix_dict, ending_key_to_exclude="_nokg"):
    """Takes the average of the context matrices where their key does not end
    in ending_key_to_exclude.

    Args:
        context_matrix_dict: Dict of batch x M x K x H matrices of embeddings for each candidate
        ending_key_to_exclude: key to exclude from average

    Returns:
        tensor average of call tensors in context_matrix_dict unless key ends in ending_key_to_exclude
    """
    new_ctx = []
    for key, val in context_matrix_dict.items():
        if not key.endswith(ending_key_to_exclude):
            new_ctx.append(val)
    assert len(new_ctx) > 0, (
        f"You have provided a context matrix dict with only keys ending with _nokg. We average the context matrices "
        f"that do not end in _nokg as the final context matrix. Please rename the final matrix."
    )
    return torch.sum(torch.stack(new_ctx), dim=0) / len(new_ctx)


def init_embeddings_to_vec(module, pad_idx, vec=None):
    """Initializes module of nn.Embedding to have the value specified in vec.

    Args:
        module: nn.Embedding module to initialize
        pad_idx: pad index
        vec: vector to intialize (will randomly generate if None)

    Returns: the vector to initialize
    """
    assert (
        pad_idx == 0 or pad_idx == -1
    ), f"Only accept pads of 0 or -1; you gave {pad_idx}"
    assert isinstance(module, (nn.Embedding))
    embedding_dim = module.embedding_dim

    if vec is None:
        # Follows how nn.Embedding intializes their weights
        vec = torch.Tensor(1, embedding_dim)
        init.normal_(vec)
    # Copy the pad row via clone
    pad_row = module.weight.data[pad_idx].clone()
    test_equal = False
    if not torch.equal(pad_row, vec):
        test_equal = True
    module.weight.data[:] = vec
    module.weight.data[pad_idx] = pad_row
    # Assert the pad row is different from the vec
    if test_equal:
        assert not torch.equal(pad_row, vec)
    # We want the pad row to stay the same as it was before (i.e., all zeros) and not become a tail embedding
    assert torch.equal(pad_row, module.weight.data[pad_idx][:])
    return vec


def get_stage_head_name(layer_idx):
    """Wrapper function for giving a name to each layer in the model. These are
    used as keys to store intermediate outputs.

    Args:
        layer_idx: layer index

    Returns: name of layer
    """
    return f"{FINAL_LOSS}_stage_{layer_idx}"
