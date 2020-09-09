"""Model utils"""
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import logging

# Adds padding to a batch of tensors with pad up to pad_len
from bootleg.utils.utils import flatten


def pad_tensor(batch, pad, pad_len=None):
    assert type(batch) == list and len(batch) > 0  # non-empty batch
    assert all(len(x.size()) == 2 for x in batch)  # each element is a 2-dimensional tensor
    assert len(set([x.size(1) for x in batch])) == 1  # all elements have same hidden dimension
    assert pad_len == None or all(x.size(1) <= pad_len for x in batch)  # no tensor exceeds pad_len
    if pad_len:
        device = batch[0].get_device() if batch[0].is_cuda else 'cpu'
        batch.append(torch.zeros((pad_len, batch[0].size(1)), device=device, dtype=batch[0].dtype))
        return pad_sequence(batch, batch_first=True)[:-1]
    else:
        return pad_sequence(batch, batch_first=True)

# Mask a batch by mask value
# Useful if wanting to mask again after attentions
def mask(batch, mask, mask_value):
    batch[mask] = mask_value
    pass

def count_parameters(model, requires_grad, logger):
    for p in [p for p in model.named_parameters() if p[1].requires_grad is requires_grad]:
        logger.debug("{:s} {:d} {:.2f} MB".format(p[0], p[1].numel(), p[1].numel()*4/1024**2))
    return sum(p.numel() for p in model.parameters() if p.requires_grad is requires_grad)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def select_alias_word_sent(alias_idx_pair_sent, sent_embedding, index):
    alias_idx_sent = alias_idx_pair_sent[index]
    # get alias words from sent embedding
    # batch x seq_len x hidden_size -> batch x M x hidden_size
    batch_size, M = alias_idx_sent.shape
    _, seq_len, hidden_size = sent_embedding.tensor.shape

    # expand so we can use gather
    sent_tensor = sent_embedding.tensor.unsqueeze(1).expand(batch_size, M, seq_len, hidden_size)
    # gather can't take negative values so we set them to the first word in the sequence
    # we mask these out later
    alias_idx_sent_mask = alias_idx_sent == -1
    alias_idx_sent[alias_idx_sent_mask] = 0
    alias_word_tensor = torch.gather(sent_tensor, 2, alias_idx_sent.long().unsqueeze(-1).unsqueeze(-1).expand(
        batch_size, M, 1, hidden_size)).squeeze(2)
    alias_word_tensor[alias_idx_sent_mask] = 0
    return alias_word_tensor

# Mask of True means keep
def selective_avg(ids, mask, embeds):
    num_valid = mask.sum(-1)
    # replace zero by one to avoid divide by zero errors
    num_valid = torch.where(num_valid == 0, torch.ones_like(num_valid), num_valid)
    # mask out the padded values before sum
    embeds_masked = torch.where(mask.unsqueeze(-1), embeds, torch.zeros_like(embeds))
    total_k = embeds_masked.sum(-2)
    average_val = total_k / num_valid.unsqueeze(-1)
    return average_val

def normalize_matrix(mat, dim=3, p=2):
    mat_divided = F.normalize(mat, p=p, dim=dim)
    return mat_divided

# Masks an entire row of the embedding (2d mask). Each row gets masked with probability mask_perc.
def emb_2d_dropout(training, mask_perc, tensor):
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

# This masks each row of an embedding matrix (type, kg, ...) based on the weights in the regularization tensor
# This tensor is the probability of something being masked
def emb_dropout_by_tensor(training, regularization_tensor, tensor):
    assert list(regularization_tensor.size()) == list(tensor.size()[:-1]), f"{regularization_tensor.size()} should be the same size as {tensor.size()[:-1]}"
    if training:
        # randomly mask each entity embedding
        zero_mask = (torch.bernoulli(regularization_tensor) > 0).unsqueeze(-1)
        tensor = tensor.masked_fill(zero_mask, 0)
    return tensor

def remove_batched_masked_rows(embeds, unshortened_masks):
    assert len(unshortened_masks.size()) == 2, "<unshortened_masks> should have regular size Bx(M*K[*T]) for some M, K, T"
    assert len(embeds.size()) == 3
    assert unshortened_masks.size() == embeds.size()[0:2]
    unshortened_masks = ~(unshortened_masks.bool())  # In the input, False means "keep"
    lengths = unshortened_masks.sum(-1)
    max_length = max(lengths)
    embeds = [embed[mask] for mask, embed in zip(unshortened_masks, embeds)]
    embeds = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True)
    shortened_masks = torch.tensor([[False] * length + [True] * (max_length - length) for length in lengths], device=unshortened_masks.device)
    return embeds, shortened_masks


def add_batched_masked_rows(embeds, unshortened_masks):
    assert len(unshortened_masks.size()) == 2
    assert len(embeds.size()) == 3
    assert unshortened_masks.size(0) == embeds.size(0)
    assert unshortened_masks.size(1) >= embeds.size(1)
    unshortened_masks = ~(unshortened_masks.bool())
    batch_positions = torch.max(torch.tensor(0, device=unshortened_masks.device),
                          torch.cumsum(unshortened_masks.long(), dim=-1) - 1)
    unshortened_masks = unshortened_masks.unsqueeze(-1)
    embeds = [embed[positions] * mask for embed, positions, mask in zip(embeds, batch_positions, unshortened_masks)]
    embeds = torch.stack(embeds)
    return embeds


def partition(embeds, shortened_mask):
    lengths = shortened_mask.sum(-1)
    # Sort embeds (and masks) by increasing length
    lengths = lengths.sort(0)
    embeds = embeds[lengths.indices]
    shortened_mask = shortened_mask[lengths.indices]
    unpermute_idxs = lengths.indices.sort().indices
    lengths = lengths.values.tolist()
    # Partition such that largest-in-bucket is no more than 4x smallest-in-bucket
    part_offsets = [0]
    part_maxlen = [0]
    bucket_limit = max(lengths[0], 4)  # smallest bucket can always handle up to 4x4=16 length
    for idx, length in enumerate(lengths):
        if length > 4*bucket_limit:
            part_offsets.append(idx)
            bucket_limit = length
            part_maxlen.append(0)
        part_maxlen[-1] = length
    # Apply the partition to the embeds and masks
    part_offsets.append(len(embeds))
    parts = [embeds[part_offsets[idx]: part_offsets[idx+1], :part_maxlen[idx]]
             for idx in range(len(part_offsets)-1)]
    masks = [shortened_mask[part_offsets[idx]: part_offsets[idx+1], :part_maxlen[idx]]
             for idx in range(len(part_offsets)-1)]
    return parts, masks, unpermute_idxs


def unpartition(parts, masks, unpermute_idxs):
    embeds = flatten([list(part) for part in parts])
    masks = flatten([list(part) for part in masks])
    embeds = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
    embeds = embeds[unpermute_idxs]
    masks = masks[unpermute_idxs]
    return embeds, masks

def init_weights(module):
    """ Initialize the weights; Taken from Hugging Face for BERT """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def init_tail_embeddings(module, qid_count_dict, entity_symbols, pad_idx, vec=None):
    assert pad_idx == 0 or pad_idx == -1, f"Only accept pads of 0 or -1; you gave {pad_idx}"
    assert isinstance(module, (nn.Embedding))
    embedding_dim = module.embedding_dim

    # generate index to fill based on counts
    tail_idx = []
    for qid in entity_symbols.get_all_qids():
        eid = entity_symbols.get_eid(qid)
        weight = qid_count_dict.get(qid, 0)
        if weight <= 0:
            tail_idx.append(eid)
    tail_idx = torch.tensor(tail_idx)
    if vec is None:
        # Follows how nn.Embedding intializes their weights
        vec = torch.Tensor(1, embedding_dim)
        init.normal_(vec)
    pad_row = module.weight.data[pad_idx][:]
    module.weight.data[tail_idx] = vec
    # We want the pad row to stay the same as it was before (i.e., all zeros) and not become a tail embedding
    assert torch.equal(pad_row, module.weight.data[pad_idx][:])
    # Can only verify sameness for more than two embeddings
    if tail_idx.shape[0] > 2:
        assert torch.equal(module.weight.data[tail_idx[0]], module.weight.data[tail_idx[1]])
    return vec