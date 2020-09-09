"""Simple model building blocks"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_

from bootleg.utils import eval_utils, model_utils

class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention
    (adapted from https://github.com/yuhaozhang/tacred-relation/blob/master/model/layers.py#L42)
    weight is
       a = T' . tanh(Ux)
    where x is the input
    """

    def __init__(self, input_size, attn_size, feature_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning

    def forward(self, x, mask, f=None):
        """
        x : batch_size x seq_len x input_size
        mask : batch_size x seq_len - mask is true where we have valid elems, false for invalid (i.e. padded) elems
        f : batch_size x seq_len x feature_size
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)
        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
                batch_size, seq_len, self.attn_size)
            projs = [x_proj, f_proj]
        else:
            projs = [x_proj]
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(~mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs


class SoftAttn(nn.Module):
    """
    Computes a soft attention where elemns is batch x n_choices x emb_dim
    and context is batch x context_dim
    We compute a dot product weighted average where the output is batch x emb_dim
    """
    def __init__(self, emb_dim, context_dim, size, tau=0.8):
        super(SoftAttn, self).__init__()
        self.emb_linear = nn.Linear(emb_dim, size)
        self.context_linear = nn.Linear(context_dim, size)
        self.tau = tau

    # elems is batch x n_choices x emb_dim
    # context is batch x context_dim
    def forward(self, elems, context, mask):
        batch, _, emb_dim = elems.shape
        elems_norm = model_utils.normalize_matrix(self.emb_linear(elems), dim=-1)
        context_norm = model_utils.normalize_matrix(self.context_linear(context), dim=-1)
        scores = torch.bmm(elems_norm,
                           context_norm.unsqueeze(-1)).squeeze(-1)
        # mask is true where we have valid elems, false for invalid (i.e. padded) elems
        probs = eval_utils.masked_softmax(pred=scores/self.tau, mask=mask, dim=-1).unsqueeze(-1)
        # probs = eval_utils.masked_gumbel(pred=scores, mask=mask, tau=self.tau, dim=-1).unsqueeze(-1)
        out = (elems * probs).sum(1)
        assert list(out.shape) == [batch, emb_dim]
        return out

class Attn(nn.Module):
    """
    Attention layer, with option for residual connection.
    """
    def __init__(self, size, dropout, num_heads, residual = False):
        super(Attn, self).__init__()
        self.norm_q = torch.nn.LayerNorm(size)
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.attn_layer = nn.MultiheadAttention(size, num_heads)
        self.residual = residual

    def forward(self, q, x, key_mask, attn_mask):
        y = self.norm(x)
        # print(q.shape, y.shape, y.shape)
        assert y.shape[1] == q.shape[1], 'Dimensions are wrong for attention!!!!!'
        output, weights = self.attn_layer(self.norm_q(q), y, y, key_padding_mask=key_mask, attn_mask=attn_mask)
        res = self.dropout(output)
        if self.residual:
            res += q
        return res, weights


class SelfAttn(nn.Module):
    """
    Self attention layer, with the option for residual connection.
    """
    def __init__(self, size, dropout, num_heads, residual = False):
        super(SelfAttn, self).__init__()
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.attn_layer = nn.MultiheadAttention(size, num_heads)
        self.residual = residual

    def forward(self, x, key_mask, attn_mask):
        "Apply residual connection to any sublayer with the same size."
        y = self.norm(x)
        output, weights = self.attn_layer(y, y, y, key_padding_mask=key_mask, attn_mask=attn_mask)
        res = self.dropout(output)
        if self.residual:
            res += x
        return res, weights


class Feedforward(nn.Module):
    """
    Applies linear layer with option for residual connection. By default, adds no dropout.
    """
    def __init__(self, size, output, dropout=0.0, residual=False, activation=None):
        super(Feedforward, self).__init__()
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(size, output)
        self.residual = residual
        self.activation = activation

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        res = self.linear_layer(self.norm(x))
        if self.activation is not None:
            res = self.activation(res)
        res = self.dropout(res)
        if self.residual:
            res += x
        return res


class AttnBlock(nn.Module):
    """
    Attention layer with FFN (like transformer)
    """
    def __init__(self, size, ff_inner_size, dropout, num_heads):
        super(AttnBlock, self).__init__()
        self.attn = Attn(size, dropout, num_heads, residual=True)
        self.ffn1 = Feedforward(size, ff_inner_size, dropout, residual=False, activation=torch.nn.ReLU())
        self.ffn2 = Feedforward(ff_inner_size, size, dropout, residual=False, activation=None)

    def forward(self, q, x, key_mask, attn_mask):
        out, weights = self.attn(q, x, key_mask, attn_mask)
        out = out + self.ffn2(self.ffn1(out))
        return out, weights


class SelfAttnBlock(nn.Module):
    """
    Attention layer with FFN (like transformer)
    """
    def __init__(self, size, ff_inner_size, dropout, num_heads):
        super(SelfAttnBlock, self).__init__()
        self.attn = SelfAttn(size, dropout, num_heads, residual=True)
        self.ffn1 = Feedforward(size, ff_inner_size, dropout, residual=False, activation=torch.nn.ReLU())
        self.ffn2 = Feedforward(ff_inner_size, size, dropout, residual=False, activation=None)

    def forward(self, x, key_mask, attn_mask):
        out, weights = self.attn(x, key_mask, attn_mask)
        out = out + self.ffn2(self.ffn1(out))
        return out, weights


class MLP(nn.Module):
    """ Defines a MLP in terms of the number of hidden units, the output dimension, and the total number of layers """
    def __init__(self, input_size, num_hidden_units, output_size, num_layers, dropout=0.0, residual=False, activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        assert num_layers > 0
        self.layers = []
        in_dim = input_size
        for _ in range(num_layers - 1):
            self.layers.append(Feedforward(in_dim, num_hidden_units, dropout, residual, activation))
            in_dim = num_hidden_units

        # The output layer has no residual connection, no activation, and no dropout
        self.layers.append(Feedforward(in_dim, output_size, dropout=0.0, residual=False, activation=None))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        return out


class CatAndMLP(nn.Module):
    """ Concatenates, and then projects down """

    def __init__(self, size, num_hidden_units, output_size, num_layers):
        super(CatAndMLP, self).__init__()
        self.proj = MLP(size, num_hidden_units, output_size, num_layers, dropout=0, residual=False, activation=torch.nn.ReLU())

    def forward(self, x, dim=3):
        y = torch.cat(x, dim)
        res = self.proj(y)
        return res

class NormAndSum(nn.Module):
    """ Norm and sums tensors """

    def __init__(self, size):
        super(NormAndSum, self).__init__()
        self.norm = torch.nn.LayerNorm(size)

    def forward(self, in_tensors):
        out_tensor = torch.stack(in_tensors, dim=0)
        out_tensor = self.norm(out_tensor)
        out_tensor = out_tensor.sum(dim=0)
        return out_tensor


class PositionalEncoding(nn.Module):
    """ Implement the PE function. Taken from https://nlp.seas.harvard.edu/2018/04/03/attention.html."""
    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pe[torch.arange(0, max_len), 0::2] = torch.sin(position * div_term)
        # If hidden dim is odd, 1::2 will contain floor(hidden_dim/2) indexes.
        # div_term, because it starts at 0, will contain ceil(hidden_dim/2) indexes.
        # So, as cos is 1::2, must remove last index so the dimensions match.
        if hidden_dim % 2 != 0:
            pe[torch.arange(0, max_len), 1::2] = torch.cos(position * div_term)[:,:-1]
        else:
            pe[torch.arange(0, max_len), 1::2] = torch.cos(position * div_term)
        # raw numerical positional encodinf
        # pe = torch.stack(hidden_dim*[torch.arange(max_len)]).transpose(0,1).long()
        # Add one for PAD positional for unknown aliases

        # There were some weird bugs by doing assignment from dims 0 to hidden_dim if making pe original have an extra column of zeros
        pe = torch.cat([pe, torch.zeros(1, hidden_dim).float()])
        pe = pe.unsqueeze(0)
        # This allows the tensor pe to be saved in state dict but not trained
        self.register_buffer('pe', pe)

    def forward(self, x, spans):
        x = x + Variable(self.pe[:, spans], requires_grad=False)
        x = x.squeeze(0)
        return x


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding embedding."""
    def __init__(self, hidden_dim, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.position_words = nn.Embedding(max_len, hidden_dim)

    def forward(self, x, spans):
        spans_pos = torch.where(spans >= 0, spans,
                        (torch.ones_like(spans, dtype=torch.long)*(self.position_words.num_embeddings-1)))
        x = x + self.position_words(spans_pos)
        return x