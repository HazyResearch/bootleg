"""Simple model building blocks."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention
    (adapted from https://github.com/yuhaozhang/tacred-relation/blob/master/model/layers.py#L42)
    weight is
       a = T' . tanh(Ux)
    where x is the input

    Args:
        input_size: input size
        attn_size: attention size
        feature_size: features size
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
        """Initialize weights.

        Returns:
        """
        self.ulinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_()  # use zero to give uniform attention at the beginning

    def forward(self, x, mask, f=None):
        """Model forward.

        Args:
            x: B x seq_len x input_size
            mask: B x seq_len (True is for valid elements, False is for padded ones)
            f: B x seq_len x feature_size

        Returns: averaged output B x seq_len x attn_size
        """
        batch_size, seq_len, _ = x.size()
        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size
        )
        if self.wlinear is not None:
            f_proj = (
                self.wlinear(f.view(-1, self.feature_size))
                .contiguous()
                .view(batch_size, seq_len, self.attn_size)
            )
            projs = [x_proj, f_proj]
        else:
            projs = [x_proj]
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len
        )

        # mask padding
        scores.data.masked_fill_(~mask.data, -float("inf"))
        weights = F.softmax(scores, dim=1)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs


class Attn(nn.Module):
    """Attention layer, with option for residual connection.

    Args:
        size: input size
        dropout: dropout perc
        num_heads: number of heads
        residual: use residual or not
    """

    def __init__(self, size, dropout, num_heads, residual=False):
        super(Attn, self).__init__()
        self.norm_q = torch.nn.LayerNorm(size)
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.attn_layer = nn.MultiheadAttention(size, num_heads)
        self.residual = residual

    def forward(self, q, x, key_mask, attn_mask):
        """Model forward.

        Args:
            q: query (L x B x E)
            x: key/value (S x B x E)
            key_mask: key mask (B x S)
            attn_mask: attention mask (L x S)

        Returns: output tensor (L x B x E), attention weights (B x L x S)
        """
        y = self.norm(x)
        # print(q.shape, y.shape, y.shape)
        assert y.shape[1] == q.shape[1], "Dimensions are wrong for attention!!!!!"
        output, weights = self.attn_layer(
            self.norm_q(q), y, y, key_padding_mask=key_mask, attn_mask=attn_mask
        )
        res = self.dropout(output)
        if self.residual:
            res += q
        return res, weights


class SelfAttn(nn.Module):
    """Self-attention layer, with option for residual connection.

    Args:
        size: input size
        dropout: dropout perc
        num_heads: number of heads
        residual: use residual or not
    """

    def __init__(self, size, dropout, num_heads, residual=False):
        super(SelfAttn, self).__init__()
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.attn_layer = nn.MultiheadAttention(size, num_heads)
        self.residual = residual

    def forward(self, x, key_mask, attn_mask):
        """Model forward.

        Args:
            q: query/key/value (L x B x E)
            key_mask: key mask (B x L)
            attn_mask: attention mask (L x L)

        Returns: output tensor (L x B x E), attention weights (B x L x L)
        """
        y = self.norm(x)
        output, weights = self.attn_layer(
            y, y, y, key_padding_mask=key_mask, attn_mask=attn_mask
        )
        res = self.dropout(output)
        if self.residual:
            res += x
        return res, weights


class Feedforward(nn.Module):
    """Applies linear layer with option for residual connection.

    By default, adds no dropout.

    Args:
        size: input size
        output: output size
        dropout: dropout perc
        residual: use residual or not
        activation: activation function
    """

    def __init__(self, size, output, dropout=0.0, residual=False, activation=None):
        super(Feedforward, self).__init__()
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(size, output)
        self.residual = residual
        self.activation = activation

    def forward(self, x):
        """Apply residual connection to any sublayer with the same size.

        Args:
            x: input tensor (*, input size)

        Returns: output tensor (*, output size)
        """
        res = self.linear_layer(self.norm(x))
        if self.activation is not None:
            res = self.activation(res)
        res = self.dropout(res)
        if self.residual:
            res += x
        return res


class AttnBlock(nn.Module):
    """Attention layer with FFN (like Transformer)

    Args:
        size: input size
        ff_inner_size: feedforward intermediate size
        dropout: dropout perc
        num_heads: number of heads
    """

    def __init__(self, size, ff_inner_size, dropout, num_heads):
        super(AttnBlock, self).__init__()
        self.attn = Attn(size, dropout, num_heads, residual=True)
        self.ffn1 = Feedforward(
            size, ff_inner_size, dropout, residual=False, activation=torch.nn.ReLU()
        )
        self.ffn2 = Feedforward(
            ff_inner_size, size, dropout, residual=False, activation=None
        )

    def forward(self, q, x, key_mask, attn_mask):
        """Model forward.

        Args:
            q: query (L x B x E)
            x: key/value (S x B x E)
            key_mask: key mask (B x S)
            attn_mask: attention mask (L x S)

        Returns: output tensor (L x B x E), attention weights (B x L x L)
        """
        out, weights = self.attn(q, x, key_mask, attn_mask)
        out = out + self.ffn2(self.ffn1(out))
        return out, weights


class SelfAttnBlock(nn.Module):
    """Self-attention layer with FFN (like Transformer)

    Args:
        size: input size
        ff_inner_size: feedforward intermediate size
        dropout: dropout perc
        num_heads: number of heads
    """

    def __init__(self, size, ff_inner_size, dropout, num_heads):
        super(SelfAttnBlock, self).__init__()
        self.attn = SelfAttn(size, dropout, num_heads, residual=True)
        self.ffn1 = Feedforward(
            size, ff_inner_size, dropout, residual=False, activation=torch.nn.ReLU()
        )
        self.ffn2 = Feedforward(
            ff_inner_size, size, dropout, residual=False, activation=None
        )

    def forward(self, x, key_mask, attn_mask):
        """Model forward.

        Args:
            q: query/key/value (L x B x E)
            key_mask: key mask (B x L)
            attn_mask: attention mask (L x L)

        Returns: output tensor (L x B x E), attention weights (B x L x L)
        """
        out, weights = self.attn(x, key_mask, attn_mask)
        out = out + self.ffn2(self.ffn1(out))
        return out, weights


class MLP(nn.Module):
    """Defines a MLP in terms of the number of hidden units, the output
    dimension, and the total number of layers.

    Args:
        input_size: input size
        num_hidden_units: number of hidden units (None is allowed)
        output_size: output size
        num_layers: number of layers
        dropout: dropout perc
        residual: use residual
        activation: activation function
    """

    def __init__(
        self,
        input_size,
        num_hidden_units,
        output_size,
        num_layers,
        dropout=0.0,
        residual=False,
        activation=torch.nn.ReLU(),
    ):
        super(MLP, self).__init__()
        assert num_layers > 0
        if num_hidden_units is None or num_hidden_units <= 0:
            assert (
                num_layers == 1
            ), f"You must provide num_hidden_units when more than 1 layer"
        self.layers = []
        in_dim = input_size
        for _ in range(num_layers - 1):
            self.layers.append(
                Feedforward(in_dim, num_hidden_units, dropout, residual, activation)
            )
            in_dim = num_hidden_units

        # The output layer has no residual connection, no activation, and no dropout
        self.layers.append(
            Feedforward(
                in_dim, output_size, dropout=0.0, residual=False, activation=None
            )
        )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """Model forward.

        Args:
            x: (*, input size)

        Returns: tensor (*, output size)
        """
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        return out


class CatAndMLP(nn.Module):
    """Concatenate and MLP layer.

    Args:
        size: input size
        num_hidden_units: number of hidden unites
        output_size: output size
        num_layers: number of layers
    """

    def __init__(self, size, num_hidden_units, output_size, num_layers):
        super(CatAndMLP, self).__init__()
        self.proj = MLP(
            size,
            num_hidden_units,
            output_size,
            num_layers,
            dropout=0,
            residual=False,
            activation=torch.nn.ReLU(),
        )

    def forward(self, x, dim=3):
        """Model forward.

        Args:
            x: List of (*, *) tensors
            dim: Dim to concatenate

        Returns: tensor (*, output size)
        """
        y = torch.cat(x, dim)
        res = self.proj(y)
        return res


class NormAndSum(nn.Module):
    """Layer norm and sum embeddings.

    Args:
        size: input size
    """

    def __init__(self, size):
        super(NormAndSum, self).__init__()
        self.norm = torch.nn.LayerNorm(size)

    def forward(self, in_tensors):
        """Model forward.

        Args:
            in_tensors: List of input tensors (D, *)

        Returns: output tensor (*)
        """
        out_tensor = torch.stack(in_tensors, dim=0)
        out_tensor = self.norm(out_tensor)
        out_tensor = out_tensor.sum(dim=0)
        return out_tensor


class PositionalEncoding(nn.Module):
    """Positional encoding.

    Taken from https://nlp.seas.harvard.edu/2018/04/03/attention.html.

    Args:
        hidden_dim: hiddem dim
        max_len: maximum position length
    """

    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim)
        )
        pe[torch.arange(0, max_len), 0::2] = torch.sin(position * div_term)
        # If hidden dim is odd, 1::2 will contain floor(hidden_dim/2) indexes.
        # div_term, because it starts at 0, will contain ceil(hidden_dim/2) indexes.
        # So, as cos is 1::2, must remove last index so the dimensions match.
        if hidden_dim % 2 != 0:
            pe[torch.arange(0, max_len), 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[torch.arange(0, max_len), 1::2] = torch.cos(position * div_term)
        # raw numerical positional encodinf
        # pe = torch.stack(hidden_dim*[torch.arange(max_len)]).transpose(0,1).long()
        # Add one for PAD positional for unknown aliases

        # There were some weird bugs by doing assignment from dims 0 to hidden_dim if making pe original have an extra column of zeros
        pe = torch.cat([pe, torch.zeros(1, hidden_dim).float()])
        pe = pe.unsqueeze(0)
        # This allows the tensor pe to be saved in state dict but not trained
        self.register_buffer("pe", pe)

    def forward(self, x, spans):
        """Model forward.

        Args:
            x: tensor (*,*)
            spans: spans to index into positional embedding to add to x

        Returns: tensor (*,*)
        """
        x = x + Variable(self.pe[:, spans], requires_grad=False)
        x = x.squeeze(0)
        return x
