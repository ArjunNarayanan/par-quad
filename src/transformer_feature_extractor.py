import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import LayerNorm
import math
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import Linear


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len, dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]`` or ``[seq_len, embedding_dim]``
        """
        # Index the sequence axis explicitly. `pe` is [1, max_len, d_model], so
        # the old `self.pe[:x.size(0)]` sliced the batch axis -- a no-op that
        # happened to broadcast only because seq_len == max_len exactly, and
        # that grew a spurious leading axis on an unbatched input.
        x = x + self.pe[0, :x.size(-2)]
        return self.dropout(x)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Template read as an ordered sequence, with the DCEL connectivity discarded.

    The ablation counterpart to `ConvolutionFeatureExtractor`: identical input
    and output shapes, so it is a drop-in swap, but `obs["next"]`,
    `obs["previous"]` and `obs["twin"]` are never read. What it keeps is the
    template ORDER, which `Tiler.knn_half_edges` produces by a breadth-first
    walk over those same three permutations -- so the positional encoding
    carries hop distance from the centre face, but not which half-edge is
    whose twin.
    """

    def __init__(
            self,
            observation_space,
            input_features,
            output_features,
            sequence_length,
            num_heads,
            number_of_layers,
            dropout,
            dim_feedforward=2048,
            norm_first=False,
    ):
        super().__init__(observation_space, output_features)
        self.num_input_features = input_features
        self.num_output_features = output_features
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.num_layers = number_of_layers

        self.projector = Linear(input_features, output_features)
        self.positional_encoder = PositionalEncoding(output_features, sequence_length, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=output_features,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=True
        )
        encoder_norm = LayerNorm(output_features)
        self.encoder = TransformerEncoder(
            encoder_layer,
            number_of_layers,
            encoder_norm
        )

    def forward(self, obs):
        x = obs["features"]
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)

        # Padded template slots are exactly the all-zero rows, the same
        # predicate the policy pools with and the convolution packs on.
        valid = x.abs().sum(dim=-1) > 0
        key_padding_mask = ~valid

        # An episode that hit an internal exception observes `_get_blank_obs`,
        # which is all zeros. Masking every key of such a row makes the
        # attention softmax normalise over nothing and returns NaN, which would
        # spread through the batch and silently poison the PPO update. Let those
        # rows attend normally instead; their actions are -inf masked anyway.
        key_padding_mask[~valid.any(dim=-1)] = False

        x = self.projector(x)
        x = self.positional_encoder(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Hand padded slots back as zeros, which is what the convolution leaves
        # in them, so the two extractors differ only in how they mix the live ones.
        x = x * valid.unsqueeze(-1).to(x.dtype)

        return x.squeeze(0) if unbatched else x
