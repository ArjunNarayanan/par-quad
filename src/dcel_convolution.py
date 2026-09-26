import torch
from torch.nn import Linear, LayerNorm, LeakyReLU


class DCELConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cycle_boundary_vector = torch.nn.Parameter(torch.randn(self.in_channels))
        self.twin_boundary_vector = torch.nn.Parameter(torch.randn(self.in_channels))

        self.linear = Linear(4 * in_channels, out_channels, bias=use_bias)
        self.norm = LayerNorm(self.out_channels)
        self.act = LeakyReLU()

    def forward(self, features, next_indices, prev_indices, twin_indices):
        expanded_features = torch.row_stack([features, self.cycle_boundary_vector, self.twin_boundary_vector])
        next_features = expanded_features[next_indices, :]
        prev_features = expanded_features[prev_indices, :]
        twin_features = expanded_features[twin_indices, :]

        features = torch.cat([features, next_features, prev_features, twin_features], dim=1)
        out_features = self.linear(features)
        out_features = self.norm(out_features)
        out_features = self.act(out_features)

        return out_features


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model1 = DCELConvBlock(in_channels, in_channels)
        self.model2 = DCELConvBlock(in_channels, out_channels)

    def forward(self, features, next_indices, prev_indices, twin_indices):
        features = self.model1(features, next_indices, prev_indices, twin_indices)
        features = self.model2(features, next_indices, prev_indices, twin_indices)
        return features


class RecurrentConvolution(torch.nn.Module):
    def __init__(self, in_channels, num_convolutions):
        super().__init__()
        self.model = torch.nn.ModuleList([DoubleConvBlock(in_channels, in_channels) for i in range(num_convolutions)])

    def forward(self, features, next_indices, prev_indices, twin_indices):
        out = features
        for layer in self.model:
            y = layer(out, next_indices, prev_indices, twin_indices)
            out = out + y
        return out
