import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import Linear
from src.dcel_convolution import RecurrentConvolution


class ConvolutionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, input_features, output_features, number_of_layers):
        super().__init__(observation_space, output_features)
        self.input_features = input_features
        self.output_features = output_features
        self.number_of_layers = number_of_layers

        self.projector = Linear(input_features, output_features)
        self.convolution = RecurrentConvolution(output_features, number_of_layers)

    def _forward_sample(self, features, next_indices, prev_indices, twin_indices):
        # check shape consistency
        assert features.ndim == 2
        assert next_indices.ndim == prev_indices.ndim == twin_indices.ndim == 1
        sample_size, feature_size = features.shape
        assert feature_size == self.input_features
        assert next_indices.shape[0] == prev_indices.shape[0] == twin_indices.shape[0] == sample_size

        features = self.projector(features)
        features = self.convolution(features, next_indices, prev_indices, twin_indices)

        return features

    @staticmethod
    def _archive_unroll_and_offset_indices(batched_indices: torch.Tensor):
        assert batched_indices.ndim == 2
        num_batch, skip = batched_indices.shape
        offset = skip
        offset_indices = [batched_indices[0].clone()]
        for batch in range(1, num_batch):
            indices = batched_indices[batch].clone()
            indices[indices >= 0] += offset
            offset_indices.append(indices)
            offset += skip

        offset_indices = torch.cat(offset_indices)
        return offset_indices

    @staticmethod
    def unroll_and_offset_indices(batched_indices: torch.Tensor):
        assert batched_indices.ndim == 2
        num_batch, num_features = batched_indices.shape

        negative_mask = batched_indices < 0
        offset = (num_features * torch.arange(num_batch, device=batched_indices.device)).reshape(-1, 1)

        offset_indices = batched_indices + offset
        offset_indices[negative_mask] = batched_indices[negative_mask]

        unrolled_indices = offset_indices.ravel()
        return unrolled_indices

    def _forward_batch_dense(self, features, next_indices, prev_indices, twin_indices):
        """Every template slot through the convolution, padding included.

        Kept as the reference implementation; `_forward_batch` packs the valid
        slots and must agree with this on them (test_packed_convolution).
        """
        batch_size, template_size, feature_size = features.shape
        unrolled_features = features.reshape(-1, feature_size)
        offset_next = self.unroll_and_offset_indices(next_indices)
        offset_prev = self.unroll_and_offset_indices(prev_indices)
        offset_twin = self.unroll_and_offset_indices(twin_indices)

        extracted_features = self._forward_sample(unrolled_features, offset_next, offset_prev, offset_twin)
        return extracted_features.reshape(batch_size, -1, self.output_features)

    def _forward_batch(self, features, next_indices, prev_indices, twin_indices):
        """Only the valid half-edge slots go through the convolution.

        A template holds `template_size` slots but a mesh fills as few as a
        dozen of them, and the padded rows are all-zero features whose
        neighbour indices point at themselves. The convolution is row-local
        (gathers of neighbours, a linear layer, LayerNorm per row), so dropping
        the padded rows changes nothing for the valid ones -- their neighbours
        are valid slots or the negative boundary codes, never padding -- and
        the padded rows come back as zeros, which the policy masks out of the
        action logits and the pooled context anyway. On a 128-slot template
        with 40 live half-edges this is a threefold cut in the update cost.
        """
        # Check input shape consistency
        assert features.ndim == 3
        assert next_indices.ndim == prev_indices.ndim == twin_indices.ndim == 2
        batch_size, template_size, feature_size = features.shape
        assert feature_size == self.input_features
        assert next_indices.shape == torch.Size([batch_size, template_size])
        assert prev_indices.shape == torch.Size([batch_size, template_size])
        assert twin_indices.shape == torch.Size([batch_size, template_size])

        valid = (features.abs().sum(dim=-1) > 0).reshape(-1)
        if bool(valid.all()):
            return self._forward_batch_dense(features, next_indices, prev_indices, twin_indices)

        # new position of every old flat slot; padded slots are never looked up
        positions = torch.cumsum(valid.long(), dim=0) - 1

        def pack_indices(batched_indices):
            flat = self.unroll_and_offset_indices(batched_indices)
            packed = flat.clone()
            inside = flat >= 0
            packed[inside] = positions[flat[inside]]
            return packed[valid]

        packed_features = features.reshape(-1, feature_size)[valid]
        extracted = self._forward_sample(packed_features, pack_indices(next_indices),
                                         pack_indices(prev_indices), pack_indices(twin_indices))
        output = extracted.new_zeros((batch_size * template_size, self.output_features))
        output[valid] = extracted
        return output.reshape(batch_size, template_size, self.output_features)

    def forward(self, obs):
        features = obs["features"]
        next_indices = obs["next"].round().long()
        prev_indices = obs["previous"].round().long()
        twin_indices = obs["twin"].round().long()
        if features.ndim == 3:
            return self._forward_batch(features, next_indices, prev_indices, twin_indices)
        elif features.ndim == 2:
            return self._forward_sample(features, next_indices, prev_indices, twin_indices)
        else:
            raise ValueError("Expected features.ndim == 2 or 3, got ", features.ndim)
