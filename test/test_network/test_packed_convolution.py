"""The packed convolution must equal the dense one on every valid slot.

`ConvolutionFeatureExtractor._forward_batch` sends only the live half-edge
slots through the DCEL convolution. Valid rows have to come out bit-for-bit
(to float tolerance) the same as the dense reference, padded rows must be
zero, and the policy's action distribution must not change.
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.append(os.getcwd())

from envs.environment_maker import initialize_environment  # noqa: E402
from src.convolution_feature_extractor import ConvolutionFeatureExtractor  # noqa: E402
from src.policy import CustomActorCriticPolicy  # noqa: E402

ENV_CONFIG = {
    "name": "AngleEnvWithLength",
    "initializer": {"name": "RandomPolygon", "min_polygon_degree": 5,
                    "max_polygon_degree": 12, "scale": 0.6, "target_angle": 90},
    "face_desired_degree": 4,
    "template_size": 64,
    "max_edge_addition_steps": 3,
    "smooth_iterations": 5,
}


def observations(count, rng):
    env = initialize_environment(ENV_CONFIG)
    batch = []
    obs, _ = env.reset()
    for _ in range(count):
        batch.append({k: np.array(v, copy=True) for k, v in obs.items()})
        valid = np.flatnonzero(np.isfinite(obs["mask"]))
        obs, _, terminated, _, _ = env.step(int(rng.choice(valid)))
        if terminated:
            obs, _ = env.reset()
    return env, {k: torch.as_tensor(np.stack([o[k] for o in batch])) for k in batch[0]}


class TestPackedConvolution(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.env, self.batch = observations(24, np.random.default_rng(0))
        self.extractor = ConvolutionFeatureExtractor(
            self.env.observation_space, input_features=self.env.num_features,
            output_features=32, number_of_layers=2)

    def test_valid_rows_match_dense_and_padding_is_zero(self):
        features = self.batch["features"]
        nxt = self.batch["next"].long()
        prv = self.batch["previous"].long()
        twn = self.batch["twin"].long()
        valid = features.abs().sum(dim=-1) > 0
        self.assertFalse(bool(valid.all()), "the test needs some padded slots")
        with torch.no_grad():
            dense = self.extractor._forward_batch_dense(features, nxt, prv, twn)
            packed = self.extractor._forward_batch(features, nxt, prv, twn)
        torch.testing.assert_close(packed[valid], dense[valid], rtol=1e-5, atol=1e-5)
        self.assertEqual(float(packed[~valid].abs().max()), 0.0)

    def test_policy_distribution_unchanged(self):
        policy = CustomActorCriticPolicy(
            self.env.observation_space, self.env.action_space, lambda _: 1e-4,
            features_extractor_class=ConvolutionFeatureExtractor,
            features_extractor_kwargs=dict(input_features=self.env.num_features,
                                           output_features=32, number_of_layers=2))
        extractor = policy.features_extractor
        with torch.no_grad():
            packed_logits = policy.get_distribution(self.batch).distribution.logits
            extractor._forward_batch = extractor._forward_batch_dense
            dense_logits = policy.get_distribution(self.batch).distribution.logits
        finite = torch.isfinite(dense_logits)
        torch.testing.assert_close(packed_logits[finite], dense_logits[finite], rtol=1e-4, atol=1e-4)
        self.assertTrue(torch.equal(torch.isfinite(packed_logits), finite))

    def test_gradients_flow_through_packed_path(self):
        out = self.extractor(self.batch)
        out.sum().backward()
        grads = [p.grad for p in self.extractor.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
