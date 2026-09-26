"""The Transformer extractor must ignore padded template slots.

`TransformerFeatureExtractor` is the connectivity ablation, so it has to
differ from the DCEL convolution ONLY in how it mixes live half-edges. It
attends over all `template_size` slots, but a mesh fills as few as a dozen of
them, so without `src_key_padding_mask` a live half-edge's representation
would depend on how many dead slots sit beside it -- a handicap that comes
from the implementation rather than from the inductive bias.

Pinned here: masked-with-padding equals the same rows with no padding at all,
padded rows come back as zeros (as the convolution leaves them), and the
all-zero observation `AngleEnv._get_blank_obs` returns after an internal
exception does not produce NaN.
"""

import os
import sys
import unittest

import torch

sys.path.append(os.getcwd())

from src.transformer_feature_extractor import TransformerFeatureExtractor  # noqa: E402

TEMPLATE_SIZE = 160
NUM_FEATURES = 10
NUM_LIVE = 20


class DummySpace:
    """BaseFeaturesExtractor only stores the space; nothing here reads it."""


def make_extractor():
    torch.manual_seed(0)
    extractor = TransformerFeatureExtractor(
        DummySpace(),
        input_features=NUM_FEATURES,
        output_features=96,
        sequence_length=TEMPLATE_SIZE,
        num_heads=4,
        number_of_layers=4,
        dropout=0.0,
        dim_feedforward=192,
        norm_first=True,
    )
    extractor.eval()
    return extractor


class TestTransformerPadding(unittest.TestCase):
    def setUp(self):
        self.extractor = make_extractor()
        torch.manual_seed(1)
        self.rows = torch.randn(1, NUM_LIVE, NUM_FEATURES)
        self.padded = torch.zeros(1, TEMPLATE_SIZE, NUM_FEATURES)
        self.padded[0, :NUM_LIVE] = self.rows[0]

    def test_padding_does_not_reach_live_slots(self):
        """A padded template must give the live rows what no padding would."""
        with torch.no_grad():
            out_padded = self.extractor({"features": self.padded})[0, :NUM_LIVE]
            out_short = self.extractor({"features": self.rows})[0]
        self.assertTrue(torch.allclose(out_padded, out_short, atol=1e-5))

    def test_padded_rows_come_back_zero(self):
        """The convolution leaves dead slots at zero; so must this."""
        with torch.no_grad():
            out = self.extractor({"features": self.padded})
        self.assertTrue(bool((out[0, NUM_LIVE:] == 0).all()))

    def test_blank_observation_is_finite(self):
        """`_get_blank_obs` is all zeros: every key masked would divide by zero."""
        batch = torch.zeros(3, TEMPLATE_SIZE, NUM_FEATURES)
        batch[0, :NUM_LIVE] = self.rows[0]          # one live sample, two blank
        out = self.extractor({"features": batch})
        self.assertTrue(bool(torch.isfinite(out).all()))

        out.sum().backward()
        grads = [p.grad for p in self.extractor.parameters() if p.grad is not None]
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))

    def test_unbatched_keeps_its_shape(self):
        """The convolution maps [T, F] -> [T, D]; this must agree."""
        with torch.no_grad():
            out = self.extractor({"features": self.padded[0]})
        self.assertEqual(out.shape, torch.Size([TEMPLATE_SIZE, 96]))


class TestParameterMatched(unittest.TestCase):
    def test_matches_the_convolution_within_one_percent(self):
        """The ablation is only clean if capacity is not the difference."""
        from src.convolution_feature_extractor import ConvolutionFeatureExtractor

        count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        convolution = ConvolutionFeatureExtractor(
            DummySpace(), input_features=NUM_FEATURES, output_features=96, number_of_layers=4
        )
        ratio = count(make_extractor()) / count(convolution)
        self.assertLess(abs(ratio - 1.0), 0.01)


if __name__ == "__main__":
    unittest.main()
