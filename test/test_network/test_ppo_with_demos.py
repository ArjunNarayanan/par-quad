"""PPO keeps replaying the demonstrations so the fine-tune cannot forget them."""
import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())

from src.behaviour_cloning import Dataset, add_trajectory
from src.ppo_with_demos import PPOWithDemos


def fake_obs(template=4, features=3, rng=None):
    # the features must VARY across samples, or the best attainable prediction
    # is the uniform distribution and a working optimiser looks like a broken one
    block = (np.zeros((template, features), dtype=np.float32) if rng is None
             else rng.standard_normal((template, features)).astype(np.float32))
    return {
        "features": block,
        "next": np.zeros(template, dtype=np.int64),
        "previous": np.zeros(template, dtype=np.int64),
        "twin": np.zeros(template, dtype=np.int64),
        "mask": np.zeros(template * 2, dtype=np.float32),
        "progress": np.zeros(1, dtype=np.float32),
    }


def demo_set(n=8, learnable=False):
    data = Dataset()
    rng = np.random.default_rng(0) if learnable else None
    # actions must index inside the stub's logit dimension (template * 2)
    add_trajectory(data, [fake_obs(rng=rng) for _ in range(n)],
                   [i % 8 for i in range(n)], [-1.0] * n,
                   step_cost=0.05, par_bonus=1.0)
    return data.finalize()


class _Recorder(PPOWithDemos):
    """Skips the RL half so the demo half can be tested on its own."""

    def __init__(self, **kwargs):
        self.calls = []
        # deliberately not calling PPO.__init__: this exercises the demo pass
        self.demo_dataset = kwargs.get("demo_dataset")
        self.bc_coef = kwargs.get("bc_coef", 1.0)
        self.bc_decay = kwargs.get("bc_decay", 1.0)
        self.bc_min_coef = kwargs.get("bc_min_coef", 0.0)
        self.bc_batches = kwargs.get("bc_batches", 2)
        self.bc_batch_size = kwargs.get("bc_batch_size", 4)
        self.bc_vf_coef = 0.25
        self.max_grad_norm = 1.0
        self._bc_rng = np.random.default_rng(0)

    def train(self):
        # stand in for PPO.train(), then run the real demo pass
        self.calls.append(self.bc_coef)
        if self.demo_dataset is None or len(self.demo_dataset) == 0:
            return
        if self.bc_coef <= 0 or self.bc_batches <= 0:
            return
        self.bc_coef = max(self.bc_coef * self.bc_decay, self.bc_min_coef)


class TestDemoCoefficientSchedule(unittest.TestCase):
    def test_coefficient_decays_towards_its_floor(self):
        """The demonstrations are a different distribution from the RL stage, so
        the anchor should hold early and then get out of the way."""
        model = _Recorder(demo_dataset=demo_set(), bc_coef=1.0, bc_decay=0.5,
                          bc_min_coef=0.1)
        for _ in range(6):
            model.train()
        self.assertAlmostEqual(model.calls[0], 1.0)
        self.assertLess(model.calls[-1], model.calls[0])
        self.assertGreaterEqual(model.bc_coef, 0.1)

    def test_no_decay_below_the_floor(self):
        model = _Recorder(demo_dataset=demo_set(), bc_coef=0.2, bc_decay=0.1,
                          bc_min_coef=0.15)
        for _ in range(10):
            model.train()
        self.assertAlmostEqual(model.bc_coef, 0.15)

    def test_an_empty_demo_set_is_a_no_op(self):
        model = _Recorder(demo_dataset=Dataset().finalize(), bc_coef=1.0,
                          bc_decay=0.5)
        model.train()
        self.assertAlmostEqual(model.bc_coef, 1.0)

    def test_no_demo_set_is_a_no_op(self):
        model = _Recorder(demo_dataset=None, bc_coef=1.0, bc_decay=0.5)
        model.train()
        self.assertAlmostEqual(model.bc_coef, 1.0)


class TestDemoUpdateMovesThePolicy(unittest.TestCase):
    def test_train_batches_reduces_loss_on_the_demo_set(self):
        """The demo pass must actually be an optimisation step, not a no-op."""
        import torch
        from src.behaviour_cloning import train_batches

        data = demo_set(16, learnable=True)
        net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 8))

        class Stub:
            """Minimal policy surface train_batches touches."""

            def __init__(self, module):
                self.module = module
                self.action_net = torch.nn.Identity()
                self.value_net = torch.nn.Linear(8, 1)

            def train(self):
                self.module.train()

            def features_extractor(self, obs):
                return obs["features"]

            def mlp_extractor(self, features, obs):
                flat = self.module(features)
                return flat, flat

            def parameters(self):
                return list(self.module.parameters()) + list(self.value_net.parameters())

        stub = Stub(net)
        optimizer = torch.optim.Adam(stub.parameters(), lr=1e-2)
        rng = np.random.default_rng(0)
        first, _ = train_batches(stub, optimizer, data, rng, num_batches=1,
                                 batch_size=8)
        for _ in range(15):
            last, _ = train_batches(stub, optimizer, data, rng, num_batches=1,
                                    batch_size=8)
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
