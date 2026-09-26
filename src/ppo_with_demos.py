"""PPO that keeps replaying the certified demonstrations while it fine-tunes.

Warm starting PPO from a cloned policy and then letting it run free costs the
capability the warm start supplied: measured over five rounds on ten polygons,
the solve count swung 7 -> 5 -> 7 -> 4 -> 6 -> 8, and at the low point four
instances that had produced a complete mesh produced none at all. The update had
walked the policy off the manifold its warm start occupied.

Demonstrations CANNOT simply be appended to PPO's rollout buffer. PPO is
on-policy: its objective weights each sample by rho = pi(a|s) / pi_old(a|s),
where pi_old is the policy that actually collected the rollout. A certified
trajectory has no such pi_old -- it came from the generator, not from any policy
in the run -- and substituting the cloned policy's probabilities gives a ratio
that drifts further from 1 the longer training goes, at which point clipping
either kills the gradient or applies it in the wrong direction. The advantages
would be quietly wrong.

So the demonstrations enter as a supervised term instead, the same weighted
cross-entropy the cloning path uses, applied after each PPO update against the
same optimizer. This is the DQfD / DAPG / PPO-ptx pattern: keep optimising the
supervised objective alongside the RL one so the fine-tune cannot forget it.

The coefficient decays. The demonstrations are polyominoes, annuli and polar
meshes, while the RL stage runs random polygons with holes -- so the term
anchors to a distribution we are deliberately trying to move away from, and it
should hold the policy steady early and then get out of the way.
"""

import numpy as np
from stable_baselines3 import PPO

from src.behaviour_cloning import train_batches


class PPOWithDemos(PPO):
    """PPO with a decaying supervised replay of a demonstration dataset.

    The demo update runs as its own pass after `PPO.train()` rather than as a
    term inside it. Alternating rather than joint: the drift this exists to
    prevent plays out over rounds, not within one update, and keeping out of
    SB3's inner loop means a library upgrade cannot silently change the
    objective.
    """

    def __init__(self, *args, demo_dataset=None, bc_coef=1.0, bc_decay=1.0,
                 bc_min_coef=0.0, bc_batches=4, bc_batch_size=256,
                 bc_vf_coef=0.25, bc_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_dataset = demo_dataset
        self.bc_coef = float(bc_coef)
        self.bc_decay = float(bc_decay)
        self.bc_min_coef = float(bc_min_coef)
        self.bc_batches = int(bc_batches)
        self.bc_batch_size = int(bc_batch_size)
        self.bc_vf_coef = float(bc_vf_coef)
        self._bc_rng = np.random.default_rng(bc_seed)

    def train(self) -> None:
        super().train()
        if self.demo_dataset is None or len(self.demo_dataset) == 0:
            return
        if self.bc_coef <= 0 or self.bc_batches <= 0:
            return
        loss, accuracy = train_batches(
            self.policy, self.policy.optimizer, self.demo_dataset, self._bc_rng,
            num_batches=self.bc_batches, batch_size=self.bc_batch_size,
            vf_coef=self.bc_vf_coef, max_grad_norm=self.max_grad_norm,
            loss_scale=self.bc_coef)
        self.logger.record("demos/cross_entropy", float(loss))
        self.logger.record("demos/top1_accuracy", float(accuracy))
        self.logger.record("demos/coefficient", self.bc_coef)
        self.bc_coef = max(self.bc_coef * self.bc_decay, self.bc_min_coef)
