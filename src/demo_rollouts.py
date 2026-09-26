"""Inject certified trajectories into PPO's own rollout buffer.

The alternative to a separate supervised term. The expert action is FORCED at
each step of a demonstration episode, but everything else about the transition
is ordinary on-policy data: the reward comes from the real env, and the recorded
log-probability is the one the CURRENT policy assigns to that action. So there
is a valid `pi_old` -- the behaviour policy is the current policy in every
respect except which action got selected -- and the ratio starts at 1 exactly as
it does for a sampled transition.

The appeal over a fixed supervised weight is that the advantage is a LEARNED
weight. A demonstration whose action the policy already favours earns a small
advantage and contributes little; a surprising one weights heavily, with no
schedule to tune.

Two properties to keep in view:

  * PPO's clip caps how fast a forced action's probability can rise -- 1 + eps
    per update, so about 1.2x per `train()` call at the default clip_range. A
    rare-but-correct action at probability 0.01 needs roughly 21 updates to
    reach 0.5. That is thousands of environment steps, not millions, but it is a
    rate limit a cross-entropy term does not have.
  * The demonstrations only sit comfortably in the same buffer as on-policy data
    when returns are on a comparable scale. With `reward_mode="normalized"` they
    are (a solved demo returns about 1 + par_bonus against a failing rollout's
    fraction of a unit); with the raw potential they are not, and the demos
    would dominate SB3's per-batch advantage normalisation and squash the
    on-policy signal toward zero.
"""

from copy import deepcopy

import numpy as np
import torch as th
from stable_baselines3 import PPO

from envs.solved_instances import _chord_between, _half_edge_between


class DemoDriver:
    """Chooses demonstration episodes and hands back their next expert action.

    The stored moves are vertex-identity descriptors rather than half-edge ids,
    because half-edge ids do not survive a rebuild of the mesh. Resolving one
    therefore needs the live graph, and a vertex insert creates a vertex whose
    identity must be learned after the step that made it -- which is why the
    alias for it is filled in lazily on the following call.
    """

    def __init__(self, instances, base_initializer, probability=0.25, rng=None):
        self.instances = list(instances)
        self.base_initializer = base_initializer
        self.probability = float(probability)
        self.rng = rng or np.random.default_rng(0)
        self._reset_episode()

    def _reset_episode(self):
        self.moves = []
        self.alias = {}
        self.index = 0
        self.pending_vertex = None      # (half_edge, created) awaiting its alias
        self.active = False

    def __call__(self):
        """Draw a start state, sometimes a demonstration."""
        self._reset_episode()
        if self.instances and self.rng.random() < self.probability:
            instance = self.instances[int(self.rng.integers(len(self.instances)))]
            graph, desired = instance.build()
            self.moves = list(instance.moves)
            self.alias = {v: v for v in graph.vertex_list(tag=False)}
            self.active = True
            return graph, desired
        return self.base_initializer()

    def _settle_pending(self, env):
        if self.pending_vertex is None:
            return
        half_edge, created = self.pending_vertex
        self.pending_vertex = None
        try:
            self.alias[created] = env.graph.target_vertex(half_edge, tag=False)
        except Exception:
            self.abandon()

    def abandon(self):
        self.active = False
        self.moves = []

    def next_action(self, env):
        """The expert's next move as a linear action index, or None."""
        if not self.active:
            return None
        self._settle_pending(env)
        if not self.active or self.index >= len(self.moves):
            self.abandon()
            return None

        kind, u, v, created = self.moves[self.index]
        if u not in self.alias or v not in self.alias:
            self.abandon()
            return None
        graph = env.graph
        if kind == "chord":
            found = _chord_between(graph, self.alias[u], self.alias[v],
                                   env.max_edge_addition_steps, env._chord_offset,
                                   env.min_face_degree)
            if found is None:
                self.abandon()
                return None
            half_edge, local = found
        else:
            half_edge = _half_edge_between(graph, self.alias[u], self.alias[v])
            if half_edge is None:
                self.abandon()
                return None
            local = env._insert_vertex_action

        slot = env.half_edge_to_index.get(half_edge)
        if slot is None:
            # the template window moved off this half-edge; the demonstration
            # cannot be expressed as an action from here
            self.abandon()
            return None
        linear = slot * env.num_actions_per_half_edge + local
        if not np.isfinite(env._get_obs()["mask"][linear]):
            self.abandon()
            return None

        self.index += 1
        if kind == "vertex":
            self.pending_vertex = (half_edge, created)
        return int(linear)


def next_demo_action(env):
    """Module-level so a vec env can reach it by name through `env_method`."""
    driver = getattr(env, "demo_driver", None)
    if driver is None:
        return None
    return driver.next_action(env)


class PPOWithDemoRollouts(PPO):
    """PPO whose rollouts sometimes follow a certified trajectory.

    The forcing is done by wrapping the policy's forward pass for the duration
    of collection, rather than by reimplementing `collect_rollouts`: everything
    SB3 does with the transition afterwards -- rewards, values, GAE, the buffer
    -- is left exactly as it is, and a library upgrade cannot silently change
    the objective.
    """

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        original_forward = self.policy.forward

        def forward_with_demos(obs, deterministic=False):
            actions, values, log_probs = original_forward(obs, deterministic=deterministic)
            try:
                forced = env.env_method("next_demo_action_for_self")
            except Exception:
                return actions, values, log_probs
            rows = [i for i, action in enumerate(forced) if action is not None]
            if not rows:
                return actions, values, log_probs
            merged = actions.clone()
            for row in rows:
                merged[row] = int(forced[row])
            # the recorded log-probability must belong to the action actually
            # taken, or the ratio is wrong from the very first update
            new_values, new_log_probs, _ = self.policy.evaluate_actions(obs, merged)
            index = th.tensor(rows, device=actions.device, dtype=th.long)
            actions = actions.index_copy(0, index, merged.index_select(0, index))
            values = values.index_copy(0, index, new_values.index_select(0, index))
            log_probs = log_probs.index_copy(
                0, index, new_log_probs.index_select(0, index))
            return actions, values, log_probs

        self.policy.forward = forward_with_demos
        try:
            return super().collect_rollouts(env, callback, rollout_buffer,
                                            n_rollout_steps)
        finally:
            self.policy.forward = original_forward
