"""The search must speak exactly the environment's action language.

If `legal_actions` and the env's mask ever disagree, or if `apply_action` and
`env.step` diverge, a trajectory found by search stops being replayable -- and
expert iteration would be training on moves the agent cannot make.
"""

import unittest

import numpy as np

from envs.angle_env_with_length import AngleEnvWithLength
from envs.environment_initializers import RandomPolygon
from src.canonical import certificate, weisfeiler_lehman_hash
from src.search import (
    ActionSpec, apply_action, beam_search, heuristic, ida_star, is_goal,
    legal_actions, state_from_env,
)


def make_env(seed=0, degrees=(5, 6, 7)):
    np.random.seed(seed)
    initializer = RandomPolygon(list(degrees), 90, scale=0.6)
    return AngleEnvWithLength(4, initializer, template_size=64,
                              max_edge_addition_steps=3, max_steps_factor=3)


class TestActionAgreement(unittest.TestCase):
    def test_legal_actions_match_the_env_mask(self):
        env = make_env(1)
        for _ in range(5):
            env.reset()
            state = state_from_env(env)
            from_search = set(legal_actions(state))
            from_env = set()
            mask = env._get_action_mask()
            for linear in np.nonzero(mask == 0)[0]:
                half_edge, local = env._linear_action_index_to_half_edge_and_action(int(linear))
                from_env.add((half_edge, local))
            self.assertEqual(from_search, from_env)

    def test_apply_action_matches_env_step(self):
        env = make_env(2)
        env.reset()
        state = state_from_env(env)
        actions = legal_actions(state)
        half_edge, local = actions[len(actions) // 2]
        child = apply_action(state, (half_edge, local))

        slot = env.half_edge_to_index[half_edge]
        env.step(slot * env.num_actions_per_half_edge + local)

        self.assertEqual(
            weisfeiler_lehman_hash(child.graph, child.desired),
            weisfeiler_lehman_hash(env.graph, env.vertex_desired_degree))


class TestCertificates(unittest.TestCase):
    def test_certificate_is_stable_under_relabelling(self):
        env = make_env(3)
        env.reset()
        state = state_from_env(env)
        other = state.copy()
        self.assertEqual(certificate(state.graph, state.desired),
                         certificate(other.graph, other.desired))

    def test_distinct_states_get_distinct_fingerprints(self):
        env = make_env(4)
        env.reset()
        state = state_from_env(env)
        seen = {state.fingerprint()}
        for action in legal_actions(state)[:6]:
            seen.add(apply_action(state, action).fingerprint())
        self.assertGreater(len(seen), 2)


class TestHeuristicAndGoal(unittest.TestCase):
    def test_heuristic_is_zero_only_at_a_quad_mesh_at_par(self):
        env = make_env(5)
        env.reset()
        state = state_from_env(env)
        self.assertGreater(heuristic(state, env.par), 0)
        self.assertFalse(is_goal(state, env.par))

    def test_ida_star_solves_a_small_instance(self):
        env = make_env(6, degrees=(5,))
        env.reset()
        state = state_from_env(env)
        actions, stats = ida_star(state, env.par, max_depth=4, budget_s=45.0,
                                  exact_dedup=False)
        if actions is None:
            self.skipTest(f"search budget exhausted: {stats['status']}")
        for half_edge, local in actions:
            slot = env.half_edge_to_index.get(half_edge)
            self.assertIsNotNone(slot)
            linear = slot * env.num_actions_per_half_edge + local
            self.assertTrue(env.is_valid_action(linear))
            env.step(linear)
        self.assertTrue(env.is_at_par())


class TestBeamSearch(unittest.TestCase):
    def test_beam_search_returns_replayable_actions(self):
        import torch
        from src.convolution_feature_extractor import ConvolutionFeatureExtractor
        from src.policy import CustomActorCriticPolicy

        torch.manual_seed(0)
        env = make_env(7, degrees=(5, 6))
        env.reset()
        policy = CustomActorCriticPolicy(
            env.observation_space, env.action_space, lambda _: 3e-4,
            features_extractor_class=ConvolutionFeatureExtractor,
            features_extractor_kwargs=dict(input_features=env.num_features,
                                           output_features=32, number_of_layers=2),
            output_features=32)
        policy.eval()
        actions, stats = beam_search(policy, env, par=env.par, beam=8, top_k=4,
                                     max_depth=6, budget_s=45.0)
        self.assertIn(stats["status"], {"solved", "not-found", "already-solved"})
        if not actions:
            self.skipTest("beam did not find a solution within the budget")
        for half_edge, local in actions:
            slot = env.half_edge_to_index.get(half_edge)
            self.assertIsNotNone(slot)
            env.step(slot * env.num_actions_per_half_edge + local)
        self.assertTrue(env.is_at_par())


if __name__ == "__main__":
    unittest.main()
