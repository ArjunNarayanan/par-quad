"""The critic sees the step budget; the actor does not.

The episode has a step limit, so the return from a state depends on how much
of it remains -- a state five moves from a solution is worth one thing with ten
moves left and another with one. A critic that cannot tell those apart is being
asked to fit two numbers to one input, so it keeps the horizon.

The actor must not. `max_steps` is `max(min_max_steps, factor * half_edges)`, a
training setting rather than a property of the domain, so a policy conditioned
on it is conditioned on the budget we happened to choose. A larger domain
needing more moves than training ever allowed then reads as out of
distribution for a reason that has nothing to do with its geometry.
"""

import unittest

import torch

from src.convolution_feature_extractor import ConvolutionFeatureExtractor
from src.policy import CustomActorCriticPolicy


def _env():
    import envs.polygon_utils as utils
    from envs.environment_initializers import RandomPolygon
    from envs.global_angle_env import AngleEnv
    env = AngleEnv(face_desired_degree=4,
                   graph_initializer=RandomPolygon(range(8, 12),
                                                   utils.average_face_angle(4)),
                   template_size=48, resample_if_at_par=False)
    env.reset()
    return env


def _policy(env, hide):
    torch.manual_seed(0)
    policy = CustomActorCriticPolicy(
        env.observation_space, env.action_space, lambda _: 3e-4,
        features_extractor_class=ConvolutionFeatureExtractor,
        features_extractor_kwargs=dict(input_features=env.num_features,
                                       output_features=32, number_of_layers=2),
        output_features=32, hide_budget_from_actor=hide)
    policy.eval()
    return policy


def _latents(policy, observation, bump):
    tensors = {k: torch.as_tensor(v).unsqueeze(0) for k, v in observation.items()}
    if bump:
        tensors["progress"] = tensors["progress"] + 0.5
    features = policy.extract_features(tensors)
    with torch.no_grad():
        return policy.mlp_extractor(features, tensors)


class TestBudgetVisibility(unittest.TestCase):
    def test_the_actor_ignores_the_budget_when_hidden(self):
        env = _env()
        policy = _policy(env, hide=True)
        observation = env._get_obs()
        before, _ = _latents(policy, observation, bump=False)
        after, _ = _latents(policy, observation, bump=True)
        self.assertTrue(torch.allclose(before, after),
                        "the actor still responds to the step budget")

    def test_the_critic_always_sees_the_budget(self):
        env = _env()
        observation = env._get_obs()
        for hide in (False, True):
            policy = _policy(env, hide=hide)
            _, before = _latents(policy, observation, bump=False)
            _, after = _latents(policy, observation, bump=True)
            self.assertFalse(torch.allclose(before, after),
                             f"the critic lost the horizon with hide={hide}")

    def test_the_default_keeps_the_old_behaviour(self):
        env = _env()
        policy = _policy(env, hide=False)
        observation = env._get_obs()
        before, _ = _latents(policy, observation, bump=False)
        after, _ = _latents(policy, observation, bump=True)
        self.assertFalse(torch.allclose(before, after),
                         "the actor should see the budget unless asked not to")

    def test_the_budget_is_not_duplicated_in_the_observation(self):
        """It used to arrive twice: obs['global'][3] and obs['progress']."""
        env = _env()
        observation = env._get_obs()
        progress = float(observation["progress"][0])
        for index, value in enumerate(observation["global"]):
            if abs(float(value) - progress) < 1e-12 and progress > 0:
                self.fail(f"global[{index}] duplicates progress ({progress})")
