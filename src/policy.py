from stable_baselines3.common.policies import ActorCriticPolicy
import torch


class CustomNetwork(torch.nn.Module):
    """Per-half-edge trunk plus a whole-mesh context vector.

    The convolution only ever sees a few hops of the template, so on its own it
    cannot say how far the mesh is from par, how many odd faces are left, or
    how much time remains. Those live in `obs["global"]`; pooling them together
    with mean- and max-pooled half-edge features gives one context vector that
    is broadcast back onto every half-edge before the action head and is the
    critic's only input.
    """

    def __init__(self, input_features, output_features=None, global_features=0,
                 context_features=None):
        super().__init__()
        if output_features is None:
            output_features = input_features
        if context_features is None:
            context_features = output_features

        self.latent_dim_pi = output_features
        self.latent_dim_vf = output_features
        self.global_features = global_features
        # off by default: every existing checkpoint was trained with the actor
        # seeing the budget, and turning this on changes what it is shown
        self.hide_budget_from_actor = False

        # mean-pool + max-pool over valid half-edges, the global vector, progress
        context_input = 2 * input_features + global_features + 1
        self.context_net = torch.nn.Sequential(
            torch.nn.Linear(context_input, context_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(context_features, context_features),
            torch.nn.LeakyReLU(),
        )

        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(input_features, output_features),
            torch.nn.LeakyReLU()
        )
        self.context_to_policy = torch.nn.Linear(context_features, output_features)

        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(context_features, output_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(output_features, output_features),
            torch.nn.LeakyReLU(),
        )

    @staticmethod
    def _valid_mask(obs):
        """Padded template slots are exactly the all-zero rows of the raw features."""
        raw = obs["features"]
        return (raw.abs().sum(dim=-1) > 0).unsqueeze(-1).to(raw.dtype)

    def _context(self, features, obs, hide_budget=False):
        """The whole-mesh context vector.

        `hide_budget` zeroes `obs["progress"]`, which is `num_steps/max_steps`.

        The CRITIC must see it. The episode has a step limit, so the return
        from a state depends on how much of the limit is left: a state five
        moves from a solution is worth one thing with ten moves remaining and
        another with one, and a value function that cannot tell them apart is
        being asked to fit two different numbers to one input.

        The ACTOR should not. `max_steps` is
        `max(min_max_steps, factor * half_edges)` -- a training setting, not a
        property of the domain. A policy conditioned on it is conditioned on
        how much budget WE chose to give it, so a larger domain that needs more
        moves than anything in training reads as out of distribution for a
        reason that has nothing to do with its geometry. Measured on the curved
        agent, changing the factor alone made the deterministic trajectories
        diverge on half the domains.
        """
        valid = self._valid_mask(obs)
        count = valid.sum(dim=-2).clamp(min=1.0)
        mean_pool = (features * valid).sum(dim=-2) / count
        max_pool = features.masked_fill(valid == 0, float("-inf")).max(dim=-2).values
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)

        parts = [mean_pool, max_pool]
        if self.global_features > 0:
            parts.append(obs["global"])
        progress = obs["progress"]
        if hide_budget and self.hide_budget_from_actor:
            progress = torch.zeros_like(progress)
        parts.append(progress)
        return self.context_net(torch.cat(parts, dim=-1))

    def forward_actor(self, features, obs):
        context = self._context(features, obs, hide_budget=True)
        latent = self.policy_net(features)
        return latent + self.context_to_policy(context).unsqueeze(-2)

    def forward_critic(self, features, obs):
        return self.value_net(self._context(features, obs))

    def forward(self, features, obs):
        # two contexts, because the actor and the critic are shown different
        # things; identical when `hide_budget_from_actor` is off
        latent_pi = self.policy_net(features) + self.context_to_policy(
            self._context(features, obs, hide_budget=True)).unsqueeze(-2)
        latent_vf = self.value_net(self._context(features, obs))
        return latent_pi, latent_vf


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
    ):
        self.input_features = kwargs["features_extractor_kwargs"]["output_features"]
        self.output_features = kwargs.get("output_features", self.input_features)
        self.context_features = kwargs.get("context_features", None)
        self.use_critic = kwargs.get("use_critic", True)
        # config `policy: hide_budget_from_actor: true`
        self.hide_budget_from_actor = bool(kwargs.get("hide_budget_from_actor", False))

        if "global" in observation_space.spaces:
            self.global_features = int(observation_space.spaces["global"].shape[0])
        else:
            self.global_features = 0

        # remove keys that are not needed for super class
        kwargs.pop("output_features", None)
        kwargs.pop("context_features", None)
        kwargs.pop("use_critic", None)
        kwargs.pop("hide_budget_from_actor", None)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            normalize_images=False,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomNetwork(
            self.input_features,
            self.output_features,
            global_features=self.global_features,
            context_features=self.context_features,
        )
        self.mlp_extractor.hide_budget_from_actor = self.hide_budget_from_actor

    def _get_masked_action_dist_from_latent(self, latent_pi, mask):
        # latent_pi.shape == [batch_size, num_halfedges, num_features]
        assert mask.ndim == 2
        assert latent_pi.ndim == 3
        batch_size, num_halfedges, num_features = latent_pi.shape
        assert mask.shape[0] == batch_size

        action_logits = self.action_net(latent_pi)  # [batch_size, num_halfedges, num_actions_per_halfedge]
        action_logits = action_logits.reshape(batch_size, -1)  # [batch_size, num_actions_per_sample]
        action_logits = action_logits + mask

        return self.action_dist.proba_distribution(action_logits=action_logits)

    def predict_values(self, obs):
        features = self.features_extractor(obs)
        latent_vf = self.mlp_extractor.forward_critic(features, obs)
        values = self.value_net(latent_vf)
        return values

    def get_distribution(self, obs):
        features = self.features_extractor(obs)
        mask = obs["mask"]

        latent_pi = self.mlp_extractor.forward_actor(features, obs)
        distribution = self._get_masked_action_dist_from_latent(latent_pi, mask)
        return distribution

    def evaluate_actions(self, obs, actions):
        features = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(features, obs)

        mask = obs["mask"]
        distribution = self._get_masked_action_dist_from_latent(latent_pi, mask)
        log_prob = distribution.log_prob(actions)

        values = self.value_net(latent_vf)
        if not self.use_critic:
            values = torch.zeros_like(values)

        entropy = distribution.entropy()
        return values, log_prob, entropy

    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)

        latent_pi, latent_vf = self.mlp_extractor(features, obs)

        values = self.value_net(latent_vf)
        if not self.use_critic:
            values = torch.zeros_like(values)

        mask = obs["mask"]
        distribution = self._get_masked_action_dist_from_latent(latent_pi, mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob
