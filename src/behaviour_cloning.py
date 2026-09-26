"""Shared machinery for cloning certified solutions.

Both the plain behaviour-cloning workflow and the expert-iteration loop turn
verified trajectories into (observation, action, return) triples and take
gradient steps on them, so the dataset, the collection pass and the update
live here rather than in either workflow.
"""

import numpy as np
import torch
import torch.nn.functional as F

from envs.environment_maker import initialize_environment, get_env_feature_size
from envs.solved_instances import replay
from src.feature_extractor import feature_extractor_initializer
from src.policy import CustomActorCriticPolicy
from src.utils import learning_rate_schedule
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


INT_KEYS = ("next", "previous", "twin")
# how many equally-optimal moves a single state may record; states with more
# than this are truncated, which only softens the target a little
MAX_PLAYABLE = 24


class Dataset:
    """Compact store of (observation, action, value target) triples."""

    def __init__(self):
        self.features = []
        self.next = []
        self.previous = []
        self.twin = []
        self.mask = []
        self.progress = []
        self.global_features = []
        self.actions = []
        self.playable = []
        self.values = []
        self.weights = []

    def add(self, obs, action, value, playable=None, weight=1.0):
        self.features.append(obs["features"].astype(np.float32))
        # int16, not int8: the template index can exceed 127 on a wide window
        self.next.append(obs["next"].astype(np.int16))
        self.previous.append(obs["previous"].astype(np.int16))
        self.twin.append(obs["twin"].astype(np.int16))
        self.mask.append(np.isfinite(obs["mask"]))
        self.progress.append(obs["progress"].astype(np.float32))
        if "global" in obs:
            self.global_features.append(obs["global"].astype(np.float32))
        self.actions.append(action)
        targets = np.full(MAX_PLAYABLE, -1, dtype=np.int32)
        options = [action] if playable is None else list(playable)[:MAX_PLAYABLE]
        targets[:len(options)] = options
        self.playable.append(targets)
        self.values.append(value)
        # a trajectory that only got close still teaches how to finish a mesh;
        # weighting is what keeps it from also teaching that close is good enough
        self.weights.append(weight)

    def finalize(self):
        if not self.actions:
            # an empty set is legitimate -- a round can harvest nothing, and the
            # demo replay is handed whatever the round produced
            self.features = np.zeros((0, 0, 0), dtype=np.float32)
            for name in ("next", "previous", "twin", "mask", "progress",
                         "global_features", "playable"):
                setattr(self, name, np.zeros((0, 0), dtype=np.float32))
            self.actions = np.zeros(0, dtype=np.int64)
            self.values = np.zeros(0, dtype=np.float32)
            self.weights = np.zeros(0, dtype=np.float32)
            return self
        self.features = np.stack(self.features)
        self.next = np.stack(self.next)
        self.previous = np.stack(self.previous)
        self.twin = np.stack(self.twin)
        self.mask = np.stack(self.mask)
        self.progress = np.stack(self.progress)
        if self.global_features:
            self.global_features = np.stack(self.global_features)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.playable = np.stack(self.playable)
        self.values = np.asarray(self.values, dtype=np.float32)
        self.weights = np.asarray(self.weights, dtype=np.float32)
        return self

    def __len__(self):
        return len(self.actions)

    def batch(self, index):
        mask = np.where(self.mask[index], 0.0, -np.inf).astype(np.float32)
        obs = {
            "features": torch.as_tensor(self.features[index]),
            "next": torch.as_tensor(self.next[index].astype(np.int64)),
            "previous": torch.as_tensor(self.previous[index].astype(np.int64)),
            "twin": torch.as_tensor(self.twin[index].astype(np.int64)),
            "mask": torch.as_tensor(mask),
            "progress": torch.as_tensor(self.progress[index]),
        }
        if len(self.global_features):
            obs["global"] = torch.as_tensor(self.global_features[index])
        return (obs,
                torch.as_tensor(self.actions[index]),
                torch.as_tensor(self.playable[index].astype(np.int64)),
                torch.as_tensor(self.values[index]),
                torch.as_tensor(self.weights[index]))


def drop_holdout_domains(instances, levels=None, targets=(3, 4), verbose=True):
    """Remove every instance posing the same problem as a held-out level.

    The generator never sees the levels, but it does build them: L-shape,
    T-bracket, Z-shape and Plus are small polyominoes, and the polar family at
    three quads is a triangle with par 1 -- which is exactly the Triangle
    level. Matching on the outline signature (the cyclic sequence of desired
    degrees, up to rotation and reflection) is the conservative rule: it drops
    the domain, not merely the coordinates, so a level cannot be solved by
    having practised the same problem in a different pose.

    A domain's signature depends on the target it is meshed for, so the ban
    list is built per target and an instance is checked against its own.
    """
    from src.holdout import DEFAULT_ENV_CONFIG, canonical_signature, holdout_signatures

    banned = {}
    for target in targets:
        config = dict(DEFAULT_ENV_CONFIG, face_desired_degree=target)
        banned[target] = set(holdout_signatures(levels, config).values())

    kept, dropped = [], 0
    for instance in instances:
        signature = canonical_signature([instance.desired[v] for v in instance.loop])
        if signature in banned.get(instance.target, set()):
            dropped += 1
            continue
        kept.append(instance)
    if verbose:
        print(f"  dropped {dropped} instances sharing a held-out outline signature "
              f"({dropped / max(len(instances), 1):.1%}); {len(kept)} remain")
    return kept


def balance_replays(instances, base_replays, max_replays=20):
    """Replay counts that flatten the (outline size, par, hole) distribution.

    The generator's natural output is dominated by small par-0 outlines, but
    the levels that are hard are the large ones, the high-par ones and the ones
    with holes. Replaying the rare buckets more often is the cheapest way to
    stop the clone from simply ignoring them.
    """
    from collections import Counter

    def bucket(instance):
        degrees = {instance.desired[v] for v in instance.loop}
        # whether the outline wants a single parity of degree: a rectilinear
        # domain asked for triangles wants only {3,5}, and the triforce ring
        # only {2,4,6}, which the seed families produce far more rarely than
        # the mixed outlines they mostly make
        parity = ("odd" if all(d % 2 for d in degrees)
                  else "even" if not any(d % 2 for d in degrees) else "mixed")
        return (instance.target, min(instance.polygon_degree // 4, 6),
                min(instance.par, 4), instance.has_hole, parity)

    counts = Counter(bucket(i) for i in instances)
    target = float(np.median(list(counts.values())))
    return [int(min(max(1, round(base_replays * target / counts[bucket(i)])), max_replays))
            for i in instances]


def collect(env, instances, replays, rng, dataset=None, max_samples=None):
    """Replay solutions through the env, recording states and optimal moves.

    `env` is either one environment or a mapping from face target to
    environment, for a dataset mixing quad and triangle instances. `replays` is
    either a count for every instance or one count per instance (see
    `balance_replays`).
    """
    dataset = dataset or Dataset()
    environments = env if isinstance(env, dict) else None
    per_instance = replays if isinstance(replays, (list, tuple, np.ndarray)) else None
    for position, instance in enumerate(instances):
        if environments is not None:
            if instance.target not in environments:
                continue
            env = environments[instance.target]
        step_cost = env.step_cost
        par_bonus = env.par_bonus
        replays_here = per_instance[position] if per_instance is not None else replays
        # the env would time out before the solution finishes; such a state is
        # not one the agent can be asked to solve, so it is not training data
        budget = max(env.min_max_steps,
                     int(np.ceil(env.max_steps_factor * instance.polygon_degree)))
        if instance.cost_to_go > budget:
            continue
        for attempt in range(replays_here):
            order_rng = rng if attempt or replays_here > 1 else None
            observations, actions, potentials, playable, played, ok = replay(
                env, instance, rng=order_rng)
            if not ok or played != len(instance.moves) or not env.is_at_par():
                continue
            total = len(actions)
            for step, (obs, action, potential, options) in enumerate(
                    zip(observations, actions, potentials, playable)):
                remaining = total - step
                # the return of the certified continuation under the env's own
                # potential-based reward with gamma = 1: the walk ends at par,
                # where the potential is zero
                value = -potential - step_cost * remaining + par_bonus
                dataset.add(obs, action, value, options)
            if max_samples and len(dataset.actions) >= max_samples:
                return dataset
    return dataset


def build_model(config, env_config, output_dir):
    env = make_vec_env(lambda: initialize_environment(env_config), 1, vec_env_cls=DummyVecEnv)
    features_extractor_class, features_extractor_kwargs = feature_extractor_initializer(config)
    features_extractor_kwargs.update({"input_features": get_env_feature_size(env_config)})
    policy_kwargs = dict(config["policy"])
    policy_kwargs["features_extractor_class"] = features_extractor_class
    policy_kwargs["features_extractor_kwargs"] = features_extractor_kwargs
    ppo_config = dict(config["PPO"])
    ppo_config["learning_rate"] = learning_rate_schedule(ppo_config["learning_rate"])
    ppo_config["policy_kwargs"] = policy_kwargs
    ppo_config["verbose"] = 0
    ppo_config["tensorboard_log"] = output_dir
    return PPO(CustomActorCriticPolicy, env, **ppo_config)


def train_epoch(policy, optimizer, dataset, rng, batch_size=256, vf_coef=0.25,
                max_grad_norm=1.0):
    """One pass over the dataset. Returns (cross entropy, top-1 accuracy)."""
    num_samples = len(dataset)
    order = rng.permutation(num_samples)
    batches = [order[start:start + batch_size]
               for start in range(0, num_samples - batch_size + 1, batch_size)]
    return _run_batches(policy, optimizer, dataset, batches, vf_coef, max_grad_norm)


def train_batches(policy, optimizer, dataset, rng, num_batches, batch_size=256,
                  vf_coef=0.25, max_grad_norm=1.0, loss_scale=1.0):
    """A few minibatches drawn at random, rather than a full pass.

    This is what lets a demonstration set be replayed alongside an RL update
    without the cost of sweeping it every iteration.
    """
    num_samples = len(dataset)
    if num_samples < batch_size:
        batch_size = num_samples
    if batch_size <= 0:
        return float("nan"), float("nan")
    batches = [rng.choice(num_samples, size=batch_size, replace=False)
               for _ in range(num_batches)]
    return _run_batches(policy, optimizer, dataset, batches, vf_coef,
                        max_grad_norm, loss_scale)


def _run_batches(policy, optimizer, dataset, batches, vf_coef, max_grad_norm,
                 loss_scale=1.0):
    policy.train()
    losses, accuracies = [], []
    for index in batches:
        obs, actions, playable, values, weights = dataset.batch(index)
        features = policy.features_extractor(obs)
        latent_pi, latent_vf = policy.mlp_extractor(features, obs)
        logits = policy.action_net(latent_pi)
        logits = logits.reshape(logits.shape[0], -1) + obs["mask"]
        predicted = policy.value_net(latent_vf).squeeze(-1)

        # cross-entropy against a uniform distribution over the moves that are
        # equally optimal here, not against one arbitrary member of that set
        log_probs = F.log_softmax(logits, dim=1)
        valid = playable >= 0
        gathered = log_probs.gather(1, playable.clamp(min=0))
        # padding slots point at index 0, which may be a masked action whose
        # log-probability is -inf; select rather than multiply, or 0 * -inf
        # poisons the whole batch with NaN
        gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
        per_sample = -gathered.sum(dim=1).div(valid.sum(dim=1).clamp(min=1))
        # weighted mean, so a batch of mostly low-weight samples still yields a
        # gradient of the usual magnitude rather than a vanishing one
        policy_loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
        value_loss = (F.mse_loss(predicted, values, reduction="none") * weights
                      ).sum() / weights.sum().clamp(min=1e-6)
        loss = loss_scale * (policy_loss + vf_coef * value_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        losses.append(policy_loss.item())
        chosen = logits.argmax(1, keepdim=True)
        hit = ((playable == chosen) & valid).any(dim=1).float().mean()
        accuracies.append(hit.item())
    if not losses:
        return float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(accuracies))


def excess_weight(excess, beta=1.0):
    """Weight for a complete-but-suboptimal trajectory, 1.0 at par.

    Reward-weighted-regression form: the preference for par is continuous rather
    than a cliff, so a near-miss still teaches how to finish a mesh without also
    teaching that a near-miss is good enough.
    """
    return float(np.exp(-beta * max(excess, 0)))


def add_trajectory(dataset, observations, actions, potentials, step_cost, par_bonus,
                   weight=1.0, at_par=True):
    """Record a trajectory; the value target is its own exact return.

    `at_par` is not cosmetic. The return of a trajectory that never reached par
    does not include the bonus, and recording it as though it did would teach
    the critic to expect a payment the episode never made.
    """
    total = len(actions)
    bonus = par_bonus if at_par else 0.0
    for step, (obs, action, potential) in enumerate(zip(observations, actions, potentials)):
        remaining = total - step
        dataset.add(obs, action, -potential - step_cost * remaining + bonus,
                    weight=weight)
    return dataset
