"""Warm start a policy by cloning, then improve it with PPO on harder shapes.

    venv/bin/python workflows/train_warm_ppo.py \
        -config experiments/self-play/unified/warm-ppo-v1/config.yml \
        -checkpoint experiments/self-play/unified/unified-bc-v4/bc_model.zip \
        -num_envs 8

WHERE THE SHAPES COME FROM is entirely a config decision. The environment's
`initializer` block is dispatched by `envs/environment_maker.get_env_initializer`,
and an initializer is just a callable with this contract:

    initializer() -> (Tiler, {untagged vertex id: desired degree})

A generator that lives outside this repository plugs in with no code change
here:

    initializer:
      name: Custom
      factory: "my_package.shapes:make_initializer"
      # ... any other keys are passed through as keyword arguments

and several sources can be blended without either knowing about the other:

    initializer:
      name: Mixture
      weights: [0.5, 0.2, 0.3]
      components:
        - {name: Custom, factory: "my_package.shapes:make_initializer"}
        - {name: RandomPolygonWithHole, min_polygon_degree: 6, max_polygon_degree: 14}
        - {name: SolvedInstances, path: data/quad_instances_v3.pkl}

THE CRITIC IS CALIBRATED BEFORE THE POLICY MOVES, and this is not optional. A
cloned checkpoint's value head was fit on the value targets cloning used; if the
RL stage runs a different `reward_mode` the two scales need not agree at all.
Measured on a cloned checkpoint against `reward_mode: normalized`, the critic
predicted 8.53 where the actual return was 0.80 -- so every advantage came out
near -7.7, uniformly negative, and the first updates pushed down on everything
the policy did. The collapse that looks like catastrophic forgetting is mostly
this. `-critic_warmup` fits the value head with the rest of the network frozen
until the advantages mean something.
"""

import argparse
import datetime
import os
import pickle
import sys

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

sys.path.append(os.getcwd())

from envs.environment_maker import get_env_feature_size, initialize_environment  # noqa: E402
from src.behaviour_cloning import Dataset, add_trajectory, drop_holdout_domains  # noqa: E402
from src.demo_rollouts import DemoDriver, PPOWithDemoRollouts  # noqa: E402
from src.feature_extractor import feature_extractor_initializer  # noqa: E402
from src.policy import CustomActorCriticPolicy  # noqa: E402
from src.ppo_with_demos import PPOWithDemos  # noqa: E402
from src.solve_rate_callback import SolveRateCallback  # noqa: E402
from src.utils import learning_rate_schedule, load_yaml_config  # noqa: E402

ANCHORS = ("none", "mixture", "inject", "aux")


def with_demo_domains(env_config, dataset, weight, generator=None):
    """Put the certified domains into the training distribution itself.

    The cheapest anchor there is, and the first one to try: the policy keeps
    being asked to solve the problems cloning taught it, so the reward keeps
    reinforcing that behaviour without any special machinery. No forcing, no
    supervised term -- just the domains, alongside the random ones.

    `generator`, when given, replaces the pickle with a live generator. The
    geo2d half of the mixture draws a new domain every episode, so a fixed
    pickle makes the two halves incomparable: over a 4M-step run the random
    side sees about 124,000 unseen domains while the certified side recycles
    9,000 about seven times each -- and since geo2d only builds par 0 domains,
    every par > 0 instance the agent meets comes from that finite list.
    """
    initializer = env_config.get("initializer", {})
    demos = ({"name": "GeneratedInstances", **generator} if generator
             else {"name": "SolvedInstances", "dataset": dataset})
    if initializer.get("name") == "Mixture":
        merged = dict(initializer)
        components = list(merged.get("components", []))
        weights = list(merged.get("weights", [1.0] * len(components)))
        total = sum(weights) or 1.0
        merged["components"] = components + [demos]
        merged["weights"] = [w * (1.0 - weight) / total for w in weights] + [weight]
    else:
        merged = {"name": "Mixture",
                  "components": [initializer, demos],
                  "weights": [1.0 - weight, weight]}
    return dict(env_config, initializer=merged)


def build_env(env_config, num_envs, demo_instances=None, anchor="none",
              demo_probability=0.25, seed=0):
    def factory(rank):
        def make():
            env = initialize_environment(env_config)
            if anchor == "inject" and demo_instances:
                driver = DemoDriver(demo_instances, env.graph_initializer,
                                    probability=demo_probability,
                                    rng=np.random.default_rng(seed + rank))
                env.graph_initializer = driver
                env.demo_driver = driver
            return env
        return make

    cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    return cls([factory(rank) for rank in range(num_envs)])


def calibrate_critic(model, steps, epochs, learning_rate, verbose=True):
    """Fit the value head to the reward actually in use, policy frozen.

    Without this the advantages are dominated by the mismatch between the value
    scale cloning produced and the one this run pays out, and the first updates
    move the policy on noise.
    """
    # the critic's own layers: SB3's final linear readout and the two-layer MLP
    # in CustomNetwork that feeds it. The MLP is critic-only -- the actor reads
    # the shared context through `context_to_policy`, never through it -- so
    # fitting it cannot move the policy, and without it a single linear layer
    # has to absorb a tenfold change of return scale, which it cannot.
    critic_only = ("value_net.", "mlp_extractor.value_net.")
    frozen, trainable = [], []
    for name, parameter in model.policy.named_parameters():
        if name.startswith(critic_only):
            trainable.append(parameter)
        else:
            frozen.append((parameter, parameter.requires_grad))
            parameter.requires_grad_(False)
    optimizer = th.optim.Adam(trainable, lr=learning_rate)
    try:
        # SB3 needs its callback plumbing initialised before collect_rollouts
        _, callback = model._setup_learn(steps, None)
        callback.on_training_start(locals(), globals())
        model.collect_rollouts(model.env, callback, model.rollout_buffer,
                               n_rollout_steps=model.n_steps)
        callback.on_training_end()

        before, after = [], []
        for epoch in range(epochs):
            losses = []
            for batch in model.rollout_buffer.get(model.batch_size):
                values = model.policy.predict_values(batch.observations).flatten()
                loss = th.nn.functional.mse_loss(batch.returns.flatten(), values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss))
            if epoch == 0:
                before = list(losses)
            after = list(losses)
        if verbose and before:
            print(f"  critic calibration: value loss "
                  f"{np.mean(before):.3f} -> {np.mean(after):.3f}")
    finally:
        for parameter, required in frozen:
            parameter.requires_grad_(required)


def space_overrides(checkpoint, env):
    """Let a checkpoint trained at one template size warm start at another.

    The network is per-half-edge with pooling, so nothing in the weights depends
    on `template_size`; only the spaces stored in the checkpoint do, and SB3
    refuses to load against an env whose spaces differ unless they are
    re-declared through `custom_objects`. The global feature vector is different:
    its width is baked into the context net, so a mismatch there (from
    `target_in_observation`) cannot be papered over and is reported instead.
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    data, _, _ = load_from_zip_file(checkpoint, load_data=True)
    saved = data["observation_space"].spaces
    live = env.observation_space.spaces
    if saved["global"].shape != live["global"].shape:
        raise SystemExit(
            f"checkpoint has {saved['global'].shape[0]} global features, the env "
            f"{live['global'].shape[0]}: set `target_in_observation` in the config to "
            f"match the checkpoint it was trained with")
    if saved["features"].shape == live["features"].shape:
        return {}
    print(f"  checkpoint template {saved['features'].shape[0]} -> "
          f"env template {live['features'].shape[0]}; re-declaring the spaces")
    return {"observation_space": env.observation_space,
            "action_space": env.action_space}


def load_demo_instances(path, limit, exclude_holdout=True):
    with open(path, "rb") as handle:
        instances = pickle.load(handle)
    if exclude_holdout:
        instances = drop_holdout_domains(instances, verbose=False)
    return instances[:limit]


def demo_dataset_from(instances, env_config, limit=200):
    """Certified trajectories as (observation, action) pairs for the aux term."""
    scratch = initialize_environment(env_config)
    data = Dataset()
    kept = 0
    for instance in instances:
        solo = DemoDriver([instance], None, probability=1.0,
                          rng=np.random.default_rng(0))
        graph, desired = solo()
        scratch._reset_to_state(graph, dict(desired))
        scratch.demo_driver = solo
        observations, actions, potentials = [], [], []
        for _ in range(scratch.max_steps):
            linear = solo.next_action(scratch)
            if linear is None:
                break
            observations.append({k: np.array(v, copy=True)
                                 for k, v in scratch._get_obs().items()})
            actions.append(linear)
            potentials.append(scratch.potential)
            scratch.step(linear)
        if actions and scratch.is_at_par():
            add_trajectory(data, observations, actions, potentials,
                           env_config.get("step_cost", 0.05),
                           env_config.get("par_bonus", 1.0))
            kept += 1
        if kept >= limit:
            break
    data.finalize()
    return data, kept



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-config", required=True)
    parser.add_argument("-checkpoint", default=None,
                        help="cloned policy to warm start from")
    parser.add_argument("-num_envs", default=4, type=int)
    parser.add_argument("-total_timesteps", default=None, type=int)
    parser.add_argument("-anchor", default="mixture", choices=ANCHORS,
                        help="how the certified data keeps influencing the run: "
                             "'mixture' just includes those domains in the "
                             "training distribution, 'inject' forces their "
                             "trajectories into the rollouts, 'aux' replays them "
                             "as a supervised term, 'none' does nothing")
    parser.add_argument("-demos", default="data/quad_instances_v3.pkl")
    parser.add_argument("-demo_limit", default=2000, type=int)
    parser.add_argument("-demo_probability", default=0.25, type=float)
    parser.add_argument("-demo_generator", action="store_true",
                        help="draw the certified anchor from the generator each "
                             "batch instead of a fixed pickle, so the certified "
                             "half of the mixture is as fresh as the random half")
    parser.add_argument("-demo_batch", default=256, type=int,
                        help="instances per generated batch; a batch costs about "
                             "twenty seconds and lasts a few hundred episodes")
    parser.add_argument("-demo_hole_probability", default=0.30, type=float)
    parser.add_argument("-demo_pinwheel_probability", default=0.0, type=float)
    parser.add_argument("-demo_gate", default="untangle",
                        choices=("laplacian", "untangle", "topology"))
    parser.add_argument("-demo_min_cells", default=2, type=int)
    parser.add_argument("-demo_max_cells", default=14, type=int)
    parser.add_argument("-keep_holdout_domains", action="store_true")
    parser.add_argument("-critic_warmup", default=60, type=int,
                        help="epochs of value-only fitting before the policy moves")
    parser.add_argument("-critic_lr", default=1.0e-3, type=float)
    parser.add_argument("-seed", default=0, type=int)
    parser.add_argument("-torch_threads", default=None, type=int,
                        help="intra-op threads for the update step (the env workers "
                             "are separate processes); defaults to torch's own choice")
    parser.add_argument("-out", default=None)
    args = parser.parse_args()
    if args.torch_threads:
        th.set_num_threads(args.torch_threads)

    config = load_yaml_config(args.config)
    output_dir = args.out or os.path.dirname(os.path.abspath(args.config))
    os.makedirs(output_dir, exist_ok=True)
    env_config = dict(config["environment"])
    env_config.setdefault("logdir", output_dir)

    generator = None
    if args.demo_generator:
        generator = {
            "batch_size": args.demo_batch,
            "cell_range": [args.demo_min_cells, args.demo_max_cells],
            "hole_probability": args.demo_hole_probability,
            "pinwheel_probability": args.demo_pinwheel_probability,
            "gate": args.demo_gate,
        }
    if args.anchor == "mixture":
        if generator is None and not os.path.exists(args.demos):
            raise SystemExit(f"-anchor mixture needs {args.demos}")
        env_config = with_demo_domains(env_config, args.demos,
                                       args.demo_probability, generator=generator)
        source = ("the generator, fresh every "
                  f"{args.demo_batch} instances" if generator else args.demos)
        print(f"training distribution includes the certified domains from "
              f"{source} at weight {args.demo_probability}")

    demo_instances = []
    if args.anchor in ("inject", "aux") and os.path.exists(args.demos):
        demo_instances = load_demo_instances(
            args.demos, args.demo_limit,
            exclude_holdout=not args.keep_holdout_domains)
        print(f"demonstrations: {len(demo_instances)} certified instances")

    env = build_env(env_config, args.num_envs, demo_instances, args.anchor,
                    args.demo_probability, args.seed)

    features_extractor_class, features_extractor_kwargs = feature_extractor_initializer(config)
    features_extractor_kwargs.update({"input_features": get_env_feature_size(env_config)})
    policy_kwargs = dict(config["policy"])
    policy_kwargs["features_extractor_class"] = features_extractor_class
    policy_kwargs["features_extractor_kwargs"] = features_extractor_kwargs

    ppo_config = dict(config["PPO"])
    ppo_config["learning_rate"] = learning_rate_schedule(ppo_config["learning_rate"])
    ppo_config["policy_kwargs"] = policy_kwargs
    ppo_config["verbose"] = 1
    ppo_config["tensorboard_log"] = output_dir
    ppo_config["seed"] = args.seed

    algorithm = PPO
    extra = {}
    if args.anchor == "inject":
        algorithm = PPOWithDemoRollouts
    elif args.anchor == "aux":
        algorithm = PPOWithDemos
        data, kept = demo_dataset_from(demo_instances, env_config)
        if kept == 0:
            raise SystemExit("no certified trajectory replayed; the aux anchor "
                             "would silently be plain PPO")
        print(f"  demo dataset: {kept} trajectories, {len(data)} pairs")
        extra = dict(demo_dataset=data, bc_coef=1.0, bc_decay=0.8,
                     bc_min_coef=0.1, bc_batches=4, bc_batch_size=128)

    if args.checkpoint:
        print(f"\n\tWARM STARTING FROM : {args.checkpoint}\n")
        ppo_config.update(space_overrides(args.checkpoint, env))
        model = algorithm.load(args.checkpoint, env=env,
                               custom_objects=ppo_config, **extra)
    else:
        model = algorithm(CustomActorCriticPolicy, env, **ppo_config, **extra)

    if args.critic_warmup > 0 and args.checkpoint:
        print("calibrating the critic to this run's reward scale")
        calibrate_critic(model, model.n_steps, args.critic_warmup, args.critic_lr)

    total_timesteps = args.total_timesteps or config.get("total_timesteps", 1_000_000)
    # CheckpointCallback counts callback calls, one per vectorised step, so the
    # interval has to be divided by the number of envs to mean timesteps
    checkpoint_every = config.get("checkpoint_every", max(total_timesteps // 40, 1))
    callbacks = [SolveRateCallback(window=config.get("solve_rate_window", 10000)),
                 CheckpointCallback(save_freq=max(checkpoint_every // args.num_envs, 1),
                                    save_path=output_dir,
                                    name_prefix="warm_ppo")]

    print(f"PPO START : {datetime.datetime.now()}  anchor={args.anchor}  "
          f"envs={args.num_envs}  steps={total_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks),
                reset_num_timesteps=False)
    destination = os.path.join(output_dir, "warm_ppo_model.zip")
    model.save(destination)
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
