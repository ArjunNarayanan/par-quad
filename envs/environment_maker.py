from envs.substep_angle_env import AngleEnv as SubstepAngleEnv
from envs.random_polygon_tiler_env import RandomPolygonEnv
from envs.global_angle_env import AngleEnv as GlobalAngleEnv
from envs.angle_env_with_length import AngleEnvWithLength
from envs.evaluator_env import EvaluatorAngleEnv
from envs.topological_env import TopologicalEnv
import envs.environment_initializers as init


def get_env_initializer(init_config):
    name = init_config["name"]
    angle = init_config.get("target_angle", 90)
    if name == "LEnv":
        return init.LEnv(angle)
    elif name == "Hexagon":
        return init.Hexagon(angle)
    elif name == "CenterCrack":
        return init.CenterCrack(angle)
    elif name == "SquareHole":
        return init.SquareHole(angle)
    elif name == "RandomPolygon":
        min_degree = init_config["min_polygon_degree"]
        max_degree = init_config["max_polygon_degree"]
        scale = init_config.get("scale", 0.5)
        min_quality = init_config.get("min_quality", 0.4)
        degree_range = list(range(min_degree, max_degree + 1))
        return init.RandomPolygon(degree_range, angle, scale=scale, min_quality=min_quality)
    elif name == "RandomPolygonWithHole":
        min_degree = init_config["min_polygon_degree"]
        max_degree = init_config["max_polygon_degree"]
        degree_range = list(range(min_degree, max_degree + 1))
        hole_range = (init_config.get("min_hole_degree", 4),
                      init_config.get("max_hole_degree", 8))
        return init.RandomPolygonWithHole(
            degree_range, angle,
            scale=init_config.get("scale", 0.5),
            hole_degree_range=hole_range,
            hole_scale=init_config.get("hole_scale", 0.35),
            hole_margin=init_config.get("hole_margin", 0.9),
            min_quality=init_config.get("min_quality", 0.4))
    elif name == "FixedRandomPolygon":
        polygon_degree = init_config["polygon_degree"]
        scale = init_config.get("scale", 0.5)
        return init.FixedRandomPolygon(polygon_degree, angle, scale=scale)
    elif name == "Custom":
        # An initializer supplied from outside this repository. `factory` is a
        # dotted path "package.module:callable"; it is called with the rest of
        # the config as keyword arguments and must return a callable with the
        # initializer contract:
        #
        #     initializer() -> (Tiler, {untagged vertex id: desired degree})
        #
        # The Tiler is a single-face mesh of the domain -- the raw outline the
        # agent starts from -- and the mapping gives each ORIGINAL corner the
        # degree its angle asks for. A domain with a hole is one face with a
        # slit, so its two slit vertices appear twice in the face loop; see
        # envs/environment_initializers.RandomPolygonWithHole for a worked
        # example. Anything else the env needs it derives itself.
        import importlib

        spec = init_config["factory"]
        module_path, _, attribute = spec.partition(":")
        if not attribute:
            raise ValueError(
                f"Custom initializer factory must be 'module:callable', got {spec!r}")
        factory = getattr(importlib.import_module(module_path), attribute)
        options = {k: v for k, v in init_config.items()
                   if k not in ("name", "factory")}
        options.setdefault("target_angle", angle)
        return factory(**options)
    elif name == "SolvedInstances":
        from envs.solved_instances import SolvedInstanceInitializer
        return SolvedInstanceInitializer.from_config(init_config)
    elif name == "GeneratedInstances":
        from envs.solved_instances import GeneratedInstanceInitializer
        return GeneratedInstanceInitializer.from_config(init_config)
    elif name == "Mixture":
        components = [get_env_initializer(sub) for sub in init_config["components"]]
        weights = init_config.get("weights", None)
        return init.MixtureInitializer(components, weights)
    else:
        raise ValueError("Unexpected env initializer with name : ", name)


def initialize_environment(env_config):
    env_config = env_config.copy()

    env_name = env_config["name"]
    if "initializer" in env_config:
        initializer = get_env_initializer(env_config["initializer"])
        env_config["graph_initializer"] = initializer

    if env_name == "RandomPolygonEnv":
        return RandomPolygonEnv.from_config(env_config)
    elif env_name == "SubstepAngleEnv":
        return SubstepAngleEnv.from_config(env_config)
    elif env_name == "GlobalAngleEnv":
        return GlobalAngleEnv.from_config(env_config)
    elif env_name == "AngleEnvWithLength":
        return AngleEnvWithLength.from_config(env_config)
    elif env_name == "EvaluatorAngleEnv":
        return EvaluatorAngleEnv.from_config(env_config)
    elif env_name == "TopologicalEnv":
        return TopologicalEnv.from_config(env_config)
    else:
        raise TypeError("Unexpected environment name : ", env_name)


def get_env_feature_size(env_config):
    env_name = env_config["name"]
    if env_name == "RandomPolygonEnv":
        return RandomPolygonEnv.get_feature_size()
    elif env_name == "SubstepAngleEnv":
        return SubstepAngleEnv.get_feature_size()
    elif env_name == "GlobalAngleEnv":
        return GlobalAngleEnv.get_feature_size()
    elif env_name == "AngleEnvWithLength":
        return AngleEnvWithLength.get_feature_size()
    elif env_name == "EvaluatorAngleEnv":
        return EvaluatorAngleEnv.get_feature_size()
    elif env_name == "TopologicalEnv":
        return TopologicalEnv.get_feature_size()
    else:
        raise TypeError("Unexpected environment name : ", env_name)
