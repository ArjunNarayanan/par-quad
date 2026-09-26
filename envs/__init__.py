from envs.hex_env import HexEnv
from gymnasium.envs.registration import register

register(
    id="envs/HexEnv-v0",
    entry_point="envs:HexEnv",
)
