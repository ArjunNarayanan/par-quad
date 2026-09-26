"""Log solve rate, not just return.

Once the reward is potential-based the mean episode return says little about
whether episodes are actually being won, and it is not comparable between runs
with different step costs. Solve rate is.

Both numbers describe the TRAJECTORY, not its last state. With
`best_so_far_reward` the episode deliberately runs past its own goal to
max_steps, so the final mesh is whatever the monotone vocabulary did after the
good one: `at_par` read there is near zero however well the agent is doing, and
`num_steps` is the step cap for every episode alike. The env reports
`best_at_par` and `moves_to_best` for exactly this reason, and they fall back
to the old fields when it does not.
"""

from stable_baselines3.common.callbacks import BaseCallback


class SolveRateCallback(BaseCallback):
    def __init__(self, window=10000, verbose=0):
        super().__init__(verbose)
        self.window = window
        self._solved = 0
        self._episodes = 0
        self._steps = 0
        self._overflow = 0
        self._last_log = 0

    def _on_step(self):
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if not done:
                continue
            self._episodes += 1
            solved = info.get("best_at_par", info.get("at_par", False))
            self._solved += int(bool(solved))
            self._steps += int(info.get("moves_to_best", info.get("num_steps", 0)))

        if self.num_timesteps - self._last_log < self.window or self._episodes == 0:
            return True
        self._last_log = self.num_timesteps
        self.logger.record("rollout/solve_rate", self._solved / self._episodes)
        self.logger.record("rollout/mean_episode_moves", self._steps / self._episodes)
        self._solved = self._episodes = self._steps = 0
        return True
