"""Geometry stays in the loop; only the win test moves to the evaluator.

`AngleEnv.is_at_par` asks three questions -- every face a quad, vertex score
equal to par, and min scaled Jacobian at least `quality_threshold` -- and asks
the third of the LAPLACIAN-smoothed mesh. A Laplacian sits at its own fixed
point and cannot open a folded element, so that third question measures the
smoother rather than the topology. Measured: a hand-played solution of Diamond
in square is at par with min quality 0.196 and stays there under two hundred
further sweeps, while geo2d's untangler takes the same topology to 0.660.

The pinwheel family makes this decisive rather than merely untidy. The k=4
pinwheel -- the diamond-in-square connectivity, the one the released agent
cannot find -- sits at 0.203 under a Laplacian, so the old test refuses it while
accepting k=5 (0.516) and k=6 (0.672). Training on that family under the old
test would teach the structure in cloning and then punish it in reinforcement:
no par bonus, no termination, and a quality penalty at time-out, for producing
exactly the right answer.

So this env keeps everything `AngleEnvWithLength` has -- the ten features
including the two geometric ones, the Laplacian between moves, the same
observation, the same eight global entries -- and changes one method:

  topology at par?  ->  untangle IN PLACE  ->  above the DEGENERACY bar?  -> solved
                                            -> otherwise carry on from there

The untangled mesh is KEPT. `resmooth_env` reverts its own result whenever it is
worse than what it was handed, so its output is never worse: there is nothing to
protect the episode from, and handing the agent a better drawing of its own
topology beats telling it the drawing was bad. That also means no extra
observation entry and no flag -- a state at par with poor quality already says so
through the quality scalar the global vector has always carried, and what the
agent does about it is keep going. Checkpoints stay compatible.

The bar is low on purpose. It is there to refuse a topology that admits no
non-degenerate embedding -- a vertex inserted on the edge of a triangle forces a
180-degree corner, and no smoother repairs that -- not to demand a good mesh. A
mesh at 0.25 is smoothing and refinement's problem, downstream of the agent.

Cost: the untangler runs only when the topology is already at par, which is once
or twice an episode, and the verdict is cached per step so `is_terminated` and
the reward path do not pay for it twice. The in-loop settings are deliberately
cheaper than the ones used to SCORE a finished run: four sweeps with a Laplacian
slide cost about 2 ms and put the k=4 pinwheel at 0.414, where twelve with the
optimised slide cost 101 ms and reach 0.660. Both clear a 0.1 bar by a wide
margin, and the cheap one errs toward refusing, which is the safe direction for
a gate. An episode is roughly 35 ms of stepping.
"""

from envs.angle_env_with_length import AngleEnvWithLength


class EvaluatorAngleEnv(AngleEnvWithLength):
    """`AngleEnvWithLength` with the quality gate moved to the untangler."""

    def __init__(self, *args, degeneracy_threshold=0.1, evaluator="untangle",
                 untangle_iterations=None, warm_start_iterations=None,
                 evaluator_slide=None, **kwargs):
        self.degeneracy_threshold = degeneracy_threshold
        self.evaluator = evaluator
        self.evaluator_slide = evaluator_slide
        self.untangle_iterations = untangle_iterations
        self.warm_start_iterations = warm_start_iterations
        self._verdict_step = None
        self._verdict = None
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        env = super().from_config(config)
        env.degeneracy_threshold = config.get("degeneracy_threshold", 0.1)
        env.evaluator = config.get("evaluator", "untangle")
        # None means "use the shared setting", so the gate and the evaluator
        # cannot drift apart by default
        env.untangle_iterations = config.get("untangle_iterations")
        env.evaluator_slide = config.get("evaluator_slide")
        env.warm_start_iterations = config.get("warm_start_iterations")
        return env

    def evaluate_candidate(self):
        """Untangle this mesh in place, then ask whether it is degenerate.

        Do NOT deepcopy the env to do this. `graph_initializer` may be a
        geo2d-backed Mixture holding a module reference, so `deepcopy(self)`
        raises `cannot pickle 'module' object`; caught by a broad `except`, that
        silently degrades the gate to raw angles and nothing says so.
        """
        if self.evaluator == "none":
            return True
        if self.evaluator == "raw":
            return self.min_element_quality() >= self.degeneracy_threshold
        try:
            from src.geo2d_bridge import resmooth_env
            # The gate must smooth EXACTLY as the evaluator does, or the agent
            # is trained against a different mesh from the one it is scored on.
            # The config keys are honoured when set so an experiment can still
            # vary them deliberately; left at their defaults they resolve to
            # `geo2d_bridge.SMOOTHING`, the one shared setting. The warm start
            # now lives inside `resmooth_env` for the same reason.
            resmooth_env(self, iters=self.untangle_iterations,
                         method="optimize", slide=self.evaluator_slide,
                         warm_start=self.warm_start_iterations)
        except Exception:
            # geogen is optional and the optimiser can fail on a tangle; the raw
            # angles are strictly the harsher test, so this never accepts
            # something the untangler would have refused
            pass
        return self.min_element_quality() >= self.degeneracy_threshold

    def is_at_par(self):
        if not self.is_topologically_at_par():
            return False
        # `is_terminated` and the reward path both ask, on the same state
        if self._verdict_step == self.num_steps and self._verdict is not None:
            return self._verdict
        self._verdict_step = self.num_steps
        self._verdict = self.evaluate_candidate()
        return self._verdict

    def reset(self, *args, **kwargs):
        self._verdict_step = None
        self._verdict = None
        return super().reset(*args, **kwargs)
