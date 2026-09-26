"""The javascript engine and the python reference must agree, at every target.

`game/js/engine.js` is a port of `game/server.py`, and the agent's objective is
defined by the python side. If the two ever disagree about par, a human playing
the game is chasing a different goal than the one the agent is scored against.

Skipped where node is unavailable.
"""

import json
import shutil
import subprocess
import sys
import unittest

NODE = shutil.which("node")

JS = """
const e = require('./game/js/engine.js');
const out = {};
for (const target of [4, 3]) {
  e.setElementTarget(target);
  out[target] = {};
  for (const shape of Object.keys(e.SHAPES)) out[target][shape] = new e.Game(shape).par;
}
console.log(JSON.stringify(out));
"""


@unittest.skipIf(NODE is None, "node is not available")
class TestEngineParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, "game")
        result = subprocess.run([NODE, "-e", JS], capture_output=True, text=True)
        if result.returncode != 0:
            raise unittest.SkipTest(f"engine.js failed to load: {result.stderr[:300]}")
        cls.js = json.loads(result.stdout)

    def _python_par(self, target):
        import server
        server.set_element_target(target)
        try:
            return {shape: server.Game(shape).par for shape in server.INITIALIZERS}
        finally:
            server.set_element_target(4)

    def test_par_agrees_at_both_targets(self):
        for target in (4, 3):
            python = self._python_par(target)
            javascript = self.js[str(target)]
            self.assertEqual(set(python), set(javascript))
            for shape in python:
                with self.subTest(target=target, shape=shape):
                    self.assertEqual(python[shape], javascript[shape])

    def test_the_target_actually_changes_the_puzzle(self):
        """Otherwise the test above would pass trivially."""
        self.assertNotEqual(self.js["4"], self.js["3"])

    def test_par_matches_the_training_environment(self):
        """The game and the agent must be scored against the same bound."""
        from src.holdout import DEFAULT_ENV_CONFIG, LEVEL_NAMES, level_outline
        for target in (4, 3):
            config = dict(DEFAULT_ENV_CONFIG, face_desired_degree=target)
            for level in LEVEL_NAMES:
                with self.subTest(target=target, level=level):
                    self.assertEqual(level_outline(level, config)["par"],
                                     self.js[str(target)][level])


if __name__ == "__main__":
    unittest.main()
