"""Sticky centering: finish a face before moving the window.

The global-argmax rule re-picks a centre from scratch every step, so one chord
can teleport the window to an unrelated part of the mesh, and its angle term
lets Laplacian smoothing move what is really a topological decision.
"""
import os
import sys
import unittest

import numpy as np

sys.path.append(os.getcwd())

from envs.environment_maker import initialize_environment
from src.utils import load_yaml_config

CONFIG = "experiments/self-play/unified/unified-bc-v4/config.yml"


def build(mode, **overrides):
    config = load_yaml_config(CONFIG)
    return initialize_environment(
        dict(config["environment"], center_mode=mode, **overrides))


def legal_action(env):
    legal = np.flatnonzero(np.isfinite(env._get_obs()["mask"]))
    return int(legal[0]) if len(legal) else None


class TestStickyCenter(unittest.TestCase):
    def test_centre_holds_while_its_face_is_unfinished(self):
        np.random.seed(0)
        env = build("sticky_face")
        env.reset()
        for _ in range(10):
            centre = env.template_center
            face_before = env.graph.face(centre)
            if env._face_irregularity(face_before) == 0:
                break
            action = legal_action(env)
            if action is None:
                break
            env.step(action)
            if not env.graph.is_half_edge(centre):
                break
            if env._face_irregularity(env.graph.face(centre)) != 0:
                self.assertEqual(env.template_center, centre)

    def test_centre_moves_once_its_face_is_the_element_we_asked_for(self):
        """Otherwise the agent would be pinned to a finished region."""
        np.random.seed(1)
        env = build("sticky_face")
        moved = False
        for _ in range(6):
            env.reset()
            for _ in range(14):
                centre = env.template_center
                action = legal_action(env)
                if action is None:
                    break
                env.step(action)
                if (env.graph.is_half_edge(centre)
                        and env._face_irregularity(env.graph.face(centre)) == 0
                        and env.template_center != centre):
                    moved = True
                    break
            if moved:
                break
        self.assertTrue(moved, "centre never moved off a completed face")

    def test_choice_is_deterministic_for_a_state(self):
        np.random.seed(2)
        first = build("sticky_face")
        first.reset()
        graph, desired = first.graph, dict(first.vertex_desired_degree)
        from copy import deepcopy
        second = build("sticky_face")
        second._reset_to_state(deepcopy(graph), dict(desired))
        third = build("sticky_face")
        third._reset_to_state(deepcopy(graph), dict(desired))
        self.assertEqual(second.template_center, third.template_center)

    def test_it_picks_the_worst_face_then_the_worst_vertex_on_it(self):
        np.random.seed(5)
        env = build("sticky_face")
        env.reset()
        centre = env.template_center
        worst_face = max(env._face_irregularity(f) for f in env.graph.face_list())
        self.assertEqual(env._face_irregularity(env.graph.face(centre)), worst_face)
        on_face = [h for h in env.graph.half_edge_list()
                   if env._face_irregularity(env.graph.face(h)) == worst_face]
        best_defect = max(env.vertex_defect(env.graph.source_vertex(h, tag=False))
                          for h in on_face)
        self.assertEqual(
            env.vertex_defect(env.graph.source_vertex(centre, tag=False)),
            best_defect)

    def test_default_mode_is_unchanged(self):
        """Existing checkpoints were cloned under the old rule; it stays default."""
        env = build("global_worst")
        self.assertEqual(env.center_mode, "global_worst")
        plain = initialize_environment(
            dict(load_yaml_config(CONFIG)["environment"]))
        self.assertEqual(plain.center_mode, "global_worst")

    def test_sticky_is_more_stable_than_the_global_rule(self):
        stability = {}
        for mode in ("global_worst", "sticky_face"):
            np.random.seed(3)
            env = build(mode)
            held, steps = 0, 0
            for _ in range(5):
                env.reset()
                previous = env.template_center
                for _ in range(10):
                    action = legal_action(env)
                    if action is None:
                        break
                    env.step(action)
                    steps += 1
                    held += env.template_center == previous
                    previous = env.template_center
            stability[mode] = held / max(steps, 1)
        self.assertGreater(stability["sticky_face"], stability["global_worst"])




class TestStickyCenterUnderDeletes(unittest.TestCase):
    """A delete can take the centre with it; the window must stay on the region.

    `delete_half_edge` keeps the deleted edge's OWN face and absorbs its twin's
    into it, so a centre on the twin side loses its face id even though the
    region survives, merged. Falling back to a global re-pick there would give
    back exactly the teleporting this rule exists to stop.
    """

    def test_centre_stays_on_the_merged_face_when_it_is_deleted(self):
        np.random.seed(11)
        env = build("sticky_face", allow_delete=True)
        checked = 0
        for _ in range(60):
            env.reset()
            for _ in range(12):
                centre = env.template_center
                witnesses = set(env._center_face_witnesses)
                # delete the CENTRE itself, which is the case under test; a
                # randomly chosen delete almost never removes it
                slot = env.half_edge_to_index.get(centre)
                action = None
                if slot is not None:
                    candidate = (slot * env.num_actions_per_half_edge
                                 + env._delete_edge_action)
                    if np.isfinite(env._get_obs()["mask"][candidate]):
                        action = candidate
                if action is None:
                    fallback = legal_action(env)
                    if fallback is None:
                        break
                    env.step(fallback)
                    continue
                env.step(action)
                self.assertFalse(env.graph.is_half_edge(centre))
                survivors = [w for w in witnesses if env.graph.is_half_edge(w)]
                self.assertTrue(survivors, "no witness survived a single delete")
                merged = env.graph.face(survivors[0])
                self.assertEqual(env.graph.face(env.template_center), merged)
                checked += 1
                break
            if checked >= 3:
                break
        self.assertGreater(checked, 0, "no delete removed the centre; test vacuous")

    def test_it_picks_the_worst_vertex_on_the_surviving_face(self):
        np.random.seed(12)
        env = build("sticky_face", allow_delete=True)
        env.reset()
        face = env.graph.face(env.template_center)
        on_face = env._half_edges_of(face)
        chosen = env._worst_vertex_half_edge(on_face)
        best = max(env.vertex_defect(env.graph.source_vertex(h, tag=False))
                   for h in on_face)
        self.assertEqual(
            env.vertex_defect(env.graph.source_vertex(chosen, tag=False)), best)
        tied = [h for h in on_face
                if env.vertex_defect(env.graph.source_vertex(h, tag=False)) == best]
        self.assertEqual(chosen, min(tied))

    def test_witnesses_survive_a_single_delete(self):
        """A move removes at most one half-edge pair, so a face of degree three
        or more always leaves a witness to name the merged face."""
        np.random.seed(13)
        env = build("sticky_face", allow_delete=True)
        env.reset()
        self.assertGreaterEqual(len(env._center_face_witnesses), 3)


class TestStickyCenterRecentresLocallyFirst(unittest.TestCase):
    """A finished face hands the centre to the window before the whole mesh.

    Restarting globally throws away the context the agent has just built up and
    can teleport the window across the mesh. So when the centre's face becomes
    the element we asked for, the next centre is taken from the template around
    it, and only when nothing in view is still unfinished does the rule look at
    every face.
    """

    @staticmethod
    def _record(env):
        """What `_sticky_center` saw, judged at the moment it decided.

        The verdicts are computed here rather than after the rollout: the
        centre is chosen before the template is rebuilt, and both the window
        and every face id move on the next step, so half-edges recorded now
        mean something else by the end of the episode.
        """
        events = []
        decide = env._sticky_center

        def irregularity(half_edge):
            return env._face_irregularity(env.graph.face(half_edge))

        def wrapped():
            window = tuple(env._template_neighbourhood(env._center_anchor()))
            face = env._surviving_center_face()
            finished = face is None or env._face_irregularity(face) == 0
            unfinished = [h for h in window if irregularity(h) != 0]
            chosen = decide()

            event = {"finished": finished, "window_had_work": bool(unfinished),
                     "chosen_in_window": chosen in window, "worst_in_window": None}
            if unfinished and chosen is not None:
                worst = max(irregularity(h) for h in unfinished)
                tied = [h for h in unfinished if irregularity(h) == worst]
                event["worst_in_window"] = (
                    irregularity(chosen) == worst
                    and chosen == env._worst_vertex_half_edge(tied))
            events.append(event)
            return chosen

        env._sticky_center = wrapped
        return events

    @classmethod
    def _rollout(cls, seed, episodes=12, steps=20):
        np.random.seed(seed)
        env = build("sticky_face")
        events = cls._record(env)
        for _ in range(episodes):
            env.reset()
            for _ in range(steps):
                action = legal_action(env)
                if action is None:
                    break
                if env.step(action)[2]:
                    break
        return events

    def test_it_recentres_inside_the_window_when_the_window_has_work_left(self):
        local = [e for e in self._rollout(21)
                 if e["finished"] and e["window_had_work"]]
        self.assertTrue(local, "no face finished with work left in view")
        for event in local:
            self.assertTrue(event["chosen_in_window"])

    def test_the_new_centre_is_the_worst_face_then_worst_vertex_in_view(self):
        local = [e for e in self._rollout(22)
                 if e["finished"] and e["window_had_work"]]
        self.assertTrue(local, "nothing exercised the in-window re-pick")
        for event in local:
            self.assertTrue(event["worst_in_window"])

    def test_the_global_repick_waits_until_nothing_in_view_is_unfinished(self):
        """The expensive case is the last resort, not the default."""
        for event in self._rollout(23):
            if event["finished"] and not event["chosen_in_window"]:
                self.assertFalse(
                    event["window_had_work"],
                    "left the window while it still had an irregular face")

    def test_the_window_is_read_off_the_current_topology(self):
        """`index_to_half_edge` is a step stale at this point -- it is built
        after the centre is chosen -- so the neighbourhood is recomputed."""
        np.random.seed(24)
        env = build("sticky_face")
        env.reset()
        action = legal_action(env)
        self.assertIsNotNone(action)
        stale = set(env.index_to_half_edge)
        env.step(action)
        fresh = set(env._template_neighbourhood(env._center_anchor()))
        self.assertTrue(all(env.graph.is_half_edge(h) for h in fresh))
        self.assertNotEqual(fresh, stale)


if __name__ == "__main__":
    unittest.main()
