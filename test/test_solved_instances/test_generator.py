"""The backward generator must only ever emit solutions that actually work.

Every claim it makes is checkable: the seed mesh is at par, the walk ends at
the raw polygon, the recorded moves replay through the real environment, and
the replay lands on par. If any of those slips, behaviour cloning would be
training on lies.
"""

import unittest

import numpy as np

import envs.polygon_utils as utils
from envs.solved_instances import (
    SolvedInstanceInitializer, backward_walk, default_scratch_env,
    generate_instances, polygon_from_angles, polygon_from_graph,
    polyomino_mesh, random_polyomino, replay, sample_angles,
)


class TestPolyominoSeeds(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_seed_mesh_is_all_quads_and_at_par(self):
        env = default_scratch_env()
        for _ in range(20):
            cells = random_polyomino(int(self.rng.integers(2, 12)), self.rng)
            graph, desired, _ = polyomino_mesh(cells)
            degrees = [graph.face_degree(f) for f in graph.face_list()]
            self.assertTrue(all(d == 4 for d in degrees))
            env._reset_to_state(graph, dict(desired))
            self.assertEqual(env.global_face_score, 0)
            self.assertEqual(env.global_vertex_score, env.par)

    def test_boundary_desired_degree_matches_the_corner_angle(self):
        cells = random_polyomino(8, self.rng)
        graph, desired, _ = polyomino_mesh(cells)
        for vertex in graph.vertex_list(tag=False):
            if graph.is_boundary_vertex(vertex):
                self.assertEqual(desired[vertex], graph.vertex_degree(vertex))


class TestBackwardWalk(unittest.TestCase):
    def test_walk_reaches_a_single_face(self):
        rng = np.random.default_rng(1)
        reached = 0
        for _ in range(30):
            cells = random_polyomino(int(rng.integers(2, 10)), rng)
            graph, desired, _ = polyomino_mesh(cells)
            walk = backward_walk(graph, desired, rng=rng)
            if walk is None:
                continue
            polygon_graph, _, moves = walk
            self.assertEqual(len(polygon_graph.face_list()), 1)
            self.assertGreater(len(moves), 0)
            reached += 1
        self.assertGreaterEqual(reached, 15, "backward walk stalls far too often")


class TestPolygonFromAngles(unittest.TestCase):
    def test_round_trip_reproduces_the_angles(self):
        rng = np.random.default_rng(2)
        cells = random_polyomino(6, rng)
        graph, desired, _ = polyomino_mesh(cells)
        walk = backward_walk(graph, desired, rng=rng)
        self.assertIsNotNone(walk)
        polygon_graph, _, _ = walk
        loop = polygon_from_graph(polygon_graph)
        coordinates = {v: polygon_graph.vertex_coordinate(v) for v in loop}
        angles = utils.get_polygon_interior_angles(loop, coordinates)
        lengths = [float(np.linalg.norm(
            np.asarray(coordinates[loop[(i + 1) % len(loop)]]) - np.asarray(coordinates[v])))
            for i, v in enumerate(loop)]

        target = [angles[v] for v in loop]
        points = polygon_from_angles(target, rng, reference_lengths=lengths)
        self.assertIsNotNone(points)
        rebuilt = utils.get_polygon_interior_angles(
            list(range(len(loop))), {i: points[i] for i in range(len(loop))})
        for i, expected in enumerate(target):
            self.assertAlmostEqual(rebuilt[i], expected, places=3)

    def test_sampled_angles_sum_to_a_polygon(self):
        rng = np.random.default_rng(3)
        degrees = [2, 2, 4, 2, 2, 2, 2, 2]
        for bumps in ([0] * 8, [1] + [0] * 7, [1, 1] + [0] * 6):
            angles = sample_angles(degrees, bumps, rng)
            self.assertIsNotNone(angles)
            self.assertAlmostEqual(float(np.sum(angles)), 180.0 * (len(degrees) - 2), places=4)


class TestGeneratedInstances(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4)
        cls.instances, cls.stats = generate_instances(60, cell_range=(2, 10), rng=rng)

    def test_generator_produces_instances(self):
        self.assertGreaterEqual(len(self.instances), 50)

    def test_every_instance_replays_to_par(self):
        env = default_scratch_env()
        for instance in self.instances:
            _, actions, _, _, played, ok = replay(env, instance)
            self.assertTrue(ok)
            self.assertEqual(played, instance.cost_to_go)
            self.assertTrue(env.is_at_par())

    def test_shuffled_replay_also_reaches_par(self):
        env = default_scratch_env()
        rng = np.random.default_rng(5)
        solved = 0
        for instance in self.instances:
            *_, ok = replay(env, instance, rng=rng)
            solved += int(ok and env.is_at_par())
        self.assertGreater(solved / len(self.instances), 0.8)

    def test_no_instance_starts_at_par(self):
        env = default_scratch_env()
        for instance in self.instances:
            graph, desired = instance.build()
            env._reset_to_state(graph, desired)
            self.assertFalse(env.is_at_par())

    def test_reported_par_is_the_env_par(self):
        env = default_scratch_env()
        for instance in self.instances:
            graph, desired = instance.build()
            env._reset_to_state(graph, desired)
            self.assertEqual(instance.par, env.par)


class TestReverseCurriculumInitializer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(6)
        cls.instances, _ = generate_instances(40, cell_range=(3, 10), rng=rng)

    def test_cost_to_go_start_states_are_that_far_from_par(self):
        env = default_scratch_env()
        for cost in (1, 2, 3):
            initializer = SolvedInstanceInitializer(
                [i for i in self.instances if i.cost_to_go >= cost], cost_to_go=cost)
            for _ in range(10):
                graph, desired = initializer()
                env._reset_to_state(graph, desired)
                self.assertFalse(env.is_at_par())

    def test_degree_range_filter(self):
        initializer = SolvedInstanceInitializer(self.instances)
        initializer.set_degree_range([4, 5, 6])
        for _ in range(10):
            graph, _ = initializer()
            self.assertLessEqual(graph.number_of_half_edges(), 6)


if __name__ == "__main__":
    unittest.main()


class TestPolarSeeds(unittest.TestCase):
    """The family that covers what a polyomino outline can never be."""

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.env = default_scratch_env()

    def test_par_is_the_hub_defect(self):
        from envs.solved_instances import polar_mesh
        for num_quads in range(3, 8):
            seed = polar_mesh(num_quads, self.rng, flat_corner_probability=0.0)
            self.assertIsNotNone(seed, f"m={num_quads} outline failed")
            graph, desired, _ = seed
            self.env._reset_to_state(graph, dict(desired))
            self.assertEqual(self.env.global_face_score, 0)
            self.assertEqual(self.env.par, abs(4 - num_quads))
            self.assertTrue(self.env.is_at_par())

    def test_three_quads_give_a_triangle(self):
        from envs.solved_instances import backward_walk, polar_mesh
        seed = polar_mesh(3, self.rng, flat_corner_probability=0.0)
        graph, desired, _ = seed
        walk = backward_walk(graph, desired, rng=self.rng)
        self.assertIsNotNone(walk)
        polygon_graph, _, _ = walk
        self.assertEqual(len(polygon_from_graph(polygon_graph)), 3)


class TestAnnulusSeeds(unittest.TestCase):
    def test_annulus_mesh_is_at_par(self):
        from envs.solved_instances import random_annulus
        rng = np.random.default_rng(8)
        env = default_scratch_env()
        for _ in range(6):
            cells = random_annulus(int(rng.integers(8, 14)), rng)
            graph, desired, _ = polyomino_mesh(cells)
            env._reset_to_state(graph, dict(desired))
            self.assertEqual(env.global_face_score, 0)
            self.assertEqual(env.global_vertex_score, env.par)

    def test_hole_instances_carry_a_slit(self):
        rng = np.random.default_rng(9)
        instances, _ = generate_instances(120, cell_range=(8, 13), rng=rng,
                                          hole_probability=1.0, polar_probability=0.0)
        holes = [i for i in instances if i.has_hole]
        self.assertGreater(len(holes), 0, "no hole instance was generated")
        env = default_scratch_env()
        for instance in holes:
            *_, ok = replay(env, instance)
            self.assertTrue(ok)
            self.assertTrue(env.is_at_par())


class TestFamilyCoverage(unittest.TestCase):
    def test_generator_spans_odd_degrees_and_par(self):
        rng = np.random.default_rng(10)
        instances, _ = generate_instances(300, cell_range=(2, 13), rng=rng)
        degrees = {i.polygon_degree for i in instances}
        pars = {i.par for i in instances}
        self.assertTrue(any(d % 2 == 1 for d in degrees),
                        "no odd-degree outline; the held-out triangle and pentagon "
                        "would be out of distribution")
        self.assertTrue(pars - {0}, "every instance has par 0")


class TestTriangleSeeds(unittest.TestCase):
    """The triangular lattice is to triangles what the polyomino grid is to quads.

    A lattice point touched by k triangles has interior angle 60k and mesh
    degree k+1, and `rounded_desired_degree(60k, 60)` is exactly k+1, so the
    patch is at par by construction -- the same argument, one constant apart.
    """

    def setUp(self):
        from envs.solved_instances import default_scratch_env
        self.rng = np.random.default_rng(0)
        self.env = default_scratch_env(face_desired_degree=3)

    def test_polyiamond_seeds_are_at_par(self):
        from envs.solved_instances import polyiamond_mesh, random_polyiamond
        for _ in range(15):
            cells = random_polyiamond(int(self.rng.integers(4, 16)), self.rng)
            graph, desired, _ = polyiamond_mesh(cells, rng=self.rng,
                                                flat_corner_probability=0.3)
            self.env._reset_to_state(graph, dict(desired))
            self.assertEqual(self.env.global_face_score, 0)
            self.assertEqual(self.env.global_vertex_score, self.env.par)

    def test_every_seed_face_is_a_triangle(self):
        from envs.solved_instances import polyiamond_mesh, random_polyiamond
        cells = random_polyiamond(10, self.rng)
        graph, _, _ = polyiamond_mesh(cells, rng=self.rng)
        self.assertTrue(all(graph.face_degree(f) == 3 for f in graph.face_list()))

    def test_fan_par_is_the_hub_defect(self):
        """m triangles round a hub: par is |6 - m|, which is where par > 0 and
        odd outlines come from.

        The identity assumes each corner has one admissible degree. At m = 4 the
        outline is a square, whose 90 degree corners are exact ties at the
        triangle target, and freeing them drops par to 0: a square really does
        admit a perfectly regular triangulation, and the 4-fan is not it. The
        generator rejects that seed on its own `is_at_par` check.
        """
        from envs.solved_instances import triangle_fan_mesh
        for num_faces in range(4, 12):
            seed = triangle_fan_mesh(num_faces, self.rng)
            self.assertIsNotNone(seed, f"m={num_faces} outline failed")
            graph, desired, _ = seed
            self.env._reset_to_state(graph, dict(desired))
            self.assertEqual(self.env.global_face_score, 0)
            if self.env.vertex_desired_options:
                self.assertEqual(num_faces, 4, "only the square fan should tie")
                self.assertEqual(self.env.par, 0)
                self.assertFalse(self.env.is_at_par())
                continue
            self.assertEqual(self.env.par, abs(6 - num_faces))
            self.assertTrue(self.env.is_at_par())

    def test_generated_triangle_instances_replay_to_par(self):
        instances, _ = generate_instances(60, cell_range=(4, 14), rng=self.rng,
                                          face_desired_degree=3)
        self.assertGreaterEqual(len(instances), 40)
        for instance in instances:
            self.assertEqual(instance.target, 3)
            *_, ok = replay(self.env, instance)
            self.assertTrue(ok)
            self.assertTrue(self.env.is_at_par())


class TestParGeneralisation(unittest.TestCase):
    """The par coefficient is the interior vertex's generic degree, not the number 4."""

    def test_hexagon_fan_is_at_par_for_triangles(self):
        import math
        from src.tiler import Tiler
        from envs.angle_env_with_length import AngleEnvWithLength
        coords = {i: [math.cos(math.pi / 3 * i), math.sin(math.pi / 3 * i)] for i in range(6)}
        coords[6] = [0.0, 0.0]
        env = AngleEnvWithLength(
            3, lambda: (Tiler.from_face_loops([[i, (i + 1) % 6, 6] for i in range(6)], coords),
                        {**{i: 3 for i in range(6)}, 6: 6}),
            template_size=64, max_edge_addition_steps=3, resample_if_at_par=False)
        env.reset()
        self.assertEqual(env.global_vertex_score, 0)
        self.assertEqual(env.par, 0)
        self.assertTrue(env.is_at_par())
        self.assertAlmostEqual(float(env.min_element_quality()), 1.0, places=6)

    def test_quad_par_is_unchanged(self):
        """The released quad agent depends on this coefficient being 4."""
        from envs.solved_instances import default_scratch_env, polyomino_mesh, random_polyomino
        rng = np.random.default_rng(3)
        env = default_scratch_env()
        for _ in range(10):
            cells = random_polyomino(int(rng.integers(2, 10)), rng)
            graph, desired, _ = polyomino_mesh(cells)
            env._reset_to_state(graph, dict(desired))
            self.assertEqual(env.interior_vertex_desired_degree, 4)
            self.assertEqual(env.global_vertex_score, env.par)


class TestRectilinearTriangleSeeds(unittest.TestCase):
    """The family that closes the hole in the triangle distribution."""

    def setUp(self):
        from envs.solved_instances import default_scratch_env
        self.rng = np.random.default_rng(4)
        self.env = default_scratch_env(face_desired_degree=3)

    def test_polyomino_triangulation_is_at_par(self):
        """Splitting every cell along the same diagonal gives interior degree 6
        and leaves each corner inside the tie set it admits, so par is 0."""
        from envs.solved_instances import (polyomino_triangle_mesh,
                                           random_polyomino, random_annulus)
        for trial in range(24):
            with self.subTest(trial=trial):
                cells = (random_annulus(int(self.rng.integers(8, 14)), self.rng)
                         if trial % 4 == 3 else
                         random_polyomino(int(self.rng.integers(2, 11)), self.rng))
                graph, desired, _ = polyomino_triangle_mesh(cells, rng=self.rng)
                self.env._reset_to_state(graph, dict(desired))
                self.assertEqual(self.env.global_face_score, 0)
                self.assertEqual(self.env.par, 0)
                self.assertTrue(self.env.is_at_par())

    def test_rectilinear_domains_reach_the_dataset(self):
        """Without this family no rectilinear domain exists at 60 degrees at
        all: every one of its corners is a tie, and the angle sampler backs away
        from every bin edge."""
        from envs.solved_instances import generate_instances, default_scratch_env
        env = default_scratch_env(face_desired_degree=3)
        instances, _ = generate_instances(
            120, cell_range=(2, 8), env=env, rng=np.random.default_rng(7),
            face_desired_degree=3, rectilinear_probability=1.0)
        self.assertTrue(instances)
        # a rectilinear outline wants only degrees 3 and 5 at its corners
        rectilinear = [i for i in instances
                       if set(i.desired.values()) <= {2, 3, 4, 5, 6}
                       and {3, 5} & set(i.desired.values())]
        self.assertTrue(rectilinear, "no rectilinear-looking instance generated")
