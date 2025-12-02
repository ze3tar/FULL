import random
import unittest

try:  # pragma: no cover - optional dependency for lightweight CI
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    np = None  # type: ignore
    _NUMPY_ERROR = exc
else:
    _NUMPY_ERROR = None

try:  # pragma: no cover - optional dependency for lightweight CI
    from baseline_enhanced import (
        create_random_spheres,
        prune_path,
        rrt_apf_guided,
        rrt_basic,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    create_random_spheres = prune_path = rrt_apf_guided = rrt_basic = None  # type: ignore
    _BASELINE_ERROR = exc
else:
    _BASELINE_ERROR = None


class BaselinePlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _NUMPY_ERROR is not None or _BASELINE_ERROR is not None:
            raise unittest.SkipTest("Baseline planner dependencies are not installed")

    def setUp(self) -> None:
        random.seed(1)
        np.random.seed(1)
        self.start = (0.0, 0.0, 0.0)
        self.goal = (10.0, 10.0, 10.0)
        self.bounds = ((-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0))
        self.obstacles = create_random_spheres(num=2, bounds=self.bounds, rmin=2, rmax=3, seed=1)

    def test_rrt_basic_runs(self) -> None:
        path, nodes, parents, runtime = rrt_basic(
            self.start,
            self.goal,
            self.obstacles,
            self.bounds,
            max_iters=200,
            step_size=5.0,
            goal_radius=5.0,
            goal_bias=0.2,
        )
        self.assertIsInstance(nodes, list)
        self.assertGreater(len(nodes), 0)
        self.assertIsInstance(runtime, float)
        self.assertIn(0, parents)
        # Path may be None depending on randomness, but prune_path should handle it
        pruned = prune_path(path, self.obstacles)
        if path is None:
            self.assertIsNone(pruned)
        else:
            self.assertGreaterEqual(len(pruned), 2)

    def test_rrt_apf_guided_runs(self) -> None:
        path, nodes, parents, runtime = rrt_apf_guided(
            self.start,
            self.goal,
            self.obstacles,
            self.bounds,
            max_iters=200,
            r_step=5.0,
            goal_radius=5.0,
            goal_bias=0.2,
            K_att=1.0,
            K_rep=0.2,
            d0=10.0,
        )
        self.assertIsInstance(nodes, list)
        self.assertGreater(len(nodes), 0)
        self.assertIsInstance(runtime, float)
        self.assertIn(0, parents)


if __name__ == "__main__":
    unittest.main()
