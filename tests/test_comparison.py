import unittest

try:  # pragma: no cover - optional dependency for lightweight CI
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    np = None  # type: ignore
    _NUMPY_ERROR = exc
else:
    _NUMPY_ERROR = None

try:  # pragma: no cover - optional dependency for lightweight CI
    from baseline_enhanced import path_length
    from comprehensive_comparison import PlanningBenchmark
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    path_length = None  # type: ignore
    PlanningBenchmark = None  # type: ignore
    _COMPARISON_ERROR = exc
else:
    _COMPARISON_ERROR = None


class ComparisonFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _NUMPY_ERROR is not None or _COMPARISON_ERROR is not None:
            raise unittest.SkipTest("Comparison framework dependencies are not installed")

    def test_rl_planner_handles_missing_agent(self) -> None:
        benchmark = PlanningBenchmark()
        start = (0.0, 0.0, 0.0)
        goal = (100.0, 100.0, 100.0)
        obstacles = [((10.0, 10.0, 10.0), 5.0)]
        result = benchmark._run_rl_apf_rrt(start, goal, obstacles, benchmark.bounds)
        self.assertIn('success', result)
        self.assertFalse(result['success'])

    def test_path_length_helper(self) -> None:
        path = [np.zeros(3), np.ones(3)]
        length = path_length(path)
        self.assertAlmostEqual(length, np.linalg.norm(np.ones(3)))


if __name__ == "__main__":
    unittest.main()
