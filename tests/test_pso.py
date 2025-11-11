import unittest

try:  # pragma: no cover - optional dependency for lightweight CI
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    np = None  # type: ignore
    _NUMPY_ERROR = exc
else:
    _NUMPY_ERROR = None

try:  # pragma: no cover - optional dependency for lightweight CI
    from pso_path_smoother import PSOPathSmoother
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    PSOPathSmoother = None  # type: ignore
    _PSO_ERROR = exc
else:
    _PSO_ERROR = None


class PSOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _NUMPY_ERROR is not None or _PSO_ERROR is not None:
            raise unittest.SkipTest("PSO smoother dependencies are not installed")

    def test_smoothing_preserves_shape(self) -> None:
        t = np.linspace(0, 1, 6)
        path = np.column_stack([t, t**2, np.sin(t)])
        smoother = PSOPathSmoother(n_particles=6, max_iters=5, verbose=False)
        smoothed, cost, metrics = smoother.smooth(
            path,
            obstacles=[(np.zeros(3), 0.1)],
            fixed_endpoints=True,
            verbose=False,
        )
        self.assertEqual(smoothed.shape, path.shape)
        self.assertIsInstance(cost, float)
        self.assertIn('final_cost', metrics)
        self.assertIn('improvement_percent', metrics)


if __name__ == "__main__":
    unittest.main()
