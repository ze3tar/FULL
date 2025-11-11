import unittest

try:  # pragma: no cover - optional dependency for lightweight CI
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    np = None  # type: ignore
    _NUMPY_ERROR = exc
else:
    _NUMPY_ERROR = None

try:  # pragma: no cover - optional dependency for lightweight CI
    from rl_enhanced_apf_rrt import APFRRTEnv, PlannerParameters, ScenarioConfig
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    APFRRTEnv = PlannerParameters = ScenarioConfig = None  # type: ignore
    _RL_ERROR = exc
else:
    _RL_ERROR = None


class RLEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _NUMPY_ERROR is not None or _RL_ERROR is not None:
            raise unittest.SkipTest("RL environment dependencies are not installed")

    def test_reset_and_step(self) -> None:
        scenario = ScenarioConfig(difficulty="easy", max_steps=10, dynamic_probability=0.0)
        env = APFRRTEnv(scenario, seed=123)
        obs, info = env.reset()
        self.assertIsInstance(info, dict)
        self.assertEqual(obs.shape[0], env.observation_space.shape[0])

        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, reward, done, truncated, step_info = env.step(zero_action)
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("clearance", step_info)

    def test_parameter_bounds(self) -> None:
        params = PlannerParameters()
        original = params.to_array().copy()
        params.apply_delta(np.array([1.0, -1.0, 1.0, 1.0, 1.0]))
        updated = params.to_array()
        self.assertEqual(original.shape, updated.shape)


if __name__ == "__main__":
    unittest.main()
