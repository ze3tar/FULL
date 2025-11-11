import unittest

try:  # pragma: no cover - optional dependency for lightweight CI
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    np = None  # type: ignore
    _NUMPY_ERROR = exc
else:
    _NUMPY_ERROR = None

try:  # pragma: no cover - optional dependency for lightweight CI
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    torch = None  # type: ignore
    _TORCH_ERROR = exc
else:
    _TORCH_ERROR = None

try:  # pragma: no cover - optional dependency for lightweight CI
    from obstacle_predictor import ObstaclePredictorLSTM, ObstacleTrajectoryDataset
except ModuleNotFoundError as exc:  # pragma: no cover - handled via skip
    ObstaclePredictorLSTM = ObstacleTrajectoryDataset = None  # type: ignore
    _PREDICTOR_ERROR = exc
else:
    _PREDICTOR_ERROR = None


class LSTMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if any(error is not None for error in (_NUMPY_ERROR, _TORCH_ERROR, _PREDICTOR_ERROR)):
            raise unittest.SkipTest("LSTM predictor dependencies are not installed")

    def test_dataset_and_model_forward(self) -> None:
        trajectory = []
        for step in range(15):
            t = float(step)
            position = np.array([np.sin(t), np.cos(t), t * 0.1])
            velocity = np.array([np.cos(t), -np.sin(t), 0.1])
            trajectory.append(np.concatenate([[t], position, velocity]))
        dataset = ObstacleTrajectoryDataset([np.array(trajectory)], history_len=10, predict_len=3)
        self.assertGreater(len(dataset), 0)
        history, future = dataset[0]
        self.assertEqual(history.shape[-1], 6)
        self.assertEqual(future.shape[0], 9)

        model = ObstaclePredictorLSTM(input_size=6, hidden_size=32, num_layers=1, predict_steps=3)
        output = model(history.unsqueeze(0))
        self.assertEqual(output.shape, (1, 9))

        criterion = torch.nn.MSELoss()
        loss = criterion(output, future.unsqueeze(0))
        loss.backward()
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
