import unittest

try:                                                             
    import torch
except ModuleNotFoundError as exc:                                       
    torch = None                
    _TORCH_ERROR = exc
else:
    _TORCH_ERROR = None


class HardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _TORCH_ERROR is not None:
            raise unittest.SkipTest("PyTorch is not installed")

    def test_device_selection(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.ones(4, device=device)
        self.assertEqual(tensor.sum().item(), 4.0)
        self.assertEqual(tensor.device.type, "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    unittest.main()
