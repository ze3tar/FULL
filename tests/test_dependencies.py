import importlib
import importlib.util
import unittest


REQUIRED_MODULES = [
    "numpy",
    "torch",
    "gymnasium",
    "matplotlib",
    "pandas",
    "tqdm",
    "stable_baselines3",
]


class DependencyImportTests(unittest.TestCase):
    def test_required_modules_import(self) -> None:
        for module_name in REQUIRED_MODULES:
            with self.subTest(module=module_name):
                if importlib.util.find_spec(module_name) is None:
                    self.skipTest(f"Optional dependency '{module_name}' is not installed")
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
