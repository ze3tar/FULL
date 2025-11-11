import os

print("🔧 Setting up environment for Google Colab...")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Install dependencies
os.system("pip install -r requirements.txt")

# Verify key libraries
try:
    import torch
    import gymnasium as gym  # noqa: F401
    import numpy  # noqa: F401
    import matplotlib  # noqa: F401
except Exception as exc:  # pragma: no cover - setup helper
    print(f"⚠️ Warning while verifying libraries: {exc}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

print("✅ Environment ready.")
