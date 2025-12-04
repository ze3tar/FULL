import os

print("🔧 Setting up environment for Google Colab...")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

                      
os.system("pip install -r requirements.txt")

                      
try:
    import torch
    import gymnasium as gym              
    import numpy              
    import matplotlib              
except Exception as exc:                                   
    print(f"⚠️ Warning while verifying libraries: {exc}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

print("✅ Environment ready.")
