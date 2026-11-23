"""Report actual installed versions and test a small GPU tensor operation."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import torch


def main():
    """Save reproducibility records without claiming untested versions work."""
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    for package in ("torch", "torchvision", "ultralytics", "fastapi", "uvicorn", "websockets", "opencv-python", "numpy"):
        report[package] = importlib.metadata.version(package)
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        # Availability alone is weaker evidence than actually running on the GPU.
        report["gpu_test"] = (torch.ones(2, device="cuda") * 2).tolist()
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    (output / "environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    frozen = subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"], text=True)
    (output / "requirements-working.txt").write_text(frozen, encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved outputs/environment.json and outputs/requirements-working.txt")


if __name__ == "__main__":
    main()
