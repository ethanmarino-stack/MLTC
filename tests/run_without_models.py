"""Run all tests with fail-fast model/GPU guards; safe while training runs separately."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.environ.setdefault("YOLO_CONFIG_DIR", str(root / "outputs" / "ultralytics"))

if __name__ == "__main__":
    # These guards affect only this test process, never the separate trainer.
    with patch("ultralytics.YOLO", side_effect=AssertionError("Real models are disabled in this test run")), \
         patch("torch.cuda.is_available", side_effect=AssertionError("GPU access is disabled in this test run")):
        suite = unittest.defaultTestLoader.discover(str(root / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
