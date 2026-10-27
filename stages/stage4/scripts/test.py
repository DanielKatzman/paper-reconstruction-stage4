import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
