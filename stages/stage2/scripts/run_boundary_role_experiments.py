#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from paper_reconstruction_stage2.experiments_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
