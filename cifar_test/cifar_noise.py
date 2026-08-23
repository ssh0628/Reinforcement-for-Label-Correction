"""Compatibility launcher for ``cifar_test.setting.noise``."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cifar_test.setting.noise import main


if __name__ == "__main__":
    main()
