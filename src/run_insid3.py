"""INSID3 eval entry point. Thin shim around ``src.methods.insid3.run``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.insid3.run import main, parse_args
from src.protocol.masks import as_binary_mask

__all__ = ["main", "parse_args", "as_binary_mask"]


if __name__ == "__main__":
    raise SystemExit(main())
