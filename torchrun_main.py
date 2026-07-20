#!/usr/bin/env python3
"""Canonical real-pretraining entry file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from umcg.training.runner import main

if __name__ == "__main__":
    main()
