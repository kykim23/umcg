#!/usr/bin/env python3
"""Canonical native-checkpoint weight export entry file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from umcg.cli.export_weights import main

if __name__ == "__main__":
    main()
