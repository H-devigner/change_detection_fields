#!/usr/bin/env python3
"""Compatibility entry point for the agricultural field change processor."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("sample_dw_lulc_change_processor.py")),
        run_name="__main__",
    )
