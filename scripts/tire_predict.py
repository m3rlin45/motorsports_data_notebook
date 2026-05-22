#!/usr/bin/env python3
"""Thin wrapper around motorsports_data_notebook.tire_model.cli for scripting."""

from __future__ import annotations

import sys

from motorsports_data_notebook.tire_model.cli import main

if __name__ == "__main__":
    sys.exit(main())
