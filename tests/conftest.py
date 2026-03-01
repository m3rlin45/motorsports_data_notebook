"""Shared test configuration.

Adds the tests/ directory to sys.path so test modules can import helpers.
"""

import sys
from pathlib import Path

# Add tests/ to sys.path so helpers.py is importable
sys.path.insert(0, str(Path(__file__).parent))
