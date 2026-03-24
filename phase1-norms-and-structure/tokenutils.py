"""Backward-compatible re-export — canonical code lives in common/tokenutils.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.tokenutils import token_display, categorize  # noqa: F401, E402
