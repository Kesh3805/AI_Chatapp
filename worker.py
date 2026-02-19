"""Background task runner.

Default: tasks execute in local daemon threads (zero setup).
Optional: run `python worker.py` as a separate process for dedicated
background processing (uses a shared queue via filesystem).

In both modes the API is the same — call worker.submit(fn, *args).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


def submit(fn: Callable, *args, **kwargs) -> None:
    """Submit a background task.

    Runs in a daemon thread.  Fire-and-forget — exceptions are logged,
    never raised to the caller.
    """
    def _safe():
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Background task error: {e}")

    threading.Thread(target=_safe, daemon=True).start()
