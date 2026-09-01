"""
Logging configuration. Call setup_logging() once at the entry point of any
script or test session. After that, individual modules just call
logging.getLogger(__name__) and get properly formatted output.
"""

import logging
import sys


def setup_logging(level: int = logging.DEBUG) -> None:
    """
    Configure the root logger with a stderr StreamHandler.

    Using the root logger means every getLogger(__name__) call in src/ and
    scripts/ inherits this config without needing to be individually wired up.
    Calling this more than once is safe — the guard on existing handlers
    prevents duplicate output.
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured — don't add a second handler
        return

    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
