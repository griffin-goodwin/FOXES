"""Timestamp spelling at filesystem boundaries."""

import os


_WINDOWS = os.name == "nt"


def timestamp_filename(timestamp: str, suffix: str) -> str:
    """Keep existing filename conventions while avoiding colons on Windows."""
    if _WINDOWS:
        timestamp = timestamp.replace(":", "")
    return timestamp + suffix
