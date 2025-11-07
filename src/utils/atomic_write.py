"""
Atomic file write utilities to prevent data loss.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict


def atomic_write_json(file_path: Path, data: Any, indent: int = 2):
    """
    Write JSON data to file atomically.

    Uses temp file + atomic rename to ensure data is never corrupted,
    even if process crashes during write.

    Args:
        file_path: Target file path
        data: Data to write (will be JSON serialized)
        indent: JSON indentation level

    Raises:
        IOError: If write fails
    """
    file_path = Path(file_path)

    # Create temporary file in same directory as target
    # (ensures atomic rename works across filesystems)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=file_path.parent,
        prefix=f".{file_path.name}.",
        suffix=".tmp"
    )

    try:
        # Write data to temp file
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())  # Ensure written to disk

        # Atomic rename: replaces target file in single operation
        # This is atomic on POSIX systems
        os.replace(temp_path, file_path)

    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise IOError(f"Failed to write {file_path}: {e}")


def atomic_write_text(file_path: Path, content: str):
    """
    Write text to file atomically.

    Args:
        file_path: Target file path
        content: Text content to write

    Raises:
        IOError: If write fails
    """
    file_path = Path(file_path)

    # Create temporary file in same directory
    temp_fd, temp_path = tempfile.mkstemp(
        dir=file_path.parent,
        prefix=f".{file_path.name}.",
        suffix=".tmp"
    )

    try:
        # Write content to temp file
        with os.fdopen(temp_fd, 'w') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.replace(temp_path, file_path)

    except Exception as e:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
        raise IOError(f"Failed to write {file_path}: {e}")
