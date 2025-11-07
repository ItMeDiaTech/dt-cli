"""
Progress tracking and status persistence for indexing operations.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Track and persist indexing progress.

    MEDIUM PRIORITY FIX: Add status structure validation.
    """

    # MEDIUM PRIORITY FIX: Define valid status values
    VALID_STATUSES = {'not_started', 'indexing', 'complete', 'failed'}
    REQUIRED_FIELDS = {
        'indexing': ['current', 'total', 'percentage'],
        'complete': ['total_files', 'total_chunks', 'completed_at'],
        'failed': ['error', 'failed_at']
    }

    def __init__(self, status_file: str = ".rag_data/status.json"):
        """
        Initialize progress tracker.

        Args:
            status_file: Path to status file
        """
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(exist_ok=True, parents=True)

    def _validate_status(self, status: Dict[str, Any]) -> bool:
        """
        MEDIUM PRIORITY FIX: Validate status structure.

        Args:
            status: Status dictionary to validate

        Returns:
            True if valid

        Raises:
            ValueError: If status structure is invalid
        """
        # Check status field exists
        if 'status' not in status:
            raise ValueError("Status dictionary must contain 'status' field")

        status_value = status['status']

        # Check status value is valid
        if status_value not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status value: '{status_value}'. "
                f"Must be one of: {', '.join(self.VALID_STATUSES)}"
            )

        # Check required fields for this status
        if status_value in self.REQUIRED_FIELDS:
            required = self.REQUIRED_FIELDS[status_value]
            missing = [field for field in required if field not in status]

            if missing:
                raise ValueError(
                    f"Status '{status_value}' missing required fields: {', '.join(missing)}"
                )

        # Type validation for specific fields
        if 'current' in status and not isinstance(status['current'], int):
            raise ValueError(f"Field 'current' must be an integer, got {type(status['current'])}")

        if 'total' in status and not isinstance(status['total'], int):
            raise ValueError(f"Field 'total' must be an integer, got {type(status['total'])}")

        if 'percentage' in status:
            percentage = status['percentage']
            if not isinstance(percentage, (int, float)) or percentage < 0 or percentage > 100:
                raise ValueError(f"Field 'percentage' must be 0-100, got {percentage}")

        if 'errors' in status and not isinstance(status['errors'], int):
            raise ValueError(f"Field 'errors' must be an integer, got {type(status['errors'])}")

        return True

    def save_status(self, status: Dict[str, Any]):
        """
        Save current status to file.

        MEDIUM PRIORITY FIX: Add status validation.

        Args:
            status: Status dictionary

        Raises:
            ValueError: If status structure is invalid
        """
        try:
            # MEDIUM PRIORITY FIX: Validate before saving
            self._validate_status(status)

            status['last_updated'] = datetime.now().isoformat()

            # CRITICAL FIX: Use atomic write to prevent corruption on crash
            from src.utils.atomic_write import atomic_write_json
            atomic_write_json(self.status_file, status, indent=2)

        except ValueError as e:
            logger.error(f"Invalid status structure: {e}")
            raise
        except IOError as e:
            logger.error(f"Error saving status: {e}")

    def load_status(self) -> Optional[Dict[str, Any]]:
        """
        Load status from file.

        MEDIUM PRIORITY FIX: Validate loaded status structure.

        Returns:
            Status dictionary or None if not found or invalid
        """
        if self.status_file.exists():
            try:
                status = json.loads(self.status_file.read_text())

                # MEDIUM PRIORITY FIX: Validate loaded status
                try:
                    self._validate_status(status)
                    return status
                except ValueError as e:
                    logger.error(f"Loaded status has invalid structure: {e}")
                    logger.warning("Returning None for invalid status")
                    return None

            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading status: {e}")

        return None

    def update_progress(
        self,
        current: int,
        total: int,
        current_file: str = "",
        errors: int = 0,
        callback: Optional[Callable] = None
    ):
        """
        Update progress information.

        Args:
            current: Current file number
            total: Total number of files
            current_file: Current file being processed
            errors: Number of errors encountered
            callback: Optional callback function for progress updates
        """
        percentage = (current / total * 100) if total > 0 else 0

        progress = {
            'status': 'indexing',
            'current': current,
            'total': total,
            'percentage': round(percentage, 1),
            'current_file': current_file,
            'errors': errors
        }

        self.save_status(progress)

        # Call callback if provided
        if callback:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")

        logger.debug(f"Progress: {current}/{total} ({percentage:.1f}%) - {current_file}")

    def mark_complete(
        self,
        total_files: int,
        total_chunks: int,
        errors: int = 0,
        duration_seconds: float = 0
    ):
        """
        Mark indexing as complete.

        Args:
            total_files: Total files indexed
            total_chunks: Total chunks created
            errors: Number of errors
            duration_seconds: Time taken
        """
        status = {
            'status': 'complete',
            'total_files': total_files,
            'total_chunks': total_chunks,
            'errors': errors,
            'duration_seconds': round(duration_seconds, 2),
            'completed_at': datetime.now().isoformat()
        }

        self.save_status(status)
        logger.info(
            f"Indexing complete: {total_files} files, "
            f"{total_chunks} chunks, {errors} errors"
        )

    def mark_failed(self, error_message: str):
        """
        Mark indexing as failed.

        Args:
            error_message: Error message
        """
        status = {
            'status': 'failed',
            'error': error_message,
            'failed_at': datetime.now().isoformat()
        }

        self.save_status(status)
        logger.error(f"Indexing failed: {error_message}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current indexing status.

        Returns:
            Status dictionary
        """
        status = self.load_status()

        if not status:
            return {'status': 'not_started'}

        return status

    def is_indexing(self) -> bool:
        """Check if indexing is currently in progress."""
        status = self.get_status()
        return status.get('status') == 'indexing'

    def is_complete(self) -> bool:
        """Check if indexing is complete."""
        status = self.get_status()
        return status.get('status') == 'complete'

    def clear(self):
        """Clear status file."""
        if self.status_file.exists():
            self.status_file.unlink()
            logger.info("Status cleared")
