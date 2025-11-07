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
    """Track and persist indexing progress."""

    def __init__(self, status_file: str = ".rag_data/status.json"):
        """
        Initialize progress tracker.

        Args:
            status_file: Path to status file
        """
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(exist_ok=True, parents=True)

    def save_status(self, status: Dict[str, Any]):
        """
        Save current status to file.

        Args:
            status: Status dictionary
        """
        try:
            status['last_updated'] = datetime.now().isoformat()
            # CRITICAL FIX: Use atomic write to prevent corruption on crash
            from src.utils.atomic_write import atomic_write_json
            atomic_write_json(self.status_file, status, indent=2)
        except IOError as e:
            logger.error(f"Error saving status: {e}")

    def load_status(self) -> Optional[Dict[str, Any]]:
        """
        Load status from file.

        Returns:
            Status dictionary or None if not found
        """
        if self.status_file.exists():
            try:
                return json.loads(self.status_file.read_text())
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
