"""
Async task management for long-running operations.
"""

from typing import Dict, Any, Optional, Callable
from enum import Enum
import asyncio
import threading
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    """Represents an async task."""

    def __init__(self, task_id: str, func: Callable, *args, **kwargs):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.progress = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error
        }


class AsyncTaskManager:
    """
    Manages async execution of long-running tasks.
    """

    def __init__(self):
        """Initialize task manager."""
        self.tasks: Dict[str, Task] = {}
        self.executor_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self):
        """Start the task executor."""
        if self.running:
            return

        self.running = True
        self.executor_thread = threading.Thread(target=self._executor_loop, daemon=True)
        self.executor_thread.start()
        logger.info("AsyncTaskManager started")

    def stop(self):
        """Stop the task executor."""
        self.running = False
        if self.executor_thread:
            self.executor_thread.join(timeout=5)
        logger.info("AsyncTaskManager stopped")

    def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for async execution.

        Args:
            func: Function to execute
            args: Positional arguments
            task_id: Optional task ID
            kwargs: Keyword arguments

        Returns:
            Task ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        task = Task(task_id, func, *args, **kwargs)
        self.tasks[task_id] = task

        logger.info(f"Task {task_id} submitted")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.

        Args:
            task_id: Task ID

        Returns:
            Task status dictionary or None
        """
        task = self.tasks.get(task_id)
        return task.to_dict() if task else None

    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get task result if completed.

        Args:
            task_id: Task ID

        Returns:
            Task result or None
        """
        task = self.tasks.get(task_id)

        if not task:
            return None

        if task.status == TaskStatus.COMPLETED:
            return task.result

        return None

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            True if cancelled
        """
        task = self.tasks.get(task_id)

        if task and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task {task_id} cancelled")
            return True

        return False

    def _executor_loop(self):
        """Main executor loop."""
        while self.running:
            # Find pending tasks
            pending_tasks = [
                task for task in self.tasks.values()
                if task.status == TaskStatus.PENDING
            ]

            for task in pending_tasks:
                if not self.running:
                    break

                self._execute_task(task)

            # Sleep briefly
            import time
            time.sleep(0.1)

    def _execute_task(self, task: Task):
        """
        Execute a task.

        Args:
            task: Task to execute
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        logger.info(f"Executing task {task.task_id}")

        try:
            # Execute function
            result = task.func(*task.args, **task.kwargs)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            logger.info(f"Task {task.task_id} completed")

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()

            logger.error(f"Task {task.task_id} failed: {e}")

    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """
        Cleanup old completed tasks.

        Args:
            max_age_seconds: Maximum age in seconds
        """
        now = datetime.now()
        to_remove = []

        for task_id, task in self.tasks.items():
            if task.completed_at:
                age = (now - task.completed_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self.tasks[task_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old tasks")

    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(
                1 for t in self.tasks.values() if t.status == status
            )

        return {
            'total_tasks': len(self.tasks),
            'status_counts': status_counts,
            'running': self.running
        }


# Global instance
task_manager = AsyncTaskManager()
