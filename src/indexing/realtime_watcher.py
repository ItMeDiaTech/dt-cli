"""
Real-time filesystem watcher for automatic re-indexing.

Watches codebase for changes and triggers incremental indexing automatically.
Uses watchdog library for efficient filesystem monitoring.
"""

import time
from pathlib import Path
from typing import Set, Optional, Callable, Dict, Any
from datetime import datetime, timedelta
import logging
from threading import Thread, Lock
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not available - install with: pip install watchdog")


class CodeChangeHandler(FileSystemEventHandler):
    """Handles filesystem events for code changes."""

    def __init__(
        self,
        on_change_callback: Callable[[Set[str]], None],
        debounce_seconds: float = 2.0,
        file_extensions: Optional[Set[str]] = None
    ):
        """
        Initialize change handler.

        Args:
            on_change_callback: Callback function for changes
            debounce_seconds: Debounce delay to batch changes
            file_extensions: File extensions to watch (default: code files)
        """
        super().__init__()
        self.on_change_callback = on_change_callback
        self.debounce_seconds = debounce_seconds

        # Default file extensions to watch
        self.file_extensions = file_extensions or {
            '.py', '.js', '.jsx', '.ts', '.tsx',
            '.java', '.cpp', '.c', '.h', '.hpp',
            '.go', '.rs', '.rb', '.php', '.swift',
            '.kt', '.scala', '.r', '.m', '.mm',
            '.sh', '.bash', '.md', '.rst', '.txt',
            '.json', '.yaml', '.yml', '.toml', '.xml'
        }

        # Pending changes (path -> last_modified_time)
        self.pending_changes: Dict[str, float] = {}
        self.lock = Lock()

        # Debounce thread
        self.debounce_thread: Optional[Thread] = None
        self.running = False

    def should_process_file(self, file_path: str) -> bool:
        """
        Check if file should be processed.

        Args:
            file_path: File path

        Returns:
            True if should process
        """
        path = Path(file_path)

        # Check extension
        if path.suffix not in self.file_extensions:
            return False

        # Ignore hidden files and directories
        if any(part.startswith('.') for part in path.parts):
            # Allow .claude directory
            if '.claude' not in path.parts:
                return False

        # Ignore common non-code directories
        ignore_dirs = {
            '__pycache__', 'node_modules', 'venv', 'env',
            '.git', '.svn', '.hg', 'build', 'dist',
            'target', '.pytest_cache', '.mypy_cache'
        }

        if any(ignore_dir in path.parts for ignore_dir in ignore_dirs):
            return False

        return True

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_moved(self, event: FileSystemEvent):
        """Handle file move."""
        if not event.is_directory:
            self._handle_change(event.src_path)
            if hasattr(event, 'dest_path'):
                self._handle_change(event.dest_path)

    def _handle_change(self, file_path: str):
        """
        Handle a file change.

        Args:
            file_path: Changed file path
        """
        if not self.should_process_file(file_path):
            return

        with self.lock:
            self.pending_changes[file_path] = time.time()

            # Start debounce thread if not running
            if not self.running:
                self.running = True
                self.debounce_thread = Thread(
                    target=self._debounce_worker,
                    daemon=True
                )
                self.debounce_thread.start()

        logger.debug(f"Detected change: {file_path}")

    def _debounce_worker(self):
        """Worker thread for debouncing changes."""
        while self.running:
            time.sleep(0.5)

            with self.lock:
                if not self.pending_changes:
                    self.running = False
                    break

                current_time = time.time()
                ready_files = set()

                # Find files ready to process (older than debounce time)
                for file_path, change_time in list(self.pending_changes.items()):
                    if current_time - change_time >= self.debounce_seconds:
                        ready_files.add(file_path)
                        del self.pending_changes[file_path]

                if ready_files:
                    logger.info(f"Processing {len(ready_files)} changed files")

                    try:
                        self.on_change_callback(ready_files)
                    except Exception as e:
                        logger.error(f"Error processing changes: {e}")


class RealtimeIndexWatcher:
    """
    Real-time filesystem watcher for automatic indexing.
    """

    def __init__(
        self,
        query_engine,
        watch_path: Optional[Path] = None,
        debounce_seconds: float = 2.0,
        auto_start: bool = False
    ):
        """
        Initialize realtime watcher.

        Args:
            query_engine: Query engine instance
            watch_path: Path to watch (default: cwd)
            debounce_seconds: Debounce delay
            auto_start: Start watching immediately
        """
        if not WATCHDOG_AVAILABLE:
            raise ImportError("watchdog library required - install with: pip install watchdog")

        self.query_engine = query_engine
        self.watch_path = watch_path or Path.cwd()
        self.debounce_seconds = debounce_seconds

        # Statistics
        self.stats = {
            'total_changes_detected': 0,
            'total_indexing_runs': 0,
            'last_indexing_time': None,
            'errors': 0
        }

        # Observer
        self.observer: Optional[Observer] = None
        self.handler: Optional[CodeChangeHandler] = None

        if auto_start:
            self.start()

    def start(self):
        """Start watching filesystem."""
        if self.observer:
            logger.warning("Watcher already running")
            return

        logger.info(f"Starting realtime watcher for {self.watch_path}")

        # Create handler
        self.handler = CodeChangeHandler(
            on_change_callback=self._on_changes_detected,
            debounce_seconds=self.debounce_seconds
        )

        # Create observer
        self.observer = Observer()
        self.observer.schedule(
            self.handler,
            str(self.watch_path),
            recursive=True
        )
        self.observer.start()

        logger.info("Realtime watcher started")

    def stop(self):
        """Stop watching filesystem."""
        if not self.observer:
            return

        logger.info("Stopping realtime watcher...")

        self.observer.stop()
        self.observer.join(timeout=5)
        self.observer = None

        logger.info("Realtime watcher stopped")

    def _on_changes_detected(self, changed_files: Set[str]):
        """
        Handle detected changes.

        Args:
            changed_files: Set of changed file paths
        """
        self.stats['total_changes_detected'] += len(changed_files)

        logger.info(f"Re-indexing {len(changed_files)} changed files...")

        try:
            # CRITICAL FIX: Pass changed_files for true incremental indexing
            # Convert set to list for compatibility
            result = self.query_engine.index_codebase(
                incremental=True,
                use_git=False,  # We already have the file list
                changed_files=list(changed_files)  # Pass the actual changed files
            )

            self.stats['total_indexing_runs'] += 1
            self.stats['last_indexing_time'] = datetime.now()

            if result:
                logger.info(f"Incremental indexing complete: {result}")

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            self.stats['errors'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get watcher statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()

        if stats['last_indexing_time']:
            stats['last_indexing_time'] = stats['last_indexing_time'].isoformat()

        stats['is_running'] = self.observer is not None

        return stats

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self.observer is not None


# Fallback implementation without watchdog
class PollingWatcher:
    """
    Fallback polling-based watcher (less efficient).
    """

    def __init__(
        self,
        query_engine,
        watch_path: Optional[Path] = None,
        poll_interval: int = 5
    ):
        """
        Initialize polling watcher.

        Args:
            query_engine: Query engine instance
            watch_path: Path to watch
            poll_interval: Polling interval in seconds
        """
        self.query_engine = query_engine
        self.watch_path = watch_path or Path.cwd()
        self.poll_interval = poll_interval

        # File modification times
        self.file_mtimes: Dict[str, float] = {}

        # Running flag
        self.running = False
        self.poll_thread: Optional[Thread] = None

    def start(self):
        """Start polling."""
        if self.running:
            return

        logger.info(f"Starting polling watcher (interval: {self.poll_interval}s)")

        self.running = True
        self.poll_thread = Thread(target=self._poll_worker, daemon=True)
        self.poll_thread.start()

    def stop(self):
        """Stop polling."""
        self.running = False

        if self.poll_thread:
            self.poll_thread.join(timeout=self.poll_interval + 1)

        logger.info("Polling watcher stopped")

    def _poll_worker(self):
        """Polling worker thread."""
        while self.running:
            try:
                changed_files = self._check_for_changes()

                if changed_files:
                    logger.info(f"Detected {len(changed_files)} changed files")

                    # Trigger indexing
                    self.query_engine.index_codebase(incremental=True)

            except Exception as e:
                logger.error(f"Polling error: {e}")

            # Sleep
            time.sleep(self.poll_interval)

    def _check_for_changes(self) -> Set[str]:
        """
        Check for file changes.

        Returns:
            Set of changed file paths
        """
        changed = set()

        # Scan directory
        for file_path in self.watch_path.rglob('*'):
            if not file_path.is_file():
                continue

            # Check if code file
            if file_path.suffix not in {'.py', '.js', '.ts', '.java', '.cpp'}:
                continue

            try:
                current_mtime = file_path.stat().st_mtime
                file_str = str(file_path)

                if file_str in self.file_mtimes:
                    if current_mtime != self.file_mtimes[file_str]:
                        changed.add(file_str)

                self.file_mtimes[file_str] = current_mtime

            except Exception as e:
                logger.debug(f"Error checking {file_path}: {e}")

        return changed

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self.running


def create_watcher(
    query_engine,
    watch_path: Optional[Path] = None,
    use_polling: bool = False,
    **kwargs
) -> 'RealtimeIndexWatcher | PollingWatcher':
    """
    Create filesystem watcher.

    Args:
        query_engine: Query engine instance
        watch_path: Path to watch
        use_polling: Force polling mode
        **kwargs: Additional arguments

    Returns:
        Watcher instance
    """
    if use_polling or not WATCHDOG_AVAILABLE:
        if not use_polling:
            logger.warning("watchdog not available, using polling mode")

        return PollingWatcher(
            query_engine,
            watch_path=watch_path,
            poll_interval=kwargs.get('poll_interval', 5)
        )

    return RealtimeIndexWatcher(
        query_engine,
        watch_path=watch_path,
        debounce_seconds=kwargs.get('debounce_seconds', 2.0),
        auto_start=kwargs.get('auto_start', False)
    )
