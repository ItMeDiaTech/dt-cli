"""
Intelligent cache invalidation system.

Invalidates caches when:
- Files are modified
- Git commits occur
- Manual trigger
- TTL expires
"""

import time
from typing import Dict, Set, Optional, Callable, Any
from pathlib import Path
import hashlib
import logging
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)


class CacheInvalidationStrategy:
    """Base class for cache invalidation strategies."""

    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Check if cache entry should be invalidated.

        Args:
            key: Cache key
            metadata: Cache entry metadata

        Returns:
            True if should invalidate
        """
        raise NotImplementedError


class TTLInvalidationStrategy(CacheInvalidationStrategy):
    """Time-To-Live based invalidation."""

    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize TTL strategy.

        Args:
            ttl_seconds: Time to live in seconds
        """
        self.ttl_seconds = ttl_seconds

    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if TTL expired."""
        created_at = metadata.get('created_at', 0)
        age = time.time() - created_at

        return age > self.ttl_seconds


class FileModificationStrategy(CacheInvalidationStrategy):
    """Invalidate when source files are modified."""

    def __init__(self):
        """Initialize file modification strategy."""
        self.file_hashes: Dict[str, str] = {}

    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if related files were modified."""
        related_files = metadata.get('related_files', [])

        for file_path in related_files:
            if self._file_changed(file_path):
                logger.debug(f"File {file_path} changed, invalidating {key}")
                return True

        return False

    def _file_changed(self, file_path: str) -> bool:
        """
        Check if file changed since last check.

        Args:
            file_path: Path to file

        Returns:
            True if file changed
        """
        path = Path(file_path)

        if not path.exists():
            return True

        try:
            # Get current hash
            current_hash = self._hash_file(path)

            # Check if changed
            if file_path in self.file_hashes:
                if self.file_hashes[file_path] != current_hash:
                    self.file_hashes[file_path] = current_hash
                    return True
                return False
            else:
                # First time seeing this file
                self.file_hashes[file_path] = current_hash
                return False

        except Exception as e:
            logger.warning(f"Error checking file {file_path}: {e}")
            return False

    def _hash_file(self, path: Path) -> str:
        """
        Hash file contents.

        Args:
            path: File path

        Returns:
            File hash
        """
        hasher = hashlib.md5()
        hasher.update(path.read_bytes())
        return hasher.hexdigest()


class GitCommitStrategy(CacheInvalidationStrategy):
    """Invalidate when Git commits occur."""

    def __init__(self):
        """Initialize Git commit strategy."""
        self.last_commit_hash: Optional[str] = None
        self._update_commit_hash()

    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if new commits occurred."""
        current_hash = self._get_current_commit()

        if current_hash and self.last_commit_hash:
            if current_hash != self.last_commit_hash:
                logger.info("Git commit detected, invalidating caches")
                self.last_commit_hash = current_hash
                return True

        return False

    def _get_current_commit(self) -> Optional[str]:
        """
        Get current Git commit hash.

        Returns:
            Commit hash or None
        """
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _update_commit_hash(self):
        """Update stored commit hash."""
        self.last_commit_hash = self._get_current_commit()


class MemoryPressureStrategy(CacheInvalidationStrategy):
    """Invalidate when memory usage is high."""

    def __init__(self, memory_threshold_mb: int = 1024):
        """
        Initialize memory pressure strategy.

        Args:
            memory_threshold_mb: Memory threshold in MB
        """
        self.memory_threshold_mb = memory_threshold_mb

    def should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """Check if memory pressure is high."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > self.memory_threshold_mb:
                # Invalidate older entries first
                created_at = metadata.get('created_at', 0)
                age = time.time() - created_at

                # Invalidate if older than 5 minutes
                return age > 300

        except ImportError:
            pass

        return False


class IntelligentCacheManager:
    """
    Manages cache with intelligent invalidation.
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        enable_file_tracking: bool = True,
        enable_git_tracking: bool = True,
        enable_memory_pressure: bool = True
    ):
        """
        Initialize cache manager.

        Args:
            ttl_seconds: Default TTL in seconds
            enable_file_tracking: Enable file modification tracking
            enable_git_tracking: Enable Git commit tracking
            enable_memory_pressure: Enable memory pressure tracking
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()

        # Invalidation strategies
        self.strategies: list[CacheInvalidationStrategy] = [
            TTLInvalidationStrategy(ttl_seconds)
        ]

        if enable_file_tracking:
            self.strategies.append(FileModificationStrategy())

        if enable_git_tracking:
            self.strategies.append(GitCommitStrategy())

        if enable_memory_pressure:
            self.strategies.append(MemoryPressureStrategy())

        logger.info(f"Cache manager initialized with {len(self.strategies)} strategies")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self.lock:
            if key not in self.cache:
                return None

            entry = self.cache[key]

            # Check if should invalidate
            if self._should_invalidate(key, entry['metadata']):
                del self.cache[key]
                logger.debug(f"Cache invalidated for key: {key}")
                return None

            # Update access time
            entry['metadata']['last_accessed'] = time.time()
            entry['metadata']['access_count'] = entry['metadata'].get('access_count', 0) + 1

            return entry['value']

    def set(
        self,
        key: str,
        value: Any,
        related_files: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            related_files: Files related to this cache entry
            metadata: Additional metadata
        """
        with self.lock:
            entry_metadata = metadata or {}
            entry_metadata.update({
                'created_at': time.time(),
                'last_accessed': time.time(),
                'access_count': 0,
                'related_files': related_files or []
            })

            self.cache[key] = {
                'value': value,
                'metadata': entry_metadata
            }

            logger.debug(f"Cache set for key: {key}")

    def invalidate(self, key: str) -> bool:
        """
        Manually invalidate cache entry.

        Args:
            key: Cache key

        Returns:
            True if invalidated
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                logger.info(f"Cache manually invalidated: {key}")
                return True

            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Key pattern (supports wildcards)

        Returns:
            Number of entries invalidated
        """
        import fnmatch

        with self.lock:
            keys_to_remove = [
                key for key in self.cache.keys()
                if fnmatch.fnmatch(key, pattern)
            ]

            for key in keys_to_remove:
                del self.cache[key]

            if keys_to_remove:
                logger.info(f"Invalidated {len(keys_to_remove)} entries matching {pattern}")

            return len(keys_to_remove)

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cache cleared ({count} entries removed)")

    def _should_invalidate(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Check if cache entry should be invalidated.

        Args:
            key: Cache key
            metadata: Cache metadata

        Returns:
            True if should invalidate
        """
        for strategy in self.strategies:
            if strategy.should_invalidate(key, metadata):
                return True

        return False

    def cleanup_old_entries(self):
        """Remove invalidated entries from cache."""
        with self.lock:
            keys_to_remove = []

            for key, entry in self.cache.items():
                if self._should_invalidate(key, entry['metadata']):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.cache[key]

            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} invalidated entries")

            return len(keys_to_remove)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Statistics dictionary
        """
        with self.lock:
            total_entries = len(self.cache)
            total_size_bytes = 0

            access_counts = []
            ages = []

            current_time = time.time()

            for entry in self.cache.values():
                access_counts.append(entry['metadata'].get('access_count', 0))
                ages.append(current_time - entry['metadata'].get('created_at', current_time))

                # Estimate size (rough approximation)
                try:
                    import sys
                    total_size_bytes += sys.getsizeof(entry['value'])
                except Exception:
                    pass

            return {
                'total_entries': total_entries,
                'total_size_bytes': total_size_bytes,
                'total_size_mb': total_size_bytes / 1024 / 1024,
                'average_access_count': sum(access_counts) / max(len(access_counts), 1),
                'average_age_seconds': sum(ages) / max(len(ages), 1),
                'strategies_enabled': len(self.strategies)
            }


# Global cache manager instance
cache_manager = IntelligentCacheManager()
