"""
Query result caching for improved performance.
"""

from cachetools import TTLCache
import hashlib
from typing import Dict, Any, List, Optional
import logging
import threading

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Cache for query results with TTL support.
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """
        Initialize query cache.

        MEDIUM PRIORITY FIX: Add cache size validation and limits.

        Args:
            maxsize: Maximum number of cached queries
            ttl: Time to live in seconds

        Raises:
            ValueError: If maxsize or ttl are invalid
        """
        # MEDIUM PRIORITY FIX: Validate cache size limits
        if maxsize < 1:
            raise ValueError(f"Cache maxsize must be >= 1, got {maxsize}")
        if maxsize > 100000:
            logger.warning(
                f"Cache maxsize {maxsize} is very large. "
                f"This may consume significant memory. Consider using a smaller value."
            )
        if ttl < 1:
            raise ValueError(f"Cache TTL must be >= 1 second, got {ttl}")
        if ttl > 86400:  # 24 hours
            logger.warning(
                f"Cache TTL {ttl}s is very long (>24h). "
                f"Consider using a shorter TTL for fresher results."
            )

        self.maxsize = maxsize
        self.ttl = ttl
        self.query_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.embedding_cache = TTLCache(maxsize=min(maxsize // 2, 500), ttl=ttl * 2)
        self.hits = 0
        self.misses = 0
        # HIGH PRIORITY FIX: Add lock for thread-safe counter updates
        self._stats_lock = threading.Lock()

        logger.info(f"QueryCache initialized (maxsize={maxsize}, ttl={ttl}s)")

    def _generate_key(self, query_text: str, n_results: int, file_type: Optional[str]) -> str:
        """
        Generate cache key from query parameters.

        Args:
            query_text: Query string
            n_results: Number of results
            file_type: Optional file type filter

        Returns:
            Cache key
        """
        key_str = f"{query_text}:{n_results}:{file_type or ''}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query_text: str, n_results: int = 5, file_type: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Get cached query results.

        Args:
            query_text: Query string
            n_results: Number of results
            file_type: Optional file type filter

        Returns:
            Cached results or None if not in cache
        """
        cache_key = self._generate_key(query_text, n_results, file_type)

        if cache_key in self.query_cache:
            # HIGH PRIORITY FIX: Thread-safe counter increment
            with self._stats_lock:
                self.hits += 1
            logger.debug(f"Cache HIT for query: {query_text[:50]}")
            return self.query_cache[cache_key]

        # HIGH PRIORITY FIX: Thread-safe counter increment
        with self._stats_lock:
            self.misses += 1
        logger.debug(f"Cache MISS for query: {query_text[:50]}")
        return None

    def put(self, query_text: str, results: List[Dict], n_results: int = 5, file_type: Optional[str] = None):
        """
        Store query results in cache.

        Args:
            query_text: Query string
            results: Query results
            n_results: Number of results
            file_type: Optional file type filter
        """
        cache_key = self._generate_key(query_text, n_results, file_type)
        self.query_cache[cache_key] = results
        logger.debug(f"Cached results for query: {query_text[:50]}")

    def get_embedding(self, text: str) -> Optional[Any]:
        """
        Get cached embedding.

        Args:
            text: Text to get embedding for

        Returns:
            Cached embedding or None
        """
        key = hashlib.md5(text.encode()).hexdigest()
        return self.embedding_cache.get(key)

    def put_embedding(self, text: str, embedding: Any):
        """
        Store embedding in cache.

        Args:
            text: Text
            embedding: Embedding vector
        """
        key = hashlib.md5(text.encode()).hexdigest()
        self.embedding_cache[key] = embedding

    def clear(self):
        """Clear all caches."""
        self.query_cache.clear()
        self.embedding_cache.clear()
        # HIGH PRIORITY FIX: Thread-safe counter reset
        with self._stats_lock:
            self.hits = 0
            self.misses = 0
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        MEDIUM PRIORITY FIX: Include cache size limits and utilization.

        Returns:
            Dictionary with cache stats
        """
        # HIGH PRIORITY FIX: Thread-safe counter read
        with self._stats_lock:
            hits = self.hits
            misses = self.misses

        total_requests = hits + misses
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0

        query_size = len(self.query_cache)
        embedding_size = len(self.embedding_cache)

        # MEDIUM PRIORITY FIX: Add utilization metrics
        query_utilization = (query_size / self.maxsize * 100) if self.maxsize > 0 else 0
        embedding_maxsize = self.embedding_cache.maxsize
        embedding_utilization = (embedding_size / embedding_maxsize * 100) if embedding_maxsize > 0 else 0

        return {
            'hits': hits,
            'misses': misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'query_cache_size': query_size,
            'query_cache_max': self.maxsize,
            'query_cache_utilization': f"{query_utilization:.1f}%",
            'embedding_cache_size': embedding_size,
            'embedding_cache_max': embedding_maxsize,
            'embedding_cache_utilization': f"{embedding_utilization:.1f}%",
            'ttl_seconds': self.ttl
        }
