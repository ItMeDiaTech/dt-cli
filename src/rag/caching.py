"""
Query result caching for improved performance.
"""

from cachetools import TTLCache
import hashlib
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Cache for query results with TTL support.
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """
        Initialize query cache.

        Args:
            maxsize: Maximum number of cached queries
            ttl: Time to live in seconds
        """
        self.query_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.embedding_cache = TTLCache(maxsize=500, ttl=ttl * 2)
        self.hits = 0
        self.misses = 0

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
            self.hits += 1
            logger.debug(f"Cache HIT for query: {query_text[:50]}")
            return self.query_cache[cache_key]

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
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'query_cache_size': len(self.query_cache),
            'embedding_cache_size': len(self.embedding_cache)
        }
