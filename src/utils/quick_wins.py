"""
Quick wins - small features with immediate value.

Includes:
- Request timeouts
- Result deduplication
- Query validation
- Batch query API
- Config hot-reload
"""

import signal
from typing import List, Dict, Any, Optional, Callable
from functools import wraps
import hashlib
import re
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


def timeout(seconds: int):
    """
    Decorator to add timeout to function.

    Args:
        seconds: Timeout in seconds

    Example:
        @timeout(30)
        def slow_function():
            # Will timeout after 30 seconds
            pass
    """
    def decorator(func: Callable) -> Callable:
        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set alarm
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Restore old handler and cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper
    return decorator


class ResultDeduplicator:
    """
    Deduplicates search results based on content similarity.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize deduplicator.

        Args:
            similarity_threshold: Similarity threshold for duplicates (0-1)
        """
        self.similarity_threshold = similarity_threshold

    def deduplicate(
        self,
        results: List[Dict[str, Any]],
        key: str = 'content'
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate results.

        Args:
            results: List of results
            key: Key to use for deduplication

        Returns:
            Deduplicated results
        """
        if not results:
            return []

        seen_hashes = set()
        unique_results = []

        for result in results:
            content = result.get(key, '')

            # Calculate content hash
            content_hash = self._hash_content(content)

            # Check for exact duplicates
            if content_hash not in seen_hashes:
                # Check for near duplicates
                if not self._is_near_duplicate(content, unique_results, key):
                    seen_hashes.add(content_hash)
                    unique_results.append(result)
                else:
                    logger.debug(f"Filtered duplicate result: {result.get('id')}")

        removed_count = len(results) - len(unique_results)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate results")

        return unique_results

    def _hash_content(self, content: str) -> str:
        """
        Hash content for deduplication.

        Args:
            content: Content to hash

        Returns:
            Content hash
        """
        return hashlib.md5(content.encode()).hexdigest()

    def _is_near_duplicate(
        self,
        content: str,
        existing_results: List[Dict[str, Any]],
        key: str
    ) -> bool:
        """
        Check if content is near duplicate of existing results.

        Args:
            content: Content to check
            existing_results: Existing results
            key: Content key

        Returns:
            True if near duplicate
        """
        content_words = set(content.lower().split())

        if not content_words:
            return False

        for result in existing_results:
            existing_content = result.get(key, '')
            existing_words = set(existing_content.lower().split())

            if not existing_words:
                continue

            # Calculate Jaccard similarity
            intersection = content_words & existing_words
            union = content_words | existing_words

            similarity = len(intersection) / len(union) if union else 0

            if similarity >= self.similarity_threshold:
                return True

        return False


class QueryValidator:
    """
    Validates and sanitizes queries.
    """

    def __init__(
        self,
        min_length: int = 2,
        max_length: int = 500,
        allow_special_chars: bool = True
    ):
        """
        Initialize query validator.

        Args:
            min_length: Minimum query length
            max_length: Maximum query length
            allow_special_chars: Allow special characters
        """
        self.min_length = min_length
        self.max_length = max_length
        self.allow_special_chars = allow_special_chars

    def validate(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Validate query.

        Args:
            query: Query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if empty
        if not query or not query.strip():
            return False, "Query cannot be empty"

        # Check length
        if len(query) < self.min_length:
            return False, f"Query too short (minimum {self.min_length} characters)"

        if len(query) > self.max_length:
            return False, f"Query too long (maximum {self.max_length} characters)"

        # Check for malicious patterns
        if self._contains_injection_pattern(query):
            return False, "Query contains potentially malicious patterns"

        # Check special characters
        if not self.allow_special_chars:
            if not query.replace(' ', '').isalnum():
                return False, "Query contains special characters"

        return True, None

    def sanitize(self, query: str) -> str:
        """
        Sanitize query.

        Args:
            query: Query to sanitize

        Returns:
            Sanitized query
        """
        # Remove leading/trailing whitespace
        sanitized = query.strip()

        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        # Limit length
        if len(sanitized) > self.max_length:
            sanitized = sanitized[:self.max_length]

        return sanitized

    def _contains_injection_pattern(self, query: str) -> bool:
        """
        Check for SQL/NoSQL injection patterns.

        Args:
            query: Query to check

        Returns:
            True if contains injection pattern
        """
        injection_patterns = [
            r';\s*DROP\s+TABLE',
            r';\s*DELETE\s+FROM',
            r'UNION\s+SELECT',
            r'<script>',
            r'javascript:',
            r'\$ne\s*:',  # MongoDB
        ]

        query_upper = query.upper()

        return any(
            re.search(pattern, query_upper, re.IGNORECASE)
            for pattern in injection_patterns
        )


class BatchQueryExecutor:
    """
    Execute multiple queries in batch.
    """

    def __init__(self, query_engine):
        """
        Initialize batch executor.

        Args:
            query_engine: Query engine instance
        """
        self.query_engine = query_engine

    def execute_batch(
        self,
        queries: List[str],
        n_results: int = 5,
        deduplicate: bool = True,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Execute batch of queries.

        Args:
            queries: List of queries
            n_results: Results per query
            deduplicate: Deduplicate results
            parallel: Execute in parallel

        Returns:
            Batch results
        """
        logger.info(f"Executing batch of {len(queries)} queries")

        start_time = datetime.now()

        if parallel:
            results = self._execute_parallel(queries, n_results)
        else:
            results = self._execute_sequential(queries, n_results)

        # Deduplicate if requested
        if deduplicate:
            deduplicator = ResultDeduplicator()

            for query in results:
                results[query] = deduplicator.deduplicate(results[query])

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            'total_queries': len(queries),
            'execution_time_seconds': execution_time,
            'results': results,
            'avg_time_per_query': execution_time / len(queries) if queries else 0
        }

    def _execute_sequential(
        self,
        queries: List[str],
        n_results: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute queries sequentially."""
        results = {}

        for query in queries:
            try:
                query_results = self.query_engine.query(query, n_results=n_results)
                results[query] = query_results
            except Exception as e:
                logger.error(f"Error executing query '{query}': {e}")
                results[query] = []

        return results

    def _execute_parallel(
        self,
        queries: List[str],
        n_results: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute queries in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_query = {
                executor.submit(self.query_engine.query, query, n_results): query
                for query in queries
            }

            for future in as_completed(future_to_query):
                query = future_to_query[future]

                try:
                    query_results = future.result()
                    results[query] = query_results
                except Exception as e:
                    logger.error(f"Error executing query '{query}': {e}")
                    results[query] = []

        return results


class ConfigHotReloader:
    """
    Hot reload configuration without restart.
    """

    def __init__(self, config_path: Path):
        """
        Initialize config hot reloader.

        Args:
            config_path: Path to config file
        """
        self.config_path = config_path
        self.last_mtime = 0
        self.config: Dict[str, Any] = {}

        self._load_config()

    def get_config(self) -> Dict[str, Any]:
        """
        Get current config, reload if changed.

        Returns:
            Configuration dictionary
        """
        # Check if file changed
        if self._config_changed():
            logger.info("Config file changed, reloading...")
            self._load_config()

        return self.config.copy()

    def _config_changed(self) -> bool:
        """
        Check if config file changed.

        Returns:
            True if changed
        """
        if not self.config_path.exists():
            return False

        current_mtime = self.config_path.stat().st_mtime

        if current_mtime > self.last_mtime:
            return True

        return False

    def _load_config(self):
        """Load config from file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return

        try:
            self.config = json.loads(self.config_path.read_text())
            self.last_mtime = self.config_path.stat().st_mtime

            logger.info(f"Config loaded from {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def reload(self):
        """Force reload config."""
        self._load_config()


class RateLimiter:
    """
    Simple rate limiter for queries.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []

    def allow_request(self) -> bool:
        """
        Check if request is allowed.

        Returns:
            True if allowed
        """
        import time

        current_time = time.time()

        # Remove old requests outside window
        self.requests = [
            req_time for req_time in self.requests
            if current_time - req_time < self.window_seconds
        ]

        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True

        logger.warning("Rate limit exceeded")
        return False

    def get_remaining_requests(self) -> int:
        """
        Get remaining requests in current window.

        Returns:
            Number of remaining requests
        """
        import time

        current_time = time.time()

        # Remove old requests
        self.requests = [
            req_time for req_time in self.requests
            if current_time - req_time < self.window_seconds
        ]

        return self.max_requests - len(self.requests)


# Helper functions
def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Helper to deduplicate results.

    Args:
        results: List of results

    Returns:
        Deduplicated results
    """
    deduplicator = ResultDeduplicator()
    return deduplicator.deduplicate(results)


def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Helper to validate query.

    Args:
        query: Query to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = QueryValidator()
    return validator.validate(query)


def sanitize_query(query: str) -> str:
    """
    Helper to sanitize query.

    Args:
        query: Query to sanitize

    Returns:
        Sanitized query
    """
    validator = QueryValidator()
    return validator.sanitize(query)
