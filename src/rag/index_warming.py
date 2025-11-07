"""
Index warming strategies to pre-load frequently accessed data.

Improves cold-start performance by:
- Pre-loading models
- Pre-caching popular queries
- Pre-fetching frequently accessed files
"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class IndexWarmer:
    """
    Warms up index and caches for better cold-start performance.

    MEDIUM PRIORITY FIX: Added thread safety for concurrent warming operations.
    """

    def __init__(
        self,
        query_engine,
        query_learning_system=None,
        cache_manager=None
    ):
        """
        Initialize index warmer.

        Args:
            query_engine: Query engine instance
            query_learning_system: Query learning system instance
            cache_manager: Cache manager instance
        """
        self.query_engine = query_engine
        self.query_learning_system = query_learning_system
        self.cache_manager = cache_manager

        # MEDIUM PRIORITY FIX: Add lock for thread-safe concurrent warming
        self._warming_lock = threading.Lock()
        self._is_warming = False

    def warm_all(self, max_time_seconds: int = 30) -> Dict[str, Any]:
        """
        Execute all warming strategies.

        MEDIUM PRIORITY FIX: Thread-safe warming prevents concurrent operations.

        Args:
            max_time_seconds: Maximum time to spend warming

        Returns:
            Warming statistics
        """
        # MEDIUM PRIORITY FIX: Check if already warming
        with self._warming_lock:
            if self._is_warming:
                logger.warning("Index warming already in progress, skipping")
                return {'already_warming': True}
            self._is_warming = True

        try:
            logger.info("Starting index warming...")
            start_time = time.time()

            stats = {
                'models_loaded': 0,
                'queries_cached': 0,
                'files_preloaded': 0,
                'total_time_seconds': 0
            }

            # 1. Load models
            if time.time() - start_time < max_time_seconds:
                model_stats = self.warm_models()
                stats['models_loaded'] = model_stats.get('models_loaded', 0)

            # 2. Pre-cache popular queries
            if time.time() - start_time < max_time_seconds:
                query_stats = self.warm_popular_queries(
                    max_queries=10,
                    max_time_seconds=max_time_seconds - (time.time() - start_time)
                )
                stats['queries_cached'] = query_stats.get('queries_cached', 0)

            # 3. Pre-load frequently accessed files
            if time.time() - start_time < max_time_seconds:
                file_stats = self.warm_frequent_files(max_files=20)
                stats['files_preloaded'] = file_stats.get('files_preloaded', 0)

            stats['total_time_seconds'] = time.time() - start_time

            logger.info(f"Index warming complete: {stats}")
            return stats

        finally:
            # MEDIUM PRIORITY FIX: Always clear warming flag
            with self._warming_lock:
                self._is_warming = False

    def warm_models(self) -> Dict[str, Any]:
        """
        Pre-load ML models to memory.

        MEDIUM PRIORITY FIX: Validate query engine and handle errors.

        Returns:
            Model loading statistics
        """
        logger.info("Warming models...")

        models_loaded = 0

        # MEDIUM PRIORITY FIX: Validate query engine before warming
        if not self.query_engine:
            logger.error("Query engine not available for warming")
            return {'models_loaded': 0, 'error': 'No query engine'}

        try:
            # Load embedding model
            if hasattr(self.query_engine, 'embedding_engine'):
                if hasattr(self.query_engine.embedding_engine, 'model'):
                    # MEDIUM PRIORITY FIX: Handle encode failures
                    try:
                        _ = self.query_engine.embedding_engine.encode(["warmup"])
                        models_loaded += 1
                        logger.info("Embedding model loaded")
                    except Exception as e:
                        logger.error(f"Failed to load embedding model: {e}")

            # Load reranking model if exists
            if hasattr(self.query_engine, 'reranker'):
                if hasattr(self.query_engine.reranker, 'model'):
                    # Trigger reranker loading
                    try:
                        self.query_engine.reranker.rerank(
                            "warmup",
                            [{"content": "test"}],
                            top_k=1
                        )
                        models_loaded += 1
                        logger.info("Reranker model loaded")
                    except Exception as e:
                        logger.debug(f"Reranker warmup failed: {e}")

        except Exception as e:
            logger.error(f"Error warming models: {e}")

        return {'models_loaded': models_loaded}

    def warm_popular_queries(
        self,
        max_queries: int = 10,
        max_time_seconds: int = 15
    ) -> Dict[str, Any]:
        """
        Pre-cache popular queries.

        Args:
            max_queries: Maximum queries to warm
            max_time_seconds: Maximum time to spend

        Returns:
            Query caching statistics
        """
        logger.info(f"Warming {max_queries} popular queries...")

        if not self.query_learning_system:
            logger.warning("Query learning system not available for warming")
            return {'queries_cached': 0}

        queries_cached = 0
        start_time = time.time()

        try:
            # Get popular queries
            popular = self.query_learning_system.get_popular_queries(
                days=7,
                top_k=max_queries
            )

            for query_info in popular:
                # Check time limit
                if time.time() - start_time > max_time_seconds:
                    logger.info(f"Time limit reached, cached {queries_cached} queries")
                    break

                query = query_info['query']

                try:
                    # Execute query to cache results
                    _ = self.query_engine.query(query, n_results=5)
                    queries_cached += 1
                    logger.debug(f"Cached query: {query}")

                except Exception as e:
                    logger.debug(f"Error caching query '{query}': {e}")

        except Exception as e:
            logger.error(f"Error warming queries: {e}")

        return {'queries_cached': queries_cached}

    def warm_frequent_files(self, max_files: int = 20) -> Dict[str, Any]:
        """
        Pre-load frequently accessed files.

        Args:
            max_files: Maximum files to pre-load

        Returns:
            File loading statistics
        """
        logger.info(f"Warming {max_files} frequent files...")

        files_preloaded = 0

        try:
            # Get frequently accessed files from query history
            frequent_files = self._get_frequent_files(max_files)

            # Pre-load files in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._preload_file, file_path): file_path
                    for file_path in frequent_files
                }

                for future in as_completed(futures):
                    file_path = futures[future]

                    try:
                        if future.result():
                            files_preloaded += 1
                            logger.debug(f"Preloaded file: {file_path}")

                    except Exception as e:
                        logger.debug(f"Error preloading {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error warming files: {e}")

        return {'files_preloaded': files_preloaded}

    def warm_specific_queries(self, queries: List[str]) -> Dict[str, Any]:
        """
        Warm specific queries.

        Args:
            queries: List of queries to warm

        Returns:
            Warming statistics
        """
        logger.info(f"Warming {len(queries)} specific queries...")

        queries_cached = 0

        for query in queries:
            try:
                _ = self.query_engine.query(query, n_results=5)
                queries_cached += 1

            except Exception as e:
                logger.debug(f"Error warming query '{query}': {e}")

        return {'queries_cached': queries_cached}

    def _get_frequent_files(self, max_files: int) -> List[str]:
        """
        Get frequently accessed files from history.

        Args:
            max_files: Maximum files to return

        Returns:
            List of file paths
        """
        if not self.query_learning_system:
            return []

        try:
            # Get all history
            history = self.query_learning_system.history

            # Count file accesses from selected results
            file_counts: Dict[str, int] = {}

            for entry in history:
                # This is a simplification - in reality we'd need to track
                # which files were in the selected results
                pass

            # For now, return empty list
            # In a real implementation, we'd track file access patterns
            return []

        except Exception as e:
            logger.error(f"Error getting frequent files: {e}")
            return []

    def _preload_file(self, file_path: str) -> bool:
        """
        Pre-load a file into cache.

        Args:
            file_path: File path to pre-load

        Returns:
            True if successful
        """
        try:
            path = Path(file_path)

            if not path.exists():
                return False

            # Read file to OS cache
            _ = path.read_text()

            return True

        except Exception as e:
            logger.debug(f"Error preloading {file_path}: {e}")
            return False

    def warm_on_startup(self, background: bool = True):
        """
        Warm index on system startup.

        Args:
            background: Run in background thread
        """
        if background:
            import threading

            def warm_task():
                try:
                    self.warm_all(max_time_seconds=30)
                except Exception as e:
                    logger.error(f"Background warming failed: {e}")

            thread = threading.Thread(target=warm_task, daemon=True)
            thread.start()

            logger.info("Index warming started in background")

        else:
            self.warm_all(max_time_seconds=30)


class AdaptiveWarmer:
    """
    Adaptive warming that learns which data to pre-load.
    """

    def __init__(
        self,
        query_engine,
        query_learning_system=None
    ):
        """
        Initialize adaptive warmer.

        Args:
            query_engine: Query engine instance
            query_learning_system: Query learning system instance
        """
        self.query_engine = query_engine
        self.query_learning_system = query_learning_system
        self.warming_patterns: Dict[str, Any] = {}

    def learn_warming_patterns(self) -> Dict[str, Any]:
        """
        Learn optimal warming patterns from usage.

        Returns:
            Learned patterns
        """
        if not self.query_learning_system:
            return {}

        try:
            # Analyze query history
            insights = self.query_learning_system.get_learning_insights()

            # Extract warming patterns
            patterns = {
                'peak_hours': [],
                'common_query_patterns': [],
                'frequent_terms': []
            }

            # Get usage patterns
            if 'usage_patterns' in insights:
                most_active_hour = insights['usage_patterns'].get('most_active_hour')
                if most_active_hour is not None:
                    patterns['peak_hours'] = [most_active_hour]

            # Get successful patterns
            if 'success_metrics' in insights:
                success_patterns = insights['success_metrics'].get('successful_query_patterns', {})
                patterns['frequent_terms'] = success_patterns.get('common_terms', [])

            self.warming_patterns = patterns
            logger.info(f"Learned warming patterns: {patterns}")

            return patterns

        except Exception as e:
            logger.error(f"Error learning patterns: {e}")
            return {}

    def smart_warm(self) -> Dict[str, Any]:
        """
        Smart warming based on learned patterns.

        Returns:
            Warming statistics
        """
        # Learn patterns first
        patterns = self.learn_warming_patterns()

        # Warm based on patterns
        warmer = IndexWarmer(
            self.query_engine,
            self.query_learning_system
        )

        # Generate queries from patterns
        queries_to_warm = []

        for term in patterns.get('frequent_terms', [])[:5]:
            queries_to_warm.append(term)

        # Warm those queries
        return warmer.warm_specific_queries(queries_to_warm)
