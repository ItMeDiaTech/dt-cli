"""
Query prefetching - predictive pre-execution of likely next queries.

Learns query patterns and prefetches likely next queries to reduce latency.

Strategies:
- Sequential pattern learning (A → B → C)
- Time-based patterns (morning queries vs afternoon)
- Context-based suggestions
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from threading import Thread, Lock
import time

logger = logging.getLogger(__name__)


class QueryPattern:
    """Represents a learned query pattern."""

    def __init__(self, from_query: str, to_query: str, confidence: float = 0.0):
        """
        Initialize query pattern.

        Args:
            from_query: Source query
            to_query: Next query
            confidence: Confidence score (0-1)
        """
        self.from_query = from_query
        self.to_query = to_query
        self.confidence = confidence
        self.occurrences = 1

    def update_confidence(self, total_transitions: int):
        """
        Update confidence based on occurrences.

        Args:
            total_transitions: Total transitions from from_query
        """
        if total_transitions > 0:
            self.confidence = self.occurrences / total_transitions


class QueryPrefetcher:
    """
    Predictively prefetches likely next queries.
    """

    def __init__(
        self,
        query_engine,
        cache_manager=None,
        query_learning_system=None,
        min_confidence: float = 0.3
    ):
        """
        Initialize query prefetcher.

        Args:
            query_engine: Query engine instance
            cache_manager: Cache manager instance
            query_learning_system: Query learning system instance
            min_confidence: Minimum confidence to trigger prefetch
        """
        self.query_engine = query_engine
        self.cache_manager = cache_manager
        self.query_learning_system = query_learning_system
        self.min_confidence = min_confidence

        # Query transition patterns: query -> [next_queries]
        self.patterns: Dict[str, List[QueryPattern]] = defaultdict(list)

        # Recent query history (for detecting sequences)
        self.recent_queries: deque = deque(maxlen=10)

        # Lock for thread safety
        self.lock = Lock()

        # Prefetch queue
        self.prefetch_queue: List[Tuple[str, float]] = []

        # Running flag
        self.running = False
        self.prefetch_thread: Optional[Thread] = None

    def record_query(self, query: str):
        """
        Record a query and learn patterns.

        Args:
            query: Query text
        """
        with self.lock:
            if len(self.recent_queries) > 0:
                prev_query = self.recent_queries[-1]

                # Record transition pattern
                self._record_transition(prev_query, query)

            self.recent_queries.append(query)

        # Predict next queries
        self._predict_and_prefetch(query)

    def _record_transition(self, from_query: str, to_query: str):
        """
        Record query transition.

        Args:
            from_query: Previous query
            to_query: Next query
        """
        # Find existing pattern
        existing_pattern = None

        for pattern in self.patterns[from_query]:
            if pattern.to_query == to_query:
                existing_pattern = pattern
                break

        if existing_pattern:
            # Update existing pattern
            existing_pattern.occurrences += 1
        else:
            # Create new pattern
            pattern = QueryPattern(from_query, to_query)
            self.patterns[from_query].append(pattern)

        # Update confidences
        total_transitions = sum(p.occurrences for p in self.patterns[from_query])

        for pattern in self.patterns[from_query]:
            pattern.update_confidence(total_transitions)

        logger.debug(f"Recorded transition: {from_query[:30]} -> {to_query[:30]}")

    def _predict_and_prefetch(self, current_query: str):
        """
        Predict next queries and prefetch.

        Args:
            current_query: Current query
        """
        predictions = self.predict_next_queries(current_query, top_k=3)

        for next_query, confidence in predictions:
            if confidence >= self.min_confidence:
                logger.info(f"Prefetching: {next_query[:30]} (confidence: {confidence:.2f})")
                self._add_to_prefetch_queue(next_query, confidence)

    def predict_next_queries(
        self,
        current_query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict likely next queries.

        Args:
            current_query: Current query
            top_k: Number of predictions

        Returns:
            List of (query, confidence) tuples
        """
        with self.lock:
            if current_query not in self.patterns:
                return []

            patterns = self.patterns[current_query]

            # Sort by confidence
            patterns.sort(key=lambda p: p.confidence, reverse=True)

            predictions = [
                (p.to_query, p.confidence)
                for p in patterns[:top_k]
            ]

            return predictions

    def _add_to_prefetch_queue(self, query: str, priority: float):
        """
        Add query to prefetch queue.

        Args:
            query: Query to prefetch
            priority: Priority (higher = more important)
        """
        with self.lock:
            # Check if already queued
            if any(q == query for q, _ in self.prefetch_queue):
                return

            self.prefetch_queue.append((query, priority))

            # Sort by priority
            self.prefetch_queue.sort(key=lambda x: x[1], reverse=True)

            # Limit queue size
            self.prefetch_queue = self.prefetch_queue[:10]

    def start_prefetching(self):
        """Start background prefetching thread."""
        if self.running:
            logger.warning("Prefetching already running")
            return

        self.running = True
        self.prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()

        logger.info("Query prefetching started")

    def stop_prefetching(self):
        """Stop background prefetching."""
        self.running = False

        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5)

        logger.info("Query prefetching stopped")

    def _prefetch_worker(self):
        """Background worker for prefetching queries."""
        while self.running:
            try:
                with self.lock:
                    if not self.prefetch_queue:
                        # No queries to prefetch
                        time.sleep(1)
                        continue

                    # Get highest priority query
                    query, priority = self.prefetch_queue.pop(0)

                # Check if already cached
                if self.cache_manager:
                    cache_key = f"query:{query}"
                    if self.cache_manager.get(cache_key):
                        logger.debug(f"Query already cached: {query[:30]}")
                        continue

                # Execute query to cache results
                logger.info(f"Prefetching query: {query[:30]}")

                try:
                    results = self.query_engine.query(query, n_results=5)

                    # Cache results
                    if self.cache_manager:
                        self.cache_manager.set(
                            cache_key,
                            results,
                            metadata={'prefetched': True}
                        )

                    logger.debug(f"Prefetched {len(results)} results")

                except Exception as e:
                    logger.error(f"Prefetch failed for '{query[:30]}': {e}")

                # Small delay between prefetches
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                time.sleep(1)

    def learn_from_history(self):
        """Learn patterns from query history."""
        if not self.query_learning_system:
            logger.warning("Query learning system not available")
            return

        logger.info("Learning patterns from query history...")

        try:
            history = self.query_learning_system.history

            # Analyze sequential patterns
            for i in range(len(history) - 1):
                current = history[i]
                next_entry = history[i + 1]

                # Check if queries are close in time (< 5 minutes)
                current_time = current.timestamp
                next_time = next_entry.timestamp

                if (next_time - current_time) < timedelta(minutes=5):
                    self._record_transition(current.query, next_entry.query)

            # Count patterns learned
            pattern_count = sum(len(patterns) for patterns in self.patterns.values())

            logger.info(f"Learned {pattern_count} query patterns")

        except Exception as e:
            logger.error(f"Error learning from history: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get prefetching statistics.

        Returns:
            Statistics dictionary
        """
        with self.lock:
            total_patterns = sum(len(patterns) for patterns in self.patterns.values())

            high_confidence = 0
            for patterns in self.patterns.values():
                high_confidence += sum(1 for p in patterns if p.confidence >= 0.5)

            return {
                'total_queries_tracked': len(self.patterns),
                'total_patterns': total_patterns,
                'high_confidence_patterns': high_confidence,
                'prefetch_queue_size': len(self.prefetch_queue),
                'running': self.running
            }

    def clear_patterns(self):
        """Clear all learned patterns."""
        with self.lock:
            self.patterns.clear()
            self.prefetch_queue.clear()
            self.recent_queries.clear()

        logger.info("Cleared all query patterns")

    def export_patterns(self) -> Dict[str, Any]:
        """
        Export learned patterns.

        Returns:
            Patterns dictionary
        """
        with self.lock:
            exported = {}

            for from_query, patterns in self.patterns.items():
                exported[from_query] = [
                    {
                        'to_query': p.to_query,
                        'confidence': p.confidence,
                        'occurrences': p.occurrences
                    }
                    for p in patterns
                ]

            return exported

    def import_patterns(self, patterns_data: Dict[str, Any]):
        """
        Import learned patterns.

        Args:
            patterns_data: Patterns dictionary
        """
        with self.lock:
            self.patterns.clear()

            for from_query, pattern_list in patterns_data.items():
                for p_data in pattern_list:
                    pattern = QueryPattern(
                        from_query=from_query,
                        to_query=p_data['to_query'],
                        confidence=p_data.get('confidence', 0.0)
                    )
                    pattern.occurrences = p_data.get('occurrences', 1)

                    self.patterns[from_query].append(pattern)

            logger.info(f"Imported {len(self.patterns)} query patterns")


class ContextualPrefetcher:
    """
    Context-aware prefetching based on current file/function.
    """

    def __init__(self, query_engine):
        """
        Initialize contextual prefetcher.

        Args:
            query_engine: Query engine instance
        """
        self.query_engine = query_engine

    def prefetch_for_context(
        self,
        current_file: str,
        related_queries: Optional[List[str]] = None
    ):
        """
        Prefetch queries related to current context.

        Args:
            current_file: Current file path
            related_queries: Optional list of related queries
        """
        # Generate context-based queries
        queries_to_prefetch = related_queries or []

        # Add file-based queries
        file_name = current_file.split('/')[-1]
        queries_to_prefetch.append(f"how does {file_name} work")
        queries_to_prefetch.append(f"{file_name} dependencies")

        # Prefetch each query
        for query in queries_to_prefetch:
            try:
                _ = self.query_engine.query(query, n_results=3)
                logger.debug(f"Context prefetch: {query}")
            except Exception as e:
                logger.debug(f"Context prefetch failed: {e}")


# Global instance
query_prefetcher: Optional[QueryPrefetcher] = None
