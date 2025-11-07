"""
Query history and learning system.

Tracks:
- Query patterns
- Result selections
- Feedback signals
- Performance metrics

Learns:
- Popular queries
- Query reformulations
- Effective result patterns
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import logging
import threading

logger = logging.getLogger(__name__)


class QueryHistoryEntry:
    """Represents a single query history entry."""

    def __init__(
        self,
        query: str,
        results_count: int,
        selected_results: Optional[List[int]] = None,
        feedback_score: Optional[float] = None,
        execution_time_ms: float = 0,
        correlation_id: Optional[str] = None
    ):
        """
        Initialize query history entry.

        Args:
            query: Query text
            results_count: Number of results returned
            selected_results: Indices of results user selected
            feedback_score: User feedback score (0-1)
            execution_time_ms: Query execution time
            correlation_id: Optional correlation ID
        """
        self.query = query
        self.results_count = results_count
        self.selected_results = selected_results or []
        self.feedback_score = feedback_score
        self.execution_time_ms = execution_time_ms
        self.correlation_id = correlation_id
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'results_count': self.results_count,
            'selected_results': self.selected_results,
            'feedback_score': self.feedback_score,
            'execution_time_ms': self.execution_time_ms,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryHistoryEntry':
        """Create from dictionary."""
        entry = cls(
            query=data['query'],
            results_count=data['results_count'],
            selected_results=data.get('selected_results', []),
            feedback_score=data.get('feedback_score'),
            execution_time_ms=data.get('execution_time_ms', 0),
            correlation_id=data.get('correlation_id')
        )

        if 'timestamp' in data:
            entry.timestamp = datetime.fromisoformat(data['timestamp'])

        return entry


class QueryLearningSystem:
    """
    Learns from query history to improve results.

    MEDIUM PRIORITY FIX: Added thread safety and atomic operations.
    """

    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize query learning system.

        Args:
            history_file: Path to history file (default: .rag_history.json)
        """
        self.history_file = history_file or Path.home() / '.rag_history.json'
        self.history: List[QueryHistoryEntry] = []

        # MEDIUM PRIORITY FIX: Add thread safety
        self._history_lock = threading.Lock()

        self._load_history()

    def record_query(
        self,
        query: str,
        results_count: int,
        selected_results: Optional[List[int]] = None,
        feedback_score: Optional[float] = None,
        execution_time_ms: float = 0,
        correlation_id: Optional[str] = None
    ):
        """
        Record a query in history.

        MEDIUM PRIORITY FIX: Thread-safe recording with lock.

        Args:
            query: Query text
            results_count: Number of results returned
            selected_results: Indices of results user selected
            feedback_score: User feedback score (0-1)
            execution_time_ms: Query execution time
            correlation_id: Optional correlation ID
        """
        entry = QueryHistoryEntry(
            query=query,
            results_count=results_count,
            selected_results=selected_results,
            feedback_score=feedback_score,
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id
        )

        # MEDIUM PRIORITY FIX: Thread-safe append
        with self._history_lock:
            self.history.append(entry)
            should_save = len(self.history) % 10 == 0

        # Auto-save periodically (outside lock to avoid blocking)
        if should_save:
            self._save_history()

        logger.debug(f"Recorded query: {query}")

    def get_similar_queries(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar past queries.

        MEDIUM PRIORITY FIX: Thread-safe access to history.

        Args:
            query: Current query
            max_results: Maximum results to return

        Returns:
            List of similar queries with metadata
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        similar = []

        # MEDIUM PRIORITY FIX: Thread-safe iteration
        with self._history_lock:
            history_snapshot = list(self.history)

        for entry in history_snapshot:
            entry_terms = set(entry.query.lower().split())

            # Calculate Jaccard similarity
            intersection = query_terms & entry_terms
            union = query_terms | entry_terms

            if union:
                similarity = len(intersection) / len(union)

                if similarity > 0.3:  # Threshold
                    similar.append({
                        'query': entry.query,
                        'similarity': similarity,
                        'results_count': entry.results_count,
                        'feedback_score': entry.feedback_score,
                        'timestamp': entry.timestamp.isoformat()
                    })

        # Sort by similarity
        similar.sort(key=lambda x: x['similarity'], reverse=True)

        return similar[:max_results]

    def get_popular_queries(
        self,
        days: int = 30,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most popular queries.

        Args:
            days: Number of days to look back
            top_k: Number of top queries to return

        Returns:
            List of popular queries
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_queries = [
            entry.query for entry in self.history
            if entry.timestamp >= cutoff_date
        ]

        query_counts = Counter(recent_queries)

        popular = [
            {
                'query': query,
                'count': count,
                'frequency': count / len(recent_queries) if recent_queries else 0
            }
            for query, count in query_counts.most_common(top_k)
        ]

        return popular

    def get_query_suggestions(
        self,
        partial_query: str,
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Get query suggestions based on history.

        Args:
            partial_query: Partial query text
            max_suggestions: Maximum suggestions

        Returns:
            List of suggested queries
        """
        partial_lower = partial_query.lower()

        suggestions = []

        # Find queries that start with partial
        for entry in reversed(self.history):
            if entry.query.lower().startswith(partial_lower):
                if entry.query not in suggestions:
                    suggestions.append(entry.query)

                if len(suggestions) >= max_suggestions:
                    break

        # If not enough, find queries containing partial
        if len(suggestions) < max_suggestions:
            for entry in reversed(self.history):
                if partial_lower in entry.query.lower():
                    if entry.query not in suggestions:
                        suggestions.append(entry.query)

                    if len(suggestions) >= max_suggestions:
                        break

        return suggestions

    def get_performance_metrics(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get performance metrics over time period.

        Args:
            days: Number of days to analyze

        Returns:
            Performance metrics
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_entries = [
            entry for entry in self.history
            if entry.timestamp >= cutoff_date
        ]

        if not recent_entries:
            return {}

        # Calculate metrics
        total_queries = len(recent_entries)

        execution_times = [
            e.execution_time_ms for e in recent_entries
            if e.execution_time_ms > 0
        ]

        feedback_scores = [
            e.feedback_score for e in recent_entries
            if e.feedback_score is not None
        ]

        results_counts = [e.results_count for e in recent_entries]

        metrics = {
            'total_queries': total_queries,
            'queries_per_day': total_queries / days,
            'avg_execution_time_ms': sum(execution_times) / len(execution_times) if execution_times else 0,
            'p95_execution_time_ms': self._percentile(execution_times, 0.95) if execution_times else 0,
            'avg_results_count': sum(results_counts) / len(results_counts) if results_counts else 0,
            'avg_feedback_score': sum(feedback_scores) / len(feedback_scores) if feedback_scores else None,
            'queries_with_feedback': len(feedback_scores)
        }

        return metrics

    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights learned from query history.

        Returns:
            Learning insights
        """
        if not self.history:
            return {}

        # Analyze query patterns
        query_lengths = [len(e.query.split()) for e in self.history]

        # Analyze result selection patterns
        selection_positions = []
        for entry in self.history:
            selection_positions.extend(entry.selected_results)

        # Analyze time patterns
        hours = [e.timestamp.hour for e in self.history]
        hour_distribution = Counter(hours)

        # Analyze successful queries (high feedback)
        successful_queries = [
            e for e in self.history
            if e.feedback_score and e.feedback_score >= 0.7
        ]

        insights = {
            'query_characteristics': {
                'avg_query_length': sum(query_lengths) / len(query_lengths) if query_lengths else 0,
                'optimal_query_length': self._find_optimal_query_length(),
            },
            'result_selection_patterns': {
                'avg_selection_position': sum(selection_positions) / len(selection_positions) if selection_positions else None,
                'top_3_selection_rate': len([p for p in selection_positions if p < 3]) / len(selection_positions) if selection_positions else 0
            },
            'usage_patterns': {
                'most_active_hour': hour_distribution.most_common(1)[0][0] if hour_distribution else None,
                'total_sessions': len(self.history)
            },
            'success_metrics': {
                'successful_query_rate': len(successful_queries) / len(self.history) if self.history else 0,
                'successful_query_patterns': self._analyze_successful_patterns(successful_queries)
            }
        }

        return insights

    def _find_optimal_query_length(self) -> Optional[int]:
        """
        Find optimal query length based on feedback.

        Returns:
            Optimal query length or None
        """
        length_feedback = defaultdict(list)

        for entry in self.history:
            if entry.feedback_score is not None:
                length = len(entry.query.split())
                length_feedback[length].append(entry.feedback_score)

        if not length_feedback:
            return None

        # Find length with highest avg feedback
        avg_feedback_by_length = {
            length: sum(scores) / len(scores)
            for length, scores in length_feedback.items()
        }

        optimal_length = max(avg_feedback_by_length, key=avg_feedback_by_length.get)
        return optimal_length

    def _analyze_successful_patterns(
        self,
        successful_queries: List[QueryHistoryEntry]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in successful queries.

        Args:
            successful_queries: List of successful query entries

        Returns:
            Pattern analysis
        """
        if not successful_queries:
            return {}

        # Extract common terms
        all_terms = []
        for entry in successful_queries:
            all_terms.extend(entry.query.lower().split())

        term_counts = Counter(all_terms)

        # Find query types
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        questions = sum(1 for e in successful_queries if any(w in e.query.lower() for w in question_words))

        return {
            'common_terms': [term for term, count in term_counts.most_common(10)],
            'question_rate': questions / len(successful_queries) if successful_queries else 0,
            'avg_results_returned': sum(e.results_count for e in successful_queries) / len(successful_queries)
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """
        Calculate percentile of values.

        Args:
            values: List of values
            percentile: Percentile (0-1)

        Returns:
            Percentile value
        """
        if not values:
            return 0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _load_history(self):
        """
        Load history from file.

        MEDIUM PRIORITY FIX: Validate loaded history structure.
        """
        if not self.history_file.exists():
            return

        try:
            data = json.loads(self.history_file.read_text())

            # MEDIUM PRIORITY FIX: Validate structure
            if not isinstance(data, dict):
                raise ValueError(f"Invalid history format: expected dict, got {type(data)}")

            if 'entries' not in data:
                logger.warning("History file missing 'entries' field, treating as empty")
                return

            if not isinstance(data['entries'], list):
                raise ValueError(f"Invalid entries format: expected list, got {type(data['entries'])}")

            # MEDIUM PRIORITY FIX: Validate and load entries with error recovery
            loaded_count = 0
            errors = 0

            for idx, entry_data in enumerate(data['entries']):
                try:
                    # Validate entry structure
                    if not isinstance(entry_data, dict):
                        logger.warning(f"Skipping invalid entry at index {idx}: not a dict")
                        errors += 1
                        continue

                    # Check required fields
                    required_fields = ['query', 'results_count']
                    missing_fields = [f for f in required_fields if f not in entry_data]

                    if missing_fields:
                        logger.warning(
                            f"Skipping entry at index {idx}: missing fields {missing_fields}"
                        )
                        errors += 1
                        continue

                    # Load entry
                    entry = QueryHistoryEntry.from_dict(entry_data)
                    self.history.append(entry)
                    loaded_count += 1

                except Exception as e:
                    logger.warning(f"Error loading entry at index {idx}: {e}")
                    errors += 1

            logger.info(
                f"Loaded {loaded_count} query history entries "
                f"({errors} errors, {len(self.history)} total)"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted history file: {e}. Starting with empty history.")
            self.history = []

        except Exception as e:
            logger.error(f"Error loading query history: {e}. Starting with empty history.")
            self.history = []

    def _save_history(self):
        """
        Save history to file.

        MEDIUM PRIORITY FIX: Use atomic write to prevent corruption.
        """
        import tempfile
        import os

        try:
            # MEDIUM PRIORITY FIX: Thread-safe copy of entries
            with self._history_lock:
                entries_to_save = self.history[-1000:]

            data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'entries': [entry.to_dict() for entry in entries_to_save]
            }

            # MEDIUM PRIORITY FIX: Atomic write using temp file + rename
            # Ensure parent directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)

            # Serialize to JSON
            json_data = json.dumps(data, indent=2)

            # Write to temp file
            fd, temp_path = tempfile.mkstemp(
                dir=str(self.history_file.parent),
                prefix='.rag_history.tmp.',
                suffix='.json'
            )

            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(json_data)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure written to disk

                # Atomic rename
                os.replace(temp_path, str(self.history_file))

                logger.debug(f"Saved {len(entries_to_save)} query history entries")

            except Exception:
                # Cleanup temp file on error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            logger.error(f"Error saving query history: {e}")

    def export_history(self, output_path: Path):
        """
        Export full history to file.

        Args:
            output_path: Output file path
        """
        data = {
            'version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'total_entries': len(self.history),
            'entries': [entry.to_dict() for entry in self.history]
        }

        output_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Exported history to {output_path}")

    def clear_old_history(self, days: int = 90):
        """
        Clear history older than specified days.

        Args:
            days: Days to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        original_count = len(self.history)

        self.history = [
            entry for entry in self.history
            if entry.timestamp >= cutoff_date
        ]

        removed_count = original_count - len(self.history)

        if removed_count > 0:
            self._save_history()
            logger.info(f"Cleared {removed_count} old history entries")


# Global instance
query_learning_system = QueryLearningSystem()
