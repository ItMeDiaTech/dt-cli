"""
Health monitoring and metrics collection.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitor system health and collect metrics."""

    def __init__(self):
        """Initialize health monitor."""
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.last_error: Optional[Dict[str, Any]] = None
        self.query_times: list = []

    def record_request(self):
        """Record a request."""
        self.request_count += 1

    def record_error(self, error: Exception):
        """
        Record an error.

        Args:
            error: Exception that occurred
        """
        self.error_count += 1
        self.last_error = {
            'type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        logger.error(f"Error recorded: {error}")

    def record_query_time(self, duration_ms: float):
        """
        Record query execution time.

        Args:
            duration_ms: Query duration in milliseconds
        """
        self.query_times.append(duration_ms)

        # Keep only last 1000 queries
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-1000:]

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status.

        Returns:
            Health status dictionary
        """
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        error_rate = self.error_count / max(self.request_count, 1)

        # Determine overall status
        if self.error_count > 10 or error_rate > 0.1:
            status = "unhealthy"
        elif self.error_count > 5 or error_rate > 0.05:
            status = "degraded"
        else:
            status = "healthy"

        # Calculate query stats
        avg_query_time = (
            sum(self.query_times) / len(self.query_times)
            if self.query_times else 0
        )

        return {
            'status': status,
            'uptime_seconds': round(uptime_seconds, 2),
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': round(error_rate * 100, 2),
            'last_error': self.last_error,
            'avg_query_time_ms': round(avg_query_time, 2) if avg_query_time else None
        }

    def reset_stats(self):
        """Reset statistics."""
        self.request_count = 0
        self.error_count = 0
        self.last_error = None
        self.query_times = []
        logger.info("Health monitor stats reset")


class MetricsCollector:
    """Collect and track various metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            'queries_total': 0,
            'queries_cached': 0,
            'queries_miss': 0,
            'indexing_operations': 0,
            'files_indexed': 0,
            'chunks_created': 0,
            'agents_executed': 0
        }

    def increment(self, metric_name: str, value: int = 1):
        """
        Increment a metric.

        Args:
            metric_name: Name of the metric
            value: Value to increment by
        """
        if metric_name in self.metrics:
            self.metrics[metric_name] += value
        else:
            self.metrics[metric_name] = value

    def get_metrics(self) -> Dict[str, int]:
        """
        Get all metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()

    def get_metric(self, metric_name: str) -> int:
        """
        Get a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Metric value
        """
        return self.metrics.get(metric_name, 0)

    def reset(self):
        """Reset all metrics."""
        self.metrics = {key: 0 for key in self.metrics}
        logger.info("Metrics reset")
