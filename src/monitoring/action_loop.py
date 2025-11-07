"""
Monitoring Action Loop - Auto-remediation based on health checks.
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
import threading
import time

logger = logging.getLogger(__name__)


class ActionLoop:
    """
    Monitors system health and triggers automatic remediation actions.
    """

    def __init__(
        self,
        health_monitor,
        query_engine,
        orchestrator,
        check_interval: int = 60
    ):
        """
        Initialize action loop.

        Args:
            health_monitor: Health monitoring instance
            query_engine: Query engine instance
            orchestrator: Agent orchestrator instance
            check_interval: How often to check (seconds)
        """
        self.health_monitor = health_monitor
        self.query_engine = query_engine
        self.orchestrator = orchestrator
        self.check_interval = check_interval

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_remediation: Dict[str, datetime] = {}
        self.remediation_cooldown = timedelta(minutes=5)

        # Action thresholds
        self.thresholds = {
            'error_rate': 0.1,      # 10% error rate
            'error_count': 10,       # 10 errors
            'memory_percent': 80,    # 80% memory usage
            'slow_queries': 5        # 5 slow queries
        }

        logger.info("ActionLoop initialized")

    def start(self):
        """Start the monitoring action loop."""
        if self.running:
            logger.warning("ActionLoop already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("ActionLoop started")

    def stop(self):
        """Stop the monitoring action loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("ActionLoop stopped")

    def _run_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_and_act()
            except Exception as e:
                logger.error(f"ActionLoop error: {e}")

            time.sleep(self.check_interval)

    def _check_and_act(self):
        """Check health status and trigger actions if needed."""
        status = self.health_monitor.get_health_status()

        # Check error rate
        if status.get('error_rate', 0) > self.thresholds['error_rate']:
            self._trigger_action('high_error_rate', self._handle_high_error_rate)

        # Check error count
        if status.get('error_count', 0) > self.thresholds['error_count']:
            self._trigger_action('error_spike', self._handle_error_spike)

        # Check memory usage
        memory_percent = self._get_memory_usage()
        if memory_percent > self.thresholds['memory_percent']:
            self._trigger_action('high_memory', self._handle_high_memory)

        # Check slow queries
        avg_query_time = status.get('avg_query_time_ms')
        if avg_query_time and avg_query_time > 1000:  # > 1 second
            self._trigger_action('slow_queries', self._handle_slow_queries)

    def _trigger_action(self, action_name: str, action_func: Callable):
        """
        Trigger a remediation action with cooldown.

        Args:
            action_name: Name of the action
            action_func: Function to execute
        """
        # Check cooldown
        if action_name in self.last_remediation:
            time_since = datetime.now() - self.last_remediation[action_name]
            if time_since < self.remediation_cooldown:
                logger.debug(f"Action {action_name} in cooldown")
                return

        logger.warning(f"Triggering remediation: {action_name}")

        try:
            action_func()
            self.last_remediation[action_name] = datetime.now()
        except Exception as e:
            logger.error(f"Remediation action {action_name} failed: {e}")

    def _handle_high_error_rate(self):
        """Handle high error rate - clear caches and reset contexts."""
        logger.info("Remediation: Clearing caches due to high error rate")

        # Clear query cache
        if hasattr(self.query_engine, 'cache'):
            self.query_engine.cache.clear()

        # Clear old contexts
        if hasattr(self.orchestrator, 'cleanup_old_contexts'):
            self.orchestrator.cleanup_old_contexts(max_age_seconds=1800)

        # Reset health monitor stats
        self.health_monitor.reset_stats()

    def _handle_error_spike(self):
        """Handle error spike - more aggressive cleanup."""
        logger.info("Remediation: Aggressive cleanup due to error spike")

        # Clear all caches
        if hasattr(self.query_engine, 'cache'):
            self.query_engine.cache.clear()

        # Clear all contexts
        if hasattr(self.orchestrator, 'context_manager'):
            for ctx_id in list(self.orchestrator.context_manager.get_active_contexts()):
                self.orchestrator.context_manager.clear_context(ctx_id)

    def _handle_high_memory(self):
        """Handle high memory usage - unload models and clear caches."""
        logger.info("Remediation: Reducing memory usage")

        # Unload embedding model if using lazy loading
        if hasattr(self.query_engine.embedding_engine, 'unload'):
            self.query_engine.embedding_engine.unload()

        # Clear caches
        if hasattr(self.query_engine, 'cache'):
            self.query_engine.cache.clear()

        # Cleanup old contexts
        if hasattr(self.orchestrator, 'cleanup_old_contexts'):
            self.orchestrator.cleanup_old_contexts(max_age_seconds=600)

    def _handle_slow_queries(self):
        """Handle slow queries - optimize cache settings."""
        logger.info("Remediation: Optimizing for slow queries")

        # Could trigger reindexing, cache warming, etc.
        # For now, just log
        logger.info("Consider reindexing or checking vector store performance")

    def _get_memory_usage(self) -> float:
        """
        Get current memory usage percentage.

        Returns:
            Memory usage as percentage
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            return memory_percent
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
            return 0.0
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0

    def get_status(self) -> Dict[str, Any]:
        """
        Get action loop status.

        Returns:
            Status dictionary
        """
        return {
            'running': self.running,
            'check_interval': self.check_interval,
            'thresholds': self.thresholds,
            'last_remediation': {
                k: v.isoformat() for k, v in self.last_remediation.items()
            }
        }
