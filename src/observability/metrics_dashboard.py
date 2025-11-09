"""
CLI Metrics Dashboard for RAG system monitoring.

Displays:
- System health
- Query performance
- Cache statistics
- Resource usage
- Recent queries
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsDashboard:
 """
 CLI-based metrics dashboard for monitoring RAG system.
 """

 def __init__(
 self,
 query_engine=None,
 health_monitor=None,
 cache_manager=None,
 query_learning_system=None
 ):
 """
 Initialize metrics dashboard.

 Args:
 query_engine: Query engine instance
 health_monitor: Health monitor instance
 cache_manager: Cache manager instance
 query_learning_system: Query learning system instance
 """
 self.query_engine = query_engine
 self.health_monitor = health_monitor
 self.cache_manager = cache_manager
 self.query_learning_system = query_learning_system

 def display_dashboard(self, refresh_interval: Optional[int] = None):
 """
 Display interactive dashboard.

 Args:
 refresh_interval: Auto-refresh interval in seconds (None = no refresh)
 """
 try:
 while True:
 self._clear_screen()
 self._render_dashboard()

 if refresh_interval is None:
 break

 print(f"\nRefreshing in {refresh_interval}s... (Ctrl+C to stop)")
 time.sleep(refresh_interval)

 except KeyboardInterrupt:
 print("\n\nDashboard stopped.")

 def _render_dashboard(self):
 """Render complete dashboard."""
 print("=" * 80)
 print("RAG SYSTEM METRICS DASHBOARD".center(80))
 print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
 print("=" * 80)
 print()

 # System Health
 self._render_health_section()
 print()

 # Query Performance
 self._render_query_performance()
 print()

 # Cache Statistics
 self._render_cache_stats()
 print()

 # Resource Usage
 self._render_resource_usage()
 print()

 # Recent Activity
 self._render_recent_activity()

 def _render_health_section(self):
 """Render system health section."""
 print("[=] SYSTEM HEALTH")
 print("-" * 80)

 if self.health_monitor:
 try:
 metrics = self.health_monitor.get_metrics()

 # Overall status
 status = metrics.get('status', 'unknown')
 status_symbol = self._get_status_symbol(status)
 print(f"Status: {status_symbol} {status.upper()}")

 # Metrics
 error_rate = metrics.get('error_rate', 0)
 memory_usage = metrics.get('memory_usage_mb', 0)
 query_latency = metrics.get('avg_query_latency_ms', 0)

 print(f"Error Rate: {error_rate:.2%}")
 print(f"Memory Usage: {memory_usage:.1f} MB")
 print(f"Avg Query Latency: {query_latency:.1f} ms")

 # Degradation status
 if 'degradation_level' in metrics:
 deg_level = metrics['degradation_level']
 print(f"Degradation Level: {deg_level}")

 except Exception as e:
 print(f"[!] Health monitoring unavailable: {e}")
 else:
 print("[!] Health monitoring not configured")

 def _render_query_performance(self):
 """Render query performance section."""
 print("[!] QUERY PERFORMANCE")
 print("-" * 80)

 if self.query_learning_system:
 try:
 metrics = self.query_learning_system.get_performance_metrics(days=7)

 if metrics:
 print(f"Total Queries (7d): {metrics.get('total_queries', 0)}")
 print(f"Queries/Day: {metrics.get('queries_per_day', 0):.1f}")
 print(f"Avg Execution Time: {metrics.get('avg_execution_time_ms', 0):.1f} ms")
 print(f"P95 Execution Time: {metrics.get('p95_execution_time_ms', 0):.1f} ms")
 print(f"Avg Results/Query: {metrics.get('avg_results_count', 0):.1f}")

 feedback = metrics.get('avg_feedback_score')
 if feedback is not None:
 print(f"Avg Feedback Score: {feedback:.2f}/1.0")
 print(f"Queries with Feedback: {metrics.get('queries_with_feedback', 0)}")
 else:
 print("No query history available")

 except Exception as e:
 print(f"[!] Performance metrics unavailable: {e}")
 else:
 print("[!] Query learning not configured")

 def _render_cache_stats(self):
 """Render cache statistics section."""
 print("[@] CACHE STATISTICS")
 print("-" * 80)

 if self.cache_manager:
 try:
 stats = self.cache_manager.get_statistics()

 print(f"Total Entries: {stats.get('total_entries', 0)}")
 print(f"Total Size: {stats.get('total_size_mb', 0):.2f} MB")
 print(f"Avg Access Count: {stats.get('average_access_count', 0):.1f}")
 print(f"Avg Entry Age: {stats.get('average_age_seconds', 0):.1f}s")
 print(f"Strategies Enabled: {stats.get('strategies_enabled', 0)}")

 except Exception as e:
 print(f"[!] Cache stats unavailable: {e}")
 else:
 print("[!] Cache manager not configured")

 def _render_resource_usage(self):
 """Render resource usage section."""
 print("[CODE] RESOURCE USAGE")
 print("-" * 80)

 try:
 import psutil
 import os

 process = psutil.Process(os.getpid())

 # Memory
 mem_info = process.memory_info()
 mem_mb = mem_info.rss / 1024 / 1024
 print(f"Process Memory: {mem_mb:.1f} MB")

 # CPU
 cpu_percent = process.cpu_percent(interval=0.1)
 print(f"CPU Usage: {cpu_percent:.1f}%")

 # Threads
 num_threads = process.num_threads()
 print(f"Active Threads: {num_threads}")

 # System-wide
 system_mem = psutil.virtual_memory()
 print(f"System Memory: {system_mem.percent:.1f}% used")

 except ImportError:
 print("[!] psutil not available for resource monitoring")
 except Exception as e:
 print(f"[!] Resource monitoring unavailable: {e}")

 def _render_recent_activity(self):
 """Render recent activity section."""
 print("[NOTE] RECENT ACTIVITY")
 print("-" * 80)

 if self.query_learning_system:
 try:
 # Get recent queries
 popular = self.query_learning_system.get_popular_queries(days=1, top_k=5)

 if popular:
 print("Top Queries (Last 24h):")
 for idx, query_info in enumerate(popular, 1):
 query = query_info['query']
 count = query_info['count']
 # Truncate long queries
 if len(query) > 50:
 query = query[:47] + "..."
 print(f" {idx}. {query} ({count}x)")
 else:
 print("No recent activity")

 except Exception as e:
 print(f"[!] Recent activity unavailable: {e}")
 else:
 print("[!] Query learning not configured")

 def get_summary_report(self) -> Dict[str, Any]:
 """
 Get summary report as dictionary.

 Returns:
 Summary report
 """
 report = {
 'timestamp': datetime.now().isoformat(),
 'health': {},
 'performance': {},
 'cache': {},
 'resources': {}
 }

 # Health
 if self.health_monitor:
 try:
 report['health'] = self.health_monitor.get_metrics()
 except Exception as e:
 report['health']['error'] = str(e)

 # Performance
 if self.query_learning_system:
 try:
 report['performance'] = self.query_learning_system.get_performance_metrics(days=7)
 except Exception as e:
 report['performance']['error'] = str(e)

 # Cache
 if self.cache_manager:
 try:
 report['cache'] = self.cache_manager.get_statistics()
 except Exception as e:
 report['cache']['error'] = str(e)

 # Resources
 try:
 import psutil
 import os

 process = psutil.Process(os.getpid())
 mem_info = process.memory_info()

 report['resources'] = {
 'memory_mb': mem_info.rss / 1024 / 1024,
 'cpu_percent': process.cpu_percent(interval=0.1),
 'num_threads': process.num_threads()
 }
 except Exception as e:
 report['resources']['error'] = str(e)

 return report

 def export_report(self, output_path: Path):
 """
 Export summary report to file.

 Args:
 output_path: Output file path
 """
 import json

 report = self.get_summary_report()
 output_path.write_text(json.dumps(report, indent=2))

 logger.info(f"Report exported to {output_path}")
 print(f"[OK] Report exported to {output_path}")

 def _get_status_symbol(self, status: str) -> str:
 """
 Get symbol for status.

 Args:
 status: Status string

 Returns:
 Status symbol
 """
 symbols = {
 'healthy': '[OK]',
 'degraded': '[!]',
 'unhealthy': '[X]',
 'unknown': ''
 }
 return symbols.get(status.lower(), '')

 def _clear_screen(self):
 """Clear terminal screen."""
 import os
 os.system('clear' if os.name != 'nt' else 'cls')

 def display_compact(self):
 """Display compact single-line dashboard."""
 try:
 # Health status
 status = "OK"
 if self.health_monitor:
 metrics = self.health_monitor.get_metrics()
 status = metrics.get('status', 'unknown').upper()

 # Query stats
 query_count = 0
 avg_latency = 0
 if self.query_learning_system:
 perf = self.query_learning_system.get_performance_metrics(days=1)
 query_count = perf.get('total_queries', 0)
 avg_latency = perf.get('avg_execution_time_ms', 0)

 # Cache stats
 cache_entries = 0
 if self.cache_manager:
 cache_stats = self.cache_manager.get_statistics()
 cache_entries = cache_stats.get('total_entries', 0)

 # Resource usage
 mem_mb = 0
 try:
 import psutil
 import os
 process = psutil.Process(os.getpid())
 mem_mb = process.memory_info().rss / 1024 / 1024
 except Exception:
 pass

 print(f"RAG Status: {status} | Queries (24h): {query_count} | "
 f"Avg Latency: {avg_latency:.0f}ms | Cache: {cache_entries} entries | "
 f"Memory: {mem_mb:.0f}MB")

 except Exception as e:
 print(f"Dashboard error: {e}")


def create_dashboard(
 query_engine=None,
 health_monitor=None,
 cache_manager=None,
 query_learning_system=None
) -> MetricsDashboard:
 """
 Create metrics dashboard instance.

 Args:
 query_engine: Query engine instance
 health_monitor: Health monitor instance
 cache_manager: Cache manager instance
 query_learning_system: Query learning system instance

 Returns:
 MetricsDashboard instance
 """
 return MetricsDashboard(
 query_engine=query_engine,
 health_monitor=health_monitor,
 cache_manager=cache_manager,
 query_learning_system=query_learning_system
 )
