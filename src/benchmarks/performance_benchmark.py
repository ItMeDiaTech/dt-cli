"""
Performance benchmarking tools for RAG system.

Measures:
- Query latency (avg, P50, P95, P99)
- Indexing performance
- Cache hit rates
- Memory usage
- Throughput
"""

import time
import statistics
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Represents benchmark results."""

    name: str
    total_runs: int
    successful_runs: int
    failed_runs: int

    # Latency metrics (milliseconds)
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Throughput
    queries_per_second: float

    # Memory
    avg_memory_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None

    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'failed_runs': self.failed_runs,
            'latency': {
                'avg_ms': self.avg_latency_ms,
                'min_ms': self.min_latency_ms,
                'max_ms': self.max_latency_ms,
                'p50_ms': self.p50_latency_ms,
                'p95_ms': self.p95_latency_ms,
                'p99_ms': self.p99_latency_ms,
            },
            'throughput': {
                'qps': self.queries_per_second
            },
            'memory': {
                'avg_mb': self.avg_memory_mb,
                'peak_mb': self.peak_memory_mb
            },
            'timestamps': {
                'started_at': self.started_at,
                'completed_at': self.completed_at
            },
            'metadata': self.metadata
        }


class PerformanceBenchmark:
    """
    Performance benchmarking suite.
    """

    def __init__(self, query_engine, max_results: int = 100):
        """
        Initialize benchmark.

        HIGH PRIORITY FIX: Add max_results limit to prevent memory leak.

        Args:
            query_engine: Query engine instance
            max_results: Maximum number of results to keep (prevents memory leak)
        """
        self.query_engine = query_engine
        self.results: List[BenchmarkResult] = []
        self.max_results = max_results  # HIGH PRIORITY FIX: Limit result collection

    def clear_results(self):
        """
        HIGH PRIORITY FIX: Clear old results to free memory.

        Call this periodically in long-running benchmark sessions.
        """
        self.results.clear()
        logger.info("Cleared benchmark results")

    def benchmark_query_latency(
        self,
        queries: List[str],
        n_results: int = 5,
        warmup_runs: int = 2
    ) -> BenchmarkResult:
        """
        Benchmark query latency.

        Args:
            queries: List of queries to test
            n_results: Number of results per query
            warmup_runs: Number of warmup runs

        Returns:
            BenchmarkResult
        """
        logger.info(f"Benchmarking query latency ({len(queries)} queries)...")

        # Warmup
        for i in range(min(warmup_runs, len(queries))):
            try:
                self.query_engine.query(queries[i], n_results=n_results)
            except Exception:
                pass

        # Actual benchmark
        latencies = []
        memory_samples = []
        successful = 0
        failed = 0

        start_time = time.time()

        for query in queries:
            try:
                # Measure memory before
                mem_before = self._get_memory_mb()

                # Execute query
                query_start = time.time()
                _ = self.query_engine.query(query, n_results=n_results)
                query_end = time.time()

                # Measure memory after
                mem_after = self._get_memory_mb()

                # Record metrics
                latency_ms = (query_end - query_start) * 1000
                latencies.append(latency_ms)

                if mem_before and mem_after:
                    memory_samples.append(mem_after)

                successful += 1

            except Exception as e:
                logger.debug(f"Query failed: {e}")
                failed += 1

        total_time = time.time() - start_time

        # HIGH PRIORITY FIX: Handle empty latencies list edge case
        # Calculate statistics
        if latencies:
            latencies.sort()

            result = BenchmarkResult(
                name="Query Latency",
                total_runs=len(queries),
                successful_runs=successful,
                failed_runs=failed,
                avg_latency_ms=statistics.mean(latencies),
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                p50_latency_ms=self._percentile(latencies, 0.50),
                p95_latency_ms=self._percentile(latencies, 0.95),
                p99_latency_ms=self._percentile(latencies, 0.99),
                queries_per_second=successful / total_time if total_time > 0 else 0,
                avg_memory_mb=statistics.mean(memory_samples) if memory_samples else None,
                peak_memory_mb=max(memory_samples) if memory_samples else None,
                completed_at=datetime.now().isoformat()
            )

            # HIGH PRIORITY FIX: Prevent unbounded memory growth
            self.results.append(result)
            if len(self.results) > self.max_results:
                # Remove oldest result
                self.results.pop(0)
                logger.debug(f"Removed oldest result (limit: {self.max_results})")

            logger.info(f"Query latency benchmark complete: {result.avg_latency_ms:.2f}ms avg")

            return result
        else:
            # HIGH PRIORITY FIX: Return meaningful result even when all queries fail
            logger.warning("No successful queries in benchmark")
            result = BenchmarkResult(
                name="Query Latency",
                total_runs=len(queries),
                successful_runs=0,
                failed_runs=failed,
                avg_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                queries_per_second=0.0,
                completed_at=datetime.now().isoformat()
            )
            self.results.append(result)
            return result

    def benchmark_indexing(
        self,
        incremental: bool = True,
        use_git: bool = True
    ) -> BenchmarkResult:
        """
        Benchmark indexing performance.

        Args:
            incremental: Use incremental indexing
            use_git: Use git diff

        Returns:
            BenchmarkResult
        """
        logger.info("Benchmarking indexing performance...")

        runs = []
        memory_samples = []

        # Run indexing 3 times
        for i in range(3):
            try:
                mem_before = self._get_memory_mb()

                start = time.time()
                stats = self.query_engine.index_codebase(
                    incremental=incremental,
                    use_git=use_git
                )
                end = time.time()

                mem_after = self._get_memory_mb()

                duration_ms = (end - start) * 1000
                runs.append(duration_ms)

                if mem_before and mem_after:
                    memory_samples.append(mem_after)

                logger.info(f"Run {i+1}: {duration_ms:.2f}ms")

            except Exception as e:
                logger.error(f"Indexing failed: {e}")

        # HIGH PRIORITY FIX: Handle empty runs list edge case
        if runs:
            result = BenchmarkResult(
                name=f"Indexing ({'incremental' if incremental else 'full'})",
                total_runs=len(runs),
                successful_runs=len(runs),
                failed_runs=0,
                avg_latency_ms=statistics.mean(runs),
                min_latency_ms=min(runs),
                max_latency_ms=max(runs),
                p50_latency_ms=self._percentile(runs, 0.50),
                p95_latency_ms=self._percentile(runs, 0.95),
                p99_latency_ms=self._percentile(runs, 0.99),
                queries_per_second=0,  # N/A for indexing
                avg_memory_mb=statistics.mean(memory_samples) if memory_samples else None,
                peak_memory_mb=max(memory_samples) if memory_samples else None,
                completed_at=datetime.now().isoformat()
            )

            # HIGH PRIORITY FIX: Prevent unbounded memory growth
            self.results.append(result)
            if len(self.results) > self.max_results:
                # Remove oldest result
                self.results.pop(0)
                logger.debug(f"Removed oldest result (limit: {self.max_results})")

            return result
        else:
            # HIGH PRIORITY FIX: Return meaningful result even when all runs fail
            logger.warning("No successful indexing runs in benchmark")
            result = BenchmarkResult(
                name=f"Indexing ({'incremental' if incremental else 'full'})",
                total_runs=3,
                successful_runs=0,
                failed_runs=3,
                avg_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                queries_per_second=0.0,
                completed_at=datetime.now().isoformat()
            )
            self.results.append(result)
            return result

    def benchmark_cache_effectiveness(
        self,
        queries: List[str],
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark cache effectiveness.

        Args:
            queries: List of queries
            n_results: Number of results

        Returns:
            Cache statistics
        """
        logger.info("Benchmarking cache effectiveness...")

        # First pass (populate cache)
        first_pass_times = []
        for query in queries:
            try:
                start = time.time()
                _ = self.query_engine.query(query, n_results=n_results)
                end = time.time()
                first_pass_times.append((end - start) * 1000)
            except Exception:
                pass

        # Second pass (from cache)
        second_pass_times = []
        for query in queries:
            try:
                start = time.time()
                _ = self.query_engine.query(query, n_results=n_results)
                end = time.time()
                second_pass_times.append((end - start) * 1000)
            except Exception:
                pass

        if first_pass_times and second_pass_times:
            avg_first = statistics.mean(first_pass_times)
            avg_second = statistics.mean(second_pass_times)

            speedup = avg_first / avg_second if avg_second > 0 else 0

            stats = {
                'avg_uncached_ms': avg_first,
                'avg_cached_ms': avg_second,
                'speedup_factor': speedup,
                'cache_benefit_percent': ((avg_first - avg_second) / avg_first * 100) if avg_first > 0 else 0
            }

            logger.info(f"Cache speedup: {speedup:.2f}x")

            return stats

        return {}

    def run_full_benchmark_suite(
        self,
        test_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.

        Args:
            test_queries: Optional test queries

        Returns:
            Full benchmark results
        """
        logger.info("Running full benchmark suite...")

        # Default test queries
        if not test_queries:
            test_queries = [
                "authentication flow",
                "database connection",
                "error handling",
                "API endpoints",
                "configuration settings",
                "user management",
                "data validation",
                "security checks",
                "performance optimization",
                "testing framework"
            ]

        results = {}

        # 1. Query latency
        latency_result = self.benchmark_query_latency(test_queries)
        if latency_result:
            results['query_latency'] = latency_result.to_dict()

        # 2. Indexing performance
        indexing_result = self.benchmark_indexing(incremental=True)
        if indexing_result:
            results['indexing_incremental'] = indexing_result.to_dict()

        # 3. Cache effectiveness
        cache_stats = self.benchmark_cache_effectiveness(test_queries[:5])
        if cache_stats:
            results['cache_effectiveness'] = cache_stats

        # 4. Memory usage
        memory_stats = self._benchmark_memory_usage()
        if memory_stats:
            results['memory_usage'] = memory_stats

        results['summary'] = self._generate_summary(results)

        return results

    def _benchmark_memory_usage(self) -> Optional[Dict[str, Any]]:
        """
        Benchmark memory usage.

        Returns:
            Memory statistics
        """
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # Get current memory
            mem_info = process.memory_info()

            return {
                'rss_mb': mem_info.rss / 1024 / 1024,
                'vms_mb': mem_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }

        except ImportError:
            logger.warning("psutil not available for memory benchmarking")
            return None
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
            return None

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate benchmark summary.

        Args:
            results: Benchmark results

        Returns:
            Summary dict
        """
        summary = {}

        # Query latency summary
        if 'query_latency' in results:
            latency = results['query_latency']['latency']
            summary['query_performance'] = f"Avg: {latency['avg_ms']:.2f}ms, P95: {latency['p95_ms']:.2f}ms"

        # Indexing summary
        if 'indexing_incremental' in results:
            indexing = results['indexing_incremental']['latency']
            summary['indexing_performance'] = f"Avg: {indexing['avg_ms']:.2f}ms"

        # Cache summary
        if 'cache_effectiveness' in results:
            cache = results['cache_effectiveness']
            summary['cache_speedup'] = f"{cache['speedup_factor']:.2f}x"

        # Memory summary
        if 'memory_usage' in results:
            memory = results['memory_usage']
            summary['memory_usage'] = f"{memory['rss_mb']:.1f}MB"

        return summary

    def export_results(self, output_path: Path):
        """
        Export benchmark results to file.

        Args:
            output_path: Output file path
        """
        export_data = {
            'benchmark_date': datetime.now().isoformat(),
            'results': [r.to_dict() for r in self.results]
        }

        output_path.write_text(json.dumps(export_data, indent=2))
        logger.info(f"Benchmark results exported to {output_path}")

    def print_results(self, result: BenchmarkResult):
        """
        Print benchmark results.

        Args:
            result: BenchmarkResult to print
        """
        print("\n" + "=" * 70)
        print(f"BENCHMARK: {result.name}")
        print("=" * 70)
        print(f"Runs: {result.successful_runs}/{result.total_runs} successful")
        print()

        print("LATENCY:")
        print(f"  Avg:  {result.avg_latency_ms:.2f}ms")
        print(f"  Min:  {result.min_latency_ms:.2f}ms")
        print(f"  Max:  {result.max_latency_ms:.2f}ms")
        print(f"  P50:  {result.p50_latency_ms:.2f}ms")
        print(f"  P95:  {result.p95_latency_ms:.2f}ms")
        print(f"  P99:  {result.p99_latency_ms:.2f}ms")
        print()

        if result.queries_per_second > 0:
            print(f"THROUGHPUT: {result.queries_per_second:.2f} queries/second")
            print()

        if result.avg_memory_mb:
            print(f"MEMORY:")
            print(f"  Avg:  {result.avg_memory_mb:.1f}MB")
            print(f"  Peak: {result.peak_memory_mb:.1f}MB")
            print()

        print("=" * 70 + "\n")

    def _percentile(self, values: List[float], percentile: float) -> float:
        """
        Calculate percentile using linear interpolation.

        HIGH PRIORITY FIX: Use proper percentile calculation instead of
        simple index-based approach. Matches numpy.percentile behavior.

        Args:
            values: List of values (will be sorted)
            percentile: Percentile (0-1)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0

        if len(values) == 1:
            return values[0]

        sorted_values = sorted(values)

        # HIGH PRIORITY FIX: Use linear interpolation for accurate percentiles
        # Formula: position = (n - 1) * percentile, then interpolate
        n = len(sorted_values)
        position = (n - 1) * percentile

        # Get the two surrounding indices
        lower_idx = int(position)
        upper_idx = min(lower_idx + 1, n - 1)

        # Calculate interpolation weight
        weight = position - lower_idx

        # Interpolate between the two values
        result = sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight

        return result

    def _get_memory_mb(self) -> Optional[float]:
        """
        Get current memory usage.

        Returns:
            Memory in MB or None
        """
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024

        except ImportError:
            return None
        except Exception:
            return None


def run_benchmark(query_engine, output_path: Optional[Path] = None):
    """
    Run benchmark and optionally save results.

    Args:
        query_engine: Query engine instance
        output_path: Optional output path

    Returns:
        Benchmark results
    """
    benchmark = PerformanceBenchmark(query_engine)
    results = benchmark.run_full_benchmark_suite()

    if output_path:
        output_path.write_text(json.dumps(results, indent=2))

    return results
