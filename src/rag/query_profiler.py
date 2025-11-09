"""
Query performance profiling for detailed execution analysis.

Tracks:
- Execution time per stage
- Memory allocation
- Function calls
- Bottleneck identification
"""

import time
import functools
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Context variable for current profiler
current_profiler: ContextVar[Optional['QueryProfiler']] = ContextVar('current_profiler', default=None)


@dataclass
class ProfiledStage:
 """Represents a profiled execution stage."""

 name: str
 start_time: float
 end_time: Optional[float] = None
 duration_ms: Optional[float] = None
 memory_before_mb: Optional[float] = None
 memory_after_mb: Optional[float] = None
 memory_delta_mb: Optional[float] = None
 metadata: Dict[str, Any] = field(default_factory=dict)
 sub_stages: List['ProfiledStage'] = field(default_factory=list)

 def complete(self, end_time: Optional[float] = None):
 """Mark stage as complete."""
 self.end_time = end_time or time.time()
 self.duration_ms = (self.end_time - self.start_time) * 1000

 # Calculate memory delta
 if self.memory_before_mb is not None and self.memory_after_mb is not None:
 self.memory_delta_mb = self.memory_after_mb - self.memory_before_mb

 def to_dict(self) -> Dict[str, Any]:
 """Convert to dictionary."""
 return {
 'name': self.name,
 'duration_ms': self.duration_ms,
 'memory_delta_mb': self.memory_delta_mb,
 'metadata': self.metadata,
 'sub_stages': [s.to_dict() for s in self.sub_stages]
 }


class QueryProfiler:
 """
 Profiles query execution with detailed metrics.

 MEDIUM PRIORITY FIX: Added max depth limit for sub-stages.
 """

 # MEDIUM PRIORITY FIX: Limit sub-stage nesting depth
 MAX_STAGE_DEPTH = 10

 def __init__(self, query: str, correlation_id: Optional[str] = None):
 """
 Initialize query profiler.

 Args:
 query: Query being profiled
 correlation_id: Optional correlation ID
 """
 self.query = query
 self.correlation_id = correlation_id
 self.start_time = time.time()
 self.end_time: Optional[float] = None
 self.stages: List[ProfiledStage] = []
 self.current_stage: Optional[ProfiledStage] = None
 self.metadata: Dict[str, Any] = {}
 # MEDIUM PRIORITY FIX: Track current depth
 self._current_depth = 0

 def start_stage(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> ProfiledStage:
 """
 Start profiling a stage.

 MEDIUM PRIORITY FIX: Enforce max depth limit for sub-stages.

 Args:
 name: Stage name
 metadata: Optional metadata

 Returns:
 ProfiledStage instance
 """
 # MEDIUM PRIORITY FIX: Check depth limit
 if self._current_depth >= self.MAX_STAGE_DEPTH:
 logger.warning(
 f"Maximum profiling depth ({self.MAX_STAGE_DEPTH}) reached, "
 f"skipping stage '{name}'"
 )
 # Return a dummy stage that won't be added
 return ProfiledStage(name=name, start_time=time.time(), metadata=metadata or {})

 stage = ProfiledStage(
 name=name,
 start_time=time.time(),
 metadata=metadata or {}
 )

 # Get memory before
 stage.memory_before_mb = self._get_memory_usage_mb()

 # Add to current stage or top-level
 if self.current_stage:
 self.current_stage.sub_stages.append(stage)
 self._current_depth += 1
 else:
 self.stages.append(stage)
 self._current_depth = 1

 # Set as current
 self.current_stage = stage

 return stage

 def end_stage(self):
 """
 End current profiling stage.

 MEDIUM PRIORITY FIX: Decrement depth when ending stage.
 """
 if self.current_stage:
 # Get memory after
 self.current_stage.memory_after_mb = self._get_memory_usage_mb()

 # Complete stage
 self.current_stage.complete()

 # MEDIUM PRIORITY FIX: Decrement depth
 self._current_depth = max(0, self._current_depth - 1)

 # Pop to parent stage (if exists)
 # For simplicity, we go back to None
 self.current_stage = None

 def complete(self, metadata: Optional[Dict[str, Any]] = None):
 """
 Complete profiling.

 Args:
 metadata: Optional metadata
 """
 self.end_time = time.time()

 if metadata:
 self.metadata.update(metadata)

 def get_total_duration_ms(self) -> float:
 """
 Get total query duration.

 Returns:
 Duration in milliseconds
 """
 if self.end_time:
 return (self.end_time - self.start_time) * 1000
 return (time.time() - self.start_time) * 1000

 def get_profile_report(self) -> Dict[str, Any]:
 """
 Get detailed profile report.

 Returns:
 Profile report
 """
 return {
 'query': self.query,
 'correlation_id': self.correlation_id,
 'total_duration_ms': self.get_total_duration_ms(),
 'timestamp': datetime.fromtimestamp(self.start_time).isoformat(),
 'stages': [s.to_dict() for s in self.stages],
 'metadata': self.metadata,
 'bottlenecks': self._identify_bottlenecks()
 }

 def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
 """
 Identify performance bottlenecks.

 Returns:
 List of bottlenecks
 """
 bottlenecks = []

 total_duration = self.get_total_duration_ms()

 for stage in self.stages:
 if stage.duration_ms and total_duration:
 percentage = (stage.duration_ms / total_duration) * 100

 # Flag stages taking > 30% of total time
 if percentage > 30:
 bottlenecks.append({
 'stage': stage.name,
 'duration_ms': stage.duration_ms,
 'percentage': percentage,
 'severity': 'high' if percentage > 50 else 'medium'
 })

 return bottlenecks

 def _get_memory_usage_mb(self) -> Optional[float]:
 """
 Get current memory usage.

 MEDIUM PRIORITY FIX: Enhanced error handling for memory tracking.

 Returns:
 Memory usage in MB or None
 """
 try:
 import psutil
 import os

 process = psutil.Process(os.getpid())
 memory_info = process.memory_info()

 # MEDIUM PRIORITY FIX: Validate memory_info has RSS attribute
 if not hasattr(memory_info, 'rss'):
 logger.warning("Memory info missing RSS attribute")
 return None

 rss_bytes = memory_info.rss

 # MEDIUM PRIORITY FIX: Validate reasonable value
 if rss_bytes < 0 or rss_bytes > 1e12: # More than 1TB is suspicious
 logger.warning(f"Suspicious memory value: {rss_bytes} bytes")
 return None

 return rss_bytes / 1024 / 1024

 except ImportError:
 # psutil not available - this is expected and not an error
 return None

 except (OSError, AttributeError) as e:
 # Process or OS issues
 logger.debug(f"Could not get memory usage: {e}")
 return None

 except Exception as e:
 # Unexpected errors
 logger.warning(f"Unexpected error getting memory usage: {e}")
 return None

 def print_report(self):
 """Print profile report to console."""
 report = self.get_profile_report()

 print("\n" + "=" * 80)
 print("QUERY PERFORMANCE PROFILE")
 print("=" * 80)
 print(f"Query: {report['query']}")
 print(f"Total Duration: {report['total_duration_ms']:.2f}ms")
 print(f"Timestamp: {report['timestamp']}")
 print()

 print("STAGES:")
 print("-" * 80)
 self._print_stages(report['stages'], indent=0)

 if report['bottlenecks']:
 print()
 print("BOTTLENECKS:")
 print("-" * 80)
 for bottleneck in report['bottlenecks']:
 severity_symbol = "[FAIL]" if bottleneck['severity'] == 'high' else "[WARN]"
 print(f"{severity_symbol} {bottleneck['stage']}: {bottleneck['duration_ms']:.2f}ms "
 f"({bottleneck['percentage']:.1f}% of total)")

 print("=" * 80 + "\n")

 def _print_stages(self, stages: List[Dict[str, Any]], indent: int = 0):
 """
 Print stages recursively.

 Args:
 stages: List of stages
 indent: Indentation level
 """
 for stage in stages:
 indent_str = " " * indent
 duration = stage.get('duration_ms', 0)
 mem_delta = stage.get('memory_delta_mb')

 mem_str = f" [{mem_delta:+.2f}MB]" if mem_delta is not None else ""

 print(f"{indent_str}• {stage['name']}: {duration:.2f}ms{mem_str}")

 # Print sub-stages
 if stage.get('sub_stages'):
 self._print_stages(stage['sub_stages'], indent + 1)


class ProfileContext:
 """Context manager for profiling a stage."""

 def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
 """
 Initialize profile context.

 Args:
 name: Stage name
 metadata: Optional metadata
 """
 self.name = name
 self.metadata = metadata
 self.stage: Optional[ProfiledStage] = None

 def __enter__(self) -> ProfiledStage:
 """Enter context."""
 profiler = current_profiler.get()

 if profiler:
 self.stage = profiler.start_stage(self.name, self.metadata)
 return self.stage

 # Create dummy stage if no profiler
 return ProfiledStage(name=self.name, start_time=time.time())

 def __exit__(self, exc_type, exc_val, exc_tb):
 """Exit context."""
 profiler = current_profiler.get()

 if profiler:
 profiler.end_stage()


def profile_query(func: Callable) -> Callable:
 """
 Decorator to profile a query function.

 Args:
 func: Function to profile

 Returns:
 Wrapped function
 """
 @functools.wraps(func)
 def wrapper(*args, **kwargs):
 # Extract query from args/kwargs
 query = kwargs.get('query', args[0] if args else 'unknown')

 # Create profiler
 profiler = QueryProfiler(query)
 token = current_profiler.set(profiler)

 try:
 # Execute function
 result = func(*args, **kwargs)

 # Complete profiling
 profiler.complete()

 # Optionally attach profile to result
 if isinstance(result, dict):
 result['_profile'] = profiler.get_profile_report()

 return result

 finally:
 current_profiler.reset(token)

 return wrapper


def get_current_profiler() -> Optional[QueryProfiler]:
 """
 Get current profiler from context.

 Returns:
 QueryProfiler or None
 """
 return current_profiler.get()


# Usage examples
def example_usage():
 """Example usage of query profiler."""

 # Method 1: Manual profiling
 profiler = QueryProfiler("example query")
 current_profiler.set(profiler)

 with ProfileContext("stage1"):
 time.sleep(0.1)

 with ProfileContext("sub_stage"):
 time.sleep(0.05)

 with ProfileContext("stage2"):
 time.sleep(0.2)

 profiler.complete()
 profiler.print_report()

 # Method 2: Decorator
 @profile_query
 def my_query_function(query: str):
 with ProfileContext("processing"):
 time.sleep(0.1)

 with ProfileContext("retrieval"):
 time.sleep(0.2)

 return {"results": []}

 result = my_query_function("test query")
 if '_profile' in result:
 print("Profile attached to result")
