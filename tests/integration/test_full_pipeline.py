"""
Integration tests for full RAG-MAF pipeline.

Tests end-to-end workflows:
- Indexing -> Query
- Query Expansion -> Hybrid Search -> Reranking
- Multi-agent orchestration
- Component integration
- Performance regression
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import time
from typing import Dict, Any

from src.rag.enhanced_query_engine import EnhancedQueryEngine
from src.pipelines.integrated_pipeline import IntegratedQueryPipeline
from src.maf.enhanced_orchestrator import EnhancedMAFOrchestrator
from src.monitoring.action_loop import ActionLoop
from src.async_tasks.task_manager import AsyncTaskManager, TaskStatus
from src.logging_utils.structured_logging import CorrelationContext


class TestFullPipeline:
    """Test complete RAG-MAF pipeline integration."""

    @pytest.fixture
    def temp_codebase(self):
        """Create temporary codebase for testing."""
        temp_dir = tempfile.mkdtemp()

        # Create sample Python files
        (Path(temp_dir) / "main.py").write_text("""
def authenticate_user(username: str, password: str) -> bool:
    '''Authenticate user with username and password.'''
    # Authentication logic here
    return True

class UserManager:
    '''Manages user accounts and authentication.'''

    def __init__(self):
        self.users = {}

    def create_user(self, username: str, email: str):
        '''Create a new user account.'''
        self.users[username] = {'email': email}
""")

        (Path(temp_dir) / "database.py").write_text("""
import sqlite3

class DatabaseConnection:
    '''Handle database connections and queries.'''

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    def execute_query(self, query: str):
        '''Execute SQL query.'''
        cursor = self.conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
""")

        (Path(temp_dir) / "utils.py").write_text("""
def validate_email(email: str) -> bool:
    '''Validate email format.'''
    return '@' in email

def hash_password(password: str) -> str:
    '''Hash password for secure storage.'''
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()
""")

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def query_engine(self, temp_codebase):
        """Initialize query engine with test codebase."""
        config_data = {
            'codebase_path': temp_codebase,
            'db_path': str(Path(temp_codebase) / 'test_chroma.db'),
            'embedding_model': 'all-MiniLM-L6-v2',
            'use_cache': True,
            'lazy_loading': True
        }

        engine = EnhancedQueryEngine(config_data)
        engine.index_codebase(incremental=False)

        return engine

    def test_full_indexing_query_workflow(self, query_engine):
        """Test complete workflow from indexing to query results."""
        # Verify indexing worked
        assert query_engine.collection is not None

        # Query for authentication
        results = query_engine.query("authentication flow")

        assert len(results) > 0
        assert any('authenticate' in r.get('content', '').lower() for r in results)

    def test_integrated_pipeline_stages(self, query_engine):
        """Test full integrated pipeline with all stages."""
        pipeline = IntegratedQueryPipeline(
            query_engine,
            use_expansion=True,
            use_hybrid=True,
            use_reranking=True
        )

        with CorrelationContext() as corr_id:
            results = pipeline.query("user authentication", n_results=3)

        # Verify all stages executed
        assert 'pipeline_stages' in results
        assert 'expansion' in results['pipeline_stages'] or 'search' in results['pipeline_stages']
        assert 'search' in results['pipeline_stages']

        # Verify results
        assert 'results' in results
        assert len(results['results']) > 0

        # Verify metadata
        assert 'metadata' in results
        assert 'total_time_ms' in results['metadata']
        assert results['metadata']['total_time_ms'] > 0

    def test_pipeline_with_degradation(self, query_engine):
        """Test pipeline continues working when components fail."""
        # Create pipeline with some features disabled
        pipeline = IntegratedQueryPipeline(
            query_engine,
            use_expansion=False,  # Disable expansion
            use_hybrid=False,      # Disable hybrid
            use_reranking=False    # Disable reranking
        )

        results = pipeline.query("database query", n_results=3)

        # Should still get results with semantic search only
        assert len(results['results']) > 0
        assert results['pipeline_stages']['search']['type'] == 'semantic'

    def test_maf_orchestration_integration(self, query_engine):
        """Test multi-agent orchestration with real queries."""
        orchestrator = EnhancedMAFOrchestrator(query_engine)

        query = "How does user authentication work?"

        result = orchestrator.run_analysis(
            query,
            use_parallel=True,
            max_context_size=50000
        )

        # Verify orchestration completed
        assert 'final_result' in result
        assert 'agent_outputs' in result
        assert 'execution_stats' in result

        # Should have outputs from multiple agents
        assert len(result['agent_outputs']) > 0

    def test_async_task_execution(self, query_engine):
        """Test async execution of long-running operations."""
        task_manager = AsyncTaskManager()
        task_manager.start()

        try:
            # Submit indexing task
            def index_task():
                query_engine.index_codebase(incremental=True)
                return {"status": "completed"}

            task_id = task_manager.submit_task(index_task)

            # Poll for completion
            max_wait = 30
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status = task_manager.get_task_status(task_id)

                if status['status'] in ['completed', 'failed']:
                    break

                time.sleep(0.5)

            # Verify task completed
            final_status = task_manager.get_task_status(task_id)
            assert final_status['status'] == 'completed'

            result = task_manager.get_task_result(task_id)
            assert result is not None
            assert result['status'] == 'completed'

        finally:
            task_manager.stop()

    def test_correlation_id_tracing(self, query_engine):
        """Test correlation ID propagates through pipeline."""
        pipeline = IntegratedQueryPipeline(query_engine)

        custom_correlation_id = "test-correlation-123"

        results = pipeline.query(
            "database operations",
            n_results=3,
            correlation_id=custom_correlation_id
        )

        # Verify correlation ID is in results
        assert results['correlation_id'] == custom_correlation_id

    def test_performance_benchmarks(self, query_engine):
        """Test performance meets acceptable thresholds."""
        pipeline = IntegratedQueryPipeline(query_engine)

        # Test query performance
        start_time = time.time()
        results = pipeline.query("authentication", n_results=5)
        query_time = time.time() - start_time

        # Should complete in under 2 seconds
        assert query_time < 2.0

        # Verify metadata timing
        assert results['metadata']['total_time_ms'] < 2000

    def test_incremental_indexing(self, temp_codebase, query_engine):
        """Test incremental indexing only processes changed files."""
        # Initial index
        stats1 = query_engine.index_codebase(incremental=True)

        # Modify one file
        time.sleep(0.1)  # Ensure different mtime
        (Path(temp_codebase) / "main.py").write_text("""
def authenticate_user(username: str, password: str) -> bool:
    '''Authenticate user with username and password.'''
    # Updated authentication logic
    return verify_credentials(username, password)
""")

        # Re-index incrementally
        stats2 = query_engine.index_codebase(incremental=True)

        # Should have processed fewer files
        if stats1 and stats2:
            assert stats2.get('files_processed', 0) < stats1.get('files_processed', 1)

    def test_cache_effectiveness(self, query_engine):
        """Test query caching improves performance."""
        pipeline = IntegratedQueryPipeline(query_engine)

        query = "user authentication flow"

        # First query (uncached)
        start1 = time.time()
        results1 = pipeline.query(query, n_results=3)
        time1 = time.time() - start1

        # Second query (should be cached)
        start2 = time.time()
        results2 = pipeline.query(query, n_results=3)
        time2 = time.time() - start2

        # Cached query should be faster
        assert time2 < time1

        # Results should be identical
        assert len(results1['results']) == len(results2['results'])

    def test_monitoring_action_loop(self, query_engine):
        """Test monitoring action loop with auto-remediation."""
        from src.monitoring.health_monitor import HealthMonitor

        monitor = HealthMonitor()
        orchestrator = EnhancedMAFOrchestrator(query_engine)

        action_loop = ActionLoop(
            monitor,
            query_engine,
            orchestrator,
            check_interval=1
        )

        # Start monitoring
        action_loop.start()

        try:
            # Let it run for a few cycles
            time.sleep(3)

            # Verify it's running
            assert action_loop.running

            # Check metrics were collected
            metrics = monitor.get_metrics()
            assert 'timestamp' in metrics

        finally:
            action_loop.stop()

    def test_error_recovery(self, query_engine):
        """Test system recovers from component failures."""
        pipeline = IntegratedQueryPipeline(query_engine)

        # Simulate failure by passing invalid query
        try:
            results = pipeline.query("", n_results=3)

            # Should handle gracefully
            assert 'results' in results

        except Exception as e:
            # If it raises, should be a validation error, not a crash
            assert "query" in str(e).lower()

    def test_multi_query_batch(self, query_engine):
        """Test handling multiple queries efficiently."""
        pipeline = IntegratedQueryPipeline(query_engine)

        queries = [
            "authentication",
            "database connection",
            "email validation",
            "password hashing"
        ]

        results_list = []
        total_start = time.time()

        for query in queries:
            results = pipeline.query(query, n_results=2)
            results_list.append(results)

        total_time = time.time() - total_start

        # All queries should complete
        assert len(results_list) == len(queries)

        # Should complete in reasonable time
        assert total_time < 5.0

        # Each should have results
        for results in results_list:
            assert len(results['results']) > 0


class TestComponentIntegration:
    """Test integration between individual components."""

    def test_entity_extraction_to_knowledge_graph(self, tmp_path):
        """Test entity extraction feeds into knowledge graph."""
        from src.entity_extraction.extractor import EntityExtractor

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class UserService:
    def authenticate(self, username, password):
        pass

def hash_password(password):
    return password
""")

        extractor = EntityExtractor()
        entities = extractor.extract_from_file(test_file)

        # Should extract class and function
        assert len(entities) >= 2

        entity_types = [e.entity_type for e in entities]
        assert 'class' in entity_types
        assert 'function' in entity_types

    def test_structured_logging_integration(self, query_engine):
        """Test structured logging captures pipeline events."""
        import logging
        from io import StringIO

        # Capture logs
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)

        logger = logging.getLogger('src.pipelines.integrated_pipeline')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            pipeline = IntegratedQueryPipeline(query_engine)

            with CorrelationContext("test-123") as corr_id:
                results = pipeline.query("test query", n_results=2)

            # Check logs were generated
            log_output = log_stream.getvalue()
            assert len(log_output) > 0

        finally:
            logger.removeHandler(handler)


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions."""

    def test_query_latency_p95(self, query_engine):
        """Test 95th percentile query latency."""
        pipeline = IntegratedQueryPipeline(query_engine)

        latencies = []

        for i in range(20):
            start = time.time()
            pipeline.query(f"test query {i}", n_results=3)
            latencies.append(time.time() - start)

        # Calculate p95
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]

        # P95 should be under 1 second
        assert p95_latency < 1.0

    def test_memory_usage_stable(self, query_engine):
        """Test memory usage remains stable over multiple queries."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        pipeline = IntegratedQueryPipeline(query_engine)

        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple queries
        for i in range(50):
            pipeline.query(f"query {i}", n_results=3)

        # Check memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory should not grow significantly (allow 50MB growth)
        memory_growth = final_memory - baseline_memory
        assert memory_growth < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
