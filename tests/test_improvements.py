"""
Tests for all new improvements.
"""

import pytest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_config_validation():
    """Test configuration validation with Pydantic."""
    from config import PluginConfig, RAGConfig

    # Test valid config
    config = PluginConfig()
    assert config.rag.chunk_size == 1000

    # Test validation
    warnings = config.validate_config()
    assert isinstance(warnings, dict)


def test_query_cache():
    """Test query result caching."""
    from rag.caching import QueryCache

    cache = QueryCache(maxsize=10, ttl=60)

    # Test cache miss
    result = cache.get("test query")
    assert result is None

    # Test cache hit
    cache.put("test query", ["result1", "result2"])
    result = cache.get("test query")
    assert result == ["result1", "result2"]

    # Test stats
    stats = cache.get_stats()
    assert stats['hits'] == 1
    assert stats['misses'] == 1


def test_incremental_indexing():
    """Test incremental indexing."""
    from rag.incremental_indexing import IncrementalIndexer
    from pathlib import Path

    temp_dir = tempfile.mkdtemp()

    try:
        indexer = IncrementalIndexer(f"{temp_dir}/manifest.json")

        # Create test files
        test_file = Path(temp_dir) / "test.py"
        test_file.write_text("print('hello')")

        all_files = [test_file]

        # First run - should find changed
        changed = indexer.discover_changed_files(all_files, temp_dir)
        assert len(changed) == 1

        # Second run - no changes
        changed = indexer.discover_changed_files(all_files, temp_dir)
        assert len(changed) == 0

        # Modify file
        test_file.write_text("print('hello world')")

        # Should detect change
        changed = indexer.discover_changed_files(all_files, temp_dir)
        assert len(changed) == 1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_bounded_context_manager():
    """Test bounded context manager."""
    from maf.bounded_context import BoundedContextManager

    manager = BoundedContextManager(max_contexts=5)

    # Create contexts
    for i in range(10):
        manager.create_context(
            context_id=f"ctx{i}",
            query=f"query{i}",
            task_type="test"
        )

    # Should only have 5 (max)
    assert len(manager.get_active_contexts()) == 5

    # Oldest should be evicted
    assert "ctx0" not in manager.get_active_contexts()
    assert "ctx9" in manager.get_active_contexts()


def test_lazy_embedding_engine():
    """Test lazy loading of embedding model."""
    from rag.lazy_loading import LazyEmbeddingEngine

    engine = LazyEmbeddingEngine(idle_timeout=1)

    # Model not loaded initially
    assert not engine.is_loaded()

    # Encode should load model
    embeddings = engine.encode(["test"])
    assert engine.is_loaded()
    assert embeddings.shape[0] == 1


def test_query_expansion():
    """Test query expansion."""
    from rag.query_expansion import QueryExpander

    expander = QueryExpander()

    # Test expansion
    expansions = expander.expand_query("how to create a function")
    assert len(expansions) > 1
    assert any("method" in exp or "procedure" in exp for exp in expansions)

    # Test technical term extraction
    terms = expander.extract_technical_terms("MyClass.my_method()")
    assert len(terms) > 0


def test_progress_tracker():
    """Test progress tracking."""
    from rag.progress_tracker import ProgressTracker
    import tempfile

    temp_file = tempfile.mktemp(suffix=".json")

    try:
        tracker = ProgressTracker(temp_file)

        # Update progress
        tracker.update_progress(5, 10, "test.py", 0)

        status = tracker.get_status()
        assert status['status'] == 'indexing'
        assert status['current'] == 5
        assert status['total'] == 10

        # Mark complete
        tracker.mark_complete(10, 100, 0, 5.0)

        status = tracker.get_status()
        assert status['status'] == 'complete'
        assert tracker.is_complete()

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_health_monitor():
    """Test health monitoring."""
    from monitoring import HealthMonitor

    monitor = HealthMonitor()

    # Record requests
    monitor.record_request()
    monitor.record_request()

    # Record error
    monitor.record_error(Exception("Test error"))

    status = monitor.get_health_status()
    assert status['request_count'] == 2
    assert status['error_count'] == 1
    assert status['status'] in ['healthy', 'degraded', 'unhealthy']


def test_git_tracker():
    """Test Git change tracking."""
    from rag.git_tracker import GitChangeTracker

    tracker = GitChangeTracker()

    # Check if we're in a git repo
    if tracker.is_git_repo:
        # Try to get changed files
        changed = tracker.get_all_changed()
        assert isinstance(changed, set)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
