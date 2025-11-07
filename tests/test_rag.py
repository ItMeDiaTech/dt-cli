"""
Tests for RAG system components.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag import EmbeddingEngine, VectorStore, DocumentIngestion, QueryEngine


def test_embedding_engine():
    """Test embedding engine initialization and encoding."""
    engine = EmbeddingEngine()
    engine.load_model()

    # Test single text encoding
    text = "This is a test sentence."
    embedding = engine.encode(text)

    assert embedding is not None
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1

    # Test dimension
    dim = engine.get_dimension()
    assert dim == 384  # all-MiniLM-L6-v2 dimension


def test_document_ingestion():
    """Test document ingestion."""
    ingestion = DocumentIngestion(chunk_size=100, chunk_overlap=20)

    # Test text chunking
    text = "This is a test. " * 50  # Create longer text
    metadata = {"file_path": "test.py", "file_type": ".py"}

    chunks = ingestion.chunk_text(text, metadata)

    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)
    assert all('metadata' in chunk for chunk in chunks)


def test_vector_store():
    """Test vector store operations."""
    import tempfile
    import shutil

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        store = VectorStore(persist_directory=temp_dir)
        store.initialize()

        # Test adding documents
        documents = ["Test document 1", "Test document 2"]
        metadatas = [{"id": 1}, {"id": 2}]
        ids = ["doc1", "doc2"]

        store.add_documents(documents, metadatas, ids)

        # Test count
        count = store.get_count()
        assert count == 2

    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
