"""
Query Engine for RAG system.
"""

from typing import List, Dict, Any, Optional
import logging
from .embeddings import EmbeddingEngine
from .vector_store import VectorStore
from .ingestion import DocumentIngestion

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query engine that combines embeddings, vector store, and ingestion.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./.rag_data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the query engine.

        Args:
            embedding_model: Name of the embedding model
            persist_directory: Directory to persist vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.vector_store = VectorStore(persist_directory=persist_directory)
        self.ingestion = DocumentIngestion(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        logger.info("QueryEngine initialized")

    def index_codebase(self, root_path: str = "."):
        """
        Index the entire codebase.

        Args:
            root_path: Root directory of the codebase
        """
        logger.info(f"Indexing codebase at: {root_path}")

        # Ingest documents
        chunks = self.ingestion.ingest_directory(root_path)

        if not chunks:
            logger.warning("No documents found to index")
            return

        # Extract texts and metadata
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_engine.encode(
            documents,
            batch_size=32,
            show_progress_bar=True
        )

        # Store in vector database
        logger.info("Storing in vector database...")
        self.vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )

        logger.info(f"Indexing complete! Indexed {len(chunks)} chunks")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        file_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system.

        Args:
            query_text: Query string
            n_results: Number of results to return
            file_type: Optional file type filter (e.g., '.py')

        Returns:
            List of results with text, metadata, and scores
        """
        logger.info(f"Querying: {query_text}")

        # Generate query embedding
        query_embedding = self.embedding_engine.encode([query_text])

        # Build filters
        where = None
        if file_type:
            where = {"file_type": file_type}

        # Query vector store
        results = self.vector_store.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where
        )

        # Format results
        formatted_results = []

        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0,
                    'id': results['ids'][0][i] if results['ids'] else None
                }
                formatted_results.append(result)

        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the RAG system.

        Returns:
            Status information
        """
        count = self.vector_store.get_count()

        return {
            'indexed_chunks': count,
            'embedding_model': self.embedding_engine.model_name,
            'embedding_dimension': self.embedding_engine.get_dimension(),
            'status': 'ready' if count > 0 else 'not_indexed'
        }

    def reset(self):
        """Reset the RAG system (delete all indexed data)."""
        logger.info("Resetting RAG system")
        self.vector_store.reset()
        logger.info("RAG system reset complete")
