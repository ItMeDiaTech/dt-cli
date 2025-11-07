"""
Query Engine for RAG system.
"""

from typing import List, Dict, Any, Optional
import logging
from .embeddings import EmbeddingEngine
from .vector_store import VectorStore
from .ingestion import DocumentIngestion
from .caching import QueryCache

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query engine that combines embeddings, vector store, and ingestion.

    MEDIUM PRIORITY FIX: Integrate configuration management and caching.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        config_manager: Optional[Any] = None,
        enable_cache: bool = True
    ):
        """
        Initialize the query engine.

        MEDIUM PRIORITY FIX: Use ConfigManager instead of hardcoded defaults.

        Args:
            embedding_model: Name of the embedding model (overrides config)
            persist_directory: Directory to persist vector store (overrides config)
            chunk_size: Size of text chunks (overrides config)
            chunk_overlap: Overlap between chunks (overrides config)
            config_manager: Optional ConfigManager instance
            enable_cache: Enable query result caching
        """
        # MEDIUM PRIORITY FIX: Use config manager if provided
        if config_manager:
            from ..config.config_manager import ConfigManager
            self.config = config_manager
        else:
            # Try to import global config manager
            try:
                from ..config.config_manager import config_manager as global_config
                self.config = global_config
                logger.info("Using global configuration manager")
            except ImportError:
                self.config = None
                logger.warning("ConfigManager not available, using default values")

        # MEDIUM PRIORITY FIX: Remove hardcoded defaults, use config
        if self.config:
            embedding_model = embedding_model or self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            persist_directory = persist_directory or self.config.get('db_path', './.rag_data')
            chunk_size = chunk_size or self.config.get('chunk_size', 1000)
            chunk_overlap = chunk_overlap or self.config.get('chunk_overlap', 200)
        else:
            # Fallback to defaults if no config
            embedding_model = embedding_model or 'all-MiniLM-L6-v2'
            persist_directory = persist_directory or './.rag_data'
            chunk_size = chunk_size or 1000
            chunk_overlap = chunk_overlap or 200

        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.vector_store = VectorStore(persist_directory=persist_directory)
        self.ingestion = DocumentIngestion(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # MEDIUM PRIORITY FIX: Integrate query result caching
        self.cache: Optional[QueryCache] = None
        if enable_cache:
            # Get cache settings from config
            if self.config:
                cache_size = self.config.get('cache_size', 1000)
                cache_ttl = self.config.get('cache_ttl_seconds', 3600)
            else:
                cache_size = 1000
                cache_ttl = 3600

            self.cache = QueryCache(maxsize=cache_size, ttl=cache_ttl)
            logger.info(f"Query caching enabled (size={cache_size}, ttl={cache_ttl}s)")

        logger.info("QueryEngine initialized with configuration")

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
        n_results: Optional[int] = None,
        file_type: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system.

        MEDIUM PRIORITY FIX: Integrate query result caching.

        Args:
            query_text: Query string
            n_results: Number of results to return (uses config default if None)
            file_type: Optional file type filter (e.g., '.py')
            use_cache: Use query cache if available

        Returns:
            List of results with text, metadata, and scores
        """
        # MEDIUM PRIORITY FIX: Use config for n_results default
        if n_results is None:
            n_results = self.config.get('n_results', 5) if self.config else 5

        logger.info(f"Querying: {query_text}")

        # MEDIUM PRIORITY FIX: Check cache first
        if use_cache and self.cache:
            cached_results = self.cache.get(query_text, n_results, file_type)
            if cached_results is not None:
                logger.info(f"Returning {len(cached_results)} cached results")
                return cached_results

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

        # MEDIUM PRIORITY FIX: Cache results
        if use_cache and self.cache:
            self.cache.put(query_text, formatted_results, n_results, file_type)

        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the RAG system.

        MEDIUM PRIORITY FIX: Include cache statistics.

        Returns:
            Status information
        """
        count = self.vector_store.get_count()

        status = {
            'indexed_chunks': count,
            'embedding_model': self.embedding_engine.model_name,
            'embedding_dimension': self.embedding_engine.get_dimension(),
            'status': 'ready' if count > 0 else 'not_indexed'
        }

        # MEDIUM PRIORITY FIX: Add cache stats if caching enabled
        if self.cache:
            status['cache_stats'] = self.cache.get_stats()

        return status

    def reset(self):
        """
        Reset the RAG system (delete all indexed data).

        MEDIUM PRIORITY FIX: Clear cache on reset.
        """
        logger.info("Resetting RAG system")
        self.vector_store.reset()

        # MEDIUM PRIORITY FIX: Clear cache when resetting
        if self.cache:
            self.cache.clear()

        logger.info("RAG system reset complete")
