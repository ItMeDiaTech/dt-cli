"""
Enhanced Query Engine integrating all improvements.
"""

from typing import List, Dict, Any, Optional, Callable
import logging
import time
from pathlib import Path

from .embeddings import EmbeddingEngine
from .lazy_loading import LazyEmbeddingEngine
from .vector_store import VectorStore
from .ingestion import DocumentIngestion
from .incremental_indexing import IncrementalIndexer
from .git_tracker import GitChangeTracker
from .caching import QueryCache
from .hybrid_search import HybridSearchEngine
from .query_expansion import QueryExpander
from .reranking import Reranker
from .progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class EnhancedQueryEngine:
    """
    Enhanced query engine with all improvements integrated.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./.rag_data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        use_lazy_loading: bool = True,
        use_reranking: bool = True
    ):
        """
        Initialize enhanced query engine.

        Args:
            embedding_model: Name of the embedding model
            persist_directory: Directory to persist vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            cache_size: Size of query cache
            cache_ttl: Cache TTL in seconds
            use_lazy_loading: Use lazy model loading
            use_reranking: Use cross-encoder reranking
        """
        # Use lazy loading if enabled
        if use_lazy_loading:
            self.embedding_engine = LazyEmbeddingEngine(model_name=embedding_model)
        else:
            self.embedding_engine = EmbeddingEngine(model_name=embedding_model)

        self.vector_store = VectorStore(persist_directory=persist_directory)
        self.ingestion = DocumentIngestion(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # New features
        self.cache = QueryCache(maxsize=cache_size, ttl=cache_ttl)
        self.incremental_indexer = IncrementalIndexer()
        self.git_tracker = GitChangeTracker()
        self.hybrid_search = HybridSearchEngine()
        self.query_expander = QueryExpander()
        self.reranker = Reranker() if use_reranking else None
        self.progress_tracker = ProgressTracker()

        logger.info("EnhancedQueryEngine initialized with all improvements")

    def validate(self) -> bool:
        """
        MEDIUM PRIORITY FIX: Validate query engine is properly configured.

        Returns:
            True if valid

        Raises:
            RuntimeError: If validation fails
        """
        errors = []

        # Check embedding engine
        try:
            if not hasattr(self.embedding_engine, 'encode'):
                errors.append("Embedding engine missing 'encode' method")
        except Exception as e:
            errors.append(f"Embedding engine validation failed: {e}")

        # Check vector store
        try:
            if not hasattr(self.vector_store, 'query'):
                errors.append("Vector store missing 'query' method")
        except Exception as e:
            errors.append(f"Vector store validation failed: {e}")

        # Check ingestion pipeline
        try:
            if not hasattr(self.ingestion, 'discover_files'):
                errors.append("Ingestion pipeline missing 'discover_files' method")
        except Exception as e:
            errors.append(f"Ingestion pipeline validation failed: {e}")

        if errors:
            error_msg = "Query engine validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.debug("Query engine validation passed")
        return True

    def is_ready(self) -> bool:
        """
        MEDIUM PRIORITY FIX: Check if query engine is ready for queries.

        Returns:
            True if ready (has indexed documents)
        """
        try:
            count = self.vector_store.get_count()
            return count > 0
        except Exception as e:
            logger.error(f"Error checking query engine readiness: {e}")
            return False

    def index_codebase(
        self,
        root_path: str = ".",
        incremental: bool = True,
        use_git: bool = True,
        progress_callback: Optional[Callable] = None
    ):
        """
        Index codebase with incremental support and progress tracking.

        MEDIUM PRIORITY FIX: Add validation before indexing.

        Args:
            root_path: Root directory to index
            incremental: Use incremental indexing
            use_git: Use git for change detection
            progress_callback: Optional progress callback

        Raises:
            RuntimeError: If query engine is not properly configured
        """
        # MEDIUM PRIORITY FIX: Validate query engine before indexing
        try:
            self.validate()
        except RuntimeError as e:
            logger.error(f"Query engine validation failed: {e}")
            raise

        start_time = time.time()

        logger.info(f"Starting indexing: incremental={incremental}, git={use_git}")

        try:
            # Discover all files
            all_files = self.ingestion.discover_files(root_path)

            if not all_files:
                logger.warning("No files found to index")
                return

            # Determine which files to process
            if incremental and use_git and self.git_tracker.is_git_repo:
                # Git-based change detection
                git_changed = self.git_tracker.get_all_changed()

                if git_changed:
                    files_to_process = [
                        f for f in all_files
                        if str(f.relative_to(root_path)) in git_changed
                    ]
                    logger.info(f"Git detected {len(files_to_process)} changed files")
                else:
                    logger.info("No git changes detected, checking mtimes")
                    files_to_process = self.incremental_indexer.discover_changed_files(
                        all_files, root_path
                    )

            elif incremental:
                # Mtime-based incremental indexing
                files_to_process = self.incremental_indexer.discover_changed_files(
                    all_files, root_path
                )
            else:
                # Full index
                files_to_process = all_files
                logger.info(f"Full indexing: {len(files_to_process)} files")

            if not files_to_process:
                logger.info("No changed files to index")
                duration = time.time() - start_time
                self.progress_tracker.mark_complete(0, 0, 0, duration)
                return

            # Process files with progress tracking
            all_chunks = []
            errors = 0

            for i, file_path in enumerate(files_to_process, 1):
                try:
                    # Update progress
                    self.progress_tracker.update_progress(
                        current=i,
                        total=len(files_to_process),
                        current_file=str(file_path),
                        errors=errors,
                        callback=progress_callback
                    )

                    # Process file
                    chunks = self.ingestion.process_file(file_path, root_path)
                    all_chunks.extend(chunks)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    errors += 1

            if all_chunks:
                logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")

                documents = [chunk['text'] for chunk in all_chunks]
                metadatas = [chunk['metadata'] for chunk in all_chunks]
                ids = [chunk['id'] for chunk in all_chunks]

                # Generate embeddings
                embeddings = self.embedding_engine.encode(
                    documents,
                    batch_size=32,
                    show_progress_bar=True
                )

                # Store in vector database
                self.vector_store.add_documents(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings.tolist()
                )

                # Build hybrid search index
                if self.hybrid_search.is_available():
                    logger.info("Building hybrid search index...")
                    self.hybrid_search.build_keyword_index(documents, metadatas, ids)

            # Mark complete
            duration = time.time() - start_time
            self.progress_tracker.mark_complete(
                total_files=len(files_to_process),
                total_chunks=len(all_chunks),
                errors=errors,
                duration_seconds=duration
            )

            # Clear query cache after reindexing
            self.cache.clear()

            logger.info(
                f"Indexing complete in {duration:.2f}s: "
                f"{len(files_to_process)} files, {len(all_chunks)} chunks"
            )

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            self.progress_tracker.mark_failed(str(e))
            raise

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        file_type: Optional[str] = None,
        use_cache: bool = True,
        use_expansion: bool = False,
        use_hybrid: bool = False,
        use_reranking: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system with all enhancements.

        MEDIUM PRIORITY FIX: Add validation and readiness check.

        Args:
            query_text: Query string
            n_results: Number of results
            file_type: Optional file type filter
            use_cache: Use query cache
            use_expansion: Use query expansion
            use_hybrid: Use hybrid search
            use_reranking: Use cross-encoder reranking

        Returns:
            List of results

        Raises:
            RuntimeError: If query engine is not ready
        """
        # MEDIUM PRIORITY FIX: Validate input
        if not query_text or not query_text.strip():
            logger.warning("Query called with empty query text")
            return []

        # MEDIUM PRIORITY FIX: Check if query engine is ready
        if not self.is_ready():
            error_msg = (
                "Query engine is not ready. No documents have been indexed. "
                "Please run index_codebase() first."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Check cache first
        if use_cache:
            cached = self.cache.get(query_text, n_results, file_type)
            if cached:
                return cached

        # Expand query if requested
        if use_expansion:
            expanded_queries = self.query_expander.expand_query(query_text)
            logger.debug(f"Expanded to {len(expanded_queries)} queries")
        else:
            expanded_queries = [query_text]

        # CRITICAL FIX: Perform search with ALL expanded queries, not just first one
        all_results = []
        seen_ids = set()

        for query in expanded_queries:
            # HIGH PRIORITY FIX: Add error recovery for each query
            try:
                # Perform search for this query variation
                if use_hybrid and self.hybrid_search.is_available():
                    query_results = self._hybrid_query(query, n_results, file_type)
                else:
                    query_results = self._semantic_query(query, n_results, file_type)

                # Merge results, deduplicating by ID
                for result in query_results:
                    result_id = result.get('id') or result.get('text', '')[:100]
                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        all_results.append(result)

            except Exception as e:
                logger.warning(f"Query failed for '{query}': {e}")
                # Continue with next query expansion instead of failing completely
                continue

        # Sort merged results by score (descending)
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Take top n_results
        results = all_results[:n_results]

        # Rerank if requested
        # HIGH PRIORITY FIX: Add error recovery for reranking
        if use_reranking and self.reranker and self.reranker.is_available():
            try:
                results = self.reranker.rerank(
                    query_text,
                    results,
                    top_k=n_results
                )
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")
                # Continue with non-reranked results

        # Cache results
        if use_cache:
            self.cache.put(query_text, results, n_results, file_type)

        return results

    def _semantic_query(
        self,
        query_text: str,
        n_results: int,
        file_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform semantic (vector) search."""
        logger.info(f"Semantic query: {query_text}")

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
        return self._format_results(results)

    def _hybrid_query(
        self,
        query_text: str,
        n_results: int,
        file_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform hybrid (semantic + keyword) search."""
        logger.info(f"Hybrid query: {query_text}")

        # Get semantic results
        semantic_results = self._semantic_query(query_text, n_results * 2, file_type)

        # Get keyword results
        keyword_results = self.hybrid_search.keyword_search(query_text, n_results * 2)

        # Combine
        hybrid_results = self.hybrid_search.hybrid_search(
            semantic_results,
            keyword_results,
            semantic_weight=0.7,
            keyword_weight=0.3
        )

        return hybrid_results[:n_results]

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format raw vector store results."""
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

        return formatted_results

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        count = self.vector_store.get_count()

        status = {
            'indexed_chunks': count,
            'embedding_model': self.embedding_engine.model_name,
            'embedding_dimension': self.embedding_engine.get_dimension(),
            'status': 'ready' if count > 0 else 'not_indexed',
            'cache_stats': self.cache.get_stats(),
            'indexing_stats': self.incremental_indexer.get_stats(),
            'git_available': self.git_tracker.is_git_repo,
            'hybrid_search_available': self.hybrid_search.is_available(),
            'reranking_available': self.reranker.is_available() if self.reranker else False,
            'progress': self.progress_tracker.get_status()
        }

        return status

    def reset(self):
        """Reset the RAG system."""
        logger.info("Resetting RAG system")
        self.vector_store.reset()
        self.cache.clear()
        self.incremental_indexer.reset()
        self.progress_tracker.clear()
        logger.info("RAG system reset complete")
