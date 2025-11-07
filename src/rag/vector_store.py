"""
Vector Store using ChromaDB for efficient similarity search.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import logging
import os

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for efficient similarity search.
    """

    def __init__(
        self,
        persist_directory: str = "./.rag_data",
        collection_name: str = "codebase"
    ):
        """
        Initialize the vector store.

        MEDIUM PRIORITY FIX: Add path traversal validation.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use

        Raises:
            ValueError: If persist_directory is invalid or attempts path traversal
        """
        # MEDIUM PRIORITY FIX: Validate persist_directory to prevent path traversal
        from pathlib import Path

        persist_path = Path(persist_directory).resolve()

        # Check for suspicious patterns
        if ".." in persist_directory:
            logger.error(f"Path traversal detected in persist_directory: {persist_directory}")
            raise ValueError(
                f"Invalid persist_directory: path traversal detected. "
                f"Directory must not contain '..' components."
            )

        # Ensure path is within current working directory or explicitly absolute
        # If relative path, it should be safe
        if not persist_path.is_absolute():
            # Relative paths are resolved relative to cwd, which is safe
            logger.debug(f"Using relative persist_directory: {persist_directory}")
        else:
            # For absolute paths, verify they don't point to sensitive system directories
            sensitive_dirs = ['/etc', '/sys', '/proc', '/dev', '/root', '/boot']
            for sensitive in sensitive_dirs:
                try:
                    if persist_path.is_relative_to(sensitive):
                        logger.error(f"Attempt to use sensitive directory: {persist_path}")
                        raise ValueError(
                            f"Invalid persist_directory: cannot use sensitive system directory {sensitive}"
                        )
                except (ValueError, AttributeError):
                    # is_relative_to() not available in Python < 3.9, or not relative
                    # Fall back to string comparison
                    if str(persist_path).startswith(sensitive):
                        logger.error(f"Attempt to use sensitive directory: {persist_path}")
                        raise ValueError(
                            f"Invalid persist_directory: cannot use sensitive system directory {sensitive}"
                        )

        self.persist_directory = str(persist_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        logger.info(f"Initializing VectorStore at {self.persist_directory}")

    def initialize(self):
        """Initialize the ChromaDB client and collection."""
        if self.client is None:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Code repository vector store"}
            )

            logger.info(f"ChromaDB collection '{self.collection_name}' initialized")

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
            embeddings: Optional pre-computed embeddings
        """
        self.initialize()

        logger.info(f"Adding {len(documents)} documents to vector store")

        if embeddings is not None:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        logger.info("Documents added successfully")

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_embeddings: Query embedding vectors
            n_results: Number of results to return
            where: Optional metadata filters

        Returns:
            Query results with documents, metadatas, and distances
        """
        self.initialize()

        logger.debug(f"Querying vector store for top {n_results} results")

        # HIGH PRIORITY FIX: Add error handling
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where
            )
            return results

        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            # Return empty results instead of crashing
            return {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': []
            }

    def delete_collection(self):
        """Delete the entire collection."""
        self.initialize()

        # HIGH PRIORITY FIX: Add error handling
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def get_count(self) -> int:
        """Get the number of documents in the collection."""
        self.initialize()
        return self.collection.count()

    def reset(self):
        """Reset the vector store (delete all documents)."""
        self.delete_collection()
        self.collection = None
        self.initialize()
        logger.info("Vector store reset")
