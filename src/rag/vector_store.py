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

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None

        logger.info(f"Initializing VectorStore at {persist_directory}")

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

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )

        return results

    def delete_collection(self):
        """Delete the entire collection."""
        self.initialize()
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted")

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
