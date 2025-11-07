"""
Embedding Engine using sentence-transformers for local embeddings.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Local embedding generation using sentence-transformers.
    Uses all-MiniLM-L6-v2 model for fast, efficient embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding engine.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initializing EmbeddingEngine with model: {model_name}")

    def load_model(self):
        """Load the embedding model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        logger.debug(f"Encoding {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )

        return embeddings

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        self.load_model()
        return self.model.get_sentence_embedding_dimension()

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
