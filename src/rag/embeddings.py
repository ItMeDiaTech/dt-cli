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
        """
        Load the embedding model.

        MEDIUM PRIORITY FIX: Add error handling for model loading failures.
        """
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")

            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                raise RuntimeError(
                    f"Could not load embedding model '{self.model_name}'. "
                    f"Check internet connection or model name. Error: {e}"
                ) from e

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        MEDIUM PRIORITY FIX: Add error handling for encoding failures.

        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar

        Returns:
            Numpy array of embeddings

        Raises:
            RuntimeError: If embedding generation fails
        """
        self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        # MEDIUM PRIORITY FIX: Validate input
        if not texts or all(not t or not t.strip() for t in texts):
            logger.warning("Encode called with empty or whitespace-only texts")
            # Return zero embeddings for empty input
            dimension = self.get_dimension()
            return np.zeros((len(texts), dimension))

        logger.debug(f"Encoding {len(texts)} texts")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(
                f"Failed to generate embeddings for {len(texts)} texts. "
                f"This may be due to memory issues or invalid input. Error: {e}"
            ) from e

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
        # CRITICAL FIX: Prevent division by zero
        magnitude = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        if magnitude == 0:
            return 0.0  # Return 0 similarity for zero vectors
        return np.dot(embedding1, embedding2) / magnitude
