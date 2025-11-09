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

    Default model: BAAI/bge-base-en-v1.5 (optimized for code retrieval)
    - 768 dimensions (compatible with previous model)
    - Specifically trained for code and technical content
    - Supports instruction prefixes for better retrieval
    - Research shows 15-20% better performance on code tasks

    Fallback: all-MiniLM-L6-v2 (smaller, faster, general purpose)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        use_instruction_prefix: bool = True
    ):
        """
        Initialize the embedding engine.

        Args:
            model_name: Name of the sentence-transformers model to use
            use_instruction_prefix: Use instruction prefix for better retrieval
                                   (recommended for BGE models)
        """
        self.model_name = model_name
        self.use_instruction_prefix = use_instruction_prefix
        self.model = None

        # Instruction prefix for BGE models (improves retrieval quality)
        self.instruction_prefix = "Represent this code for retrieval: "

        logger.info(
            f"Initializing EmbeddingEngine with model: {model_name} "
            f"(instruction_prefix={use_instruction_prefix})"
        )

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
        show_progress_bar: bool = False,
        is_query: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            is_query: If True, add instruction prefix (for queries only)
                     For documents, no prefix is added

        Returns:
            Numpy array of embeddings

        Raises:
            RuntimeError: If embedding generation fails
        """
        self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        # Validate input
        if not texts or all(not t or not t.strip() for t in texts):
            logger.warning("Encode called with empty or whitespace-only texts")
            # Return zero embeddings for empty input
            dimension = self.get_dimension()
            return np.zeros((len(texts), dimension))

        # Add instruction prefix for queries if using BGE model
        processed_texts = texts
        if self.use_instruction_prefix and is_query and 'bge' in self.model_name.lower():
            processed_texts = [
                f"{self.instruction_prefix}{text}"
                for text in texts
            ]
            logger.debug("Added instruction prefix to query texts")

        logger.debug(f"Encoding {len(texts)} texts (is_query={is_query})")

        try:
            embeddings = self.model.encode(
                processed_texts,
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

    def encode_query(
        self,
        query: str,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode a query with instruction prefix.

        This is a convenience method that automatically adds the instruction
        prefix for better retrieval quality.

        Args:
            query: Query text
            batch_size: Batch size for encoding

        Returns:
            Query embedding
        """
        return self.encode(query, batch_size=batch_size, is_query=True)

    def encode_documents(
        self,
        documents: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode documents without instruction prefix.

        Args:
            documents: Document(s) to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar

        Returns:
            Document embeddings
        """
        return self.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            is_query=False
        )

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
