"""
Lazy model loading with automatic unloading.
"""

import threading
import time
from typing import Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LazyEmbeddingEngine:
    """
    Embedding engine with lazy loading and automatic unloading after idle period.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", idle_timeout: int = 300):
        """
        Initialize lazy embedding engine.

        Args:
            model_name: Name of the sentence-transformers model
            idle_timeout: Seconds of idleness before unloading model
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.idle_timeout = idle_timeout
        self.last_used: Optional[float] = None
        self.cleanup_thread: Optional[threading.Thread] = None
        self.model_lock = threading.Lock()
        self._stop_cleanup = False
        logger.info(f"Initialized LazyEmbeddingEngine: {model_name}")

    def load_model(self):
        """Load the embedding model if not already loaded."""
        with self.model_lock:
            if self.model is None:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("Model loaded successfully")
                self._start_cleanup_timer()

            self.last_used = time.time()

    def _start_cleanup_timer(self):
        """Start background thread to unload model after idle period."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return

        self._stop_cleanup = False
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True
        )
        self.cleanup_thread.start()

    def _cleanup_worker(self):
        """Background thread that unloads model when idle."""
        while not self._stop_cleanup:
            time.sleep(60)  # Check every minute

            if self.model and self.last_used:
                idle_time = time.time() - self.last_used

                if idle_time > self.idle_timeout:
                    with self.model_lock:
                        if self.model:  # Double-check after acquiring lock
                            logger.info(
                                f"Unloading idle model (idle for {idle_time:.0f}s)"
                            )
                            self.model = None
                            return  # Exit thread

    def encode(
        self,
        texts: list,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: List of texts to embed
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

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self.model is not None

    def unload(self):
        """Manually unload the model."""
        with self.model_lock:
            if self.model:
                logger.info("Manually unloading model")
                self.model = None
                self._stop_cleanup = True

    def __del__(self):
        """Cleanup on deletion."""
        self._stop_cleanup = True
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=1)
