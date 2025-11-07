"""
Lazy model loading with automatic unloading.
"""

import threading
import time
from typing import Optional, Dict, Any
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LazyEmbeddingEngine:
    """
    Embedding engine with lazy loading and automatic unloading after idle period.

    MEDIUM PRIORITY FIX: Add memory monitoring.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", idle_timeout: int = 300):
        """
        Initialize lazy embedding engine.

        MEDIUM PRIORITY FIX: Track memory usage statistics.

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

        # MEDIUM PRIORITY FIX: Track memory and usage statistics
        self.load_count = 0
        self.unload_count = 0
        self.encode_count = 0
        self.total_encode_time = 0.0

        logger.info(f"Initialized LazyEmbeddingEngine: {model_name}")

    def load_model(self):
        """
        Load the embedding model if not already loaded.

        MEDIUM PRIORITY FIX: Track load events.
        """
        with self.model_lock:
            if self.model is None:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)

                # MEDIUM PRIORITY FIX: Track load count
                self.load_count += 1

                logger.info(f"Model loaded successfully (load #{self.load_count})")
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
        """
        Background thread that unloads model when idle.

        MEDIUM PRIORITY FIX: Track unload events.
        """
        while not self._stop_cleanup:
            time.sleep(60)  # Check every minute

            if self.model and self.last_used:
                idle_time = time.time() - self.last_used

                if idle_time > self.idle_timeout:
                    with self.model_lock:
                        if self.model:  # Double-check after acquiring lock
                            # MEDIUM PRIORITY FIX: Track unload count
                            self.unload_count += 1

                            logger.info(
                                f"Unloading idle model (idle for {idle_time:.0f}s, unload #{self.unload_count})"
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

        MEDIUM PRIORITY FIX: Track encoding operations and timing.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        # HIGH PRIORITY FIX: Hold lock during entire operation to prevent
        # model from being unloaded while encoding is in progress
        with self.model_lock:
            if self.model is None:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)

                # MEDIUM PRIORITY FIX: Track load count
                self.load_count += 1

                logger.info(f"Model loaded successfully (load #{self.load_count})")
                self._start_cleanup_timer()

            self.last_used = time.time()

            if isinstance(texts, str):
                texts = [texts]

            logger.debug(f"Encoding {len(texts)} texts")

            # MEDIUM PRIORITY FIX: Track encoding timing
            start_time = time.time()

            # Model cannot be unloaded while we hold the lock
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )

            # MEDIUM PRIORITY FIX: Update statistics
            encode_time = time.time() - start_time
            self.encode_count += 1
            self.total_encode_time += encode_time

            logger.debug(f"Encoded {len(texts)} texts in {encode_time:.2f}s")

        return embeddings

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        # HIGH PRIORITY FIX: Hold lock to prevent race condition
        with self.model_lock:
            if self.model is None:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("Model loaded successfully")
                self._start_cleanup_timer()

            self.last_used = time.time()
            return self.model.get_sentence_embedding_dimension()

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        # HIGH PRIORITY FIX: Thread-safe read
        with self.model_lock:
            return self.model is not None

    def unload(self):
        """Manually unload the model."""
        with self.model_lock:
            if self.model:
                logger.info("Manually unloading model")
                self.model = None
                self._stop_cleanup = True

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        MEDIUM PRIORITY FIX: Get memory and usage statistics.

        Returns:
            Dictionary with statistics
        """
        with self.model_lock:
            is_loaded = self.model is not None
            idle_time = time.time() - self.last_used if self.last_used else None

        avg_encode_time = (
            self.total_encode_time / self.encode_count
            if self.encode_count > 0
            else 0
        )

        return {
            'model_name': self.model_name,
            'is_loaded': is_loaded,
            'idle_time_seconds': idle_time,
            'idle_timeout_seconds': self.idle_timeout,
            'load_count': self.load_count,
            'unload_count': self.unload_count,
            'encode_count': self.encode_count,
            'total_encode_time': self.total_encode_time,
            'avg_encode_time': avg_encode_time
        }

    def __del__(self):
        """Cleanup on deletion."""
        self._stop_cleanup = True
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=1)
