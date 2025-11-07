"""
Cross-encoder reranking for improved accuracy.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import CrossEncoder, make it optional
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("CrossEncoder not available. Reranking will be disabled.")


class Reranker:
    """
    Rerank search results using cross-encoder for better accuracy.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name
        """
        self.model_name = model_name
        self.model: Optional['CrossEncoder'] = None

    def load_model(self):
        """Lazy load the cross-encoder model."""
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("CrossEncoder not available")
            return

        if self.model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Reranker loaded successfully")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder.

        MEDIUM PRIORITY FIX: Log warnings on empty results instead of silent return.

        Args:
            query: Original query
            results: List of search results
            top_k: Number of top results to return (None = all)

        Returns:
            Reranked results
        """
        # MEDIUM PRIORITY FIX: Log warnings instead of silent returns
        if not CROSS_ENCODER_AVAILABLE:
            logger.debug("CrossEncoder not available - skipping reranking")
            return results

        if not results:
            logger.warning(
                f"Rerank called with empty results for query: '{query}'. "
                f"This may indicate an issue with the initial search."
            )
            return results

        self.load_model()

        if self.model is None:
            return results

        # Prepare query-document pairs
        pairs = [[query, result.get('text', '')] for result in results]

        # Get cross-encoder scores
        try:
            scores = self.model.predict(pairs)

            # Add rerank scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
                result['original_score'] = result.get('distance', result.get('score', 0))

            # Sort by rerank score
            reranked = sorted(
                results,
                key=lambda x: x.get('rerank_score', 0),
                reverse=True
            )

            if top_k:
                reranked = reranked[:top_k]

            logger.info(f"Reranked {len(results)} results to {len(reranked)}")

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    def is_available(self) -> bool:
        """Check if reranking is available."""
        return CROSS_ENCODER_AVAILABLE
