"""
Hybrid search combining semantic and keyword search.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try to import BM25, but make it optional
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not available. Hybrid search will be disabled.")


class HybridSearchEngine:
    """
    Combines semantic (vector) search with keyword (BM25) search.
    """

    def __init__(self):
        """Initialize hybrid search engine."""
        self.bm25_index: Optional['BM25Okapi'] = None
        self.corpus: List[str] = []
        self.corpus_metadata: List[Dict[str, Any]] = []
        self.corpus_ids: List[str] = []

    def build_keyword_index(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """
        Build BM25 keyword index.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        """
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, skipping keyword index")
            return

        self.corpus = documents
        self.corpus_metadata = metadatas
        self.corpus_ids = ids

        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in documents]

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)

        logger.info(f"Built BM25 index with {len(documents)} documents")

    def keyword_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search using BM25.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of results with scores
        """
        if not BM25_AVAILABLE or self.bm25_index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top results
        top_indices = np.argsort(scores)[::-1][:n_results]

        results = []
        for idx in top_indices:
            if idx < len(self.corpus):
                results.append({
                    'text': self.corpus[idx],
                    'metadata': self.corpus_metadata[idx] if idx < len(self.corpus_metadata) else {},
                    'id': self.corpus_ids[idx] if idx < len(self.corpus_ids) else str(idx),
                    'score': float(scores[idx]),
                    'search_type': 'keyword'
                })

        return results

    def hybrid_search(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword search results.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic results (0-1)
            keyword_weight: Weight for keyword results (0-1)

        Returns:
            Combined and reranked results
        """
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        semantic_weight /= total_weight
        keyword_weight /= total_weight

        # Normalize scores for semantic results (distances -> similarities)
        semantic_scores = self._normalize_scores(
            [1 - r.get('distance', 0) for r in semantic_results]
        )

        # Normalize scores for keyword results
        keyword_scores = self._normalize_scores(
            [r.get('score', 0) for r in keyword_results]
        )

        # Combine results
        combined: Dict[str, Dict[str, Any]] = {}

        # Add semantic results
        for result, score in zip(semantic_results, semantic_scores):
            doc_id = result.get('id', result.get('text', ''))
            combined[doc_id] = {
                'result': result,
                'combined_score': score * semantic_weight,
                'search_type': 'semantic'
            }

        # Add/boost keyword results
        for result, score in zip(keyword_results, keyword_scores):
            doc_id = result.get('id', result.get('text', ''))

            if doc_id in combined:
                # Found by both methods - boost score
                combined[doc_id]['combined_score'] += score * keyword_weight
                combined[doc_id]['search_type'] = 'hybrid'
            else:
                combined[doc_id] = {
                    'result': result,
                    'combined_score': score * keyword_weight,
                    'search_type': 'keyword'
                }

        # Sort by combined score
        ranked = sorted(
            combined.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )

        # Return results with combined scores
        final_results = []
        for item in ranked:
            result = item['result'].copy()
            result['combined_score'] = item['combined_score']
            result['search_type'] = item['search_type']
            final_results.append(result)

        return final_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range.

        Args:
            scores: List of scores

        Returns:
            Normalized scores
        """
        if not scores:
            return []

        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()

        if max_score == min_score:
            return [1.0] * len(scores)

        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()

    def is_available(self) -> bool:
        """Check if hybrid search is available."""
        return BM25_AVAILABLE and self.bm25_index is not None
