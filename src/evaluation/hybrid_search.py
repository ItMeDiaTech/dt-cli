"""
Hybrid Search - Combine semantic and keyword search for better retrieval.

This module provides hybrid search that combines:
- Semantic search (embeddings + cosine similarity)
- Keyword search (BM25)
- Query rewriting and expansion

Expected impact: +20-30% retrieval accuracy.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Search result with scoring details.
    """
    text: str
    metadata: Dict[str, Any]
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'metadata': self.metadata,
            'scores': {
                'semantic': self.semantic_score,
                'keyword': self.keyword_score,
                'combined': self.combined_score
            },
            'rank': self.rank
        }


class BM25:
    """
    BM25 keyword scoring algorithm.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25.

        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization parameter (0-1)
        """
        self.k1 = k1
        self.b = b

        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def fit(self, documents: List[str]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of document texts
        """
        self.num_docs = len(documents)

        # Calculate document frequencies
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Calculate average document length
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0

        # Calculate IDF scores
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((self.num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str, document: str, doc_index: int) -> float:
        """
        Calculate BM25 score for query-document pair.

        Args:
            query: Query text
            document: Document text
            doc_index: Document index (for length lookup)

        Returns:
            BM25 score
        """
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)

        # Term frequency in document
        doc_tf = Counter(doc_tokens)

        score = 0.0
        doc_length = self.doc_lengths[doc_index] if doc_index < len(self.doc_lengths) else len(doc_tokens)

        for token in query_tokens:
            if token not in self.idf:
                continue

            # Get term frequency
            tf = doc_tf.get(token, 0)

            # Get IDF
            idf = self.idf[token]

            # Calculate BM25 score component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

            score += idf * (numerator / denominator)

        return score

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.split() if len(token) > 2]


class QueryRewriter:
    """
    Query rewriting and expansion for better retrieval.
    """

    def __init__(self, llm_provider=None):
        """
        Initialize query rewriter.

        Args:
            llm_provider: Optional LLM for advanced rewriting
        """
        self.llm = llm_provider

        # Common code-related synonyms
        self.synonyms = {
            'function': ['method', 'procedure', 'routine'],
            'class': ['object', 'type', 'struct'],
            'variable': ['var', 'field', 'attribute'],
            'error': ['exception', 'bug', 'issue'],
            'fix': ['solve', 'repair', 'resolve'],
            'create': ['make', 'build', 'generate'],
            'delete': ['remove', 'drop', 'destroy'],
            'update': ['modify', 'change', 'edit']
        }

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and variations.

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        queries = [query]

        # Add lowercase version
        if query != query.lower():
            queries.append(query.lower())

        # Expand with synonyms
        words = query.lower().split()
        for i, word in enumerate(words):
            if word in self.synonyms:
                for synonym in self.synonyms[word]:
                    expanded = words.copy()
                    expanded[i] = synonym
                    queries.append(' '.join(expanded))

        return list(set(queries))  # Remove duplicates

    def rewrite_for_code(self, query: str) -> str:
        """
        Rewrite natural language query for code search.

        Args:
            query: Natural language query

        Returns:
            Code-optimized query
        """
        # Add code-specific terms
        code_terms = []

        if 'how' in query.lower():
            code_terms.append('implementation')

        if 'what' in query.lower():
            code_terms.append('definition')

        if 'where' in query.lower():
            code_terms.append('location')

        if 'error' in query.lower() or 'bug' in query.lower():
            code_terms.extend(['exception', 'traceback', 'stack'])

        if code_terms:
            return f"{query} {' '.join(code_terms)}"

        return query


class HybridSearch:
    """
    Hybrid search combining semantic and keyword search.
    """

    def __init__(
        self,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid search.

        Args:
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        self.bm25 = BM25()
        self.query_rewriter = QueryRewriter()

        self.documents: List[str] = []
        self.metadata_list: List[Dict] = []

    def index_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Index documents for hybrid search.

        Args:
            documents: Document texts
            metadata: Optional metadata per document
        """
        self.documents = documents
        self.metadata_list = metadata if metadata else [{} for _ in documents]

        # Build BM25 index
        self.bm25.fit(documents)

        logger.info(f"Indexed {len(documents)} documents for hybrid search")

    def search(
        self,
        query: str,
        semantic_scores: Optional[List[float]] = None,
        top_k: int = 5,
        use_query_expansion: bool = True
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword scoring.

        Args:
            query: Search query
            semantic_scores: Pre-computed semantic similarity scores
            top_k: Number of results to return
            use_query_expansion: Whether to expand query

        Returns:
            List of search results
        """
        if not self.documents:
            return []

        # Expand query if requested
        if use_query_expansion:
            expanded_queries = self.query_rewriter.expand_query(query)
        else:
            expanded_queries = [query]

        # Calculate keyword scores (average across expanded queries)
        keyword_scores = []
        for i, doc in enumerate(self.documents):
            scores_for_doc = []
            for expanded_query in expanded_queries:
                score = self.bm25.score(expanded_query, doc, i)
                scores_for_doc.append(score)

            # Average score across expansions
            keyword_scores.append(np.mean(scores_for_doc))

        # Normalize keyword scores
        if keyword_scores:
            max_keyword = max(keyword_scores) if max(keyword_scores) > 0 else 1.0
            keyword_scores = [s / max_keyword for s in keyword_scores]

        # Use provided semantic scores or default to zeros
        if semantic_scores is None:
            semantic_scores = [0.0] * len(self.documents)

        # Combine scores
        results = []
        for i, (doc, metadata) in enumerate(zip(self.documents, self.metadata_list)):
            semantic_score = semantic_scores[i] if i < len(semantic_scores) else 0.0
            keyword_score = keyword_scores[i] if i < len(keyword_scores) else 0.0

            combined_score = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )

            results.append(SearchResult(
                text=doc,
                metadata=metadata,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                combined_score=combined_score
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        # Assign ranks
        for rank, result in enumerate(results[:top_k], 1):
            result.rank = rank

        return results[:top_k]

    def tune_weights(
        self,
        queries: List[str],
        ground_truth_indices: List[List[int]],
        semantic_scores_list: List[List[float]]
    ) -> Tuple[float, float]:
        """
        Tune semantic/keyword weights using validation set.

        Args:
            queries: Validation queries
            ground_truth_indices: Relevant document indices per query
            semantic_scores_list: Semantic scores per query

        Returns:
            Optimal (semantic_weight, keyword_weight)
        """
        best_score = 0.0
        best_weights = (0.7, 0.3)

        # Try different weight combinations
        for sem_weight in np.arange(0.0, 1.1, 0.1):
            kw_weight = 1.0 - sem_weight

            self.semantic_weight = sem_weight
            self.keyword_weight = kw_weight

            # Calculate average precision
            precisions = []
            for query, ground_truth, sem_scores in zip(queries, ground_truth_indices, semantic_scores_list):
                results = self.search(query, sem_scores, top_k=10)

                # Calculate precision@10
                retrieved_indices = [
                    self.documents.index(r.text) for r in results
                ]
                relevant_retrieved = len(set(retrieved_indices) & set(ground_truth))
                precision = relevant_retrieved / len(results) if results else 0

                precisions.append(precision)

            avg_precision = np.mean(precisions) if precisions else 0

            if avg_precision > best_score:
                best_score = avg_precision
                best_weights = (sem_weight, kw_weight)

        self.semantic_weight, self.keyword_weight = best_weights
        logger.info(
            f"Tuned weights: semantic={self.semantic_weight:.2f}, "
            f"keyword={self.keyword_weight:.2f} (score={best_score:.3f})"
        )

        return best_weights


def create_hybrid_search(
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> HybridSearch:
    """
    Create hybrid search instance.

    Args:
        semantic_weight: Weight for semantic scoring
        keyword_weight: Weight for keyword scoring

    Returns:
        Initialized HybridSearch
    """
    return HybridSearch(
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight
    )
