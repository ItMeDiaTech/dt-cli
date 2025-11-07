"""
Integrated query pipeline combining all components.
"""

from typing import List, Dict, Any, Optional
import logging
import time
from ..rag.query_expansion import QueryExpander
from ..rag.hybrid_search import HybridSearchEngine
from ..rag.reranking import Reranker
from ..resilience.graceful_degradation import degradation_manager
from ..logging_utils.structured_logging import StructuredLogger, CorrelationContext

logger = StructuredLogger(__name__)


class IntegratedQueryPipeline:
    """
    Unified pipeline: Query Expansion → Hybrid Search → Reranking.
    """

    def __init__(
        self,
        query_engine,
        use_expansion: bool = True,
        use_hybrid: bool = True,
        use_reranking: bool = True
    ):
        """
        Initialize integrated pipeline.

        Args:
            query_engine: Query engine instance
            use_expansion: Enable query expansion
            use_hybrid: Enable hybrid search
            use_reranking: Enable reranking
        """
        self.query_engine = query_engine
        self.expander = QueryExpander() if use_expansion else None
        self.hybrid_search = HybridSearchEngine() if use_hybrid else None
        self.reranker = Reranker() if use_reranking else None

        self.use_expansion = use_expansion
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking

        logger.info(
            "Integrated pipeline initialized",
            expansion=use_expansion,
            hybrid=use_hybrid,
            reranking=use_reranking
        )

    @degradation_manager.with_fallback("integrated_pipeline")
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full integrated pipeline.

        Args:
            query_text: Query string
            n_results: Number of results
            correlation_id: Optional correlation ID for tracing

        Returns:
            Pipeline results with metadata
        """
        with CorrelationContext(correlation_id) as corr_id:
            start_time = time.time()

            logger.info(
                "Starting integrated pipeline",
                query=query_text,
                n_results=n_results
            )

            pipeline_results = {
                'original_query': query_text,
                'correlation_id': corr_id,
                'pipeline_stages': {},
                'results': [],
                'metadata': {}
            }

            # Stage 1: Query Expansion
            queries = [query_text]
            if self.use_expansion and self.expander:
                try:
                    queries = self._expand_query(query_text)
                    pipeline_results['pipeline_stages']['expansion'] = {
                        'enabled': True,
                        'expanded_queries': queries
                    }
                except Exception as e:
                    logger.warning("Query expansion failed", error=str(e))
                    pipeline_results['pipeline_stages']['expansion'] = {
                        'enabled': True,
                        'error': str(e)
                    }

            # Stage 2: Search (Hybrid or Semantic)
            if self.use_hybrid and self.hybrid_search:
                results = self._hybrid_search(queries[0], n_results * 2)
                pipeline_results['pipeline_stages']['search'] = {
                    'type': 'hybrid',
                    'candidates': len(results)
                }
            else:
                results = self._semantic_search(queries[0], n_results * 2)
                pipeline_results['pipeline_stages']['search'] = {
                    'type': 'semantic',
                    'candidates': len(results)
                }

            # Stage 3: Reranking
            if self.use_reranking and self.reranker and results:
                try:
                    results = self._rerank(query_text, results, n_results)
                    pipeline_results['pipeline_stages']['reranking'] = {
                        'enabled': True,
                        'final_count': len(results)
                    }
                except Exception as e:
                    logger.warning("Reranking failed", error=str(e))
                    results = results[:n_results]
                    pipeline_results['pipeline_stages']['reranking'] = {
                        'enabled': True,
                        'error': str(e),
                        'used_fallback': True
                    }
            else:
                results = results[:n_results]

            # Add results
            pipeline_results['results'] = results
            pipeline_results['metadata'] = {
                'total_time_ms': (time.time() - start_time) * 1000,
                'result_count': len(results),
                'degradation_level': degradation_manager.get_system_level()
            }

            logger.info(
                "Pipeline completed",
                result_count=len(results),
                time_ms=pipeline_results['metadata']['total_time_ms']
            )

            return pipeline_results

    @degradation_manager.with_fallback("query_expansion")
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        return self.expander.expand_query(query, max_expansions=2)

    @degradation_manager.with_fallback("hybrid_search")
    def _hybrid_search(self, query: str, n_results: int) -> List[Dict]:
        """Perform hybrid search."""
        semantic_results = self.query_engine._semantic_query(query, n_results, None)
        keyword_results = self.hybrid_search.keyword_search(query, n_results)

        return self.hybrid_search.hybrid_search(
            semantic_results,
            keyword_results
        )

    @degradation_manager.with_fallback("semantic_search")
    def _semantic_search(self, query: str, n_results: int) -> List[Dict]:
        """Perform semantic search."""
        return self.query_engine._semantic_query(query, n_results, None)

    @degradation_manager.with_fallback("reranking")
    def _rerank(self, query: str, results: List[Dict], n_results: int) -> List[Dict]:
        """Rerank results."""
        return self.reranker.rerank(query, results, top_k=n_results)
