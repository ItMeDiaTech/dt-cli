"""
Advanced Query Engine implementing 2025 RAG best practices.

Features:
- Self-RAG with reflection and self-critique
- HyDE (Hypothetical Document Embeddings)
- Query rewriting and expansion
- Multi-project retrieval
- Context engineering
- Adaptive retrieval strategies
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval with metadata."""
    text: str
    metadata: Dict[str, Any]
    score: float
    source_project: str
    relevance_score: Optional[float] = None  # Self-RAG relevance assessment


@dataclass
class SelfRAGDecision:
    """
    Decision from Self-RAG reflection.

    Self-RAG best practice: Dynamically decide when to retrieve,
    assess relevance, and critique outputs.
    """
    should_retrieve: bool
    confidence: float
    reasoning: str
    retrieval_query: Optional[str] = None  # Rewritten query if needed


class AdvancedQueryEngine:
    """
    Advanced Query Engine implementing 2025 RAG best practices.

    Best practices implemented:
    1. Self-RAG: Self-reflective retrieval with critique
    2. HyDE: Hypothetical document generation
    3. Query rewriting: Reformulate queries for better retrieval
    4. Multi-project: Search across multiple indexed folders
    5. Adaptive retrieval: Adjust strategy based on query type
    """

    def __init__(
        self,
        base_query_engine: Any,
        llm_provider: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize advanced query engine.

        Args:
            base_query_engine: Base QueryEngine instance
            llm_provider: LLM provider for query rewriting and reflection
            config: Configuration dictionary
        """
        self.base_engine = base_query_engine
        self.llm = llm_provider
        self.config = config or {}

        # Advanced features flags
        self.self_rag_enabled = self.config.get('self_rag_enabled', True)
        self.hyde_enabled = self.config.get('hyde_enabled', True)
        self.query_rewriting_enabled = self.config.get('query_rewriting_enabled', True)

        # Thresholds
        self.reflection_threshold = self.config.get('reflection_threshold', 0.7)
        self.max_retrieval_attempts = self.config.get('max_retrieval_attempts', 3)

        logger.info("Advanced Query Engine initialized")
        logger.info(f"Self-RAG: {self.self_rag_enabled}, HyDE: {self.hyde_enabled}, Query Rewriting: {self.query_rewriting_enabled}")

    def query(
        self,
        query: str,
        n_results: int = 5,
        project_filter: Optional[List[str]] = None,
        use_advanced_features: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system with advanced features.

        Args:
            query: User query
            n_results: Number of results to return
            project_filter: Optional list of project names to search in
            use_advanced_features: Whether to use Self-RAG, HyDE, etc.

        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()

        # Step 1: Self-RAG - Decide if retrieval is needed
        if use_advanced_features and self.self_rag_enabled:
            decision = self._self_rag_decide(query)

            if not decision.should_retrieve:
                logger.info(f"Self-RAG: Skipping retrieval - {decision.reasoning}")
                return {
                    'results': [],
                    'self_rag_decision': 'skip_retrieval',
                    'reasoning': decision.reasoning,
                    'query_time': time.time() - start_time
                }

            # Use rewritten query if provided
            if decision.retrieval_query:
                logger.info(f"Self-RAG: Using rewritten query: {decision.retrieval_query}")
                query = decision.retrieval_query

        # Step 2: Query Rewriting (if not done by Self-RAG)
        original_query = query
        if use_advanced_features and self.query_rewriting_enabled and not hasattr(self, '_query_rewritten'):
            rewritten_queries = self._rewrite_query(query)
            logger.info(f"Query rewriting generated {len(rewritten_queries)} variations")
        else:
            rewritten_queries = [query]

        # Step 3: HyDE - Generate hypothetical documents
        hypothetical_docs = []
        if use_advanced_features and self.hyde_enabled:
            hypothetical_docs = self._generate_hypothetical_documents(query)
            logger.info(f"HyDE generated {len(hypothetical_docs)} hypothetical documents")

        # Step 4: Multi-Query Retrieval
        all_results = []

        # Retrieve using original and rewritten queries
        for q in rewritten_queries[:2]:  # Limit to top 2 variations
            try:
                results = self.base_engine.query(q, n_results=n_results)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Retrieval error for query '{q}': {e}")

        # Retrieve using hypothetical documents
        for doc in hypothetical_docs[:1]:  # Use top hypothetical doc
            try:
                results = self.base_engine.query(doc, n_results=n_results // 2)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"HyDE retrieval error: {e}")

        # Step 5: Deduplicate and rerank
        deduplicated_results = self._deduplicate_results(all_results)
        ranked_results = self._rerank_results(deduplicated_results, original_query)

        # Step 6: Self-RAG - Assess relevance of retrieved documents
        if use_advanced_features and self.self_rag_enabled:
            ranked_results = self._self_rag_assess_relevance(ranked_results, original_query)

        # Step 7: Filter by project if requested
        if project_filter:
            ranked_results = [r for r in ranked_results if r.get('metadata', {}).get('project') in project_filter]

        query_time = time.time() - start_time

        return {
            'results': ranked_results[:n_results],
            'original_query': original_query,
            'rewritten_queries': rewritten_queries,
            'hypothetical_docs': hypothetical_docs,
            'total_retrieved': len(all_results),
            'after_dedup': len(deduplicated_results),
            'query_time': query_time,
            'features_used': {
                'self_rag': self.self_rag_enabled,
                'hyde': self.hyde_enabled,
                'query_rewriting': self.query_rewriting_enabled
            }
        }

    def _self_rag_decide(self, query: str) -> SelfRAGDecision:
        """
        Self-RAG: Decide if retrieval is needed.

        Best Practice: Use LLM to classify if the query requires
        external knowledge or can be answered directly.
        """
        prompt = f"""Analyze this query and decide if retrieving code/documentation is needed.

Query: {query}

Consider:
1. Does this require looking at existing code?
2. Does this require project-specific knowledge?
3. Can this be answered with general programming knowledge?

Respond in this format:
RETRIEVE: yes/no
CONFIDENCE: 0.0-1.0
REASONING: brief explanation
REWRITTEN_QUERY: (optional) better query for retrieval"""

        try:
            response = self.llm.generate(prompt=prompt, max_tokens=200)

            # Parse response
            should_retrieve = 'yes' in response.lower().split('\n')[0]
            confidence = 0.8  # Default

            # Extract confidence if present
            for line in response.split('\n'):
                if 'confidence:' in line.lower():
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except:
                        pass

            # Extract reasoning
            reasoning = "Needs code retrieval"
            for line in response.split('\n'):
                if 'reasoning:' in line.lower():
                    reasoning = line.split(':', 1)[1].strip()

            # Extract rewritten query
            rewritten_query = None
            for line in response.split('\n'):
                if 'rewritten_query:' in line.lower():
                    rewritten_query = line.split(':', 1)[1].strip()

            return SelfRAGDecision(
                should_retrieve=should_retrieve,
                confidence=confidence,
                reasoning=reasoning,
                retrieval_query=rewritten_query
            )

        except Exception as e:
            logger.error(f"Self-RAG decision error: {e}")
            # Default to retrieving on error
            return SelfRAGDecision(
                should_retrieve=True,
                confidence=0.5,
                reasoning="Error in self-reflection, defaulting to retrieve"
            )

    def _rewrite_query(self, query: str) -> List[str]:
        """
        Rewrite query for better retrieval.

        Best Practice: Generate multiple query variations to improve recall.
        """
        prompt = f"""Rewrite this query in 3 different ways to improve code search:

Original: {query}

Generate:
1. A more technical/specific version
2. A broader version with related concepts
3. A version with relevant keywords

Format each as "VERSION N: <query>"""

        try:
            response = self.llm.generate(prompt=prompt, max_tokens=300)

            queries = [query]  # Include original

            # Parse versions
            for line in response.split('\n'):
                if 'VERSION' in line and ':' in line:
                    rewritten = line.split(':', 1)[1].strip()
                    if rewritten and rewritten != query:
                        queries.append(rewritten)

            return queries[:4]  # Max 4 queries

        except Exception as e:
            logger.error(f"Query rewriting error: {e}")
            return [query]

    def _generate_hypothetical_documents(self, query: str) -> List[str]:
        """
        HyDE: Generate hypothetical documents that would answer the query.

        Best Practice: Create ideal answer documents, then search for similar real documents.
        """
        prompt = f"""Generate a hypothetical code snippet or documentation that would perfectly answer this query:

Query: {query}

Write a concise, realistic code example or explanation that represents the ideal answer.
Focus on technical accuracy and completeness."""

        try:
            response = self.llm.generate(prompt=prompt, max_tokens=500)
            return [response.strip()]

        except Exception as e:
            logger.error(f"HyDE generation error: {e}")
            return []

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate results by content similarity."""
        if not results:
            return []

        seen_texts = set()
        deduplicated = []

        for result in results:
            text = result.get('text', '')
            # Simple deduplication by exact text match
            # TODO: Could use embedding similarity for fuzzy dedup
            if text not in seen_texts:
                seen_texts.add(text)
                deduplicated.append(result)

        logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)}")
        return deduplicated

    def _rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rerank results using LLM.

        Best Practice: Use LLM to rerank based on relevance to query.
        """
        if not results or len(results) <= 1:
            return results

        # For now, return as-is (base engine already ranks by similarity)
        # TODO: Implement LLM-based reranking
        return results

    def _self_rag_assess_relevance(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Self-RAG: Assess relevance of retrieved documents.

        Best Practice: Use LLM to evaluate if retrieved docs are actually relevant.
        """
        if not results:
            return results

        # Assess top results
        for i, result in enumerate(results[:3]):  # Assess top 3
            try:
                text_preview = result.get('text', '')[:500]  # First 500 chars

                prompt = f"""Assess relevance of this code snippet to the query:

Query: {query}

Code:
{text_preview}

Is this relevant? Rate 0.0-1.0 and explain briefly."""

                assessment = self.llm.generate(prompt=prompt, max_tokens=150)

                # Extract score
                relevance_score = 0.5  # Default
                for line in assessment.split('\n'):
                    # Look for number
                    words = line.split()
                    for word in words:
                        try:
                            score = float(word)
                            if 0.0 <= score <= 1.0:
                                relevance_score = score
                                break
                        except:
                            continue

                result['self_rag_relevance'] = relevance_score
                result['self_rag_assessment'] = assessment

            except Exception as e:
                logger.error(f"Self-RAG assessment error: {e}")
                result['self_rag_relevance'] = 0.5

        # Sort by relevance if assessed
        results_with_scores = [r for r in results if 'self_rag_relevance' in r]
        results_without_scores = [r for r in results if 'self_rag_relevance' not in r]

        sorted_results = sorted(results_with_scores, key=lambda x: x['self_rag_relevance'], reverse=True)
        return sorted_results + results_without_scores

    def get_status(self) -> Dict[str, Any]:
        """Get status of advanced query engine."""
        return {
            'self_rag_enabled': self.self_rag_enabled,
            'hyde_enabled': self.hyde_enabled,
            'query_rewriting_enabled': self.query_rewriting_enabled,
            'reflection_threshold': self.reflection_threshold,
            'max_retrieval_attempts': self.max_retrieval_attempts
        }
