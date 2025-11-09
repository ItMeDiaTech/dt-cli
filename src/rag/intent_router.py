"""
Intent-based query routing for intelligent auto-triggering.

This module analyzes user queries and automatically routes them to the
appropriate system:
- Vector search (RAG)
- Knowledge graph queries
- Debug agent
- Code review agent
- Direct LLM (skip RAG)

Expected impact: 70% reduction in manual commands, seamless UX.
"""

from typing import Literal, Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# Query intent types
QueryIntent = Literal[
    'code_search',      # Semantic code search (use RAG)
    'graph_query',      # Code relationships (use knowledge graph)
    'debugging',        # Error analysis (use debug agent)
    'code_review',      # Code quality check (use review agent)
    'direct_answer',    # Simple question (skip RAG, use LLM directly)
    'documentation'     # Documentation query (use RAG on docs)
]


class IntentRouter:
    """
    Semantic intent router using embedding similarity.

    Routes queries to appropriate systems based on learned patterns.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.7):
        """
        Initialize intent router.

        Args:
            model_name: Model for embedding route patterns
            threshold: Minimum similarity threshold for classification
        """
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
        self.route_embeddings = None

        # Define route patterns (example queries for each intent)
        self.routes = {
            'code_search': [
                "where is the authentication code?",
                "find error handling logic",
                "show me API endpoints",
                "locate database queries",
                "find all uses of this function",
                "where is this class defined?",
                "search for logging code",
                "find configuration loading",
                "show me middleware implementations",
                "locate validation logic"
            ],
            'graph_query': [
                "what depends on this module?",
                "what imports this class?",
                "show me the call graph",
                "what functions call this?",
                "what would break if I change this?",
                "show dependencies for this file",
                "what uses this API?",
                "trace the execution flow",
                "show me all callers",
                "what tests cover this function?"
            ],
            'debugging': [
                "why is this test failing?",
                "debug this error",
                "fix this bug",
                "what's causing this exception?",
                "why doesn't this work?",
                "investigate this failure",
                "troubleshoot this issue",
                "analyze this error message",
                "why am I getting this error?",
                "help me debug this"
            ],
            'code_review': [
                "review this code",
                "check for issues",
                "any problems with this?",
                "is this code correct?",
                "can you review my changes?",
                "look for bugs in this",
                "check code quality",
                "review my implementation",
                "is there a better way to do this?",
                "suggest improvements"
            ],
            'direct_answer': [
                "fix this typo",
                "add a comment here",
                "rename this variable",
                "explain what this does",
                "what does this function do?",
                "how do I use this API?",
                "write a docstring",
                "format this code",
                "add type hints",
                "simplify this expression"
            ],
            'documentation': [
                "how do I configure this?",
                "what are the installation steps?",
                "show me the API docs",
                "how does this feature work?",
                "what's the purpose of this module?",
                "explain the architecture",
                "show me examples",
                "what are the requirements?",
                "how to get started?",
                "read the documentation"
            ]
        }

        logger.info(f"Initialized IntentRouter with {len(self.routes)} route types")

    def _load_model(self):
        """Load embedding model for route classification."""
        if self.model is None:
            logger.info(f"Loading intent router model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info("Intent router model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load intent router model: {e}")
                raise

    def _compute_route_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Pre-compute embeddings for all route patterns.

        Returns:
            Dictionary mapping intent to average embedding
        """
        self._load_model()

        route_embeddings = {}

        for intent, examples in self.routes.items():
            # Embed all examples for this intent
            embeddings = self.model.encode(examples, convert_to_numpy=True)

            # Use average embedding as representative
            avg_embedding = np.mean(embeddings, axis=0)

            route_embeddings[intent] = avg_embedding

            logger.debug(f"Computed embedding for intent '{intent}' from {len(examples)} examples")

        return route_embeddings

    def classify(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify query intent using semantic similarity.

        Args:
            query: User query to classify

        Returns:
            Tuple of (intent, confidence_score)
        """
        # Lazy load embeddings
        if self.route_embeddings is None:
            self.route_embeddings = self._compute_route_embeddings()

        # Embed query
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Compute similarity to each route
        best_intent = None
        best_score = -1.0

        for intent, route_embedding in self.route_embeddings.items():
            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, route_embedding)

            if similarity > best_score:
                best_score = similarity
                best_intent = intent

        # Check threshold
        if best_score < self.threshold:
            logger.info(
                f"Query intent unclear (best: {best_intent}, score: {best_score:.2f}), "
                f"defaulting to code_search"
            )
            return 'code_search', best_score

        logger.info(f"Classified query intent: {best_intent} (confidence: {best_score:.2f})")

        return best_intent, best_score

    def should_use_rag(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> bool:
        """
        Determine if RAG should be used for this query.

        Args:
            query: User query
            context: Optional context (files open, conversation history, etc.)

        Returns:
            True if RAG should be used, False otherwise
        """
        # Check context first
        if context:
            # Skip RAG if sufficient context already available
            if self._has_sufficient_context(context):
                logger.info("Sufficient context available, skipping RAG")
                return False

            # Skip RAG for follow-up questions
            if context.get('is_followup', False):
                logger.info("Follow-up question, using existing context")
                return False

        # Classify intent
        intent, confidence = self.classify(query)

        # Map intent to RAG usage
        rag_intents = {
            'code_search': True,
            'graph_query': True,  # Could use knowledge graph, but RAG is fallback
            'debugging': True,
            'code_review': True,
            'direct_answer': False,
            'documentation': True
        }

        use_rag = rag_intents.get(intent, True)

        logger.info(
            f"RAG decision for '{intent}': {use_rag} "
            f"(confidence: {confidence:.2f})"
        )

        return use_rag

    def route_query(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Route query to appropriate system.

        Args:
            query: User query
            context: Optional context

        Returns:
            Routing decision with metadata
        """
        # Classify intent
        intent, confidence = self.classify(query)

        # Build routing decision
        decision = {
            'intent': intent,
            'confidence': confidence,
            'use_rag': False,
            'use_graph': False,
            'use_debug_agent': False,
            'use_review_agent': False,
            'use_direct_llm': False
        }

        # Route based on intent
        if intent == 'code_search':
            decision['use_rag'] = True

        elif intent == 'graph_query':
            decision['use_graph'] = True
            decision['use_rag'] = True  # Fallback if graph not available

        elif intent == 'debugging':
            decision['use_debug_agent'] = True
            decision['use_rag'] = True  # For context

        elif intent == 'code_review':
            decision['use_review_agent'] = True
            decision['use_rag'] = True  # For similar code

        elif intent == 'direct_answer':
            decision['use_direct_llm'] = True

        elif intent == 'documentation':
            decision['use_rag'] = True

        # Apply context rules
        if context and self._has_sufficient_context(context):
            # Override - use direct LLM
            decision['use_direct_llm'] = True
            decision['use_rag'] = False
            logger.info("Overriding to direct LLM due to sufficient context")

        return decision

    def _has_sufficient_context(self, context: Dict) -> bool:
        """
        Check if sufficient context is already available.

        Args:
            context: Context dictionary

        Returns:
            True if sufficient context available
        """
        # Skip RAG if:
        # 1. Many files already in context
        if context.get('files_open', 0) >= 10:
            return True

        # 2. Specific files already loaded
        if context.get('specific_files_loaded', False):
            return True

        # 3. Conversation has established context
        if context.get('turns', 0) >= 3:
            return True

        return False

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if magnitude == 0:
            return 0.0
        return np.dot(vec1, vec2) / magnitude

    def add_route_examples(self, intent: QueryIntent, examples: List[str]):
        """
        Add new examples for a route.

        This allows the system to learn from user interactions.

        Args:
            intent: Intent type
            examples: New example queries
        """
        if intent not in self.routes:
            logger.warning(f"Unknown intent: {intent}")
            return

        # Add examples
        self.routes[intent].extend(examples)

        # Re-compute embeddings
        self.route_embeddings = None  # Force recomputation

        logger.info(f"Added {len(examples)} examples to '{intent}' route")

    def get_stats(self) -> Dict[str, any]:
        """
        Get router statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'num_routes': len(self.routes),
            'total_examples': sum(len(examples) for examples in self.routes.values()),
            'examples_per_route': {
                intent: len(examples)
                for intent, examples in self.routes.items()
            },
            'threshold': self.threshold,
            'model': self.model_name
        }


class AutoTriggerConfig:
    """
    Configuration for auto-triggering behavior.
    """

    def __init__(
        self,
        enabled: bool = True,
        threshold: float = 0.7,
        show_activity: bool = True,
        max_context_tokens: int = 8000
    ):
        """
        Initialize auto-trigger configuration.

        Args:
            enabled: Enable auto-triggering
            threshold: Confidence threshold for triggering
            show_activity: Show activity indicators to user
            max_context_tokens: Maximum context size
        """
        self.enabled = enabled
        self.threshold = threshold
        self.show_activity = show_activity
        self.max_context_tokens = max_context_tokens


# Convenience function
def create_intent_router(threshold: float = 0.7) -> IntentRouter:
    """
    Create and initialize intent router.

    Args:
        threshold: Confidence threshold

    Returns:
        Initialized router
    """
    return IntentRouter(threshold=threshold)
