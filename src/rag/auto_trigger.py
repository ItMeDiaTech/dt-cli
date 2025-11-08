"""
Auto-triggering system for intelligent RAG/MAF activation.

This module orchestrates automatic activation of:
- RAG retrieval
- Knowledge graph queries
- Debug agents
- Code review agents
- Direct LLM responses

Based on query intent classification and context analysis.
Expected impact: 70% reduction in manual /rag-query commands.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .intent_router import IntentRouter, QueryIntent

logger = logging.getLogger(__name__)


class TriggerAction(Enum):
    """Actions that can be auto-triggered."""
    RAG_SEARCH = "rag_search"
    GRAPH_QUERY = "graph_query"
    DEBUG_AGENT = "debug_agent"
    REVIEW_AGENT = "review_agent"
    DIRECT_LLM = "direct_llm"
    DOCUMENTATION_SEARCH = "documentation_search"


@dataclass
class TriggerDecision:
    """
    Decision made by auto-trigger system.
    """
    actions: List[TriggerAction]
    intent: QueryIntent
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]

    def should_use_rag(self) -> bool:
        """Check if any action requires RAG."""
        rag_actions = {
            TriggerAction.RAG_SEARCH,
            TriggerAction.DEBUG_AGENT,
            TriggerAction.REVIEW_AGENT,
            TriggerAction.DOCUMENTATION_SEARCH
        }
        return any(action in rag_actions for action in self.actions)

    def should_use_graph(self) -> bool:
        """Check if graph query should be used."""
        return TriggerAction.GRAPH_QUERY in self.actions

    def primary_action(self) -> TriggerAction:
        """Get the primary action to execute."""
        return self.actions[0] if self.actions else TriggerAction.DIRECT_LLM


@dataclass
class ConversationContext:
    """
    Tracks conversation context for better decision-making.
    """
    turn_count: int = 0
    files_in_context: List[str] = None
    last_intent: Optional[QueryIntent] = None
    last_actions: List[TriggerAction] = None
    user_preferences: Dict[str, Any] = None

    def __post_init__(self):
        if self.files_in_context is None:
            self.files_in_context = []
        if self.last_actions is None:
            self.last_actions = []
        if self.user_preferences is None:
            self.user_preferences = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for intent router."""
        return {
            'turns': self.turn_count,
            'files_open': len(self.files_in_context),
            'specific_files_loaded': len(self.files_in_context) > 0,
            'is_followup': self.turn_count > 0 and self.last_intent is not None
        }

    def update(self, intent: QueryIntent, actions: List[TriggerAction]):
        """Update context after a query."""
        self.turn_count += 1
        self.last_intent = intent
        self.last_actions = actions


class AutoTrigger:
    """
    Automatic triggering system that decides when to activate RAG/MAF components.

    This is the main orchestrator for intelligent auto-activation.
    """

    def __init__(
        self,
        intent_router: Optional[IntentRouter] = None,
        confidence_threshold: float = 0.7,
        show_activity: bool = True
    ):
        """
        Initialize auto-trigger system.

        Args:
            intent_router: Intent router instance (creates one if None)
            confidence_threshold: Minimum confidence for auto-triggering
            show_activity: Show activity indicators to user
        """
        self.intent_router = intent_router or IntentRouter(threshold=confidence_threshold)
        self.confidence_threshold = confidence_threshold
        self.show_activity = show_activity
        self.context = ConversationContext()

        logger.info(
            f"Initialized AutoTrigger (threshold={confidence_threshold}, "
            f"show_activity={show_activity})"
        )

    def decide(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TriggerDecision:
        """
        Decide which actions to trigger for a query.

        Args:
            query: User query
            context: Optional context override

        Returns:
            TriggerDecision with actions to take
        """
        # Use provided context or conversation context
        if context is None:
            context = self.context.to_dict()

        # Classify intent
        intent, confidence = self.intent_router.classify(query)

        logger.info(
            f"Query classified as '{intent}' with confidence {confidence:.2f}"
        )

        # Get routing decision
        routing = self.intent_router.route_query(query, context)

        # Map routing to actions
        actions = self._routing_to_actions(routing)

        # Build reasoning
        reasoning = self._build_reasoning(intent, confidence, routing, context)

        # Create decision
        decision = TriggerDecision(
            actions=actions,
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'routing': routing,
                'context': context,
                'query': query
            }
        )

        # Update conversation context
        self.context.update(intent, actions)

        logger.info(
            f"Auto-trigger decision: {[a.value for a in actions]} "
            f"(primary: {decision.primary_action().value})"
        )

        return decision

    def _routing_to_actions(self, routing: Dict[str, Any]) -> List[TriggerAction]:
        """
        Convert routing decision to trigger actions.

        Args:
            routing: Routing decision from intent router

        Returns:
            List of actions to trigger (ordered by priority)
        """
        actions = []

        # Priority order matters - determines execution order
        if routing.get('use_debug_agent'):
            actions.append(TriggerAction.DEBUG_AGENT)

        if routing.get('use_review_agent'):
            actions.append(TriggerAction.REVIEW_AGENT)

        if routing.get('use_graph'):
            actions.append(TriggerAction.GRAPH_QUERY)

        if routing.get('use_rag'):
            # Determine specific RAG action based on intent
            intent = routing.get('intent')
            if intent == 'documentation':
                actions.append(TriggerAction.DOCUMENTATION_SEARCH)
            else:
                actions.append(TriggerAction.RAG_SEARCH)

        if routing.get('use_direct_llm') or not actions:
            actions.append(TriggerAction.DIRECT_LLM)

        return actions

    def _build_reasoning(
        self,
        intent: QueryIntent,
        confidence: float,
        routing: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Build human-readable reasoning for the decision.

        Args:
            intent: Classified intent
            confidence: Classification confidence
            routing: Routing decision
            context: Context dictionary

        Returns:
            Reasoning string
        """
        parts = []

        # Intent classification
        parts.append(f"Detected '{intent}' intent (confidence: {confidence:.1%})")

        # Context factors
        if context.get('is_followup'):
            parts.append("Follow-up question in conversation")

        if context.get('specific_files_loaded'):
            parts.append(f"{context['files_open']} files in context")

        # Actions
        actions = []
        if routing.get('use_rag'):
            actions.append("RAG retrieval")
        if routing.get('use_graph'):
            actions.append("knowledge graph")
        if routing.get('use_debug_agent'):
            actions.append("debug agent")
        if routing.get('use_review_agent'):
            actions.append("review agent")
        if routing.get('use_direct_llm'):
            actions.append("direct LLM")

        if actions:
            parts.append(f"Using: {', '.join(actions)}")

        return " | ".join(parts)

    def add_file_to_context(self, file_path: str):
        """
        Add a file to the conversation context.

        Args:
            file_path: Path to file
        """
        if file_path not in self.context.files_in_context:
            self.context.files_in_context.append(file_path)
            logger.debug(f"Added file to context: {file_path}")

    def remove_file_from_context(self, file_path: str):
        """
        Remove a file from the conversation context.

        Args:
            file_path: Path to file
        """
        if file_path in self.context.files_in_context:
            self.context.files_in_context.remove(file_path)
            logger.debug(f"Removed file from context: {file_path}")

    def clear_context(self):
        """Clear conversation context."""
        self.context = ConversationContext()
        logger.info("Cleared conversation context")

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current context.

        Returns:
            Context summary dictionary
        """
        return {
            'turn_count': self.context.turn_count,
            'files_in_context': len(self.context.files_in_context),
            'file_paths': self.context.files_in_context,
            'last_intent': self.context.last_intent,
            'last_actions': [a.value for a in self.context.last_actions] if self.context.last_actions else []
        }

    def should_show_activity(self, action: TriggerAction) -> bool:
        """
        Determine if activity indicator should be shown for an action.

        Args:
            action: Trigger action

        Returns:
            True if activity should be shown
        """
        if not self.show_activity:
            return False

        # Always show for agents and graph queries
        show_for = {
            TriggerAction.DEBUG_AGENT,
            TriggerAction.REVIEW_AGENT,
            TriggerAction.GRAPH_QUERY,
            TriggerAction.RAG_SEARCH,
            TriggerAction.DOCUMENTATION_SEARCH
        }

        return action in show_for

    def get_activity_message(self, action: TriggerAction) -> str:
        """
        Get activity indicator message for an action.

        Args:
            action: Trigger action

        Returns:
            Activity message
        """
        messages = {
            TriggerAction.RAG_SEARCH: "🔍 Searching codebase...",
            TriggerAction.GRAPH_QUERY: "🕸️  Analyzing code relationships...",
            TriggerAction.DEBUG_AGENT: "🐛 Running debug analysis...",
            TriggerAction.REVIEW_AGENT: "✅ Reviewing code quality...",
            TriggerAction.DOCUMENTATION_SEARCH: "📚 Searching documentation...",
            TriggerAction.DIRECT_LLM: ""
        }
        return messages.get(action, "")


class TriggerStats:
    """
    Statistics tracker for auto-triggering system.
    """

    def __init__(self):
        self.total_queries = 0
        self.intent_counts: Dict[QueryIntent, int] = {}
        self.action_counts: Dict[TriggerAction, int] = {}
        self.confidence_scores: List[float] = []

    def record(self, decision: TriggerDecision):
        """
        Record a trigger decision for statistics.

        Args:
            decision: Trigger decision
        """
        self.total_queries += 1

        # Intent counts
        if decision.intent not in self.intent_counts:
            self.intent_counts[decision.intent] = 0
        self.intent_counts[decision.intent] += 1

        # Action counts
        for action in decision.actions:
            if action not in self.action_counts:
                self.action_counts[action] = 0
            self.action_counts[action] += 1

        # Confidence scores
        self.confidence_scores.append(decision.confidence)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get statistics summary.

        Returns:
            Statistics dictionary
        """
        avg_confidence = (
            sum(self.confidence_scores) / len(self.confidence_scores)
            if self.confidence_scores else 0.0
        )

        return {
            'total_queries': self.total_queries,
            'intent_distribution': {
                intent: count for intent, count in self.intent_counts.items()
            },
            'action_distribution': {
                action.value: count for action, count in self.action_counts.items()
            },
            'average_confidence': avg_confidence,
            'confidence_scores': self.confidence_scores
        }


# Convenience function
def create_auto_trigger(
    confidence_threshold: float = 0.7,
    show_activity: bool = True
) -> AutoTrigger:
    """
    Create and initialize auto-trigger system.

    Args:
        confidence_threshold: Minimum confidence for triggering
        show_activity: Show activity indicators

    Returns:
        Initialized AutoTrigger instance
    """
    return AutoTrigger(
        confidence_threshold=confidence_threshold,
        show_activity=show_activity
    )
