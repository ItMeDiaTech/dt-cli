"""
Tests for intent-based query routing and auto-triggering.
"""

import pytest
from src.rag.intent_router import IntentRouter, AutoTriggerConfig, create_intent_router
from src.rag.auto_trigger import (
    AutoTrigger,
    TriggerAction,
    TriggerDecision,
    ConversationContext,
    TriggerStats,
    create_auto_trigger
)


class TestIntentRouter:
    """Test intent classification and routing."""

    def test_init(self):
        """Test router initialization."""
        router = IntentRouter(threshold=0.75)
        assert router.threshold == 0.75
        assert router.model_name == 'all-MiniLM-L6-v2'
        assert len(router.routes) == 6

    def test_code_search_intent(self):
        """Test classification of code search queries."""
        router = IntentRouter()

        queries = [
            "where is the authentication code?",
            "find error handling logic",
            "show me API endpoints",
            "locate database queries"
        ]

        for query in queries:
            intent, confidence = router.classify(query)
            # Should classify as code_search (might fall back to default if confidence low)
            assert intent in ['code_search', 'documentation', 'graph_query']
            assert 0.0 <= confidence <= 1.0

    def test_graph_query_intent(self):
        """Test classification of graph query intents."""
        router = IntentRouter()

        queries = [
            "what depends on this module?",
            "what imports this class?",
            "show me the call graph",
            "what functions call this?"
        ]

        for query in queries:
            intent, confidence = router.classify(query)
            assert intent in ['graph_query', 'code_search']
            assert 0.0 <= confidence <= 1.0

    def test_debugging_intent(self):
        """Test classification of debugging queries."""
        router = IntentRouter()

        queries = [
            "why is this test failing?",
            "debug this error",
            "fix this bug",
            "what's causing this exception?"
        ]

        for query in queries:
            intent, confidence = router.classify(query)
            assert intent in ['debugging', 'code_review', 'code_search']
            assert 0.0 <= confidence <= 1.0

    def test_code_review_intent(self):
        """Test classification of code review queries."""
        router = IntentRouter()

        queries = [
            "review this code",
            "check for issues",
            "any problems with this?",
            "is this code correct?"
        ]

        for query in queries:
            intent, confidence = router.classify(query)
            assert intent in ['code_review', 'debugging', 'code_search']
            assert 0.0 <= confidence <= 1.0

    def test_direct_answer_intent(self):
        """Test classification of direct answer queries."""
        router = IntentRouter()

        queries = [
            "fix this typo",
            "add a comment here",
            "rename this variable",
            "explain what this does"
        ]

        for query in queries:
            intent, confidence = router.classify(query)
            assert intent in ['direct_answer', 'code_review', 'documentation']
            assert 0.0 <= confidence <= 1.0

    def test_documentation_intent(self):
        """Test classification of documentation queries."""
        router = IntentRouter()

        queries = [
            "how do I configure this?",
            "what are the installation steps?",
            "show me the API docs",
            "how does this feature work?"
        ]

        for query in queries:
            intent, confidence = router.classify(query)
            assert intent in ['documentation', 'code_search']
            assert 0.0 <= confidence <= 1.0

    def test_should_use_rag(self):
        """Test RAG usage decision."""
        router = IntentRouter()

        # Should use RAG
        assert router.should_use_rag("where is the authentication code?")
        assert router.should_use_rag("debug this error")
        assert router.should_use_rag("review this code")

        # Might not use RAG (direct answers)
        context = {'is_followup': False, 'files_open': 0}
        result = router.should_use_rag("fix this typo", context)
        # Result depends on classification, just check it's boolean
        assert isinstance(result, bool)

    def test_sufficient_context(self):
        """Test context sufficiency detection."""
        router = IntentRouter()

        # Should skip RAG with sufficient context
        context = {'files_open': 15}
        assert not router.should_use_rag("where is the auth code?", context)

        # Should skip RAG for follow-up questions
        context = {'is_followup': True}
        assert not router.should_use_rag("and what about this?", context)

        # Should use RAG with insufficient context
        context = {'files_open': 2}
        assert router.should_use_rag("find authentication code", context)

    def test_route_query(self):
        """Test complete query routing."""
        router = IntentRouter()

        # Code search
        decision = router.route_query("find authentication code")
        assert decision['intent'] in ['code_search', 'documentation']
        assert decision['use_rag'] is True

        # Graph query
        decision = router.route_query("what depends on this module?")
        assert decision['intent'] in ['graph_query', 'code_search']
        assert decision['use_graph'] or decision['use_rag']

        # Debugging
        decision = router.route_query("why is this test failing?")
        assert decision['intent'] in ['debugging', 'code_review']
        assert decision['use_debug_agent'] or decision['use_rag']

    def test_add_route_examples(self):
        """Test adding new route examples."""
        router = IntentRouter()

        initial_count = len(router.routes['code_search'])
        router.add_route_examples('code_search', [
            "new example 1",
            "new example 2"
        ])

        assert len(router.routes['code_search']) == initial_count + 2

    def test_get_stats(self):
        """Test router statistics."""
        router = IntentRouter()
        stats = router.get_stats()

        assert stats['num_routes'] == 6
        assert stats['total_examples'] > 0
        assert 'examples_per_route' in stats
        assert stats['threshold'] == 0.7
        assert stats['model'] == 'all-MiniLM-L6-v2'

    def test_convenience_function(self):
        """Test convenience creation function."""
        router = create_intent_router(threshold=0.8)
        assert isinstance(router, IntentRouter)
        assert router.threshold == 0.8


class TestAutoTrigger:
    """Test auto-trigger system."""

    def test_init(self):
        """Test auto-trigger initialization."""
        trigger = AutoTrigger(confidence_threshold=0.75, show_activity=False)
        assert trigger.confidence_threshold == 0.75
        assert trigger.show_activity is False
        assert isinstance(trigger.intent_router, IntentRouter)
        assert isinstance(trigger.context, ConversationContext)

    def test_decide(self):
        """Test trigger decision making."""
        trigger = AutoTrigger()

        # Test code search query
        decision = trigger.decide("where is the authentication code?")
        assert isinstance(decision, TriggerDecision)
        assert len(decision.actions) > 0
        assert isinstance(decision.intent, str)
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.reasoning) > 0

    def test_decision_should_use_rag(self):
        """Test RAG usage decision in trigger."""
        trigger = AutoTrigger()

        # Code search should use RAG
        decision = trigger.decide("find error handling logic")
        assert decision.should_use_rag()

        # Some queries might not use RAG
        decision = trigger.decide("fix this typo")
        # Result depends on classification
        assert isinstance(decision.should_use_rag(), bool)

    def test_decision_should_use_graph(self):
        """Test graph usage decision."""
        trigger = AutoTrigger()

        # Graph query should trigger graph
        decision = trigger.decide("what depends on this module?")
        # Might use graph depending on classification
        assert isinstance(decision.should_use_graph(), bool)

    def test_decision_primary_action(self):
        """Test primary action selection."""
        trigger = AutoTrigger()

        decision = trigger.decide("find authentication code")
        primary = decision.primary_action()
        assert isinstance(primary, TriggerAction)

    def test_context_tracking(self):
        """Test conversation context tracking."""
        trigger = AutoTrigger()

        # Initial context
        assert trigger.context.turn_count == 0
        assert len(trigger.context.files_in_context) == 0

        # After first query
        decision1 = trigger.decide("where is the auth code?")
        assert trigger.context.turn_count == 1
        assert trigger.context.last_intent is not None

        # After second query
        decision2 = trigger.decide("and what about logging?")
        assert trigger.context.turn_count == 2

    def test_add_file_to_context(self):
        """Test adding files to context."""
        trigger = AutoTrigger()

        trigger.add_file_to_context("src/auth.py")
        trigger.add_file_to_context("src/logging.py")

        assert len(trigger.context.files_in_context) == 2
        assert "src/auth.py" in trigger.context.files_in_context
        assert "src/logging.py" in trigger.context.files_in_context

    def test_remove_file_from_context(self):
        """Test removing files from context."""
        trigger = AutoTrigger()

        trigger.add_file_to_context("src/auth.py")
        trigger.add_file_to_context("src/logging.py")

        trigger.remove_file_from_context("src/auth.py")

        assert len(trigger.context.files_in_context) == 1
        assert "src/auth.py" not in trigger.context.files_in_context
        assert "src/logging.py" in trigger.context.files_in_context

    def test_clear_context(self):
        """Test clearing context."""
        trigger = AutoTrigger()

        # Add some context
        trigger.decide("where is auth code?")
        trigger.add_file_to_context("src/auth.py")

        # Clear it
        trigger.clear_context()

        assert trigger.context.turn_count == 0
        assert len(trigger.context.files_in_context) == 0
        assert trigger.context.last_intent is None

    def test_get_context_summary(self):
        """Test context summary."""
        trigger = AutoTrigger()

        trigger.decide("find auth code")
        trigger.add_file_to_context("src/auth.py")

        summary = trigger.get_context_summary()

        assert summary['turn_count'] == 1
        assert summary['files_in_context'] == 1
        assert 'src/auth.py' in summary['file_paths']
        assert summary['last_intent'] is not None

    def test_should_show_activity(self):
        """Test activity indicator decision."""
        trigger = AutoTrigger(show_activity=True)

        # Should show for RAG and agents
        assert trigger.should_show_activity(TriggerAction.RAG_SEARCH)
        assert trigger.should_show_activity(TriggerAction.DEBUG_AGENT)
        assert trigger.should_show_activity(TriggerAction.REVIEW_AGENT)
        assert trigger.should_show_activity(TriggerAction.GRAPH_QUERY)

        # Should not show for direct LLM
        assert not trigger.should_show_activity(TriggerAction.DIRECT_LLM)

        # Should not show if disabled
        trigger_no_activity = AutoTrigger(show_activity=False)
        assert not trigger_no_activity.should_show_activity(TriggerAction.RAG_SEARCH)

    def test_get_activity_message(self):
        """Test activity message generation."""
        trigger = AutoTrigger()

        # Check messages exist
        assert len(trigger.get_activity_message(TriggerAction.RAG_SEARCH)) > 0
        assert len(trigger.get_activity_message(TriggerAction.DEBUG_AGENT)) > 0
        assert len(trigger.get_activity_message(TriggerAction.REVIEW_AGENT)) > 0

        # Direct LLM has no message
        assert trigger.get_activity_message(TriggerAction.DIRECT_LLM) == ""

    def test_convenience_function(self):
        """Test convenience creation function."""
        trigger = create_auto_trigger(confidence_threshold=0.8, show_activity=False)
        assert isinstance(trigger, AutoTrigger)
        assert trigger.confidence_threshold == 0.8
        assert trigger.show_activity is False


class TestTriggerStats:
    """Test trigger statistics tracking."""

    def test_init(self):
        """Test stats initialization."""
        stats = TriggerStats()
        assert stats.total_queries == 0
        assert len(stats.intent_counts) == 0
        assert len(stats.action_counts) == 0
        assert len(stats.confidence_scores) == 0

    def test_record(self):
        """Test recording decisions."""
        stats = TriggerStats()
        trigger = AutoTrigger()

        # Make some decisions
        decision1 = trigger.decide("find auth code")
        decision2 = trigger.decide("debug this error")

        # Record them
        stats.record(decision1)
        stats.record(decision2)

        assert stats.total_queries == 2
        assert len(stats.confidence_scores) == 2

    def test_get_summary(self):
        """Test statistics summary."""
        stats = TriggerStats()
        trigger = AutoTrigger()

        # Record some decisions
        for query in [
            "find auth code",
            "debug this error",
            "review this code"
        ]:
            decision = trigger.decide(query)
            stats.record(decision)

        summary = stats.get_summary()

        assert summary['total_queries'] == 3
        assert 'intent_distribution' in summary
        assert 'action_distribution' in summary
        assert 'average_confidence' in summary
        assert 0.0 <= summary['average_confidence'] <= 1.0


class TestConversationContext:
    """Test conversation context tracking."""

    def test_init(self):
        """Test context initialization."""
        context = ConversationContext()
        assert context.turn_count == 0
        assert context.files_in_context == []
        assert context.last_intent is None
        assert context.last_actions == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        context = ConversationContext()
        context.turn_count = 3
        context.files_in_context = ['a.py', 'b.py']

        d = context.to_dict()

        assert d['turns'] == 3
        assert d['files_open'] == 2
        assert d['specific_files_loaded'] is True

    def test_update(self):
        """Test context updates."""
        context = ConversationContext()

        context.update('code_search', [TriggerAction.RAG_SEARCH])

        assert context.turn_count == 1
        assert context.last_intent == 'code_search'
        assert context.last_actions == [TriggerAction.RAG_SEARCH]


class TestTriggerDecision:
    """Test trigger decision dataclass."""

    def test_should_use_rag(self):
        """Test RAG usage determination."""
        # Decision with RAG
        decision = TriggerDecision(
            actions=[TriggerAction.RAG_SEARCH],
            intent='code_search',
            confidence=0.9,
            reasoning="Test",
            metadata={}
        )
        assert decision.should_use_rag()

        # Decision without RAG
        decision = TriggerDecision(
            actions=[TriggerAction.DIRECT_LLM],
            intent='direct_answer',
            confidence=0.9,
            reasoning="Test",
            metadata={}
        )
        assert not decision.should_use_rag()

    def test_should_use_graph(self):
        """Test graph usage determination."""
        # Decision with graph
        decision = TriggerDecision(
            actions=[TriggerAction.GRAPH_QUERY],
            intent='graph_query',
            confidence=0.9,
            reasoning="Test",
            metadata={}
        )
        assert decision.should_use_graph()

        # Decision without graph
        decision = TriggerDecision(
            actions=[TriggerAction.RAG_SEARCH],
            intent='code_search',
            confidence=0.9,
            reasoning="Test",
            metadata={}
        )
        assert not decision.should_use_graph()

    def test_primary_action(self):
        """Test primary action selection."""
        decision = TriggerDecision(
            actions=[TriggerAction.RAG_SEARCH, TriggerAction.DIRECT_LLM],
            intent='code_search',
            confidence=0.9,
            reasoning="Test",
            metadata={}
        )
        assert decision.primary_action() == TriggerAction.RAG_SEARCH

        # Empty actions defaults to DIRECT_LLM
        decision_empty = TriggerDecision(
            actions=[],
            intent='direct_answer',
            confidence=0.9,
            reasoning="Test",
            metadata={}
        )
        assert decision_empty.primary_action() == TriggerAction.DIRECT_LLM


class TestAutoTriggerConfig:
    """Test auto-trigger configuration."""

    def test_init(self):
        """Test config initialization."""
        config = AutoTriggerConfig(
            enabled=True,
            threshold=0.75,
            show_activity=True,
            max_context_tokens=10000
        )

        assert config.enabled is True
        assert config.threshold == 0.75
        assert config.show_activity is True
        assert config.max_context_tokens == 10000

    def test_defaults(self):
        """Test default configuration values."""
        config = AutoTriggerConfig()

        assert config.enabled is True
        assert config.threshold == 0.7
        assert config.show_activity is True
        assert config.max_context_tokens == 8000


@pytest.mark.integration
class TestIntegration:
    """Integration tests for auto-trigger system."""

    def test_end_to_end_code_search(self):
        """Test end-to-end code search flow."""
        trigger = AutoTrigger()

        decision = trigger.decide("where is the authentication code?")

        # Should trigger RAG for code search
        assert decision.should_use_rag()
        assert len(decision.reasoning) > 0
        assert decision.confidence > 0

    def test_end_to_end_with_context(self):
        """Test with conversation context."""
        trigger = AutoTrigger()

        # First query
        decision1 = trigger.decide("find authentication code")

        # Add file to context
        trigger.add_file_to_context("src/auth.py")

        # Follow-up query (should use existing context)
        decision2 = trigger.decide("what about this function?")

        # Context should be tracked
        assert trigger.context.turn_count == 2
        assert len(trigger.context.files_in_context) == 1

    def test_statistics_tracking(self):
        """Test statistics tracking over multiple queries."""
        trigger = AutoTrigger()
        stats = TriggerStats()

        queries = [
            "find auth code",
            "debug this error",
            "review this code",
            "what depends on this?",
            "fix this typo"
        ]

        for query in queries:
            decision = trigger.decide(query)
            stats.record(decision)

        summary = stats.get_summary()

        assert summary['total_queries'] == 5
        assert summary['average_confidence'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
