"""
Tests for Multi-Agent Framework components.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from maf import (
    AgentOrchestrator,
    CodeAnalyzerAgent,
    DocumentationRetrieverAgent,
    ContextManager
)


def test_context_manager():
    """Test context manager."""
    manager = ContextManager()

    # Create context
    context = manager.create_context(
        context_id="test1",
        query="test query",
        task_type="general"
    )

    assert context is not None
    assert context.query == "test query"
    assert context.task_type == "general"

    # Get context
    retrieved = manager.get_context("test1")
    assert retrieved is not None
    assert retrieved.query == "test query"

    # Update results
    manager.update_results("test1", "TestAgent", {"result": "test"})

    merged = manager.merge_results("test1")
    assert "agent_results" in merged
    assert "TestAgent" in merged["agent_results"]


def test_code_analyzer_agent():
    """Test code analyzer agent."""
    agent = CodeAnalyzerAgent()

    context = {"query": "test query"}
    result = agent.execute(context)

    assert result is not None
    assert "agent" in result
    assert result["agent"] == "CodeAnalyzer"
    assert "findings" in result


def test_documentation_retriever_agent():
    """Test documentation retriever agent."""
    agent = DocumentationRetrieverAgent()

    context = {"query": "test query"}
    result = agent.execute(context)

    assert result is not None
    assert "agent" in result
    assert result["agent"] == "DocumentationRetriever"
    assert "documentation" in result


def test_orchestrator_initialization():
    """Test orchestrator initialization."""
    orchestrator = AgentOrchestrator()

    assert orchestrator is not None
    assert orchestrator.agents is not None
    assert len(orchestrator.agents) == 4

    status = orchestrator.get_status()
    assert "agents" in status
    assert "active_contexts" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
