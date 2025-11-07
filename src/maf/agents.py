"""
Agent implementations for the Multi-Agent Framework.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(self, name: str, rag_engine: Any = None):
        """
        Initialize the agent.

        Args:
            name: Agent name
            rag_engine: RAG query engine for information retrieval
        """
        self.name = name
        self.rag_engine = rag_engine
        logger.info(f"Agent initialized: {name}")

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.

        Args:
            context: Execution context

        Returns:
            Agent results
        """
        pass


class CodeAnalyzerAgent(BaseAgent):
    """
    Agent that analyzes code structure and patterns.
    """

    def __init__(self, rag_engine: Any = None):
        super().__init__(name="CodeAnalyzer", rag_engine=rag_engine)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze code based on the query.

        Args:
            context: Execution context with query

        Returns:
            Analysis results
        """
        query = context.get('query', '')
        logger.info(f"CodeAnalyzer analyzing: {query}")

        results = {
            'agent': self.name,
            'analysis_type': 'code_structure',
            'findings': []
        }

        if self.rag_engine:
            # Query RAG for code patterns
            rag_results = self.rag_engine.query(
                query_text=query,
                n_results=5,
                file_type='.py'  # Focus on Python files
            )

            results['findings'] = [
                {
                    'file': r['metadata'].get('file_path', 'unknown'),
                    'relevance_score': 1 - r['distance'],
                    'snippet': r['text'][:200]
                }
                for r in rag_results
            ]

        logger.info(f"CodeAnalyzer found {len(results['findings'])} code patterns")
        return results


class DocumentationRetrieverAgent(BaseAgent):
    """
    Agent that retrieves relevant documentation.
    """

    def __init__(self, rag_engine: Any = None):
        super().__init__(name="DocumentationRetriever", rag_engine=rag_engine)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant documentation.

        Args:
            context: Execution context with query

        Returns:
            Documentation results
        """
        query = context.get('query', '')
        logger.info(f"DocumentationRetriever searching: {query}")

        results = {
            'agent': self.name,
            'documentation': []
        }

        if self.rag_engine:
            # Query RAG for documentation files
            rag_results = self.rag_engine.query(
                query_text=query,
                n_results=5,
                file_type='.md'  # Focus on markdown docs
            )

            results['documentation'] = [
                {
                    'file': r['metadata'].get('file_path', 'unknown'),
                    'relevance_score': 1 - r['distance'],
                    'content': r['text'][:300]
                }
                for r in rag_results
            ]

        logger.info(f"DocumentationRetriever found {len(results['documentation'])} docs")
        return results


class ContextSynthesizerAgent(BaseAgent):
    """
    Agent that synthesizes context from multiple sources.
    """

    def __init__(self, rag_engine: Any = None):
        super().__init__(name="ContextSynthesizer", rag_engine=rag_engine)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize context from multiple agent results.

        Args:
            context: Execution context with agent results

        Returns:
            Synthesized context
        """
        query = context.get('query', '')
        agent_results = context.get('agent_results', {})

        logger.info(f"ContextSynthesizer synthesizing results for: {query}")

        # Combine results from other agents
        code_findings = []
        documentation = []

        if 'CodeAnalyzer' in agent_results:
            code_findings = agent_results['CodeAnalyzer'].get('findings', [])

        if 'DocumentationRetriever' in agent_results:
            documentation = agent_results['DocumentationRetriever'].get('documentation', [])

        # Create synthesized context
        synthesis = {
            'agent': self.name,
            'query': query,
            'code_context': {
                'total_findings': len(code_findings),
                'top_files': [f['file'] for f in code_findings[:3]]
            },
            'documentation_context': {
                'total_docs': len(documentation),
                'top_docs': [d['file'] for d in documentation[:3]]
            },
            'summary': self._generate_summary(query, code_findings, documentation)
        }

        logger.info("ContextSynthesizer completed synthesis")
        return synthesis

    def _generate_summary(
        self,
        query: str,
        code_findings: List[Dict],
        documentation: List[Dict]
    ) -> str:
        """Generate a summary of the synthesized context."""
        summary_parts = [f"Query: {query}"]

        if code_findings:
            summary_parts.append(
                f"Found {len(code_findings)} relevant code patterns"
            )

        if documentation:
            summary_parts.append(
                f"Found {len(documentation)} relevant documentation files"
            )

        return " | ".join(summary_parts)


class SuggestionGeneratorAgent(BaseAgent):
    """
    Agent that generates context-aware suggestions.
    """

    def __init__(self, rag_engine: Any = None):
        super().__init__(name="SuggestionGenerator", rag_engine=rag_engine)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate suggestions based on synthesized context.

        Args:
            context: Execution context with synthesized results

        Returns:
            Generated suggestions
        """
        query = context.get('query', '')
        agent_results = context.get('agent_results', {})

        logger.info(f"SuggestionGenerator generating suggestions for: {query}")

        synthesis = agent_results.get('ContextSynthesizer', {})

        suggestions = {
            'agent': self.name,
            'suggestions': [],
            'related_files': [],
            'next_steps': []
        }

        # Extract related files
        code_context = synthesis.get('code_context', {})
        doc_context = synthesis.get('documentation_context', {})

        suggestions['related_files'] = (
            code_context.get('top_files', []) +
            doc_context.get('top_docs', [])
        )

        # Generate suggestions based on context
        if code_context.get('total_findings', 0) > 0:
            suggestions['suggestions'].append(
                "Review the identified code patterns for implementation details"
            )

        if doc_context.get('total_docs', 0) > 0:
            suggestions['suggestions'].append(
                "Check the documentation for API usage and best practices"
            )

        # Generate next steps
        suggestions['next_steps'] = [
            "Examine the top related files",
            "Consider the patterns found in the codebase",
            "Review relevant documentation"
        ]

        logger.info(f"SuggestionGenerator created {len(suggestions['suggestions'])} suggestions")
        return suggestions
