"""
Multi-Agent Framework (MAF) for intelligent task orchestration.
"""

from .orchestrator import AgentOrchestrator
from .agents import (
    CodeAnalyzerAgent,
    DocumentationRetrieverAgent,
    ContextSynthesizerAgent,
    SuggestionGeneratorAgent
)
from .context_manager import ContextManager

__all__ = [
    'AgentOrchestrator',
    'CodeAnalyzerAgent',
    'DocumentationRetrieverAgent',
    'ContextSynthesizerAgent',
    'SuggestionGeneratorAgent',
    'ContextManager'
]
