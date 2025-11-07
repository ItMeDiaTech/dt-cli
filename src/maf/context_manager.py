"""
Context Manager for managing agent context and state.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for agent execution."""
    query: str
    task_type: str
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class ContextManager:
    """
    Manages context and state for multi-agent orchestration.
    """

    def __init__(self):
        """Initialize the context manager."""
        self.contexts: Dict[str, AgentContext] = {}
        logger.info("ContextManager initialized")

    def create_context(
        self,
        context_id: str,
        query: str,
        task_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """
        Create a new context.

        Args:
            context_id: Unique context identifier
            query: Query or task description
            task_type: Type of task
            metadata: Optional metadata

        Returns:
            Created context
        """
        context = AgentContext(
            query=query,
            task_type=task_type,
            metadata=metadata or {}
        )

        self.contexts[context_id] = context
        logger.info(f"Created context: {context_id}")

        return context

    def get_context(self, context_id: str) -> Optional[AgentContext]:
        """
        Get a context by ID.

        Args:
            context_id: Context identifier

        Returns:
            Context or None if not found
        """
        return self.contexts.get(context_id)

    def update_results(
        self,
        context_id: str,
        agent_name: str,
        results: Any
    ):
        """
        Update context with agent results.

        Args:
            context_id: Context identifier
            agent_name: Name of the agent
            results: Results from the agent
        """
        context = self.get_context(context_id)
        if context:
            context.results[agent_name] = results

            # Add to history
            context.history.append({
                'agent': agent_name,
                'timestamp': datetime.now().isoformat(),
                'results': results
            })

            logger.debug(f"Updated context {context_id} with results from {agent_name}")

    def merge_results(self, context_id: str) -> Dict[str, Any]:
        """
        Merge all agent results for a context.

        Args:
            context_id: Context identifier

        Returns:
            Merged results
        """
        context = self.get_context(context_id)
        if not context:
            return {}

        merged = {
            'query': context.query,
            'task_type': context.task_type,
            'agent_results': context.results,
            'history': context.history,
            'created_at': context.created_at.isoformat()
        }

        return merged

    def clear_context(self, context_id: str):
        """
        Clear a context.

        Args:
            context_id: Context identifier
        """
        if context_id in self.contexts:
            del self.contexts[context_id]
            logger.info(f"Cleared context: {context_id}")

    def get_active_contexts(self) -> List[str]:
        """
        Get list of active context IDs.

        Returns:
            List of context IDs
        """
        return list(self.contexts.keys())
