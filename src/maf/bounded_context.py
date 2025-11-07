"""
Bounded context manager with memory limits.
"""

from collections import OrderedDict
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


class BoundedContextManager:
    """
    Context manager with bounded memory (LRU eviction).
    """

    def __init__(self, max_contexts: int = 1000):
        """
        Initialize bounded context manager.

        Args:
            max_contexts: Maximum number of contexts to keep
        """
        self.contexts: OrderedDict[str, AgentContext] = OrderedDict()
        self.max_contexts = max_contexts
        self.eviction_count = 0

    def create_context(
        self,
        context_id: str,
        query: str,
        task_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """
        Create a new context with automatic eviction.

        Args:
            context_id: Unique context identifier
            query: Query or task description
            task_type: Type of task
            metadata: Optional metadata

        Returns:
            Created context
        """
        # Evict oldest if at capacity
        if len(self.contexts) >= self.max_contexts:
            oldest_id = next(iter(self.contexts))
            del self.contexts[oldest_id]
            self.eviction_count += 1
            logger.debug(f"Evicted context: {oldest_id} (total evictions: {self.eviction_count})")

        context = AgentContext(
            query=query,
            task_type=task_type,
            metadata=metadata or {}
        )

        self.contexts[context_id] = context
        logger.info(f"Created context: {context_id} (total: {len(self.contexts)})")

        return context

    def get_context(self, context_id: str) -> Optional[AgentContext]:
        """
        Get a context by ID (moves to end for LRU).

        Args:
            context_id: Context identifier

        Returns:
            Context or None if not found
        """
        if context_id in self.contexts:
            # Move to end (most recently used)
            self.contexts.move_to_end(context_id)
            return self.contexts[context_id]
        return None

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
        Clear a specific context.

        Args:
            context_id: Context identifier
        """
        if context_id in self.contexts:
            del self.contexts[context_id]
            logger.info(f"Cleared context: {context_id}")

    def clear_old_contexts(self, max_age_seconds: int = 3600):
        """
        Clear contexts older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds
        """
        now = datetime.now()
        to_remove = []

        for context_id, context in self.contexts.items():
            age = (now - context.created_at).total_seconds()
            if age > max_age_seconds:
                to_remove.append(context_id)

        for context_id in to_remove:
            del self.contexts[context_id]

        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old contexts")

    def get_active_contexts(self) -> List[str]:
        """
        Get list of active context IDs.

        Returns:
            List of context IDs
        """
        return list(self.contexts.keys())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about context usage.

        Returns:
            Dictionary with stats
        """
        return {
            'active_contexts': len(self.contexts),
            'max_contexts': self.max_contexts,
            'eviction_count': self.eviction_count,
            'utilization': f"{len(self.contexts) / self.max_contexts * 100:.1f}%"
        }
