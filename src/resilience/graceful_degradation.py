"""
Graceful Degradation - Fallback strategies for component failures.
"""

from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class DegradationLevel:
    """Degradation levels for system components."""
    FULL = "full"           # All features working
    DEGRADED = "degraded"   # Some features disabled
    MINIMAL = "minimal"     # Only core features
    FAILED = "failed"       # Component failed


class GracefulDegradation:
    """
    Manages graceful degradation of system components.
    """

    def __init__(self):
        """Initialize graceful degradation manager."""
        self.component_status: Dict[str, DegradationLevel] = {}
        self.fallback_strategies: Dict[str, Callable] = {}

    def with_fallback(self, component_name: str, fallback_func: Optional[Callable] = None):
        """
        Decorator for functions with fallback behavior.

        Args:
            component_name: Name of the component
            fallback_func: Optional fallback function

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    self._mark_success(component_name)
                    return result
                except Exception as e:
                    logger.warning(
                        f"Component {component_name} failed: {e}, using fallback"
                    )
                    self._mark_degraded(component_name)

                    if fallback_func:
                        try:
                            return fallback_func(*args, **kwargs)
                        except Exception as e2:
                            logger.error(f"Fallback for {component_name} also failed: {e2}")
                            self._mark_failed(component_name)
                            raise

                    # Return empty/default result if no fallback
                    return self._get_default_result(func)

            return wrapper
        return decorator

    def _mark_success(self, component_name: str):
        """Mark component as fully operational."""
        self.component_status[component_name] = DegradationLevel.FULL

    def _mark_degraded(self, component_name: str):
        """Mark component as degraded."""
        self.component_status[component_name] = DegradationLevel.DEGRADED

    def _mark_failed(self, component_name: str):
        """Mark component as failed."""
        self.component_status[component_name] = DegradationLevel.FAILED

    def _get_default_result(self, func: Callable) -> Any:
        """Get default result based on function return type."""
        return_annotation = func.__annotations__.get('return')

        if return_annotation == list or return_annotation == List:
            return []
        elif return_annotation == dict or return_annotation == Dict:
            return {}
        elif return_annotation == str:
            return ""
        elif return_annotation == int:
            return 0
        elif return_annotation == bool:
            return False
        else:
            return None

    def get_system_level(self) -> DegradationLevel:
        """
        Get overall system degradation level.

        Returns:
            System degradation level
        """
        if not self.component_status:
            return DegradationLevel.FULL

        failed_count = sum(
            1 for status in self.component_status.values()
            if status == DegradationLevel.FAILED
        )

        degraded_count = sum(
            1 for status in self.component_status.values()
            if status == DegradationLevel.DEGRADED
        )

        total = len(self.component_status)

        if failed_count >= total / 2:
            return DegradationLevel.FAILED
        elif failed_count > 0 or degraded_count >= total / 2:
            return DegradationLevel.DEGRADED
        elif degraded_count > 0:
            return DegradationLevel.MINIMAL
        else:
            return DegradationLevel.FULL

    def get_status(self) -> Dict[str, Any]:
        """
        Get degradation status.

        Returns:
            Status dictionary
        """
        return {
            'system_level': self.get_system_level(),
            'components': self.component_status.copy()
        }


# Global instance
degradation_manager = GracefulDegradation()


# Fallback implementations for common operations
def fallback_semantic_search(query_text: str, **kwargs) -> List[Dict]:
    """Fallback for semantic search - return empty results."""
    logger.info("Using fallback: semantic search unavailable")
    return []


def fallback_hybrid_search(query_text: str, **kwargs) -> List[Dict]:
    """Fallback for hybrid search - use semantic only."""
    logger.info("Using fallback: hybrid search → semantic only")
    # This would call the semantic search function
    return []


def fallback_reranking(query: str, results: List[Dict], **kwargs) -> List[Dict]:
    """Fallback for reranking - return original results."""
    logger.info("Using fallback: reranking unavailable, using original order")
    return results


def fallback_query_expansion(query: str, **kwargs) -> List[str]:
    """Fallback for query expansion - return original query only."""
    logger.info("Using fallback: query expansion unavailable")
    return [query]


def fallback_agent_execution(query: str, **kwargs) -> Dict[str, Any]:
    """Fallback for agent execution - return minimal response."""
    logger.info("Using fallback: agent execution unavailable")
    return {
        'query': query,
        'status': 'degraded',
        'message': 'Advanced analysis unavailable, using basic search'
    }
