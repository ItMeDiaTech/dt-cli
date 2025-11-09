"""
Base LLM Provider interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (Ollama, vLLM, Claude) must implement this interface.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self.model_name = config.get('model_name', 'default')
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4096)
        self.timeout = config.get('timeout', 60)

        logger.info(f"Initialized {self.__class__.__name__} with model {self.model_name}")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            context: Optional retrieved context from RAG
            system_prompt: Optional system prompt
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated response string
        """
        pass

    @abstractmethod
    def generate_streaming(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Generate a streaming response from the LLM.

        Args:
            prompt: User prompt
            context: Optional retrieved context from RAG
            system_prompt: Optional system prompt
            **kwargs: Additional provider-specific parameters

        Yields:
            Response chunks
        """
        pass

    @abstractmethod
    def check_health(self) -> bool:
        """
        Check if the LLM provider is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the provider.

        Returns:
            Provider information dictionary
        """
        pass

    def format_context(self, context: List[str]) -> str:
        """
        Format retrieved context for injection into prompt.

        Args:
            context: List of context strings from RAG

        Returns:
            Formatted context string
        """
        if not context:
            return ""

        formatted = ["## Retrieved Context\n"]
        for i, ctx in enumerate(context, 1):
            formatted.append(f"### Context {i}\n{ctx}\n")

        return "\n".join(formatted)

    def build_full_prompt(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build the full prompt with system prompt and context.

        Args:
            prompt: User prompt
            context: Optional context from RAG
            system_prompt: Optional system prompt

        Returns:
            Full formatted prompt
        """
        parts = []

        if system_prompt:
            parts.append(f"## System Instructions\n{system_prompt}\n")

        if context:
            parts.append(self.format_context(context))

        parts.append(f"## User Query\n{prompt}")

        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
