"""
Claude Code Provider (Optional).

This provider integrates with Claude Code / Anthropic API.
It is OPTIONAL and only for users who:
- Already have an Anthropic subscription
- Prefer Claude for certain tasks
- Want a hybrid setup (Ollama + Claude)

Note: This is NOT required for dt-cli to function.
The open source providers (Ollama, vLLM) provide equal or better
performance for coding tasks.
"""

from typing import Dict, Any, List, Optional
import logging

from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class ClaudeProvider(BaseLLMProvider):
    """
    Claude Code / Anthropic API provider.

    OPTIONAL: Only use if you have an Anthropic subscription.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Claude provider.

        Config options:
            api_key: Anthropic API key (required)
            model_name: Model to use (default: claude-sonnet-4.5)
            temperature: Sampling temperature (default: 0.1)
            max_tokens: Maximum tokens to generate (default: 4096)
            timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(config)

        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("Claude provider requires 'api_key' in config")

        # Use Claude Sonnet 4.5 as default
        if self.model_name == 'default':
            self.model_name = 'claude-sonnet-4.5'

        logger.warning(
            "Using Claude provider (proprietary). "
            "Consider using Ollama or vLLM for 100% open source setup."
        )

    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Claude API.

        Args:
            prompt: User prompt
            context: Optional RAG context
            system_prompt: Optional system instructions
            **kwargs: Additional Claude parameters

        Returns:
            Generated response
        """
        try:
            # Try to import anthropic
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self.api_key)

        # Build system message
        system_parts = []
        if system_prompt:
            system_parts.append(system_prompt)

        if context:
            system_parts.append(self.format_context(context))

        system_message = "\n\n".join(system_parts) if system_parts else None

        # Build user message
        messages = [{
            "role": "user",
            "content": prompt
        }]

        try:
            response = client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                system=system_message,
                messages=messages
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise RuntimeError(f"Failed to generate response from Claude: {e}")

    def generate_streaming(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Generate streaming response using Claude API.

        Args:
            prompt: User prompt
            context: Optional RAG context
            system_prompt: Optional system instructions
            **kwargs: Additional Claude parameters

        Yields:
            Response chunks
        """
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self.api_key)

        # Build system message
        system_parts = []
        if system_prompt:
            system_parts.append(system_prompt)

        if context:
            system_parts.append(self.format_context(context))

        system_message = "\n\n".join(system_parts) if system_parts else None

        messages = [{
            "role": "user",
            "content": prompt
        }]

        try:
            with client.messages.stream(
                model=self.model_name,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                system=system_message,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            raise RuntimeError(f"Failed to stream from Claude: {e}")

    def check_health(self) -> bool:
        """
        Check if Claude API is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            # Simple test request
            client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True

        except Exception as e:
            logger.warning(f"Claude health check failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about Claude provider.

        Returns:
            Provider information
        """
        return {
            'provider': 'claude',
            'model': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'open_source': False,
            'local': False,
            'cost': 'Paid (Anthropic subscription)',
            'note': 'Optional provider - not required for dt-cli'
        }
