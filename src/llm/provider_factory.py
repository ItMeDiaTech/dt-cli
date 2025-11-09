"""
LLM Provider Factory.

Creates and manages LLM provider instances based on configuration.
"""

from typing import Dict, Any, Optional
import logging

from .base_provider import BaseLLMProvider
from .ollama_provider import OllamaProvider
from .vllm_provider import VLLMProvider
from .claude_provider import ClaudeProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Supports:
    - ollama: Local development (recommended, 100% open source)
    - vllm: Production deployment (recommended, 100% open source)
    - claude: Optional Anthropic integration
    """

    _providers = {
        'ollama': OllamaProvider,
        'vllm': VLLMProvider,
        'claude': ClaudeProvider
    }

    @classmethod
    def create(
        cls,
        provider_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_type: Provider type ('ollama', 'vllm', 'claude')
            config: Provider configuration

        Returns:
            Initialized provider instance

        Raises:
            ValueError: If provider_type is unknown
        """
        if config is None:
            config = {}

        provider_type = provider_type.lower()

        if provider_type not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available providers: {available}"
            )

        provider_class = cls._providers[provider_type]

        logger.info(f"Creating {provider_type} provider")

        try:
            provider = provider_class(config)

            # Check health on creation
            if not provider.check_health():
                logger.warning(
                    f"{provider_type} provider created but health check failed. "
                    f"The provider may not be accessible."
                )

            return provider

        except Exception as e:
            logger.error(f"Failed to create {provider_type} provider: {e}")
            raise

    @classmethod
    def create_from_config(
        cls,
        config: Dict[str, Any]
    ) -> BaseLLMProvider:
        """
        Create provider from configuration dictionary.

        Config format:
        {
            'provider': 'ollama',  # or 'vllm', 'claude'
            'model_name': 'qwen3-coder',
            'base_url': 'http://localhost:11434',
            'temperature': 0.1,
            ...
        }

        Args:
            config: Configuration dictionary

        Returns:
            Initialized provider instance
        """
        provider_type = config.get('provider', 'ollama')

        # Log if using non-open-source provider
        if provider_type == 'claude':
            logger.warning(
                "Using Claude provider (proprietary). "
                "dt-cli works best with open source providers like Ollama or vLLM."
            )

        return cls.create(provider_type, config)

    @classmethod
    def get_default_config(cls, provider_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a provider type.

        Args:
            provider_type: Provider type

        Returns:
            Default configuration dictionary
        """
        defaults = {
            'ollama': {
                'provider': 'ollama',
                'model_name': 'qwen3-coder',
                'base_url': 'http://localhost:11434',
                'temperature': 0.1,
                'max_tokens': 4096,
                'timeout': 60
            },
            'vllm': {
                'provider': 'vllm',
                'model_name': 'qwen3-coder',
                'base_url': 'http://localhost:8000',
                'api_key': 'not-needed',
                'temperature': 0.1,
                'max_tokens': 4096,
                'timeout': 120
            },
            'claude': {
                'provider': 'claude',
                'model_name': 'claude-sonnet-4.5',
                'api_key': '<your-api-key>',
                'temperature': 0.1,
                'max_tokens': 4096,
                'timeout': 60
            }
        }

        return defaults.get(provider_type.lower(), {})

    @classmethod
    def get_recommended_provider(cls) -> str:
        """
        Get the recommended provider for the current environment.

        Returns:
            Recommended provider name
        """
        # Check if Ollama is running locally
        try:
            ollama_config = cls.get_default_config('ollama')
            provider = OllamaProvider(ollama_config)
            if provider.check_health():
                logger.info("Detected running Ollama instance - recommended for use")
                return 'ollama'
        except Exception:
            pass

        # Check if vLLM is running
        try:
            vllm_config = cls.get_default_config('vllm')
            provider = VLLMProvider(vllm_config)
            if provider.check_health():
                logger.info("Detected running vLLM instance - recommended for use")
                return 'vllm'
        except Exception:
            pass

        # Default to Ollama (user needs to install)
        logger.warning(
            "No LLM provider detected. "
            "Please install Ollama: https://ollama.com/download"
        )
        return 'ollama'

    @classmethod
    def list_available_providers(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available providers and their status.

        Returns:
            Dictionary of provider information
        """
        providers_info = {}

        for provider_type in cls._providers.keys():
            try:
                config = cls.get_default_config(provider_type)
                provider_class = cls._providers[provider_type]
                provider = provider_class(config)

                providers_info[provider_type] = {
                    'available': provider.check_health(),
                    'info': provider.get_info(),
                    'recommended': provider_type in ['ollama', 'vllm']
                }
            except Exception as e:
                providers_info[provider_type] = {
                    'available': False,
                    'error': str(e),
                    'recommended': provider_type in ['ollama', 'vllm']
                }

        return providers_info
