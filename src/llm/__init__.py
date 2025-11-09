"""
LLM Provider abstraction layer for dt-cli.

Supports multiple LLM backends:
- Ollama (recommended for development, 100% open source)
- vLLM (recommended for production, 100% open source)
- Claude Code (optional, for users who prefer Anthropic)
"""

from .base_provider import BaseLLMProvider
from .ollama_provider import OllamaProvider
from .vllm_provider import VLLMProvider
from .claude_provider import ClaudeProvider
from .provider_factory import LLMProviderFactory

__all__ = [
    'BaseLLMProvider',
    'OllamaProvider',
    'VLLMProvider',
    'ClaudeProvider',
    'LLMProviderFactory'
]
