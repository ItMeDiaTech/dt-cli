"""
Ollama LLM Provider.

Ollama is the recommended provider for development:
- 100% open source
- Runs locally on your machine
- Easy to setup (brew/apt install)
- Supports all major open source code models
- Zero cost

Recommended models:
- qwen3-coder (best for agentic coding workflows)
- deepseek-v3 (best for reasoning)
- starcoder2 (fast, good for completion)
"""

from typing import Dict, Any, List, Optional
import httpx
import json
import logging

from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider.

    Connects to local or remote Ollama instance.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Ollama provider.

        Config options:
            model_name: Model to use (e.g., 'qwen3-coder', 'deepseek-v3')
            base_url: Ollama API URL (default: http://localhost:11434)
            temperature: Sampling temperature (default: 0.1)
            max_tokens: Maximum tokens to generate (default: 4096)
            timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(config)

        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"

        # Use recommended model if not specified
        if self.model_name == 'default':
            self.model_name = 'qwen3-coder'
            logger.info(f"Using default Ollama model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Ollama.

        Args:
            prompt: User prompt
            context: Optional RAG context
            system_prompt: Optional system instructions
            **kwargs: Additional Ollama parameters

        Returns:
            Generated response
        """
        full_prompt = self.build_full_prompt(prompt, context, system_prompt)

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": kwargs.get('max_tokens', self.max_tokens),
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.api_url, json=payload)
                response.raise_for_status()

                result = response.json()
                return result.get('response', '')

        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise RuntimeError(f"Failed to generate response from Ollama: {e}")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    def generate_streaming(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Generate streaming response using Ollama.

        Args:
            prompt: User prompt
            context: Optional RAG context
            system_prompt: Optional system instructions
            **kwargs: Additional Ollama parameters

        Yields:
            Response chunks
        """
        full_prompt = self.build_full_prompt(prompt, context, system_prompt)

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": kwargs.get('max_tokens', self.max_tokens),
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", self.api_url, json=payload) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                yield chunk['response']

                            if chunk.get('done', False):
                                break

        except httpx.HTTPError as e:
            logger.error(f"Ollama streaming error: {e}")
            raise RuntimeError(f"Failed to stream from Ollama: {e}")
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise

    def check_health(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about Ollama and available models.

        Returns:
            Provider information
        """
        info = {
            'provider': 'ollama',
            'model': self.model_name,
            'base_url': self.base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'open_source': True,
            'local': True,
            'cost': 0
        }

        # Try to get available models
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    info['available_models'] = [
                        model['name'] for model in data.get('models', [])
                    ]
        except Exception as e:
            logger.warning(f"Could not fetch available models: {e}")
            info['available_models'] = []

        return info

    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Pull/download a model from Ollama registry.

        Args:
            model_name: Model to pull (uses current model if not specified)

        Returns:
            True if successful, False otherwise
        """
        model = model_name or self.model_name

        logger.info(f"Pulling Ollama model: {model}")

        try:
            with httpx.Client(timeout=300) as client:  # 5 min timeout for downloads
                response = client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model}
                )
                response.raise_for_status()
                logger.info(f"Successfully pulled model: {model}")
                return True

        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False
