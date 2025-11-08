"""
vLLM LLM Provider.

vLLM is the recommended provider for production:
- 100% open source
- 3.2x higher throughput than Ollama
- PagedAttention for memory efficiency
- Continuous batching for concurrent requests
- OpenAI-compatible API

Deployment:
- Can run on single GPU or multi-GPU (tensor parallelism)
- Supports models up to 480B parameters
- Production-grade performance
"""

from typing import Dict, Any, List, Optional
import httpx
import json
import logging

from .base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class VLLMProvider(BaseLLMProvider):
    """
    vLLM LLM provider.

    Uses OpenAI-compatible API endpoint from vLLM server.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vLLM provider.

        Config options:
            model_name: Model name (must match vLLM server model)
            base_url: vLLM server URL (default: http://localhost:8000)
            api_key: Optional API key (default: 'not-needed')
            temperature: Sampling temperature (default: 0.1)
            max_tokens: Maximum tokens to generate (default: 4096)
            timeout: Request timeout in seconds (default: 120)
        """
        super().__init__(config)

        self.base_url = config.get('base_url', 'http://localhost:8000')
        self.api_key = config.get('api_key', 'not-needed')

        # vLLM uses OpenAI-compatible endpoints
        self.completions_url = f"{self.base_url}/v1/completions"
        self.chat_url = f"{self.base_url}/v1/chat/completions"

        # Use recommended model if not specified
        if self.model_name == 'default':
            self.model_name = 'qwen3-coder'
            logger.info(f"Using default vLLM model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response using vLLM.

        Args:
            prompt: User prompt
            context: Optional RAG context
            system_prompt: Optional system instructions
            **kwargs: Additional vLLM parameters

        Returns:
            Generated response
        """
        # Build messages for chat completion
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Add context if provided
        if context:
            context_text = self.format_context(context)
            messages.append({
                "role": "system",
                "content": context_text
            })

        # Add user message
        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.chat_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()

                result = response.json()
                return result['choices'][0]['message']['content']

        except httpx.HTTPError as e:
            logger.error(f"vLLM HTTP error: {e}")
            raise RuntimeError(f"Failed to generate response from vLLM: {e}")
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            raise

    def generate_streaming(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Generate streaming response using vLLM.

        Args:
            prompt: User prompt
            context: Optional RAG context
            system_prompt: Optional system instructions
            **kwargs: Additional vLLM parameters

        Yields:
            Response chunks
        """
        # Build messages
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        if context:
            context_text = self.format_context(context)
            messages.append({
                "role": "system",
                "content": context_text
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', self.temperature),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "stream": True
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    self.chat_url,
                    json=payload,
                    headers=headers
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix

                            if data.strip() == '[DONE]':
                                break

                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue

        except httpx.HTTPError as e:
            logger.error(f"vLLM streaming error: {e}")
            raise RuntimeError(f"Failed to stream from vLLM: {e}")
        except Exception as e:
            logger.error(f"vLLM streaming error: {e}")
            raise

    def check_health(self) -> bool:
        """
        Check if vLLM server is running and accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            with httpx.Client(timeout=5) as client:
                # Try health endpoint
                health_url = f"{self.base_url}/health"
                response = client.get(health_url)
                if response.status_code == 200:
                    return True

                # Fallback: try models endpoint
                models_url = f"{self.base_url}/v1/models"
                response = client.get(models_url)
                return response.status_code == 200

        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about vLLM server.

        Returns:
            Provider information
        """
        info = {
            'provider': 'vllm',
            'model': self.model_name,
            'base_url': self.base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'open_source': True,
            'local': self.base_url.startswith('http://localhost') or
                     self.base_url.startswith('http://127.0.0.1'),
            'production_ready': True,
            'performance': 'High (3.2x vs Ollama)'
        }

        # Try to get available models
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            with httpx.Client(timeout=5) as client:
                response = client.get(
                    f"{self.base_url}/v1/models",
                    headers=headers
                )
                if response.status_code == 200:
                    data = response.json()
                    info['available_models'] = [
                        model['id'] for model in data.get('data', [])
                    ]
        except Exception as e:
            logger.warning(f"Could not fetch available models: {e}")
            info['available_models'] = []

        return info
