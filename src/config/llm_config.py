"""
LLM Configuration Manager.

Loads and manages LLM configuration from YAML files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMConfig:
    """
    LLM configuration manager.

    Loads configuration from llm-config.yaml or environment variables.
    """

    DEFAULT_CONFIG_PATHS = [
        'llm-config.yaml',
        '.claude/llm-config.yaml',
        'config/llm-config.yaml',
        os.path.expanduser('~/.dt-cli/llm-config.yaml')
    ]

    DEFAULT_CONFIG = {
        'provider': 'ollama',
        'llm': {
            'model_name': 'qwen3-coder',
            'base_url': 'http://localhost:11434',
            'temperature': 0.1,
            'max_tokens': 4096,
            'timeout': 60
        },
        'rag': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_results': 5,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'maf': {
            'enabled': True,
            'max_iterations': 10,
            'timeout': 300
        },
        'auto_trigger': {
            'enabled': True,
            'threshold': 0.7,
            'show_activity': True
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to config file (optional)
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _find_config_file(self) -> Optional[str]:
        """
        Find configuration file in default locations.

        Returns:
            Path to config file or None
        """
        if self.config_path and os.path.exists(self.config_path):
            return self.config_path

        # Search default locations
        for path in self.DEFAULT_CONFIG_PATHS:
            if os.path.exists(path):
                logger.info(f"Found config file: {path}")
                return path

        return None

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.

        Returns:
            Configuration dictionary
        """
        # Try to find config file
        config_file = self._find_config_file()

        if config_file:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                logger.info(f"Loaded configuration from {config_file}")

                # Merge with defaults
                return self._merge_configs(self.DEFAULT_CONFIG, config)

            except Exception as e:
                logger.error(f"Failed to load config from {config_file}: {e}")
                logger.info("Using default configuration")
                return self.DEFAULT_CONFIG.copy()
        else:
            logger.info("No config file found, using defaults")
            return self.DEFAULT_CONFIG.copy()

    def _merge_configs(
        self,
        default: Dict[str, Any],
        custom: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge custom config with defaults.

        Args:
            default: Default configuration
            custom: Custom configuration

        Returns:
            Merged configuration
        """
        result = default.copy()

        for key, value in custom.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def get_provider_type(self) -> str:
        """
        Get the configured provider type.

        Returns:
            Provider type ('ollama', 'vllm', 'claude')
        """
        return self.config.get('provider', 'ollama')

    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM provider configuration.

        Returns:
            LLM configuration with provider type included
        """
        llm_config = self.config.get('llm', {}).copy()
        llm_config['provider'] = self.get_provider_type()

        # Override from environment variables if present
        env_overrides = self._get_env_overrides()
        llm_config.update(env_overrides)

        return llm_config

    def _get_env_overrides(self) -> Dict[str, Any]:
        """
        Get configuration overrides from environment variables.

        Returns:
            Environment variable overrides
        """
        overrides = {}

        # LLM provider from env
        if os.getenv('DT_CLI_PROVIDER'):
            overrides['provider'] = os.getenv('DT_CLI_PROVIDER')

        if os.getenv('DT_CLI_MODEL'):
            overrides['model_name'] = os.getenv('DT_CLI_MODEL')

        if os.getenv('DT_CLI_BASE_URL'):
            overrides['base_url'] = os.getenv('DT_CLI_BASE_URL')

        if os.getenv('DT_CLI_API_KEY'):
            overrides['api_key'] = os.getenv('DT_CLI_API_KEY')

        if os.getenv('DT_CLI_TEMPERATURE'):
            try:
                overrides['temperature'] = float(os.getenv('DT_CLI_TEMPERATURE'))
            except ValueError:
                logger.warning("Invalid DT_CLI_TEMPERATURE value")

        return overrides

    def get_rag_config(self) -> Dict[str, Any]:
        """
        Get RAG configuration.

        Returns:
            RAG configuration
        """
        return self.config.get('rag', {})

    def get_maf_config(self) -> Dict[str, Any]:
        """
        Get MAF configuration.

        Returns:
            MAF configuration
        """
        return self.config.get('maf', {})

    def get_auto_trigger_config(self) -> Dict[str, Any]:
        """
        Get auto-trigger configuration.

        Returns:
            Auto-trigger configuration
        """
        return self.config.get('auto_trigger', {})

    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        logger.info("Configuration reloaded")

    def save(self, path: Optional[str] = None):
        """
        Save current configuration to file.

        Args:
            path: Path to save to (uses current path if None)
        """
        save_path = path or self.config_path or 'llm-config.yaml'

        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def __repr__(self) -> str:
        return f"LLMConfig(provider={self.get_provider_type()}, model={self.get_llm_config().get('model_name')})"
