"""
Enhanced configuration management system.

Features:
- Environment-based configuration (dev, prod, test)
- Configuration validation
- Hot-reload support
- Configuration profiles
- Secure credential management
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import logging
from datetime import datetime

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """RAG system configuration."""

    # Paths
    codebase_path: str = "."
    db_path: str = "./chroma_db"
    cache_dir: str = "./.rag_cache"

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: Optional[str] = None

    # Query settings
    n_results: int = 5
    use_cache: bool = True
    use_hybrid: bool = True
    use_reranking: bool = False

    # Performance
    batch_size: int = 32
    lazy_loading: bool = True
    max_workers: int = 4

    # Cache settings
    cache_ttl_seconds: int = 3600
    enable_file_tracking: bool = True
    enable_git_tracking: bool = True

    # Indexing
    incremental_indexing: bool = True
    use_git_diff: bool = True
    ignore_dirs: List[str] = field(default_factory=lambda: [
        '__pycache__', 'node_modules', 'venv', '.git'
    ])

    # Server
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000

    # Monitoring
    enable_monitoring: bool = True
    enable_profiling: bool = False

    # Advanced
    enable_prefetching: bool = False
    enable_warming: bool = True
    enable_realtime_indexing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class ConfigManager:
    """
    Manages configuration with environment support.
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        environment: str = "development"
    ):
        """
        Initialize config manager.

        Args:
            config_dir: Configuration directory (default: .rag_config)
            environment: Environment name (development, production, test)
        """
        self.config_dir = config_dir or Path.home() / '.rag_config'
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.environment = environment
        self.config: RAGConfig = RAGConfig()

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load configuration from files."""
        # Load default config
        default_path = self.config_dir / 'default.json'
        if default_path.exists():
            self._merge_config(default_path)

        # Load environment-specific config
        env_path = self.config_dir / f'{self.environment}.json'
        if env_path.exists():
            self._merge_config(env_path)

        # Load from environment variables
        self._load_from_env()

        logger.info(f"Configuration loaded (environment: {self.environment})")

        # HIGH PRIORITY FIX: Validate configuration after loading
        validation_errors = self.validate()
        if validation_errors:
            for error in validation_errors:
                logger.warning(f"Configuration validation error: {error}")
            logger.warning("Some configuration validation errors were found")
        else:
            logger.debug("Configuration validation passed")

    def _merge_config(self, config_path: Path):
        """
        Merge configuration from file.

        Args:
            config_path: Path to config file
        """
        try:
            data = json.loads(config_path.read_text())

            # Update config
            for key, value in data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            logger.info(f"Merged config from {config_path}")

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Map environment variables to config fields
        env_mapping = {
            'RAG_CODEBASE_PATH': 'codebase_path',
            'RAG_DB_PATH': 'db_path',
            'RAG_EMBEDDING_MODEL': 'embedding_model',
            'RAG_N_RESULTS': 'n_results',
            'RAG_USE_CACHE': 'use_cache',
            'RAG_MCP_HOST': 'mcp_host',
            'RAG_MCP_PORT': 'mcp_port',
        }

        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)

            if value is not None:
                # Type conversion
                if config_key in ['n_results', 'mcp_port', 'batch_size']:
                    value = int(value)
                elif config_key in ['use_cache', 'use_hybrid', 'lazy_loading']:
                    value = value.lower() in ('true', '1', 'yes')

                setattr(self.config, config_key, value)
                logger.debug(f"Loaded {config_key} from environment")

    def save_config(self, environment: Optional[str] = None):
        """
        Save configuration to file.

        Args:
            environment: Environment to save to (default: current)
        """
        env = environment or self.environment
        config_path = self.config_dir / f'{env}.json'

        try:
            config_path.write_text(
                json.dumps(self.config.to_dict(), indent=2)
            )

            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value

        Returns:
            Configuration value
        """
        return getattr(self.config, key, default)

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            logger.debug(f"Set {key} = {value}")
        else:
            logger.warning(f"Unknown configuration key: {key}")

    def reload(self):
        """Reload configuration from files."""
        logger.info("Reloading configuration...")
        self._load_config()

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required paths exist
        codebase_path = Path(self.config.codebase_path)
        if not codebase_path.exists():
            errors.append(f"Codebase path does not exist: {self.config.codebase_path}")

        # Check port range
        if not (1024 <= self.config.mcp_port <= 65535):
            errors.append(f"Invalid port number: {self.config.mcp_port}")

        # Check batch size
        if self.config.batch_size < 1:
            errors.append(f"Invalid batch size: {self.config.batch_size}")

        # Check n_results
        if self.config.n_results < 1:
            errors.append(f"Invalid n_results: {self.config.n_results}")

        return errors

    def create_profile(self, name: str, overrides: Dict[str, Any]):
        """
        Create configuration profile.

        Args:
            name: Profile name
            overrides: Configuration overrides
        """
        profile_path = self.config_dir / f'profile_{name}.json'

        try:
            profile_path.write_text(json.dumps(overrides, indent=2))
            logger.info(f"Created profile: {name}")

        except Exception as e:
            logger.error(f"Error creating profile: {e}")

    def load_profile(self, name: str):
        """
        Load configuration profile.

        Args:
            name: Profile name
        """
        profile_path = self.config_dir / f'profile_{name}.json'

        if profile_path.exists():
            self._merge_config(profile_path)
            logger.info(f"Loaded profile: {name}")
        else:
            logger.warning(f"Profile not found: {name}")

    def export_config(self, output_path: Path):
        """
        Export current configuration.

        Args:
            output_path: Output file path
        """
        config_data = {
            'environment': self.environment,
            'exported_at': datetime.now().isoformat(),
            'config': self.config.to_dict()
        }

        output_path.write_text(json.dumps(config_data, indent=2))
        logger.info(f"Configuration exported to {output_path}")

    def import_config(self, input_path: Path):
        """
        Import configuration.

        Args:
            input_path: Input file path
        """
        try:
            data = json.loads(input_path.read_text())

            config_data = data.get('config', {})

            for key, value in config_data.items():
                self.set(key, value)

            logger.info(f"Configuration imported from {input_path}")

        except Exception as e:
            logger.error(f"Error importing config: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.

        Returns:
            Configuration summary
        """
        return {
            'environment': self.environment,
            'config_dir': str(self.config_dir),
            'codebase_path': self.config.codebase_path,
            'embedding_model': self.config.embedding_model,
            'mcp_endpoint': f"http://{self.config.mcp_host}:{self.config.mcp_port}",
            'features': {
                'caching': self.config.use_cache,
                'hybrid_search': self.config.use_hybrid,
                'reranking': self.config.use_reranking,
                'lazy_loading': self.config.lazy_loading,
                'prefetching': self.config.enable_prefetching,
                'warming': self.config.enable_warming,
                'realtime_indexing': self.config.enable_realtime_indexing,
            }
        }


class SecureConfigManager(ConfigManager):
    """
    Configuration manager with secure credential handling.
    """

    def __init__(self, *args, **kwargs):
        """Initialize secure config manager."""
        super().__init__(*args, **kwargs)

        # Separate file for credentials
        self.credentials_path = self.config_dir / '.credentials.json'

        # Ensure proper permissions
        if self.credentials_path.exists():
            os.chmod(self.credentials_path, 0o600)

    def set_credential(self, key: str, value: str):
        """
        Set secure credential.

        Args:
            key: Credential key
            value: Credential value
        """
        credentials = {}

        if self.credentials_path.exists():
            credentials = json.loads(self.credentials_path.read_text())

        credentials[key] = value

        # CRITICAL FIX: Create file with restricted permissions from the start
        # Prevents race condition where file is temporarily world-readable
        fd = os.open(
            self.credentials_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            mode=0o600  # Restricted permissions from creation
        )
        with os.fdopen(fd, 'w') as f:
            json.dump(credentials, f, indent=2)

        logger.info(f"Credential set: {key}")

    def get_credential(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secure credential.

        Args:
            key: Credential key
            default: Default value

        Returns:
            Credential value or default
        """
        if not self.credentials_path.exists():
            return default

        try:
            credentials = json.loads(self.credentials_path.read_text())
            return credentials.get(key, default)

        except Exception as e:
            logger.error(f"Error reading credentials: {e}")
            return default


# Global config manager instance
config_manager = ConfigManager()
