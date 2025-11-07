"""
Configuration management with Pydantic validation.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RAGConfig(BaseModel):
    """RAG system configuration."""
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_results: int = Field(default=5, ge=1, le=100)
    persist_directory: str = Field(default="./.rag_data")
    cache_size: int = Field(default=1000, ge=0)
    cache_ttl: int = Field(default=3600, ge=0)

    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v, info):
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class MAFConfig(BaseModel):
    """Multi-Agent Framework configuration."""
    enabled: bool = True
    max_contexts: int = Field(default=1000, ge=1)
    agents: Dict[str, bool] = Field(default_factory=lambda: {
        "code_analyzer": True,
        "doc_retriever": True,
        "synthesizer": True,
        "suggestion_generator": True
    })


class MCPConfig(BaseModel):
    """MCP server configuration."""
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8765, ge=1024, le=65535)
    auto_start: bool = True
    timeout: int = Field(default=30, ge=1)


class IndexingConfig(BaseModel):
    """Indexing configuration."""
    auto_index_on_start: bool = True
    incremental: bool = True
    use_git: bool = True
    file_extensions: List[str] = Field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
        ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
        ".sh", ".bash", ".sql", ".html", ".css", ".scss", ".less",
        ".json", ".yaml", ".yml", ".xml", ".md", ".rst", ".txt"
    ])
    ignore_directories: List[str] = Field(default_factory=lambda: [
        "node_modules", ".git", ".venv", "venv", "__pycache__",
        ".pytest_cache", "dist", "build", ".next", ".nuxt",
        "coverage", ".coverage", ".rag_data", ".claude"
    ])


class PluginConfig(BaseModel):
    """Complete plugin configuration."""
    rag: RAGConfig = Field(default_factory=RAGConfig)
    maf: MAFConfig = Field(default_factory=MAFConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)

    @classmethod
    def load_from_file(cls, config_path: str = ".claude/rag-config.json") -> "PluginConfig":
        """
        Load and validate configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Validated plugin configuration

        Raises:
            ValueError: If configuration is invalid
        """
        path = Path(config_path)

        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        try:
            config_data = json.loads(path.read_text())
            config = cls(**config_data)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config: {e}")
            raise ValueError(f"Configuration file is not valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def save_to_file(self, config_path: str = ".claude/rag-config.json"):
        """
        Save configuration to file.

        Args:
            config_path: Path to save configuration
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        logger.info(f"Saved configuration to {config_path}")

    def validate_config(self) -> Dict[str, str]:
        """
        Validate configuration and return any warnings.

        Returns:
            Dictionary of validation warnings
        """
        warnings = {}

        # Check chunk overlap
        if self.rag.chunk_overlap > self.rag.chunk_size * 0.5:
            warnings['chunk_overlap'] = "Chunk overlap is > 50% of chunk size"

        # Check cache settings
        if self.rag.cache_size == 0:
            warnings['cache_size'] = "Query caching is disabled"

        # Check incremental indexing
        if not self.indexing.incremental:
            warnings['incremental'] = "Incremental indexing is disabled (slower)"

        return warnings
