"""
Advanced RAG Configuration Manager with Multi-Project Support

This module implements best practices from 2025 RAG research:
- Auto-detection of current working directory
- Multi-folder project indexing
- Query rewriting and expansion (HyDE technique)
- Self-RAG capabilities
- Dynamic LLM provider switching
- Context engineering and management
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    """Configuration for a single project/folder."""
    path: str
    name: str
    indexed: bool = False
    last_indexed: Optional[str] = None
    file_count: int = 0
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", "node_modules", ".venv", "venv",
        "*.egg-info", "dist", "build", ".pytest_cache", ".mypy_cache"
    ])
    include_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".cpp", ".c", ".h", ".hpp", ".cs", ".rb", ".php", ".swift"
    ])


@dataclass
class WorkspaceConfig:
    """
    Workspace configuration managing multiple projects.

    Best Practice: Auto-detect current directory and allow
    adding additional folders for comprehensive indexing.
    """
    current_dir: str
    projects: List[ProjectConfig] = field(default_factory=list)
    active_llm_provider: str = "ollama"
    config_file: str = ".dt-cli-workspace.json"

    def __post_init__(self):
        """Initialize workspace with current directory."""
        if not self.projects:
            # Auto-detect current directory as primary project
            self.add_project(self.current_dir, name="current", auto_detect=True)

    def add_project(self, path: str, name: Optional[str] = None, auto_detect: bool = False) -> ProjectConfig:
        """
        Add a project folder to the workspace.

        Args:
            path: Path to project directory
            name: Optional project name (auto-generated if not provided)
            auto_detect: Whether this is auto-detected current directory

        Returns:
            ProjectConfig for the added project
        """
        abs_path = os.path.abspath(path)

        # Check if already exists
        for project in self.projects:
            if os.path.abspath(project.path) == abs_path:
                logger.info(f"Project already exists: {abs_path}")
                return project

        # Generate name if not provided
        if not name:
            name = os.path.basename(abs_path) or "project"
            if auto_detect:
                name = "current"

        project = ProjectConfig(path=abs_path, name=name)
        self.projects.append(project)
        logger.info(f"Added project: {name} ({abs_path})")

        return project

    def remove_project(self, name_or_path: str) -> bool:
        """
        Remove a project from workspace.

        Args:
            name_or_path: Project name or path

        Returns:
            True if removed, False if not found
        """
        abs_path = os.path.abspath(name_or_path)

        for i, project in enumerate(self.projects):
            if project.name == name_or_path or os.path.abspath(project.path) == abs_path:
                self.projects.pop(i)
                logger.info(f"Removed project: {project.name}")
                return True

        return False

    def get_all_paths(self) -> List[str]:
        """Get all project paths."""
        return [p.path for p in self.projects]

    def save(self):
        """Save workspace configuration to file."""
        config_path = os.path.join(self.current_dir, self.config_file)

        with open(config_path, 'w') as f:
            json.dump({
                'current_dir': self.current_dir,
                'active_llm_provider': self.active_llm_provider,
                'projects': [asdict(p) for p in self.projects]
            }, f, indent=2)

        logger.info(f"Saved workspace config to {config_path}")

    @classmethod
    def load(cls, current_dir: Optional[str] = None) -> 'WorkspaceConfig':
        """
        Load workspace configuration from file.

        Args:
            current_dir: Current directory (defaults to os.getcwd())

        Returns:
            WorkspaceConfig instance
        """
        current_dir = current_dir or os.getcwd()
        config_file = ".dt-cli-workspace.json"
        config_path = os.path.join(current_dir, config_file)

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)

                workspace = cls(current_dir=data.get('current_dir', current_dir))
                workspace.active_llm_provider = data.get('active_llm_provider', 'ollama')
                workspace.projects = [ProjectConfig(**p) for p in data.get('projects', [])]

                logger.info(f"Loaded workspace config from {config_path}")
                return workspace
            except Exception as e:
                logger.error(f"Failed to load workspace config: {e}")

        # Create new workspace
        workspace = cls(current_dir=current_dir)
        workspace.save()
        return workspace


class AdvancedRAGConfig:
    """
    Advanced RAG Configuration implementing 2025 best practices.

    Features:
    - Query rewriting and expansion (HyDE)
    - Self-RAG with reflection
    - Multi-project indexing
    - Dynamic LLM provider switching
    - Context engineering
    """

    def __init__(self, config_file: str = "llm-config.yaml", workspace_dir: Optional[str] = None):
        """
        Initialize advanced RAG configuration.

        Args:
            config_file: Path to main config file
            workspace_dir: Workspace directory (defaults to current directory)
        """
        self.config_file = config_file
        self.workspace_dir = workspace_dir or os.getcwd()

        # Load main config
        self.main_config = self._load_main_config()

        # Load or create workspace config
        self.workspace = WorkspaceConfig.load(self.workspace_dir)

        # Advanced RAG settings
        self.self_rag_enabled = True
        self.query_rewriting_enabled = True
        self.hyde_enabled = True  # Hypothetical Document Embeddings
        self.max_retrieval_attempts = 3
        self.reflection_threshold = 0.7

        logger.info(f"Advanced RAG config initialized in {self.workspace_dir}")
        logger.info(f"Active LLM provider: {self.workspace.active_llm_provider}")
        logger.info(f"Projects: {len(self.workspace.projects)}")

    def _load_main_config(self) -> Dict[str, Any]:
        """Load main configuration from YAML file."""
        if not os.path.exists(self.config_file):
            logger.warning(f"Config file not found: {self.config_file}")
            return {}

        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for active provider."""
        provider = self.workspace.active_llm_provider

        config = {
            'provider': provider,
            'model_name': 'qwen3-coder',
            'temperature': 0.1,
            'max_tokens': 4096
        }

        # Get provider-specific settings from main config
        if provider in self.main_config:
            config.update(self.main_config[provider])
        elif 'llm' in self.main_config:
            config.update(self.main_config['llm'])

        # Get provider from main config
        if 'provider' in self.main_config:
            config['provider'] = self.main_config['provider']

        return config

    def switch_llm_provider(self, provider: str) -> bool:
        """
        Switch LLM provider at runtime.

        Args:
            provider: Provider name (ollama, vllm, claude, etc.)

        Returns:
            True if switched successfully
        """
        valid_providers = ['ollama', 'vllm', 'claude', 'openai', 'local']

        if provider not in valid_providers:
            logger.error(f"Invalid provider: {provider}. Valid: {valid_providers}")
            return False

        self.workspace.active_llm_provider = provider
        self.workspace.save()
        logger.info(f"Switched LLM provider to: {provider}")
        return True

    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration with advanced features."""
        base_config = self.main_config.get('rag', {})

        # Add advanced features
        advanced_config = {
            **base_config,
            'self_rag_enabled': self.self_rag_enabled,
            'query_rewriting_enabled': self.query_rewriting_enabled,
            'hyde_enabled': self.hyde_enabled,
            'max_retrieval_attempts': self.max_retrieval_attempts,
            'reflection_threshold': self.reflection_threshold,
            'multi_project_paths': self.workspace.get_all_paths()
        }

        return advanced_config

    def get_auto_trigger_config(self) -> Dict[str, Any]:
        """Get auto-trigger configuration."""
        return self.main_config.get('auto_trigger', {
            'enabled': True,
            'threshold': 0.7,
            'show_activity': True
        })

    def add_project_folder(self, path: str, name: Optional[str] = None) -> bool:
        """
        Add a project folder to the workspace.

        Args:
            path: Path to project directory
            name: Optional project name

        Returns:
            True if added successfully
        """
        try:
            self.workspace.add_project(path, name)
            self.workspace.save()
            return True
        except Exception as e:
            logger.error(f"Failed to add project folder: {e}")
            return False

    def remove_project_folder(self, name_or_path: str) -> bool:
        """
        Remove a project folder from workspace.

        Args:
            name_or_path: Project name or path

        Returns:
            True if removed successfully
        """
        if self.workspace.remove_project(name_or_path):
            self.workspace.save()
            return True
        return False

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects in workspace."""
        return [asdict(p) for p in self.workspace.projects]

    def get_current_directory(self) -> str:
        """Get current working directory."""
        return self.workspace.current_dir


# Global instance
advanced_rag_config = None


def get_advanced_config(workspace_dir: Optional[str] = None) -> AdvancedRAGConfig:
    """
    Get or create global advanced RAG config instance.

    Args:
        workspace_dir: Workspace directory (defaults to current directory)

    Returns:
        AdvancedRAGConfig instance
    """
    global advanced_rag_config

    if advanced_rag_config is None:
        advanced_rag_config = AdvancedRAGConfig(workspace_dir=workspace_dir)

    return advanced_rag_config
