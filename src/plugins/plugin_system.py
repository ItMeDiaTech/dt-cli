"""
Plugin/Extension system for RAG.

Allows extending RAG functionality with custom:
- Query processors
- Result filters
- Custom commands
- Index transformers
- Export formatters
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Type
import importlib.util
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PluginBase(ABC):
    """Base class for all plugins."""

    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration.

        Args:
            config: Plugin configuration
        """
        pass


class QueryProcessorPlugin(PluginBase):
    """Base class for query processor plugins."""

    @abstractmethod
    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        """
        Process/transform query before execution.

        Args:
            query: Original query
            context: Query context

        Returns:
            Processed query
        """
        pass


class ResultFilterPlugin(PluginBase):
    """Base class for result filter plugins."""

    @abstractmethod
    def filter_results(
        self,
        results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter/transform search results.

        Args:
            results: Original results
            context: Filter context

        Returns:
            Filtered results
        """
        pass


class CommandPlugin(PluginBase):
    """Base class for command plugins."""

    @abstractmethod
    def get_command_name(self) -> str:
        """Get command name."""
        pass

    @abstractmethod
    def execute(self, args: List[str], context: Dict[str, Any]) -> Any:
        """
        Execute command.

        Args:
            args: Command arguments
            context: Execution context

        Returns:
            Command result
        """
        pass


@dataclass
class PluginMetadata:
    """Plugin metadata."""

    name: str
    version: str
    author: str
    description: str
    plugin_type: str
    enabled: bool = True


class PluginManager:
    """
    Manages plugins and extensions.
    """

    def __init__(self, plugin_dir: Optional[Path] = None):
        """
        Initialize plugin manager.

        Args:
            plugin_dir: Directory for plugins
        """
        self.plugin_dir = plugin_dir or Path.home() / '.rag_plugins'
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        # Plugin storage
        self.query_processors: Dict[str, QueryProcessorPlugin] = {}
        self.result_filters: Dict[str, ResultFilterPlugin] = {}
        self.commands: Dict[str, CommandPlugin] = {}

        # Metadata
        self.metadata: Dict[str, PluginMetadata] = {}

        # Load plugins
        self._discover_plugins()

    def register_query_processor(
        self,
        plugin: QueryProcessorPlugin,
        metadata: Optional[PluginMetadata] = None
    ):
        """
        Register query processor plugin.

        Args:
            plugin: Plugin instance
            metadata: Plugin metadata
        """
        name = plugin.get_name()

        self.query_processors[name] = plugin

        if metadata:
            self.metadata[name] = metadata

        logger.info(f"Registered query processor: {name}")

    def register_result_filter(
        self,
        plugin: ResultFilterPlugin,
        metadata: Optional[PluginMetadata] = None
    ):
        """
        Register result filter plugin.

        Args:
            plugin: Plugin instance
            metadata: Plugin metadata
        """
        name = plugin.get_name()

        self.result_filters[name] = plugin

        if metadata:
            self.metadata[name] = metadata

        logger.info(f"Registered result filter: {name}")

    def register_command(
        self,
        plugin: CommandPlugin,
        metadata: Optional[PluginMetadata] = None
    ):
        """
        Register command plugin.

        Args:
            plugin: Plugin instance
            metadata: Plugin metadata
        """
        name = plugin.get_command_name()

        self.commands[name] = plugin

        if metadata:
            self.metadata[name] = metadata

        logger.info(f"Registered command: {name}")

    def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process query through all enabled processors.

        Args:
            query: Original query
            context: Query context

        Returns:
            Processed query
        """
        processed_query = query
        context = context or {}

        for name, processor in self.query_processors.items():
            # Check if enabled
            if not self._is_enabled(name):
                continue

            try:
                processed_query = processor.process_query(processed_query, context)
                logger.debug(f"Query processed by {name}")

            except Exception as e:
                logger.error(f"Error in query processor {name}: {e}")

        return processed_query

    def filter_results(
        self,
        results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter results through all enabled filters.

        Args:
            results: Original results
            context: Filter context

        Returns:
            Filtered results
        """
        filtered_results = results
        context = context or {}

        for name, filter_plugin in self.result_filters.items():
            # Check if enabled
            if not self._is_enabled(name):
                continue

            try:
                filtered_results = filter_plugin.filter_results(filtered_results, context)
                logger.debug(f"Results filtered by {name}")

            except Exception as e:
                logger.error(f"Error in result filter {name}: {e}")

        return filtered_results

    def execute_command(
        self,
        command_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute command plugin.

        Args:
            command_name: Command name
            args: Command arguments
            context: Execution context

        Returns:
            Command result
        """
        if command_name not in self.commands:
            raise ValueError(f"Unknown command: {command_name}")

        command = self.commands[command_name]

        # Check if enabled
        if not self._is_enabled(command.get_name()):
            raise ValueError(f"Command disabled: {command_name}")

        context = context or {}

        try:
            return command.execute(args, context)

        except Exception as e:
            logger.error(f"Error executing command {command_name}: {e}")
            raise

    def list_plugins(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List registered plugins.

        Args:
            plugin_type: Filter by type

        Returns:
            List of plugin information
        """
        plugins = []

        # Add query processors
        if not plugin_type or plugin_type == 'query_processor':
            for name, processor in self.query_processors.items():
                plugins.append({
                    'name': name,
                    'type': 'query_processor',
                    'version': processor.get_version(),
                    'enabled': self._is_enabled(name)
                })

        # Add result filters
        if not plugin_type or plugin_type == 'result_filter':
            for name, filter_plugin in self.result_filters.items():
                plugins.append({
                    'name': name,
                    'type': 'result_filter',
                    'version': filter_plugin.get_version(),
                    'enabled': self._is_enabled(name)
                })

        # Add commands
        if not plugin_type or plugin_type == 'command':
            for name, command in self.commands.items():
                plugins.append({
                    'name': name,
                    'type': 'command',
                    'version': command.get_version(),
                    'enabled': self._is_enabled(name),
                    'command': command.get_command_name()
                })

        return plugins

    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            True if enabled
        """
        if plugin_name in self.metadata:
            self.metadata[plugin_name].enabled = True
            logger.info(f"Enabled plugin: {plugin_name}")
            return True

        return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """
        Disable plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            True if disabled
        """
        if plugin_name in self.metadata:
            self.metadata[plugin_name].enabled = False
            logger.info(f"Disabled plugin: {plugin_name}")
            return True

        return False

    def load_plugin_from_file(self, plugin_file: Path) -> bool:
        """
        Load plugin from Python file.

        Args:
            plugin_file: Path to plugin file

        Returns:
            True if loaded successfully
        """
        try:
            # Load module
            spec = importlib.util.spec_from_file_location("plugin", plugin_file)
            if not spec or not spec.loader:
                logger.error(f"Could not load plugin spec from {plugin_file}")
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for plugin class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Check if it's a plugin class
                if (isinstance(attr, type) and
                    issubclass(attr, PluginBase) and
                    attr != PluginBase):

                    # Instantiate and register
                    plugin_instance = attr()

                    if isinstance(plugin_instance, QueryProcessorPlugin):
                        self.register_query_processor(plugin_instance)
                    elif isinstance(plugin_instance, ResultFilterPlugin):
                        self.register_result_filter(plugin_instance)
                    elif isinstance(plugin_instance, CommandPlugin):
                        self.register_command(plugin_instance)

                    return True

            logger.warning(f"No plugin class found in {plugin_file}")
            return False

        except Exception as e:
            logger.error(f"Error loading plugin from {plugin_file}: {e}")
            return False

    def _discover_plugins(self):
        """Discover and load plugins from plugin directory."""
        if not self.plugin_dir.exists():
            return

        for plugin_file in self.plugin_dir.glob('*.py'):
            if plugin_file.name.startswith('_'):
                continue

            logger.info(f"Loading plugin: {plugin_file.name}")
            self.load_plugin_from_file(plugin_file)

    def _is_enabled(self, plugin_name: str) -> bool:
        """
        Check if plugin is enabled.

        Args:
            plugin_name: Plugin name

        Returns:
            True if enabled
        """
        if plugin_name in self.metadata:
            return self.metadata[plugin_name].enabled

        # Default to enabled if no metadata
        return True


# Example built-in plugins

class LowercaseQueryProcessor(QueryProcessorPlugin):
    """Example: Convert query to lowercase."""

    def get_name(self) -> str:
        return "lowercase_processor"

    def get_version(self) -> str:
        return "1.0.0"

    def initialize(self, config: Dict[str, Any]):
        pass

    def process_query(self, query: str, context: Dict[str, Any]) -> str:
        return query.lower()


class DeduplicateResultsFilter(ResultFilterPlugin):
    """Example: Remove duplicate results."""

    def get_name(self) -> str:
        return "deduplicate_filter"

    def get_version(self) -> str:
        return "1.0.0"

    def initialize(self, config: Dict[str, Any]):
        pass

    def filter_results(
        self,
        results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        seen = set()
        unique_results = []

        for result in results:
            result_id = result.get('id') or result.get('content', '')[:50]

            if result_id not in seen:
                seen.add(result_id)
                unique_results.append(result)

        return unique_results


# Global instance
plugin_manager = PluginManager()
