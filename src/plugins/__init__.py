"""
Plugin and extension system.
"""

from .plugin_system import (
    PluginBase,
    QueryProcessorPlugin,
    ResultFilterPlugin,
    CommandPlugin,
    PluginMetadata,
    PluginManager,
    LowercaseQueryProcessor,
    DeduplicateResultsFilter,
    plugin_manager
)

__all__ = [
    'PluginBase',
    'QueryProcessorPlugin',
    'ResultFilterPlugin',
    'CommandPlugin',
    'PluginMetadata',
    'PluginManager',
    'LowercaseQueryProcessor',
    'DeduplicateResultsFilter',
    'plugin_manager'
]
