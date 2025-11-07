"""
Intelligent caching with auto-invalidation.
"""

from .invalidation import (
    IntelligentCacheManager,
    CacheInvalidationStrategy,
    TTLInvalidationStrategy,
    FileModificationStrategy,
    GitCommitStrategy,
    MemoryPressureStrategy,
    cache_manager
)

__all__ = [
    'IntelligentCacheManager',
    'CacheInvalidationStrategy',
    'TTLInvalidationStrategy',
    'FileModificationStrategy',
    'GitCommitStrategy',
    'MemoryPressureStrategy',
    'cache_manager'
]
