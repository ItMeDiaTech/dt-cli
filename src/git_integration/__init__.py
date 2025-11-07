"""
Git integration for auto-indexing.
"""

from .hooks import GitHookInstaller, install_git_hooks, uninstall_git_hooks

__all__ = ['GitHookInstaller', 'install_git_hooks', 'uninstall_git_hooks']
