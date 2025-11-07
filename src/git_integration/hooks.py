"""
Git hooks for automatic re-indexing on code changes.

Installs post-commit and post-merge hooks to trigger incremental indexing.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict
import logging
import sys
import os

logger = logging.getLogger(__name__)


class GitHookInstaller:
    """
    Install and manage Git hooks for auto-indexing.
    """

    def __init__(self, repo_path: Optional[Path] = None):
        """
        Initialize Git hook installer.

        Args:
            repo_path: Path to Git repository (default: current directory)
        """
        self.repo_path = repo_path or Path.cwd()
        self.hooks_dir = self.repo_path / '.git' / 'hooks'

    def is_git_repo(self) -> bool:
        """
        Check if directory is a Git repository.

        Returns:
            True if Git repo
        """
        return (self.repo_path / '.git').exists()

    def install_hooks(self, python_executable: Optional[str] = None) -> bool:
        """
        Install Git hooks for auto-indexing.

        Args:
            python_executable: Path to Python executable (default: current)

        Returns:
            True if installed successfully
        """
        if not self.is_git_repo():
            logger.error(f"{self.repo_path} is not a Git repository")
            return False

        if python_executable is None:
            python_executable = sys.executable

        # Ensure hooks directory exists
        self.hooks_dir.mkdir(parents=True, exist_ok=True)

        hooks_installed = []

        # Install post-commit hook
        if self._install_post_commit_hook(python_executable):
            hooks_installed.append('post-commit')

        # Install post-merge hook
        if self._install_post_merge_hook(python_executable):
            hooks_installed.append('post-merge')

        # Install post-checkout hook
        if self._install_post_checkout_hook(python_executable):
            hooks_installed.append('post-checkout')

        logger.info(f"Installed Git hooks: {', '.join(hooks_installed)}")
        return len(hooks_installed) > 0

    def _install_post_commit_hook(self, python_executable: str) -> bool:
        """
        Install post-commit hook.

        Args:
            python_executable: Path to Python executable

        Returns:
            True if installed
        """
        hook_path = self.hooks_dir / 'post-commit'

        hook_content = f"""#!/bin/bash
# Auto-indexing post-commit hook
# Triggered after git commit

echo "Running RAG auto-indexing..."

{python_executable} -c "
import sys
sys.path.insert(0, '{self.repo_path}')

try:
    from src.rag.enhanced_query_engine import EnhancedQueryEngine

    # Run incremental indexing
    engine = EnhancedQueryEngine()
    stats = engine.index_codebase(incremental=True, use_git=True)

    if stats:
        print(f'Indexed {{stats.get(\\\"files_processed\\\", 0)}} files')
except Exception as e:
    print(f'Auto-indexing failed: {{e}}')
    # Don't block commit
    pass
" &

# Run in background to not block commit
exit 0
"""

        try:
            hook_path.write_text(hook_content)
            hook_path.chmod(0o755)
            logger.info(f"Installed post-commit hook at {hook_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to install post-commit hook: {e}")
            return False

    def _install_post_merge_hook(self, python_executable: str) -> bool:
        """
        Install post-merge hook.

        Args:
            python_executable: Path to Python executable

        Returns:
            True if installed
        """
        hook_path = self.hooks_dir / 'post-merge'

        hook_content = f"""#!/bin/bash
# Auto-indexing post-merge hook
# Triggered after git merge or pull

echo "Running RAG auto-indexing after merge..."

{python_executable} -c "
import sys
sys.path.insert(0, '{self.repo_path}')

try:
    from src.rag.enhanced_query_engine import EnhancedQueryEngine

    # Run incremental indexing
    engine = EnhancedQueryEngine()
    stats = engine.index_codebase(incremental=True, use_git=True)

    if stats:
        print(f'Indexed {{stats.get(\\\"files_processed\\\", 0)}} files after merge')
except Exception as e:
    print(f'Auto-indexing failed: {{e}}')
    pass
" &

exit 0
"""

        try:
            hook_path.write_text(hook_content)
            hook_path.chmod(0o755)
            logger.info(f"Installed post-merge hook at {hook_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to install post-merge hook: {e}")
            return False

    def _install_post_checkout_hook(self, python_executable: str) -> bool:
        """
        Install post-checkout hook.

        Args:
            python_executable: Path to Python executable

        Returns:
            True if installed
        """
        hook_path = self.hooks_dir / 'post-checkout'

        hook_content = f"""#!/bin/bash
# Auto-indexing post-checkout hook
# Triggered after git checkout (branch switch)

# Check if it's a branch switch (not file checkout)
if [ "$3" = "1" ]; then
    echo "Running RAG auto-indexing after branch switch..."

    {python_executable} -c "
import sys
sys.path.insert(0, '{self.repo_path}')

try:
    from src.rag.enhanced_query_engine import EnhancedQueryEngine

    # Run full re-indexing on branch switch
    engine = EnhancedQueryEngine()
    stats = engine.index_codebase(incremental=False)

    if stats:
        print(f'Re-indexed {{stats.get(\\\"files_processed\\\", 0)}} files')
except Exception as e:
    print(f'Auto-indexing failed: {{e}}')
    pass
" &
fi

exit 0
"""

        try:
            hook_path.write_text(hook_content)
            hook_path.chmod(0o755)
            logger.info(f"Installed post-checkout hook at {hook_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to install post-checkout hook: {e}")
            return False

    def uninstall_hooks(self) -> bool:
        """
        Uninstall auto-indexing Git hooks.

        Returns:
            True if uninstalled successfully
        """
        hooks_removed = []

        for hook_name in ['post-commit', 'post-merge', 'post-checkout']:
            hook_path = self.hooks_dir / hook_name

            if hook_path.exists():
                try:
                    # Check if it's our hook
                    content = hook_path.read_text()
                    if 'Auto-indexing' in content:
                        hook_path.unlink()
                        hooks_removed.append(hook_name)
                        logger.info(f"Removed {hook_name} hook")
                except Exception as e:
                    logger.error(f"Failed to remove {hook_name} hook: {e}")

        return len(hooks_removed) > 0

    def check_hooks_installed(self) -> Dict[str, bool]:
        """
        Check which hooks are installed.

        Returns:
            Dictionary of hook statuses
        """
        hooks = {}

        for hook_name in ['post-commit', 'post-merge', 'post-checkout']:
            hook_path = self.hooks_dir / hook_name

            if hook_path.exists():
                try:
                    content = hook_path.read_text()
                    hooks[hook_name] = 'Auto-indexing' in content
                except Exception:
                    hooks[hook_name] = False
            else:
                hooks[hook_name] = False

        return hooks


def install_git_hooks(repo_path: Optional[Path] = None) -> bool:
    """
    Helper function to install Git hooks.

    Args:
        repo_path: Path to Git repository

    Returns:
        True if installed successfully
    """
    installer = GitHookInstaller(repo_path)
    return installer.install_hooks()


def uninstall_git_hooks(repo_path: Optional[Path] = None) -> bool:
    """
    Helper function to uninstall Git hooks.

    Args:
        repo_path: Path to Git repository

    Returns:
        True if uninstalled successfully
    """
    installer = GitHookInstaller(repo_path)
    return installer.uninstall_hooks()
