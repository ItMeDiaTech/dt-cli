"""
Git integration for smart change detection.
"""

import subprocess
from typing import List, Set, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GitChangeTracker:
    """Track file changes using Git."""

    def __init__(self, repo_path: str = "."):
        """
        Initialize Git tracker.

        Args:
            repo_path: Path to Git repository
        """
        self.repo_path = Path(repo_path)
        self.is_git_repo = self._check_git_repo()

    def _check_git_repo(self) -> bool:
        """Check if directory is a git repository."""
        try:
            subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_path,
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_changed_files(self, since_commit: str = "HEAD~1") -> Set[str]:
        """
        Get files changed since a specific commit.

        Args:
            since_commit: Git reference to compare against

        Returns:
            Set of changed file paths
        """
        if not self.is_git_repo:
            return set()

        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', since_commit],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )

            files = set(result.stdout.strip().split('\n'))
            return {f for f in files if f}  # Remove empty strings

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Git diff failed: {e}")
            return set()

    def get_untracked_files(self) -> Set[str]:
        """
        Get untracked files in repository.

        Returns:
            Set of untracked file paths
        """
        if not self.is_git_repo:
            return set()

        try:
            result = subprocess.run(
                ['git', 'ls-files', '--others', '--exclude-standard'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )

            files = set(result.stdout.strip().split('\n'))
            return {f for f in files if f}

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Git ls-files failed: {e}")
            return set()

    def get_modified_files(self) -> Set[str]:
        """
        Get modified but not committed files.

        Returns:
            Set of modified file paths
        """
        if not self.is_git_repo:
            return set()

        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )

            files = set(result.stdout.strip().split('\n'))
            return {f for f in files if f}

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Git diff failed: {e}")
            return set()

    def get_staged_files(self) -> Set[str]:
        """
        Get staged files.

        Returns:
            Set of staged file paths
        """
        if not self.is_git_repo:
            return set()

        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )

            files = set(result.stdout.strip().split('\n'))
            return {f for f in files if f}

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Git diff --cached failed: {e}")
            return set()

    def get_all_changed(self) -> Set[str]:
        """
        Get all changed, untracked, and modified files.

        Returns:
            Set of all changed file paths
        """
        return (
            self.get_changed_files() |
            self.get_untracked_files() |
            self.get_modified_files() |
            self.get_staged_files()
        )

    def get_last_commit_hash(self) -> Optional[str]:
        """
        Get the last commit hash.

        Returns:
            Commit hash or None
        """
        if not self.is_git_repo:
            return None

        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )

            return result.stdout.strip()

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Git rev-parse failed: {e}")
            return None
