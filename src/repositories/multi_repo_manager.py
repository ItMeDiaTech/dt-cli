"""
Multi-repository support for RAG system.

Features:
- Index multiple repositories simultaneously
- Cross-repository search
- Repository isolation and filtering
- Repository groups/workspaces
- Unified knowledge graph across repos
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Repository:
    """Represents a repository configuration."""

    id: str
    name: str
    path: str
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    ignore_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_indexed: Optional[str] = None
    file_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Repository':
        """Create from dictionary."""
        return cls(**data)


class MultiRepositoryManager:
    """
    Manages multiple repositories for unified indexing and search.

    HIGH PRIORITY FIX: Added context manager support for proper resource management.
    """

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize multi-repository manager.

        Args:
            config_file: Path to repositories config file
        """
        self.config_file = config_file or Path.home() / '.rag_repositories.json'
        self.repositories: Dict[str, Repository] = {}

        self._load_repositories()

    def __enter__(self):
        """
        HIGH PRIORITY FIX: Context manager entry.

        Allows usage like:
            with MultiRepositoryManager() as manager:
                manager.index_all_repositories()
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        HIGH PRIORITY FIX: Context manager exit.

        Ensures repositories are saved on exit.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised

        Returns:
            False to propagate exceptions
        """
        try:
            # Save repositories on exit
            self._save_repositories()
            logger.debug("Saved repositories on context manager exit")
        except Exception as e:
            logger.error(f"Error saving repositories on exit: {e}")

        # Don't suppress exceptions
        return False

    def add_repository(
        self,
        name: str,
        path: str,
        tags: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ) -> Repository:
        """
        Add repository to manager.

        Args:
            name: Repository name
            path: Repository path
            tags: Optional tags
            ignore_patterns: Optional ignore patterns

        Returns:
            Repository instance
        """
        import hashlib

        # HIGH PRIORITY FIX: Validate inputs
        if not name or not isinstance(name, str):
            raise ValueError("Repository name must be a non-empty string")

        if not path or not isinstance(path, str):
            raise ValueError("Repository path must be a non-empty string")

        # Verify path exists and is a directory
        repo_path = Path(path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {path}")

        if not repo_path.is_dir():
            raise ValueError(f"Repository path must be a directory: {path}")

        # Generate ID from path
        repo_id = hashlib.md5(path.encode()).hexdigest()[:12]

        # Check if already exists
        if repo_id in self.repositories:
            logger.warning(f"Repository already exists: {name}")
            return self.repositories[repo_id]

        # Create repository
        repo = Repository(
            id=repo_id,
            name=name,
            path=str(repo_path.resolve()),
            tags=tags or [],
            ignore_patterns=ignore_patterns or []
        )

        self.repositories[repo_id] = repo
        self._save_repositories()

        logger.info(f"Added repository: {name} ({repo_id})")
        return repo

    def remove_repository(self, repo_id: str) -> bool:
        """
        Remove repository.

        Args:
            repo_id: Repository ID

        Returns:
            True if removed
        """
        if repo_id in self.repositories:
            repo_name = self.repositories[repo_id].name
            del self.repositories[repo_id]
            self._save_repositories()

            logger.info(f"Removed repository: {repo_name}")
            return True

        return False

    def get_repository(self, repo_id: str) -> Optional[Repository]:
        """
        Get repository by ID.

        Args:
            repo_id: Repository ID

        Returns:
            Repository or None
        """
        return self.repositories.get(repo_id)

    def list_repositories(
        self,
        tags: Optional[List[str]] = None,
        enabled_only: bool = True
    ) -> List[Repository]:
        """
        List repositories.

        Args:
            tags: Filter by tags
            enabled_only: Only return enabled repositories

        Returns:
            List of repositories
        """
        repos = list(self.repositories.values())

        # Filter by enabled
        if enabled_only:
            repos = [r for r in repos if r.enabled]

        # Filter by tags
        if tags:
            tag_set = set(tags)
            repos = [r for r in repos if tag_set & set(r.tags)]

        return repos

    def index_all_repositories(
        self,
        query_engine,
        incremental: bool = True,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Index all enabled repositories.

        Args:
            query_engine: Query engine instance
            incremental: Use incremental indexing
            parallel: Index in parallel

        Returns:
            Indexing statistics
        """
        logger.info("Indexing all repositories...")

        enabled_repos = self.list_repositories(enabled_only=True)

        if not enabled_repos:
            logger.warning("No enabled repositories to index")
            return {}

        results = {}

        if parallel:
            results = self._index_parallel(query_engine, enabled_repos, incremental)
        else:
            results = self._index_sequential(query_engine, enabled_repos, incremental)

        # Update last indexed time
        for repo_id, stats in results.items():
            if repo_id in self.repositories and stats.get('success'):
                self.repositories[repo_id].last_indexed = datetime.now().isoformat()
                if 'files_processed' in stats:
                    self.repositories[repo_id].file_count = stats['files_processed']

        self._save_repositories()

        return results

    def _index_sequential(
        self,
        query_engine,
        repositories: List[Repository],
        incremental: bool
    ) -> Dict[str, Any]:
        """
        Index repositories sequentially.

        Args:
            query_engine: Query engine instance
            repositories: List of repositories
            incremental: Use incremental indexing

        Returns:
            Indexing results
        """
        results = {}

        for repo in repositories:
            logger.info(f"Indexing repository: {repo.name}")

            try:
                # Update query engine codebase path
                original_path = query_engine.config.get('codebase_path')
                query_engine.config['codebase_path'] = repo.path

                # Index repository
                stats = query_engine.index_codebase(
                    incremental=incremental,
                    use_git=True
                )

                # Restore original path
                query_engine.config['codebase_path'] = original_path

                results[repo.id] = {
                    'success': True,
                    'name': repo.name,
                    'stats': stats or {}
                }

            except Exception as e:
                logger.error(f"Error indexing {repo.name}: {e}")
                results[repo.id] = {
                    'success': False,
                    'name': repo.name,
                    'error': str(e)
                }

        return results

    def _index_parallel(
        self,
        query_engine,
        repositories: List[Repository],
        incremental: bool
    ) -> Dict[str, Any]:
        """
        Index repositories in parallel with thread safety.

        Args:
            query_engine: Query engine instance
            repositories: List of repositories
            incremental: Use incremental indexing

        Returns:
            Indexing results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        # CRITICAL FIX: Add lock for thread-safe access to shared query_engine
        config_lock = threading.Lock()

        def index_repo(repo: Repository):
            try:
                logger.info(f"Indexing repository: {repo.name}")

                # CRITICAL FIX: Thread-safe config modification
                with config_lock:
                    # Save original path
                    original_path = query_engine.config.get('codebase_path')

                    # Set repository path
                    query_engine.config['codebase_path'] = repo.path

                    try:
                        # Index with repository path set
                        stats = query_engine.index_codebase(
                            incremental=incremental,
                            use_git=True
                        )
                    finally:
                        # Always restore original path
                        query_engine.config['codebase_path'] = original_path

                return repo.id, {
                    'success': True,
                    'name': repo.name,
                    'stats': stats or {}
                }

            except Exception as e:
                logger.error(f"Error indexing {repo.name}: {e}")
                return repo.id, {
                    'success': False,
                    'name': repo.name,
                    'error': str(e)
                }

        # Limit parallelism to avoid overwhelming the system
        max_workers = min(3, len(repositories))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(index_repo, repo): repo for repo in repositories}

            for future in as_completed(futures):
                repo_id, result = future.result()
                results[repo_id] = result

        return results

    def search_repositories(
        self,
        query: str,
        query_engine,
        repository_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search across multiple repositories.

        Args:
            query: Search query
            query_engine: Query engine instance
            repository_ids: Specific repository IDs (None = all)
            tags: Filter by repository tags
            n_results: Results per repository

        Returns:
            Search results grouped by repository
        """
        # Determine which repositories to search
        if repository_ids:
            repos = [self.repositories[rid] for rid in repository_ids if rid in self.repositories]
        else:
            repos = self.list_repositories(tags=tags, enabled_only=True)

        if not repos:
            logger.warning("No repositories to search")
            return {}

        logger.info(f"Searching {len(repos)} repositories for: {query}")

        results = {}

        for repo in repos:
            try:
                # Update query engine path
                original_path = query_engine.config.get('codebase_path')
                query_engine.config['codebase_path'] = repo.path

                # Execute query
                repo_results = query_engine.query(query, n_results=n_results)

                # Restore original path
                query_engine.config['codebase_path'] = original_path

                # Add repository metadata to results
                for result in repo_results:
                    if 'metadata' not in result:
                        result['metadata'] = {}
                    result['metadata']['repository'] = repo.name
                    result['metadata']['repository_id'] = repo.id

                results[repo.id] = {
                    'repository': repo.name,
                    'results': repo_results,
                    'count': len(repo_results)
                }

            except Exception as e:
                logger.error(f"Error searching {repo.name}: {e}")
                results[repo.id] = {
                    'repository': repo.name,
                    'error': str(e),
                    'count': 0
                }

        return results

    def get_repository_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all repositories.

        Returns:
            Repository statistics
        """
        enabled_repos = self.list_repositories(enabled_only=True)
        all_repos = list(self.repositories.values())

        stats = {
            'total_repositories': len(all_repos),
            'enabled_repositories': len(enabled_repos),
            'disabled_repositories': len(all_repos) - len(enabled_repos),
            'total_files_indexed': sum(r.file_count for r in all_repos),
            'repositories': [
                {
                    'id': r.id,
                    'name': r.name,
                    'enabled': r.enabled,
                    'file_count': r.file_count,
                    'last_indexed': r.last_indexed,
                    'tags': r.tags
                }
                for r in all_repos
            ]
        }

        # Group by tags
        tag_counts = {}
        for repo in all_repos:
            for tag in repo.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        stats['tags'] = tag_counts

        return stats

    def enable_repository(self, repo_id: str) -> bool:
        """
        Enable repository.

        Args:
            repo_id: Repository ID

        Returns:
            True if enabled
        """
        if repo_id in self.repositories:
            self.repositories[repo_id].enabled = True
            self._save_repositories()
            return True

        return False

    def disable_repository(self, repo_id: str) -> bool:
        """
        Disable repository.

        Args:
            repo_id: Repository ID

        Returns:
            True if disabled
        """
        if repo_id in self.repositories:
            self.repositories[repo_id].enabled = False
            self._save_repositories()
            return True

        return False

    def export_configuration(self, output_path: Path):
        """
        Export repository configuration.

        Args:
            output_path: Output file path
        """
        config_data = {
            'version': '1.0.0',
            'exported_at': datetime.now().isoformat(),
            'repositories': [r.to_dict() for r in self.repositories.values()]
        }

        output_path.write_text(json.dumps(config_data, indent=2))
        logger.info(f"Repository configuration exported to {output_path}")

    def import_configuration(self, input_path: Path, merge: bool = True):
        """
        Import repository configuration.

        Args:
            input_path: Input file path
            merge: Merge with existing (True) or replace (False)
        """
        # HIGH PRIORITY FIX: Validate input and provide detailed error messages
        if not input_path.exists():
            raise ValueError(f"Import file does not exist: {input_path}")

        if not input_path.is_file():
            raise ValueError(f"Import path must be a file: {input_path}")

        try:
            config_data = json.loads(input_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in import file: {e}")
        except Exception as e:
            raise ValueError(f"Error reading import file: {e}")

        # HIGH PRIORITY FIX: Validate structure
        if not isinstance(config_data, dict):
            raise ValueError("Import file must contain a JSON object")

        repositories = config_data.get('repositories', [])
        if not isinstance(repositories, list):
            raise ValueError("'repositories' field must be a list")

        if not merge:
            self.repositories.clear()

        # HIGH PRIORITY FIX: Validate each repository before importing
        imported_count = 0
        errors = []

        for i, repo_data in enumerate(repositories):
            try:
                if not isinstance(repo_data, dict):
                    raise ValueError(f"Repository entry {i} must be an object")

                # Validate required fields
                required_fields = ['id', 'name', 'path']
                for field in required_fields:
                    if field not in repo_data:
                        raise ValueError(f"Missing required field: {field}")

                repo = Repository.from_dict(repo_data)
                self.repositories[repo.id] = repo
                imported_count += 1

            except Exception as e:
                error_msg = f"Error importing repository {i}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        self._save_repositories()

        if errors:
            logger.warning(f"Imported {imported_count} repositories with {len(errors)} errors")
            # Raise exception with all errors
            raise ValueError(f"Import completed with errors:\n" + "\n".join(errors))

        logger.info(f"Imported {imported_count} repositories successfully")

    def _load_repositories(self):
        """Load repositories from config file."""
        if not self.config_file.exists():
            return

        try:
            # HIGH PRIORITY FIX: Explicit JSON error handling
            try:
                config_data = json.loads(self.config_file.read_text())
            except json.JSONDecodeError as e:
                logger.error(f"Malformed JSON in repository config: {e}")
                raise ValueError(f"Repository configuration file contains invalid JSON") from e

            # HIGH PRIORITY FIX: Validate structure
            if not isinstance(config_data, dict):
                raise ValueError("Repository configuration must be a JSON object")

            repositories = config_data.get('repositories', [])
            if not isinstance(repositories, list):
                raise ValueError("'repositories' field must be a list")

            for repo_data in repositories:
                try:
                    repo = Repository.from_dict(repo_data)
                    self.repositories[repo.id] = repo
                except Exception as e:
                    logger.warning(f"Skipping malformed repository entry: {e}")

            logger.info(f"Loaded {len(self.repositories)} repositories")

        except ValueError:
            # Re-raise ValueError (JSON errors)
            raise
        except Exception as e:
            logger.error(f"Error loading repositories: {e}")
            raise

    def _save_repositories(self):
        """Save repositories to config file."""
        try:
            config_data = {
                'version': '1.0.0',
                'last_updated': datetime.now().isoformat(),
                'repositories': [r.to_dict() for r in self.repositories.values()]
            }

            self.config_file.write_text(json.dumps(config_data, indent=2))

        except Exception as e:
            logger.error(f"Error saving repositories: {e}")


# Global instance
multi_repo_manager = MultiRepositoryManager()
