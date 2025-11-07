"""
Saved searches and query bookmarks.

Allows users to:
- Save frequently used queries
- Organize searches by tags
- Share search collections
- Quick access to favorite queries
"""

import json
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class SavedSearch:
    """Represents a saved search query."""

    id: str
    name: str
    query: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    n_results: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    use_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SavedSearch':
        """Create from dictionary."""
        return cls(**data)


class SavedSearchManager:
    """
    Manages saved searches and bookmarks.

    MEDIUM PRIORITY FIX: Added thread safety and atomic operations.
    """

    def __init__(self, storage_file: Optional[Path] = None):
        """
        Initialize saved search manager.

        Args:
            storage_file: Path to storage file (default: .rag_saved_searches.json)
        """
        self.storage_file = storage_file or Path.home() / '.rag_saved_searches.json'
        self.searches: Dict[str, SavedSearch] = {}

        # MEDIUM PRIORITY FIX: Add thread safety
        self._searches_lock = threading.RLock()

        self._load_searches()

    def save_search(
        self,
        name: str,
        query: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> SavedSearch:
        """
        Save a search query.

        MEDIUM PRIORITY FIX: Thread-safe save operation.

        Args:
            name: Search name
            query: Query text
            description: Optional description
            tags: Optional tags
            n_results: Number of results
            filters: Optional filters

        Returns:
            SavedSearch instance
        """
        # Generate ID
        search_id = self._generate_id(name)

        # Create search
        search = SavedSearch(
            id=search_id,
            name=name,
            query=query,
            description=description,
            tags=tags or [],
            n_results=n_results,
            filters=filters or {}
        )

        # MEDIUM PRIORITY FIX: Thread-safe dictionary update
        with self._searches_lock:
            self.searches[search_id] = search
            self._save_searches()

        logger.info(f"Saved search: {name}")
        return search

    def get_search(self, search_id: str) -> Optional[SavedSearch]:
        """
        Get saved search by ID.

        Args:
            search_id: Search ID

        Returns:
            SavedSearch or None
        """
        return self.searches.get(search_id)

    def get_search_by_name(self, name: str) -> Optional[SavedSearch]:
        """
        Get saved search by name.

        Args:
            name: Search name

        Returns:
            SavedSearch or None
        """
        for search in self.searches.values():
            if search.name == name:
                return search

        return None

    def delete_search(self, search_id: str) -> bool:
        """
        Delete saved search.

        MEDIUM PRIORITY FIX: Thread-safe delete operation.

        Args:
            search_id: Search ID

        Returns:
            True if deleted
        """
        # MEDIUM PRIORITY FIX: Thread-safe dictionary delete
        with self._searches_lock:
            if search_id in self.searches:
                del self.searches[search_id]
                self._save_searches()
                logger.info(f"Deleted search: {search_id}")
                return True

        return False

    def list_searches(
        self,
        tags: Optional[List[str]] = None,
        sort_by: str = 'name'
    ) -> List[SavedSearch]:
        """
        List saved searches.

        Args:
            tags: Filter by tags
            sort_by: Sort by field ('name', 'created_at', 'use_count')

        Returns:
            List of saved searches
        """
        searches = list(self.searches.values())

        # Filter by tags
        if tags:
            tag_set = set(tags)
            searches = [
                s for s in searches
                if tag_set & set(s.tags)
            ]

        # Sort
        if sort_by == 'name':
            searches.sort(key=lambda s: s.name)
        elif sort_by == 'created_at':
            searches.sort(key=lambda s: s.created_at, reverse=True)
        elif sort_by == 'use_count':
            searches.sort(key=lambda s: s.use_count, reverse=True)

        return searches

    def execute_search(
        self,
        search_id: str,
        query_engine
    ) -> Dict[str, Any]:
        """
        Execute a saved search.

        Args:
            search_id: Search ID
            query_engine: Query engine instance

        Returns:
            Query results
        """
        search = self.get_search(search_id)

        if not search:
            raise ValueError(f"Search not found: {search_id}")

        # Update usage stats
        search.last_used = datetime.now().isoformat()
        search.use_count += 1
        self._save_searches()

        # Execute query
        results = query_engine.query(
            search.query,
            n_results=search.n_results
        )

        return {
            'search': search.to_dict(),
            'results': results
        }

    def update_search(
        self,
        search_id: str,
        **updates
    ) -> Optional[SavedSearch]:
        """
        Update saved search.

        Args:
            search_id: Search ID
            **updates: Fields to update

        Returns:
            Updated SavedSearch or None
        """
        search = self.get_search(search_id)

        if not search:
            return None

        # Update fields
        for key, value in updates.items():
            if hasattr(search, key):
                setattr(search, key, value)

        self._save_searches()

        logger.info(f"Updated search: {search_id}")
        return search

    def add_tag(self, search_id: str, tag: str) -> bool:
        """
        Add tag to search.

        Args:
            search_id: Search ID
            tag: Tag to add

        Returns:
            True if added
        """
        search = self.get_search(search_id)

        if search and tag not in search.tags:
            search.tags.append(tag)
            self._save_searches()
            return True

        return False

    def remove_tag(self, search_id: str, tag: str) -> bool:
        """
        Remove tag from search.

        Args:
            search_id: Search ID
            tag: Tag to remove

        Returns:
            True if removed
        """
        search = self.get_search(search_id)

        if search and tag in search.tags:
            search.tags.remove(tag)
            self._save_searches()
            return True

        return False

    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags.

        Returns:
            List of tags
        """
        tags: Set[str] = set()

        for search in self.searches.values():
            tags.update(search.tags)

        return sorted(list(tags))

    def export_searches(self, output_path: Path):
        """
        Export searches to file.

        Args:
            output_path: Output file path
        """
        data = {
            'version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'searches': [s.to_dict() for s in self.searches.values()]
        }

        output_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Exported {len(self.searches)} searches to {output_path}")

    def import_searches(self, input_path: Path, merge: bool = True):
        """
        Import searches from file.

        Args:
            input_path: Input file path
            merge: Merge with existing (True) or replace (False)
        """
        data = json.loads(input_path.read_text())

        imported_searches = [
            SavedSearch.from_dict(s)
            for s in data.get('searches', [])
        ]

        if not merge:
            self.searches.clear()

        for search in imported_searches:
            self.searches[search.id] = search

        self._save_searches()

        logger.info(f"Imported {len(imported_searches)} searches from {input_path}")

    def get_popular_searches(self, top_k: int = 10) -> List[SavedSearch]:
        """
        Get most frequently used searches.

        Args:
            top_k: Number of top searches

        Returns:
            List of popular searches
        """
        searches = list(self.searches.values())
        searches.sort(key=lambda s: s.use_count, reverse=True)

        return searches[:top_k]

    def search_by_query(self, query_text: str) -> List[SavedSearch]:
        """
        Find saved searches by query text.

        Args:
            query_text: Query text to search for

        Returns:
            List of matching searches
        """
        query_lower = query_text.lower()

        matches = []

        for search in self.searches.values():
            if (query_lower in search.name.lower() or
                query_lower in search.query.lower() or
                query_lower in search.description.lower()):
                matches.append(search)

        return matches

    def _generate_id(self, name: str) -> str:
        """
        Generate unique ID for search.

        Args:
            name: Search name

        Returns:
            Unique ID
        """
        import hashlib

        # Use name + timestamp for uniqueness
        timestamp = datetime.now().isoformat()
        hash_input = f"{name}_{timestamp}"

        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _load_searches(self):
        """
        Load searches from file.

        MEDIUM PRIORITY FIX: Validate loaded search structure.
        """
        if not self.storage_file.exists():
            return

        try:
            data = json.loads(self.storage_file.read_text())

            # MEDIUM PRIORITY FIX: Validate structure
            if not isinstance(data, dict):
                raise ValueError(f"Invalid searches format: expected dict, got {type(data)}")

            if 'searches' not in data:
                logger.warning("Searches file missing 'searches' field, treating as empty")
                return

            if not isinstance(data['searches'], list):
                raise ValueError(f"Invalid searches format: expected list, got {type(data['searches'])}")

            # MEDIUM PRIORITY FIX: Validate and load searches with error recovery
            loaded_count = 0
            errors = 0

            for idx, search_data in enumerate(data['searches']):
                try:
                    # Validate search structure
                    if not isinstance(search_data, dict):
                        logger.warning(f"Skipping invalid search at index {idx}: not a dict")
                        errors += 1
                        continue

                    # Check required fields
                    required_fields = ['id', 'name', 'query']
                    missing_fields = [f for f in required_fields if f not in search_data]

                    if missing_fields:
                        logger.warning(
                            f"Skipping search at index {idx}: missing fields {missing_fields}"
                        )
                        errors += 1
                        continue

                    # Load search
                    search = SavedSearch.from_dict(search_data)
                    self.searches[search.id] = search
                    loaded_count += 1

                except Exception as e:
                    logger.warning(f"Error loading search at index {idx}: {e}")
                    errors += 1

            logger.info(
                f"Loaded {loaded_count} saved searches "
                f"({errors} errors, {len(self.searches)} total)"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted searches file: {e}. Starting with empty searches.")
            self.searches = {}

        except Exception as e:
            logger.error(f"Error loading saved searches: {e}. Starting with empty searches.")
            self.searches = {}

    def _save_searches(self):
        """
        Save searches to file.

        MEDIUM PRIORITY FIX: Use atomic write to prevent corruption.
        Note: Must be called within _searches_lock context.
        """
        import tempfile
        import os

        try:
            # Prepare data (must be called within lock, so we have consistent snapshot)
            data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'searches': [s.to_dict() for s in self.searches.values()]
            }

            # MEDIUM PRIORITY FIX: Atomic write using temp file + rename
            # Ensure parent directory exists
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)

            # Serialize to JSON
            json_data = json.dumps(data, indent=2)

            # Write to temp file
            fd, temp_path = tempfile.mkstemp(
                dir=str(self.storage_file.parent),
                prefix='.rag_saved_searches.tmp.',
                suffix='.json'
            )

            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(json_data)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure written to disk

                # Atomic rename
                os.replace(temp_path, str(self.storage_file))

                logger.debug(f"Saved {len(self.searches)} searches")

            except Exception:
                # Cleanup temp file on error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            logger.error(f"Error saving searches: {e}")


class SearchCollection:
    """
    Collection of related saved searches.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize search collection.

        Args:
            name: Collection name
            description: Collection description
        """
        self.name = name
        self.description = description
        self.search_ids: List[str] = []

    def add_search(self, search_id: str):
        """
        Add search to collection.

        Args:
            search_id: Search ID
        """
        if search_id not in self.search_ids:
            self.search_ids.append(search_id)

    def remove_search(self, search_id: str):
        """
        Remove search from collection.

        Args:
            search_id: Search ID
        """
        if search_id in self.search_ids:
            self.search_ids.remove(search_id)

    def get_searches(self, manager: SavedSearchManager) -> List[SavedSearch]:
        """
        Get all searches in collection.

        Args:
            manager: SavedSearchManager instance

        Returns:
            List of searches
        """
        searches = []

        for search_id in self.search_ids:
            search = manager.get_search(search_id)
            if search:
                searches.append(search)

        return searches

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'search_ids': self.search_ids
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchCollection':
        """Create from dictionary."""
        collection = cls(
            name=data['name'],
            description=data.get('description', '')
        )
        collection.search_ids = data.get('search_ids', [])

        return collection


# Global instance
saved_search_manager = SavedSearchManager()
