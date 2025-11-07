"""
Code snippet management system.

Features:
- Save code snippets with metadata
- Tag and categorize snippets
- Search snippets
- Link snippets to queries
- Export/import snippet libraries
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CodeSnippet:
    """Represents a code snippet."""

    id: str
    title: str
    code: str
    language: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    source_repo: Optional[str] = None
    line_range: Optional[tuple] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    use_count: int = 0
    related_query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.line_range:
            data['line_range'] = list(self.line_range)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeSnippet':
        """Create from dictionary."""
        if 'line_range' in data and data['line_range']:
            data['line_range'] = tuple(data['line_range'])
        return cls(**data)


class SnippetManager:
    """
    Manages code snippets.
    """

    def __init__(self, storage_file: Optional[Path] = None):
        """
        Initialize snippet manager.

        Args:
            storage_file: Path to snippets storage file
        """
        self.storage_file = storage_file or Path.home() / '.rag_snippets.json'
        self.snippets: Dict[str, CodeSnippet] = {}

        self._load_snippets()

    def add_snippet(
        self,
        title: str,
        code: str,
        language: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        source_file: Optional[str] = None,
        source_repo: Optional[str] = None,
        line_range: Optional[tuple] = None,
        related_query: Optional[str] = None
    ) -> CodeSnippet:
        """
        Add code snippet.

        Args:
            title: Snippet title
            code: Code content
            language: Programming language
            description: Description
            tags: Optional tags
            source_file: Source file path
            source_repo: Source repository
            line_range: Line range (start, end)
            related_query: Related search query

        Returns:
            CodeSnippet instance
        """
        # Generate ID from code content
        snippet_id = self._generate_id(code, title)

        # Check if already exists
        if snippet_id in self.snippets:
            logger.warning(f"Snippet already exists: {title}")
            return self.snippets[snippet_id]

        # Create snippet
        snippet = CodeSnippet(
            id=snippet_id,
            title=title,
            code=code,
            language=language,
            description=description,
            tags=tags or [],
            source_file=source_file,
            source_repo=source_repo,
            line_range=line_range,
            related_query=related_query
        )

        self.snippets[snippet_id] = snippet
        self._save_snippets()

        logger.info(f"Added snippet: {title}")
        return snippet

    def get_snippet(self, snippet_id: str) -> Optional[CodeSnippet]:
        """
        Get snippet by ID.

        Args:
            snippet_id: Snippet ID

        Returns:
            CodeSnippet or None
        """
        snippet = self.snippets.get(snippet_id)

        if snippet:
            # Update usage statistics
            snippet.last_used = datetime.now().isoformat()
            snippet.use_count += 1
            self._save_snippets()

        return snippet

    def search_snippets(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> List[CodeSnippet]:
        """
        Search snippets.

        Args:
            query: Search query
            tags: Filter by tags
            language: Filter by language

        Returns:
            List of matching snippets
        """
        query_lower = query.lower()

        matches = []

        for snippet in self.snippets.values():
            # Filter by language
            if language and snippet.language != language:
                continue

            # Filter by tags
            if tags and not (set(tags) & set(snippet.tags)):
                continue

            # Search in title, description, and code
            if (query_lower in snippet.title.lower() or
                query_lower in snippet.description.lower() or
                query_lower in snippet.code.lower()):
                matches.append(snippet)

        # Sort by relevance (use_count as proxy)
        matches.sort(key=lambda s: s.use_count, reverse=True)

        return matches

    def list_snippets(
        self,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        sort_by: str = 'created_at'
    ) -> List[CodeSnippet]:
        """
        List snippets.

        Args:
            tags: Filter by tags
            language: Filter by language
            sort_by: Sort field ('created_at', 'title', 'use_count')

        Returns:
            List of snippets
        """
        snippets = list(self.snippets.values())

        # Filter by tags
        if tags:
            tag_set = set(tags)
            snippets = [s for s in snippets if tag_set & set(s.tags)]

        # Filter by language
        if language:
            snippets = [s for s in snippets if s.language == language]

        # Sort
        if sort_by == 'title':
            snippets.sort(key=lambda s: s.title)
        elif sort_by == 'use_count':
            snippets.sort(key=lambda s: s.use_count, reverse=True)
        elif sort_by == 'created_at':
            snippets.sort(key=lambda s: s.created_at, reverse=True)

        return snippets

    def delete_snippet(self, snippet_id: str) -> bool:
        """
        Delete snippet.

        Args:
            snippet_id: Snippet ID

        Returns:
            True if deleted
        """
        if snippet_id in self.snippets:
            title = self.snippets[snippet_id].title
            del self.snippets[snippet_id]
            self._save_snippets()

            logger.info(f"Deleted snippet: {title}")
            return True

        return False

    def update_snippet(
        self,
        snippet_id: str,
        **updates
    ) -> Optional[CodeSnippet]:
        """
        Update snippet.

        Args:
            snippet_id: Snippet ID
            **updates: Fields to update

        Returns:
            Updated snippet or None
        """
        snippet = self.snippets.get(snippet_id)

        if not snippet:
            return None

        # Update fields
        for key, value in updates.items():
            if hasattr(snippet, key):
                setattr(snippet, key, value)

        self._save_snippets()

        logger.info(f"Updated snippet: {snippet.title}")
        return snippet

    def add_tag(self, snippet_id: str, tag: str) -> bool:
        """
        Add tag to snippet.

        Args:
            snippet_id: Snippet ID
            tag: Tag to add

        Returns:
            True if added
        """
        snippet = self.snippets.get(snippet_id)

        if snippet and tag not in snippet.tags:
            snippet.tags.append(tag)
            self._save_snippets()
            return True

        return False

    def remove_tag(self, snippet_id: str, tag: str) -> bool:
        """
        Remove tag from snippet.

        Args:
            snippet_id: Snippet ID
            tag: Tag to remove

        Returns:
            True if removed
        """
        snippet = self.snippets.get(snippet_id)

        if snippet and tag in snippet.tags:
            snippet.tags.remove(tag)
            self._save_snippets()
            return True

        return False

    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags.

        Returns:
            List of tags
        """
        tags = set()

        for snippet in self.snippets.values():
            tags.update(snippet.tags)

        return sorted(list(tags))

    def get_all_languages(self) -> List[str]:
        """
        Get all unique languages.

        Returns:
            List of languages
        """
        languages = set(s.language for s in self.snippets.values())
        return sorted(list(languages))

    def get_popular_snippets(self, top_k: int = 10) -> List[CodeSnippet]:
        """
        Get most popular snippets.

        Args:
            top_k: Number of top snippets

        Returns:
            List of popular snippets
        """
        snippets = list(self.snippets.values())
        snippets.sort(key=lambda s: s.use_count, reverse=True)

        return snippets[:top_k]

    def get_recent_snippets(self, count: int = 10) -> List[CodeSnippet]:
        """
        Get recently added snippets.

        Args:
            count: Number of snippets

        Returns:
            List of recent snippets
        """
        snippets = list(self.snippets.values())
        snippets.sort(key=lambda s: s.created_at, reverse=True)

        return snippets[:count]

    def export_snippets(self, output_path: Path, snippet_ids: Optional[List[str]] = None):
        """
        Export snippets to file.

        Args:
            output_path: Output file path
            snippet_ids: Specific snippet IDs (None = all)
        """
        if snippet_ids:
            snippets_to_export = [self.snippets[sid] for sid in snippet_ids if sid in self.snippets]
        else:
            snippets_to_export = list(self.snippets.values())

        export_data = {
            'version': '1.0.0',
            'exported_at': datetime.now().isoformat(),
            'count': len(snippets_to_export),
            'snippets': [s.to_dict() for s in snippets_to_export]
        }

        output_path.write_text(json.dumps(export_data, indent=2))

        logger.info(f"Exported {len(snippets_to_export)} snippets to {output_path}")

    def import_snippets(self, input_path: Path, merge: bool = True):
        """
        Import snippets from file.

        Args:
            input_path: Input file path
            merge: Merge with existing (True) or replace (False)
        """
        import_data = json.loads(input_path.read_text())

        if not merge:
            self.snippets.clear()

        imported_count = 0

        for snippet_data in import_data.get('snippets', []):
            snippet = CodeSnippet.from_dict(snippet_data)

            if merge and snippet.id in self.snippets:
                # Skip existing snippets in merge mode
                continue

            self.snippets[snippet.id] = snippet
            imported_count += 1

        self._save_snippets()

        logger.info(f"Imported {imported_count} snippets from {input_path}")

    def create_from_search_result(
        self,
        result: Dict[str, Any],
        title: Optional[str] = None
    ) -> CodeSnippet:
        """
        Create snippet from search result.

        Args:
            result: Search result dictionary
            title: Optional title (generated if not provided)

        Returns:
            CodeSnippet instance
        """
        # Extract information from result
        code = result.get('content', '')
        metadata = result.get('metadata', {})

        # Generate title if not provided
        if not title:
            # Use first line or file name
            first_line = code.split('\n')[0] if code else 'Untitled'
            title = first_line[:50] + ('...' if len(first_line) > 50 else '')

        # Detect language from file extension
        file_path = metadata.get('file_path', '')
        language = self._detect_language(file_path)

        # Create snippet
        return self.add_snippet(
            title=title,
            code=code,
            language=language,
            source_file=file_path,
            source_repo=metadata.get('repository'),
            line_range=metadata.get('line_range')
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get snippet statistics.

        Returns:
            Statistics dictionary
        """
        total_snippets = len(self.snippets)

        if total_snippets == 0:
            return {'total_snippets': 0}

        # Language distribution
        language_counts = {}
        for snippet in self.snippets.values():
            lang = snippet.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        # Tag distribution
        tag_counts = {}
        for snippet in self.snippets.values():
            for tag in snippet.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Usage statistics
        use_counts = [s.use_count for s in self.snippets.values()]

        return {
            'total_snippets': total_snippets,
            'languages': language_counts,
            'tags': tag_counts,
            'total_uses': sum(use_counts),
            'avg_uses_per_snippet': sum(use_counts) / total_snippets if total_snippets > 0 else 0,
            'most_popular': self.get_popular_snippets(5)
        }

    def _generate_id(self, code: str, title: str) -> str:
        """
        Generate unique ID for snippet.

        HIGH PRIORITY FIX: Removed timestamp from ID generation.
        Identical code+title should get same ID for deduplication.

        Args:
            code: Code content
            title: Snippet title

        Returns:
            Unique ID
        """
        # HIGH PRIORITY FIX: Remove timestamp - identical snippets should have same ID
        content = f"{title}_{code}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _detect_language(self, file_path: str) -> str:
        """
        Detect programming language from file path.

        Args:
            file_path: File path

        Returns:
            Language name
        """
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sh': 'bash',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
        }

        ext = Path(file_path).suffix.lower()
        return extension_map.get(ext, 'text')

    def _load_snippets(self):
        """Load snippets from storage file."""
        if not self.storage_file.exists():
            return

        try:
            data = json.loads(self.storage_file.read_text())

            for snippet_data in data.get('snippets', []):
                snippet = CodeSnippet.from_dict(snippet_data)
                self.snippets[snippet.id] = snippet

            logger.info(f"Loaded {len(self.snippets)} snippets")

        except Exception as e:
            logger.error(f"Error loading snippets: {e}")

    def _save_snippets(self):
        """Save snippets to storage file."""
        try:
            data = {
                'version': '1.0.0',
                'last_updated': datetime.now().isoformat(),
                'count': len(self.snippets),
                'snippets': [s.to_dict() for s in self.snippets.values()]
            }

            self.storage_file.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"Error saving snippets: {e}")


# Global instance
snippet_manager = SnippetManager()
