"""
Multi-language parser setup for AST-based code analysis.

Supports multiple programming languages using tree-sitter.
This enables syntactically-aware code chunking that never breaks
code structure.
"""

import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ParserRegistry:
    """
    Registry for tree-sitter parsers across multiple languages.

    Lazy-loads parsers on demand to avoid startup overhead.
    """

    def __init__(self):
        self._parsers = {}
        self._languages = {}
        self._initialized = False

    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        if self._initialized:
            return

        try:
            from tree_sitter import Language, Parser
            import tree_sitter_python as tspython
            import tree_sitter_javascript as tsjs
            import tree_sitter_typescript as tsts

            # Python
            self._languages['.py'] = Language(tspython.language())
            logger.info("Loaded Python parser")

            # JavaScript
            self._languages['.js'] = Language(tsjs.language())
            self._languages['.jsx'] = Language(tsjs.language())
            logger.info("Loaded JavaScript parser")

            # TypeScript
            ts_lang = Language(tsts.language_typescript())
            tsx_lang = Language(tsts.language_tsx())

            self._languages['.ts'] = ts_lang
            self._languages['.tsx'] = tsx_lang
            logger.info("Loaded TypeScript parser")

            self._initialized = True
            logger.info(f"Initialized parsers for {len(self._languages)} file types")

        except ImportError as e:
            logger.warning(
                f"Failed to load tree-sitter parsers: {e}. "
                "AST-based chunking will fall back to text chunking. "
                "Install with: pip install tree-sitter tree-sitter-python "
                "tree-sitter-javascript tree-sitter-typescript"
            )
            self._initialized = True  # Mark as initialized to avoid repeated attempts

    def get_parser(self, file_extension: str) -> Optional['Parser']:
        """
        Get parser for a file extension.

        Args:
            file_extension: File extension (e.g., '.py', '.js')

        Returns:
            Parser instance or None if not supported
        """
        self._initialize_parsers()

        if file_extension not in self._languages:
            return None

        # Create or get cached parser
        if file_extension not in self._parsers:
            try:
                from tree_sitter import Parser
                parser = Parser()
                parser.set_language(self._languages[file_extension])
                self._parsers[file_extension] = parser
            except Exception as e:
                logger.error(f"Failed to create parser for {file_extension}: {e}")
                return None

        return self._parsers[file_extension]

    def get_language(self, file_extension: str) -> Optional['Language']:
        """
        Get Language object for a file extension.

        Args:
            file_extension: File extension

        Returns:
            Language object or None
        """
        self._initialize_parsers()
        return self._languages.get(file_extension)

    def is_supported(self, file_path: str) -> bool:
        """
        Check if a file is supported for AST parsing.

        Args:
            file_path: Path to file

        Returns:
            True if supported, False otherwise
        """
        ext = os.path.splitext(file_path)[1].lower()
        self._initialize_parsers()
        return ext in self._languages

    def get_supported_extensions(self) -> list:
        """
        Get list of supported file extensions.

        Returns:
            List of extensions (e.g., ['.py', '.js', '.ts'])
        """
        self._initialize_parsers()
        return list(self._languages.keys())


# Global parser registry instance
_parser_registry = ParserRegistry()


def get_parser(file_extension: str):
    """Get parser for a file extension."""
    return _parser_registry.get_parser(file_extension)


def get_language(file_extension: str):
    """Get Language object for a file extension."""
    return _parser_registry.get_language(file_extension)


def is_supported(file_path: str) -> bool:
    """Check if a file is supported for AST parsing."""
    return _parser_registry.is_supported(file_path)


def get_supported_extensions() -> list:
    """Get list of supported file extensions."""
    return _parser_registry.get_supported_extensions()
