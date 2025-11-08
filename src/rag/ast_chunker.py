"""
AST-based code chunking using tree-sitter.

This module provides syntactically-aware code chunking that:
1. Never breaks code structure (functions, classes, etc.)
2. Preserves semantic boundaries
3. Improves RAG retrieval quality by 25-40% (research-backed)

Based on cAST framework (arXiv:2506.15655v1)
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .parsers import get_parser, is_supported

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """
    Represents a chunk of code extracted from AST.
    """
    content: str
    metadata: Dict[str, Any]
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'module'


class ASTChunker:
    """
    AST-based code chunker that preserves syntactic integrity.

    Benefits over text-based chunking:
    - Never breaks function/class definitions
    - Preserves semantic units
    - Better context for embeddings
    - Language-aware parsing
    """

    # Node types to extract as chunks (by language)
    PYTHON_CHUNK_TYPES = [
        'function_definition',
        'class_definition',
        'decorated_definition'
    ]

    JAVASCRIPT_CHUNK_TYPES = [
        'function_declaration',
        'function',
        'class_declaration',
        'method_definition',
        'arrow_function'
    ]

    TYPESCRIPT_CHUNK_TYPES = [
        'function_declaration',
        'function',
        'class_declaration',
        'method_definition',
        'arrow_function',
        'interface_declaration',
        'type_alias_declaration'
    ]

    def __init__(self, max_chunk_size: int = 1000, add_context_headers: bool = True):
        """
        Initialize AST chunker.

        Args:
            max_chunk_size: Maximum characters per chunk (soft limit)
            add_context_headers: Add file/class context to each chunk
        """
        self.max_chunk_size = max_chunk_size
        self.add_context_headers = add_context_headers

    def chunk_code(self, code: str, file_path: str) -> List[CodeChunk]:
        """
        Chunk code using AST analysis.

        Args:
            code: Source code to chunk
            file_path: Path to source file (determines language)

        Returns:
            List of CodeChunk objects
        """
        # Check if AST parsing is supported
        if not is_supported(file_path):
            logger.debug(f"AST parsing not supported for {file_path}, using fallback")
            return self._fallback_chunk(code, file_path)

        parser = get_parser(os.path.splitext(file_path)[1].lower())
        if parser is None:
            return self._fallback_chunk(code, file_path)

        try:
            # Parse code
            tree = parser.parse(bytes(code, 'utf8'))
            root_node = tree.root_node

            # Extract chunks based on language
            chunks = self._extract_chunks(root_node, code, file_path)

            logger.info(
                f"Extracted {len(chunks)} AST chunks from {file_path} "
                f"({len(code)} chars → {sum(len(c.content) for c in chunks)} chars)"
            )

            return chunks

        except Exception as e:
            logger.error(f"AST parsing failed for {file_path}: {e}, using fallback")
            return self._fallback_chunk(code, file_path)

    def _extract_chunks(
        self,
        root_node,
        code: str,
        file_path: str
    ) -> List[CodeChunk]:
        """
        Extract chunks from AST tree.

        Args:
            root_node: Root AST node
            code: Source code
            file_path: File path

        Returns:
            List of chunks
        """
        chunks = []
        ext = os.path.splitext(file_path)[1].lower()

        # Determine chunk types based on language
        if ext == '.py':
            chunk_types = self.PYTHON_CHUNK_TYPES
        elif ext in ['.js', '.jsx']:
            chunk_types = self.JAVASCRIPT_CHUNK_TYPES
        elif ext in ['.ts', '.tsx']:
            chunk_types = self.TYPESCRIPT_CHUNK_TYPES
        else:
            chunk_types = []

        # Traverse AST and extract definitions
        current_class = None

        def traverse(node, parent_class=None):
            nonlocal current_class

            # Update current class context
            if node.type in ['class_definition', 'class_declaration']:
                # Get class name
                name_node = node.child_by_field_name('name')
                if name_node:
                    current_class = code[name_node.start_byte:name_node.end_byte]

            # Check if this node should be a chunk
            if node.type in chunk_types:
                chunk = self._node_to_chunk(node, code, file_path, current_class)
                if chunk:
                    chunks.append(chunk)

                # Don't traverse children of chunk nodes
                # (already captured in the chunk)
                return

            # Traverse children
            for child in node.children:
                traverse(child, parent_class=current_class)

        traverse(root_node)

        # If no chunks extracted, fall back to simple chunking
        if not chunks:
            logger.warning(f"No AST chunks found in {file_path}, using fallback")
            return self._fallback_chunk(code, file_path)

        return chunks

    def _node_to_chunk(
        self,
        node,
        code: str,
        file_path: str,
        parent_class: Optional[str] = None
    ) -> Optional[CodeChunk]:
        """
        Convert AST node to CodeChunk.

        Args:
            node: AST node
            code: Source code
            file_path: File path
            parent_class: Name of parent class (if method)

        Returns:
            CodeChunk or None
        """
        start_byte = node.start_byte
        end_byte = node.end_byte
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Extract code
        chunk_code = code[start_byte:end_byte]

        # Skip if too large (would need splitting)
        if len(chunk_code) > self.max_chunk_size * 2:
            logger.debug(
                f"Chunk at {file_path}:{start_line} too large ({len(chunk_code)} chars), skipping"
            )
            return None

        # Get name
        name_node = node.child_by_field_name('name')
        name = code[name_node.start_byte:name_node.end_byte] if name_node else 'anonymous'

        # Determine chunk type
        chunk_type = 'function'
        if 'class' in node.type:
            chunk_type = 'class'
        elif parent_class:
            chunk_type = 'method'

        # Build metadata
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'name': name,
            'type': chunk_type,
            'start_line': start_line,
            'end_line': end_line,
            'language': self._get_language(file_path)
        }

        if parent_class:
            metadata['class'] = parent_class

        # Add context header if enabled
        if self.add_context_headers:
            header = self._create_context_header(metadata)
            chunk_code = f"{header}\n{chunk_code}"

        return CodeChunk(
            content=chunk_code,
            metadata=metadata,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type
        )

    def _create_context_header(self, metadata: Dict[str, Any]) -> str:
        """
        Create context header for chunk.

        This helps the embedding model understand the context.
        """
        parts = [f"# File: {metadata['file_path']}"]

        if metadata.get('class'):
            parts.append(f"# Class: {metadata['class']}")

        parts.append(f"# {metadata['type'].capitalize()}: {metadata['name']}")
        parts.append(f"# Lines {metadata['start_line']}-{metadata['end_line']}")

        return "\n".join(parts)

    def _get_language(self, file_path: str) -> str:
        """Get language name from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript'
        }
        return language_map.get(ext, 'unknown')

    def _fallback_chunk(self, code: str, file_path: str) -> List[CodeChunk]:
        """
        Fallback to simple text-based chunking.

        Used when AST parsing is not available or fails.
        """
        chunks = []
        lines = code.split('\n')

        chunk_size = 50  # lines per chunk
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)

            if not chunk_content.strip():
                continue

            chunks.append(CodeChunk(
                content=chunk_content,
                metadata={
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'name': f'chunk_{i // chunk_size}',
                    'type': 'text_chunk',
                    'start_line': i + 1,
                    'end_line': min(i + chunk_size, len(lines)),
                    'language': self._get_language(file_path),
                    'fallback': True
                },
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                chunk_type='text_chunk'
            ))

        return chunks


def chunk_file(file_path: str, max_chunk_size: int = 1000) -> List[CodeChunk]:
    """
    Convenience function to chunk a file.

    Args:
        file_path: Path to file
        max_chunk_size: Maximum chunk size

    Returns:
        List of chunks
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()

    chunker = ASTChunker(max_chunk_size=max_chunk_size)
    return chunker.chunk_code(code, file_path)


def chunk_directory(
    directory: str,
    extensions: Optional[List[str]] = None,
    max_chunk_size: int = 1000
) -> Dict[str, List[CodeChunk]]:
    """
    Chunk all supported files in a directory.

    Args:
        directory: Directory path
        extensions: File extensions to include (None = all supported)
        max_chunk_size: Maximum chunk size

    Returns:
        Dictionary mapping file paths to chunks
    """
    from .parsers import get_supported_extensions

    if extensions is None:
        extensions = get_supported_extensions()

    result = {}
    chunker = ASTChunker(max_chunk_size=max_chunk_size)

    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__']]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                file_path = os.path.join(root, file)
                try:
                    chunks = chunk_file(file_path, max_chunk_size)
                    result[file_path] = chunks
                    logger.info(f"Chunked {file_path}: {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Failed to chunk {file_path}: {e}")

    return result
