"""
Entity extraction from code for knowledge graph.
"""

from typing import List, Dict, Any, Set, Optional
import re
import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeEntity:
    """Represents an entity in code."""

    def __init__(
        self,
        name: str,
        entity_type: str,
        file_path: str,
        line_number: int = 0,
        metadata: Optional[Dict] = None
    ):
        self.name = name
        self.entity_type = entity_type
        self.file_path = file_path
        self.line_number = line_number
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.entity_type,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'metadata': self.metadata
        }


class EntityExtractor:
    """
    Extract entities (classes, functions, variables) from code.
    """

    def __init__(self):
        """Initialize entity extractor."""
        self.supported_extensions = {'.py', '.js', '.ts', '.java'}

    def extract_from_file(self, file_path: Path) -> List[CodeEntity]:
        """
        Extract entities from a file.

        Args:
            file_path: Path to file

        Returns:
            List of extracted entities
        """
        if file_path.suffix not in self.supported_extensions:
            return []

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')

            if file_path.suffix == '.py':
                return self._extract_python(content, str(file_path))
            else:
                return self._extract_generic(content, str(file_path))

        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {e}")
            return []

    def _extract_python(self, content: str, file_path: str) -> List[CodeEntity]:
        """
        Extract entities from Python code using AST.

        Args:
            content: File content
            file_path: File path

        Returns:
            List of entities
        """
        entities = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                # Classes
                if isinstance(node, ast.ClassDef):
                    entities.append(CodeEntity(
                        name=node.name,
                        entity_type='class',
                        file_path=file_path,
                        line_number=node.lineno,
                        metadata={
                            'bases': [b.id for b in node.bases if isinstance(b, ast.Name)],
                            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
                        }
                    ))

                # Functions
                elif isinstance(node, ast.FunctionDef):
                    entities.append(CodeEntity(
                        name=node.name,
                        entity_type='function',
                        file_path=file_path,
                        line_number=node.lineno,
                        metadata={
                            'args': [arg.arg for arg in node.args.args],
                            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
                        }
                    ))

                # Imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        entities.append(CodeEntity(
                            name=alias.name,
                            entity_type='import',
                            file_path=file_path,
                            line_number=node.lineno
                        ))

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        entities.append(CodeEntity(
                            name=node.module,
                            entity_type='import',
                            file_path=file_path,
                            line_number=node.lineno,
                            metadata={'from_import': True}
                        ))

        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {file_path}: {e}")

        return entities

    def _extract_generic(self, content: str, file_path: str) -> List[CodeEntity]:
        """
        Extract entities using regex (for non-Python files).

        Args:
            content: File content
            file_path: File path

        Returns:
            List of entities
        """
        entities = []

        # Classes (JS/TS/Java)
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                entity_type='class',
                file_path=file_path,
                line_number=line_num
            ))

        # Functions (JS/TS)
        func_pattern = r'function\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                entity_type='function',
                file_path=file_path,
                line_number=line_num
            ))

        # Arrow functions (JS/TS)
        arrow_pattern = r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
        for match in re.finditer(arrow_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                entity_type='function',
                file_path=file_path,
                line_number=line_num
            ))

        return entities

    def extract_from_directory(
        self,
        root_path: Path,
        ignore_dirs: Set[str] = None
    ) -> List[CodeEntity]:
        """
        Extract entities from all files in directory.

        Args:
            root_path: Root directory path
            ignore_dirs: Directories to ignore

        Returns:
            List of all entities
        """
        if ignore_dirs is None:
            ignore_dirs = {'node_modules', '.git', 'venv', '__pycache__'}

        all_entities = []

        for file_path in root_path.rglob('*'):
            # Skip directories and ignored paths
            if file_path.is_dir():
                continue

            if any(ignored in file_path.parts for ignored in ignore_dirs):
                continue

            entities = self.extract_from_file(file_path)
            all_entities.extend(entities)

        logger.info(f"Extracted {len(all_entities)} entities from {root_path}")
        return all_entities
