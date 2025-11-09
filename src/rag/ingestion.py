"""
Document Ingestion Pipeline for processing and indexing code.
"""

from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import hashlib
import logging
import tiktoken

logger = logging.getLogger(__name__)


class DocumentIngestion:
 """
 Document ingestion pipeline for processing code files.
 """

 # File extensions to index
 CODE_EXTENSIONS = {
 '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
 '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
 '.sh', '.bash', '.sql', '.html', '.css', '.scss', '.less',
 '.json', '.yaml', '.yml', '.xml', '.md', '.rst', '.txt'
 }

 # Directories to ignore
 IGNORE_DIRS = {
 'node_modules', '.git', '.venv', 'venv', '__pycache__',
 '.pytest_cache', 'dist', 'build', '.next', '.nuxt',
 'coverage', '.coverage', '.rag_data', '.claude'
 }

 def __init__(
 self,
 chunk_size: int = 1000,
 chunk_overlap: int = 200,
 encoding_name: str = "cl100k_base",
 max_file_size: int = 10 * 1024 * 1024 # 10MB default
 ):
 """
 Initialize the document ingestion pipeline.

 MEDIUM PRIORITY FIX: Add file size limits to prevent memory issues.

 Args:
 chunk_size: Maximum size of text chunks in tokens
 chunk_overlap: Number of overlapping tokens between chunks
 encoding_name: Tokenizer encoding to use
 max_file_size: Maximum file size in bytes (default: 10MB)
 """
 self.chunk_size = chunk_size
 self.chunk_overlap = chunk_overlap
 self.encoding = tiktoken.get_encoding(encoding_name)
 self.max_file_size = max_file_size

 logger.info(
 f"Initialized DocumentIngestion "
 f"(chunk_size={chunk_size}, max_file_size={max_file_size // 1024 // 1024}MB)"
 )

 def discover_files(
 self,
 root_path: str,
 extensions: Optional[set] = None
 ) -> List[Path]:
 """
 Discover all code files in the directory.

 Args:
 root_path: Root directory to search
 extensions: File extensions to include (defaults to CODE_EXTENSIONS)

 Returns:
 List of file paths
 """
 if extensions is None:
 extensions = self.CODE_EXTENSIONS

 root = Path(root_path)
 files = []

 logger.info(f"Discovering files in {root_path}")

 for path in root.rglob('*'):
 # Skip ignored directories
 if any(ignored in path.parts for ignored in self.IGNORE_DIRS):
 continue

 # HIGH PRIORITY FIX: Skip symlinks to prevent traversal outside root
 if path.is_symlink():
 logger.debug(f"Skipping symlink: {path}")
 continue

 # Check extension
 if path.is_file() and path.suffix in extensions:
 # Additional security: Verify path is still within root
 try:
 path.resolve().relative_to(root.resolve())
 except ValueError:
 logger.warning(f"Skipping file outside root: {path}")
 continue

 files.append(path)

 logger.info(f"Discovered {len(files)} files")
 return files

 def read_file(self, file_path: Path) -> Optional[str]:
 """
 Read file content with error handling.

 MEDIUM PRIORITY FIX: Add file size limits to prevent memory issues.

 Args:
 file_path: Path to the file

 Returns:
 File content or None if error
 """
 try:
 # MEDIUM PRIORITY FIX: Check file size before reading
 file_size = file_path.stat().st_size

 if file_size > self.max_file_size:
 logger.warning(
 f"Skipping large file {file_path}: "
 f"{file_size / 1024 / 1024:.2f}MB exceeds limit of "
 f"{self.max_file_size / 1024 / 1024:.2f}MB"
 )
 return None

 if file_size == 0:
 logger.debug(f"Skipping empty file: {file_path}")
 return None

 # HIGH PRIORITY FIX: Use 'replace' instead of 'ignore'
 # 'ignore' silently drops invalid characters, corrupting documents
 # 'replace' replaces them with so we can detect corruption
 with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
 content = f.read()

 # Log if encoding errors were detected
 if '\ufffd' in content:
 logger.warning(
 f"Encoding errors detected in {file_path} "
 f"(invalid UTF-8 characters replaced with )"
 )

 return content
 except Exception as e:
 logger.warning(f"Error reading {file_path}: {e}")
 return None

 def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
 """
 Chunk text into smaller pieces with overlap.

 Args:
 text: Text to chunk
 metadata: Metadata for the document

 Returns:
 List of chunks with metadata
 """
 tokens = self.encoding.encode(text)
 chunks = []

 start = 0
 chunk_num = 0

 while start < len(tokens):
 end = start + self.chunk_size
 chunk_tokens = tokens[start:end]
 chunk_text = self.encoding.decode(chunk_tokens)

 chunk_metadata = {
 **metadata,
 'chunk_num': chunk_num,
 'start_token': start,
 'end_token': end
 }

 chunks.append({
 'text': chunk_text,
 'metadata': chunk_metadata
 })

 start += self.chunk_size - self.chunk_overlap
 chunk_num += 1

 return chunks

 def generate_id(self, file_path: str, chunk_num: int = 0) -> str:
 """
 Generate a unique ID for a document chunk.

 Args:
 file_path: Path to the file
 chunk_num: Chunk number

 Returns:
 Unique document ID
 """
 content = f"{file_path}:{chunk_num}"
 return hashlib.md5(content.encode()).hexdigest()

 def process_file(self, file_path: Path, root_path: str) -> List[Dict[str, Any]]:
 """
 Process a single file into chunks.

 Args:
 file_path: Path to the file
 root_path: Root directory path

 Returns:
 List of document chunks
 """
 content = self.read_file(file_path)
 if content is None:
 return []

 # Create relative path
 rel_path = str(file_path.relative_to(root_path))

 metadata = {
 'file_path': rel_path,
 'file_name': file_path.name,
 'file_type': file_path.suffix,
 'file_size': len(content)
 }

 chunks = self.chunk_text(content, metadata)

 # Add unique IDs
 for chunk in chunks:
 chunk['id'] = self.generate_id(rel_path, chunk['metadata']['chunk_num'])

 return chunks

 def ingest_directory(self, root_path: str) -> List[Dict[str, Any]]:
 """
 Ingest all files in a directory.

 Args:
 root_path: Root directory to ingest

 Returns:
 List of all document chunks
 """
 logger.info(f"Starting ingestion of directory: {root_path}")

 files = self.discover_files(root_path)
 all_chunks = []

 for file_path in files:
 chunks = self.process_file(file_path, root_path)
 all_chunks.extend(chunks)

 logger.info(f"Ingested {len(all_chunks)} chunks from {len(files)} files")
 return all_chunks
