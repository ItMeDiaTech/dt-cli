"""
Incremental indexing with file change tracking.
"""

import json
import os
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class IncrementalIndexer:
    """
    Track file changes and enable incremental indexing.
    """

    def __init__(self, manifest_path: str = ".rag_data/manifest.json"):
        """
        Initialize incremental indexer.

        Args:
            manifest_path: Path to store file manifest
        """
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(exist_ok=True)
        # HIGH PRIORITY FIX: Add lock for thread-safe manifest operations
        self._manifest_lock = threading.Lock()

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        HIGH PRIORITY FIX: Compute content hash for reliable change detection.

        Args:
            file_path: Path to file

        Returns:
            MD5 hash of file content
        """
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Error hashing {file_path}: {e}")
            return ""

    def load_manifest(self) -> Dict[str, any]:
        """
        Load file modification times and hashes from last indexing.

        Returns:
            Dictionary mapping file paths to mtime or {mtime, hash} dict
        """
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading manifest: {e}")
                return {}
        return {}

    def save_manifest(self, manifest: Dict[str, float]):
        """
        Save current file modification times atomically.

        Args:
            manifest: Dictionary of file paths to modification times
        """
        try:
            # CRITICAL FIX: Use atomic write to prevent corruption on crash
            from src.utils.atomic_write import atomic_write_json
            atomic_write_json(self.manifest_path, manifest, indent=2)
        except IOError as e:
            logger.error(f"Error saving manifest: {e}")

    def discover_changed_files(
        self,
        all_files: List[Path],
        root_path: str
    ) -> List[Path]:
        """
        Find files that changed since last indexing.

        HIGH PRIORITY FIX: Uses hybrid mtime + content hash for reliable detection.
        HIGH PRIORITY FIX: Thread-safe manifest operations with lock.

        Args:
            all_files: List of all files in codebase
            root_path: Root directory path

        Returns:
            List of changed file paths
        """
        # HIGH PRIORITY FIX: Protect entire manifest operation with lock
        with self._manifest_lock:
            manifest = self.load_manifest()
            changed = []
            updated_manifest = {}

            for file_path in all_files:
                try:
                    rel_path = str(file_path.relative_to(root_path))
                    mtime = file_path.stat().st_mtime

                    # HIGH PRIORITY FIX: Hybrid approach
                    # Step 1: Fast mtime check
                    if rel_path not in manifest:
                        # New file
                        changed.append(file_path)
                        content_hash = self._compute_file_hash(file_path)
                        updated_manifest[rel_path] = {'mtime': mtime, 'hash': content_hash}
                        logger.debug(f"New file: {rel_path}")
                    else:
                        # Existing file: check if mtime changed
                        old_data = manifest[rel_path]
                        # Support old format (just float) and new format (dict)
                        old_mtime = old_data if isinstance(old_data, (int, float)) else old_data.get('mtime', 0)

                        if mtime != old_mtime:
                            # mtime changed, verify with content hash
                            content_hash = self._compute_file_hash(file_path)
                            old_hash = old_data.get('hash', '') if isinstance(old_data, dict) else ''

                            if content_hash != old_hash:
                                # Content actually changed
                                changed.append(file_path)
                                logger.debug(f"Changed (hash differs): {rel_path}")
                            else:
                                # mtime changed but content is same (e.g., touch command)
                                logger.debug(f"mtime changed but content unchanged: {rel_path}")

                            updated_manifest[rel_path] = {'mtime': mtime, 'hash': content_hash}
                        else:
                            # mtime unchanged, assume file unchanged
                            updated_manifest[rel_path] = old_data

                except (OSError, ValueError) as e:
                    logger.warning(f"Error checking {file_path}: {e}")

            # Remove deleted files from manifest
            current_files = {str(f.relative_to(root_path)) for f in all_files}
            deleted = set(manifest.keys()) - current_files

            if deleted:
                logger.info(f"Detected {len(deleted)} deleted files")

            # Save updated manifest
            self.save_manifest(updated_manifest)

            logger.info(
                f"Found {len(changed)} changed files out of {len(all_files)} total"
            )

            return changed

    def get_stats(self) -> Dict[str, int]:
        """
        Get indexing statistics.

        Returns:
            Dictionary with stats
        """
        manifest = self.load_manifest()

        return {
            'total_files': len(manifest),
            'manifest_exists': self.manifest_path.exists()
        }

    def reset(self):
        """Reset the manifest (force full re-index next time)."""
        if self.manifest_path.exists():
            self.manifest_path.unlink()
            logger.info("Manifest reset - next index will be full")
