"""
Incremental indexing with file change tracking.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Optional
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

    def load_manifest(self) -> Dict[str, float]:
        """
        Load file modification times from last indexing.

        Returns:
            Dictionary mapping file paths to modification times
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
        Save current file modification times.

        Args:
            manifest: Dictionary of file paths to modification times
        """
        try:
            self.manifest_path.write_text(json.dumps(manifest, indent=2))
        except IOError as e:
            logger.error(f"Error saving manifest: {e}")

    def discover_changed_files(
        self,
        all_files: List[Path],
        root_path: str
    ) -> List[Path]:
        """
        Find files that changed since last indexing.

        Args:
            all_files: List of all files in codebase
            root_path: Root directory path

        Returns:
            List of changed file paths
        """
        manifest = self.load_manifest()
        changed = []
        updated_manifest = {}

        for file_path in all_files:
            try:
                rel_path = str(file_path.relative_to(root_path))
                mtime = file_path.stat().st_mtime

                # Include if new file or modified
                if rel_path not in manifest or manifest[rel_path] != mtime:
                    changed.append(file_path)
                    logger.debug(f"Changed: {rel_path}")

                # Update manifest
                updated_manifest[rel_path] = mtime

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
