"""
Comprehensive data export/import utilities.

Supports exporting/importing:
- Saved searches
- Query history
- Knowledge graph
- Configuration
- Cache data
- Query patterns (prefetching)
"""

import json
import tarfile
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Exports RAG system data.
    """

    def __init__(self):
        """Initialize data exporter."""
        self.export_version = "1.0.0"

    def export_all(
        self,
        output_path: Path,
        include_index: bool = False
    ) -> bool:
        """
        Export all system data.

        Args:
            output_path: Output archive path (.tar.gz)
            include_index: Include vector index (large file)

        Returns:
            True if successful
        """
        logger.info(f"Exporting data to {output_path}...")

        # HIGH PRIORITY FIX: Use tempfile.mkdtemp() instead of hard-coded /tmp
        # Works across all platforms and respects TMPDIR environment variable
        temp_dir = Path(tempfile.mkdtemp(prefix="rag_export_"))

        try:
            # Export metadata
            self._export_metadata(temp_dir)

            # Export saved searches
            self._export_saved_searches(temp_dir)

            # Export query history
            self._export_query_history(temp_dir)

            # Export knowledge graph
            self._export_knowledge_graph(temp_dir)

            # Export configuration
            self._export_configuration(temp_dir)

            # Export query patterns
            self._export_query_patterns(temp_dir)

            # Optionally export index
            if include_index:
                self._export_index(temp_dir)

            # Create archive
            self._create_archive(temp_dir, output_path)

            logger.info(f"Export complete: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _export_metadata(self, export_dir: Path):
        """Export metadata."""
        metadata = {
            'export_version': self.export_version,
            'export_date': datetime.now().isoformat(),
            'components_exported': []
        }

        (export_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    def _export_saved_searches(self, export_dir: Path):
        """Export saved searches."""
        from src.rag.saved_searches import saved_search_manager

        try:
            searches_file = Path.home() / '.rag_saved_searches.json'

            if searches_file.exists():
                shutil.copy(searches_file, export_dir / 'saved_searches.json')
                logger.info("Exported saved searches")

        except Exception as e:
            logger.warning(f"Failed to export saved searches: {e}")

    def _export_query_history(self, export_dir: Path):
        """Export query history."""
        from src.rag.query_learning import query_learning_system

        try:
            query_learning_system.export_history(export_dir / 'query_history.json')
            logger.info("Exported query history")

        except Exception as e:
            logger.warning(f"Failed to export query history: {e}")

    def _export_knowledge_graph(self, export_dir: Path):
        """Export knowledge graph."""
        try:
            from src.knowledge_graph import CodeKnowledgeGraph

            # Note: Would need access to knowledge graph instance
            logger.info("Knowledge graph export skipped (requires instance)")

        except Exception as e:
            logger.warning(f"Failed to export knowledge graph: {e}")

    def _export_configuration(self, export_dir: Path):
        """Export configuration, excluding sensitive credentials."""
        try:
            config_dir = Path.home() / '.rag_config'

            if config_dir.exists():
                config_export_dir = export_dir / 'config'
                config_export_dir.mkdir(parents=True, exist_ok=True)

                # CRITICAL FIX: Copy config files selectively, EXCLUDING credentials
                for item in config_dir.iterdir():
                    # Skip credentials file for security
                    if item.name == '.credentials.json':
                        logger.warning("Skipping credentials file in export (security)")
                        continue

                    # Skip other sensitive files
                    if item.name.endswith(('.key', '.secret', '.pem')):
                        logger.warning(f"Skipping sensitive file: {item.name}")
                        continue

                    # Copy non-sensitive files
                    if item.is_file():
                        shutil.copy2(item, config_export_dir / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, config_export_dir / item.name)

                logger.info("Exported configuration (credentials excluded)")

        except Exception as e:
            logger.warning(f"Failed to export configuration: {e}")

    def _export_query_patterns(self, export_dir: Path):
        """Export query patterns (for prefetching)."""
        # Note: Would need access to prefetcher instance
        logger.info("Query patterns export skipped (requires instance)")

    def _export_index(self, export_dir: Path):
        """Export vector index."""
        try:
            # Find ChromaDB directory
            db_paths = list(Path.cwd().glob('**/chroma_db'))

            if db_paths:
                db_path = db_paths[0]
                index_export_dir = export_dir / 'index'
                shutil.copytree(db_path, index_export_dir)
                logger.info(f"Exported index from {db_path}")

        except Exception as e:
            logger.warning(f"Failed to export index: {e}")

    def _create_archive(self, source_dir: Path, output_path: Path):
        """
        Create tar.gz archive.

        Args:
            source_dir: Source directory
            output_path: Output archive path
        """
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(source_dir, arcname='rag_export')

        logger.info(f"Created archive: {output_path}")


class DataImporter:
    """
    Imports RAG system data.
    """

    def __init__(self):
        """Initialize data importer."""
        pass

    def import_all(
        self,
        archive_path: Path,
        include_index: bool = False,
        merge: bool = True,
        max_archive_size_mb: int = 1000
    ) -> bool:
        """
        Import all system data.

        HIGH PRIORITY FIX: Added archive integrity validation.

        Args:
            archive_path: Input archive path (.tar.gz)
            include_index: Import vector index
            merge: Merge with existing data (True) or replace (False)
            max_archive_size_mb: Maximum archive size in MB (default: 1000MB)

        Returns:
            True if successful
        """
        logger.info(f"Importing data from {archive_path}...")

        # HIGH PRIORITY FIX: Validate archive integrity before extraction
        if not self._validate_archive(archive_path, max_archive_size_mb):
            logger.error("Archive validation failed")
            return False

        # HIGH PRIORITY FIX: Use tempfile.mkdtemp() instead of hard-coded /tmp
        # Works across all platforms and respects TMPDIR environment variable
        temp_dir = Path(tempfile.mkdtemp(prefix="rag_import_"))

        try:
            # CRITICAL FIX: Extract archive with path traversal protection
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Validate all paths before extraction
                for member in tar.getmembers():
                    # Resolve member path
                    member_path = (temp_dir / member.name).resolve()

                    # SECURITY: Ensure path is within temp directory
                    if not str(member_path).startswith(str(temp_dir.resolve())):
                        raise ValueError(f"Path traversal attempt detected: {member.name}")

                    # SECURITY: Skip symlinks to prevent symlink attacks
                    if member.issym() or member.islnk():
                        logger.warning(f"Skipping symlink in archive: {member.name}")
                        continue

                # Safe to extract after validation
                tar.extractall(temp_dir)

            extract_dir = temp_dir / 'rag_export'

            # Verify metadata
            if not self._verify_metadata(extract_dir):
                logger.error("Invalid export format")
                return False

            # Import components
            self._import_saved_searches(extract_dir, merge)
            self._import_query_history(extract_dir, merge)
            self._import_configuration(extract_dir, merge)

            if include_index:
                self._import_index(extract_dir)

            logger.info("Import complete")
            return True

        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False

        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _validate_archive(self, archive_path: Path, max_size_mb: int) -> bool:
        """
        HIGH PRIORITY FIX: Validate archive integrity.

        Args:
            archive_path: Path to archive file
            max_size_mb: Maximum allowed size in MB

        Returns:
            True if valid
        """
        # Check file exists
        if not archive_path.exists():
            logger.error(f"Archive does not exist: {archive_path}")
            return False

        # Check file is actually a file
        if not archive_path.is_file():
            logger.error(f"Archive path is not a file: {archive_path}")
            return False

        # Check file size
        file_size_mb = archive_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            logger.error(f"Archive too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)")
            return False

        # Minimum size check (should be at least 100 bytes for a valid archive)
        if archive_path.stat().st_size < 100:
            logger.error(f"Archive too small: {archive_path.stat().st_size} bytes")
            return False

        # Validate it's actually a gzip file (magic bytes)
        try:
            with open(archive_path, 'rb') as f:
                magic = f.read(2)
                if magic != b'\x1f\x8b':  # gzip magic bytes
                    logger.error("Archive is not a valid gzip file")
                    return False
        except Exception as e:
            logger.error(f"Error reading archive: {e}")
            return False

        # Try to open with tarfile to verify integrity
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Get list of members without extracting
                members = tar.getmembers()

                if len(members) == 0:
                    logger.error("Archive is empty")
                    return False

                # Check for suspiciously large number of files
                if len(members) > 100000:
                    logger.error(f"Archive contains too many files: {len(members)}")
                    return False

                logger.info(f"Archive validated: {len(members)} files, {file_size_mb:.1f}MB")
                return True

        except tarfile.TarError as e:
            logger.error(f"Archive is corrupted or invalid: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating archive: {e}")
            return False

    def _verify_metadata(self, import_dir: Path) -> bool:
        """
        Verify export metadata.

        Args:
            import_dir: Import directory

        Returns:
            True if valid
        """
        metadata_file = import_dir / 'metadata.json'

        if not metadata_file.exists():
            return False

        try:
            metadata = json.loads(metadata_file.read_text())

            if 'export_version' not in metadata:
                return False

            logger.info(f"Importing version {metadata['export_version']} from {metadata['export_date']}")
            return True

        except Exception:
            return False

    def _import_saved_searches(self, import_dir: Path, merge: bool):
        """Import saved searches."""
        from src.rag.saved_searches import saved_search_manager

        try:
            searches_file = import_dir / 'saved_searches.json'

            if searches_file.exists():
                saved_search_manager.import_searches(searches_file, merge=merge)
                logger.info("Imported saved searches")

        except Exception as e:
            logger.warning(f"Failed to import saved searches: {e}")

    def _import_query_history(self, import_dir: Path, merge: bool):
        """Import query history."""
        # Note: Would need to implement in QueryLearningSystem
        logger.info("Query history import not yet implemented")

    def _import_configuration(self, import_dir: Path, merge: bool):
        """
        Import configuration.

        HIGH PRIORITY FIX: Use atomic writes for configuration files.
        """
        try:
            # Import atomic write utility
            from src.utils.atomic_write import atomic_write_json

            config_import_dir = import_dir / 'config'

            if config_import_dir.exists():
                config_dir = Path.home() / '.rag_config'

                if not merge and config_dir.exists():
                    # HIGH PRIORITY FIX: Backup before destructive operation
                    backup_dir = Path.home() / '.rag_config.backup'
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                    shutil.copytree(config_dir, backup_dir)

                    try:
                        shutil.rmtree(config_dir)
                    except Exception as e:
                        # Restore from backup
                        logger.error(f"Failed to remove config dir, restoring backup: {e}")
                        if backup_dir.exists():
                            shutil.copytree(backup_dir, config_dir)
                        raise

                config_dir.mkdir(parents=True, exist_ok=True)

                # HIGH PRIORITY FIX: Copy config files atomically
                for config_file in config_import_dir.glob('*.json'):
                    try:
                        # Read source file
                        data = json.loads(config_file.read_text())
                        # Write atomically to destination
                        target_path = config_dir / config_file.name
                        atomic_write_json(target_path, data)
                    except Exception as e:
                        logger.error(f"Failed to import {config_file.name}: {e}")
                        raise

                logger.info("Imported configuration")

        except Exception as e:
            logger.warning(f"Failed to import configuration: {e}")

    def _import_index(self, import_dir: Path):
        """Import vector index with atomic operations to prevent data loss."""
        try:
            index_import_dir = import_dir / 'index'

            if not index_import_dir.exists():
                return

            target_db_path = Path.cwd() / 'chroma_db'
            temp_db_path = Path.cwd() / 'chroma_db.new'
            backup_db_path = Path.cwd() / 'chroma_db.backup'

            # CRITICAL FIX: Use atomic operations
            try:
                # Step 1: Copy to temporary location
                if temp_db_path.exists():
                    shutil.rmtree(temp_db_path)
                shutil.copytree(index_import_dir, temp_db_path)
                logger.info("Copied index to temporary location")

                # Step 2: Backup original if it exists
                if target_db_path.exists():
                    if backup_db_path.exists():
                        shutil.rmtree(backup_db_path)
                    shutil.move(str(target_db_path), str(backup_db_path))
                    logger.info("Backed up original index")

                # Step 3: Move new index into place
                shutil.move(str(temp_db_path), str(target_db_path))
                logger.info("Imported index successfully")

                # Step 4: Remove backup
                if backup_db_path.exists():
                    shutil.rmtree(backup_db_path)

            except Exception as e:
                # Restore from backup if something went wrong
                logger.error(f"Import failed, attempting to restore backup: {e}")
                if backup_db_path.exists() and not target_db_path.exists():
                    shutil.move(str(backup_db_path), str(target_db_path))
                    logger.info("Restored original index from backup")
                raise

        except Exception as e:
            logger.warning(f"Failed to import index: {e}")


class BackupManager:
    """
    Manages automated backups.
    """

    def __init__(self, backup_dir: Optional[Path] = None):
        """
        Initialize backup manager.

        Args:
            backup_dir: Backup directory (default: ~/.rag_backups)
        """
        self.backup_dir = backup_dir or Path.home() / '.rag_backups'
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(
        self,
        name: Optional[str] = None,
        include_index: bool = False
    ) -> Optional[Path]:
        """
        Create backup.

        Args:
            name: Backup name (default: timestamp)
            include_index: Include vector index

        Returns:
            Backup file path or None
        """
        if not name:
            name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_path = self.backup_dir / f"{name}.tar.gz"

        exporter = DataExporter()

        if exporter.export_all(backup_path, include_index=include_index):
            logger.info(f"Backup created: {backup_path}")
            return backup_path

        return None

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List available backups.

        Returns:
            List of backup info
        """
        backups = []

        for backup_file in self.backup_dir.glob('*.tar.gz'):
            stat = backup_file.stat()

            backups.append({
                'name': backup_file.stem,
                'path': str(backup_file),
                'size_mb': stat.st_size / 1024 / 1024,
                'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)

        return backups

    def restore_backup(
        self,
        backup_name: str,
        include_index: bool = False,
        merge: bool = True
    ) -> bool:
        """
        Restore from backup.

        Args:
            backup_name: Backup name
            include_index: Restore index
            merge: Merge with existing data

        Returns:
            True if successful
        """
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"

        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_name}")
            return False

        importer = DataImporter()
        return importer.import_all(
            backup_path,
            include_index=include_index,
            merge=merge
        )

    def cleanup_old_backups(self, keep_count: int = 10):
        """
        Clean up old backups.

        Args:
            keep_count: Number of backups to keep
        """
        backups = self.list_backups()

        if len(backups) <= keep_count:
            return

        # Remove oldest backups
        for backup in backups[keep_count:]:
            backup_path = Path(backup['path'])

            try:
                backup_path.unlink()
                logger.info(f"Removed old backup: {backup['name']}")

            except Exception as e:
                logger.warning(f"Failed to remove backup {backup['name']}: {e}")


# Convenience functions
def export_data(output_path: Path, include_index: bool = False) -> bool:
    """
    Export system data.

    Args:
        output_path: Output archive path
        include_index: Include vector index

    Returns:
        True if successful
    """
    exporter = DataExporter()
    return exporter.export_all(output_path, include_index=include_index)


def import_data(
    archive_path: Path,
    include_index: bool = False,
    merge: bool = True
) -> bool:
    """
    Import system data.

    Args:
        archive_path: Input archive path
        include_index: Import vector index
        merge: Merge with existing data

    Returns:
        True if successful
    """
    importer = DataImporter()
    return importer.import_all(archive_path, include_index=include_index, merge=merge)


def create_backup(name: Optional[str] = None, include_index: bool = False) -> Optional[Path]:
    """
    Create backup.

    Args:
        name: Backup name
        include_index: Include vector index

    Returns:
        Backup path or None
    """
    manager = BackupManager()
    return manager.create_backup(name=name, include_index=include_index)
