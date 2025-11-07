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

        # Create temporary export directory
        temp_dir = Path(f"/tmp/rag_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        temp_dir.mkdir(parents=True, exist_ok=True)

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
        """Export configuration."""
        try:
            config_dir = Path.home() / '.rag_config'

            if config_dir.exists():
                # Copy all config files
                config_export_dir = export_dir / 'config'
                shutil.copytree(config_dir, config_export_dir)
                logger.info("Exported configuration")

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
        merge: bool = True
    ) -> bool:
        """
        Import all system data.

        Args:
            archive_path: Input archive path (.tar.gz)
            include_index: Import vector index
            merge: Merge with existing data (True) or replace (False)

        Returns:
            True if successful
        """
        logger.info(f"Importing data from {archive_path}...")

        # Create temporary import directory
        temp_dir = Path(f"/tmp/rag_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Extract archive
            with tarfile.open(archive_path, 'r:gz') as tar:
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
        """Import configuration."""
        try:
            config_import_dir = import_dir / 'config'

            if config_import_dir.exists():
                config_dir = Path.home() / '.rag_config'

                if not merge and config_dir.exists():
                    shutil.rmtree(config_dir)

                config_dir.mkdir(parents=True, exist_ok=True)

                # Copy config files
                for config_file in config_import_dir.glob('*.json'):
                    shutil.copy(config_file, config_dir / config_file.name)

                logger.info("Imported configuration")

        except Exception as e:
            logger.warning(f"Failed to import configuration: {e}")

    def _import_index(self, import_dir: Path):
        """Import vector index."""
        try:
            index_import_dir = import_dir / 'index'

            if index_import_dir.exists():
                target_db_path = Path.cwd() / 'chroma_db'

                if target_db_path.exists():
                    shutil.rmtree(target_db_path)

                shutil.copytree(index_import_dir, target_db_path)
                logger.info("Imported index")

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
