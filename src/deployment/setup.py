"""
Deployment and setup utilities for RAG system.

Automates:
- Initial setup and configuration
- Dependency installation
- Git hooks installation
- Index initialization
- MCP server configuration
- Health checks
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


class SetupManager:
    """
    Manages RAG system setup and deployment.
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize setup manager.

        Args:
            project_root: Project root directory (default: cwd)
        """
        self.project_root = project_root or Path.cwd()
        self.steps_completed: List[str] = []
        self.errors: List[str] = []

    def run_full_setup(self, skip_dependencies: bool = False) -> bool:
        """
        Run complete setup process.

        Args:
            skip_dependencies: Skip dependency installation

        Returns:
            True if successful
        """
        print("=" * 70)
        print("RAG-MAF PLUGIN SETUP".center(70))
        print("=" * 70)
        print()

        # Step 1: Check Python version
        if not self._check_python_version():
            return False

        # Step 2: Install dependencies
        if not skip_dependencies:
            if not self._install_dependencies():
                print("⚠️  Dependency installation failed, continuing...")

        # Step 3: Create configuration
        if not self._create_configuration():
            return False

        # Step 4: Install Git hooks
        if not self._install_git_hooks():
            print("⚠️  Git hooks installation failed, continuing...")

        # Step 5: Initialize index
        if not self._initialize_index():
            return False

        # Step 6: Verify setup
        if not self._verify_setup():
            return False

        # Print summary
        self._print_summary()

        return True

    def _check_python_version(self) -> bool:
        """
        Check Python version.

        Returns:
            True if version is compatible
        """
        print("🐍 Checking Python version...")

        version = sys.version_info
        required = (3, 8)

        if version >= required:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
            self.steps_completed.append("Python version check")
            return True

        print(f"❌ Python {required[0]}.{required[1]}+ required (found {version.major}.{version.minor})")
        self.errors.append("Incompatible Python version")
        return False

    def _install_dependencies(self) -> bool:
        """
        Install Python dependencies.

        Returns:
            True if successful
        """
        print("\n📦 Installing dependencies...")

        requirements_file = self.project_root / 'requirements.txt'

        if not requirements_file.exists():
            print("⚠️  requirements.txt not found")
            return False

        try:
            # HIGH PRIORITY FIX: Stream output instead of suppressing it
            # Users need to see pip progress and error messages for debugging
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
                check=True,
                # Remove capture_output to show output to user
                text=True
            )

            print("✅ Dependencies installed")
            self.steps_completed.append("Dependencies installation")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            print("Check pip output above for details")
            self.errors.append(f"Dependency installation: {e}")
            return False

    def _create_configuration(self) -> bool:
        """
        Create initial configuration.

        Returns:
            True if successful
        """
        print("\n⚙️  Creating configuration...")

        config_dir = Path.home() / '.rag_config'
        config_dir.mkdir(parents=True, exist_ok=True)

        # Create default configuration
        default_config = {
            'codebase_path': str(self.project_root),
            'db_path': str(self.project_root / 'chroma_db'),
            'embedding_model': 'all-MiniLM-L6-v2',
            'n_results': 5,
            'use_cache': True,
            'use_hybrid': True,
            'mcp_host': '0.0.0.0',
            'mcp_port': 8000,
        }

        config_file = config_dir / 'default.json'

        try:
            config_file.write_text(json.dumps(default_config, indent=2))
            print(f"✅ Configuration created: {config_file}")
            self.steps_completed.append("Configuration creation")
            return True

        except Exception as e:
            print(f"❌ Configuration creation failed: {e}")
            self.errors.append(f"Configuration: {e}")
            return False

    def _install_git_hooks(self) -> bool:
        """
        Install Git hooks.

        Returns:
            True if successful
        """
        print("\n🪝 Installing Git hooks...")

        if not (self.project_root / '.git').exists():
            print("⚠️  Not a Git repository, skipping hooks")
            return True

        try:
            from src.git_integration import install_git_hooks

            success = install_git_hooks(self.project_root)

            if success:
                print("✅ Git hooks installed")
                self.steps_completed.append("Git hooks installation")
                return True
            else:
                print("⚠️  Git hooks installation failed")
                return False

        except Exception as e:
            print(f"⚠️  Git hooks error: {e}")
            return False

    def _initialize_index(self) -> bool:
        """
        Initialize RAG index.

        Returns:
            True if successful
        """
        print("\n📚 Initializing index...")

        try:
            from src.rag.enhanced_query_engine import EnhancedQueryEngine

            print("   Building initial index (this may take a moment)...")

            config = {
                'codebase_path': str(self.project_root),
                'db_path': str(self.project_root / 'chroma_db')
            }

            engine = EnhancedQueryEngine(config)
            stats = engine.index_codebase(incremental=False)

            if stats:
                files_processed = stats.get('files_processed', 0)
                print(f"✅ Index created ({files_processed} files)")
                self.steps_completed.append("Index initialization")
                return True
            else:
                print("✅ Index created")
                self.steps_completed.append("Index initialization")
                return True

        except Exception as e:
            print(f"❌ Indexing failed: {e}")
            self.errors.append(f"Index initialization: {e}")
            return False

    def _verify_setup(self) -> bool:
        """
        Verify setup is complete.

        Returns:
            True if verified
        """
        print("\n🔍 Verifying setup...")

        checks = []

        # Check configuration
        config_file = Path.home() / '.rag_config' / 'default.json'
        checks.append(("Configuration", config_file.exists()))

        # Check index
        db_path = self.project_root / 'chroma_db'
        checks.append(("Index database", db_path.exists()))

        # Print checks
        all_passed = True
        for name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {name}")

            if not passed:
                all_passed = False

        if all_passed:
            self.steps_completed.append("Setup verification")

        return all_passed

    def _print_summary(self):
        """Print setup summary."""
        print("\n" + "=" * 70)
        print("SETUP COMPLETE!".center(70))
        print("=" * 70)
        print()

        print("✅ Steps completed:")
        for step in self.steps_completed:
            print(f"   • {step}")

        if self.errors:
            print("\n⚠️  Warnings/Errors:")
            for error in self.errors:
                print(f"   • {error}")

        print("\n📖 Next steps:")
        print("   1. Query the codebase: /rag-query <your question>")
        print("   2. View metrics: /rag-metrics")
        print("   3. Save common queries: /rag-save <name> | <query>")
        print()

        print("📚 Documentation:")
        print("   • User Guide: USER_GUIDE.md")
        print("   • Implementation Summary: IMPLEMENTATION_SUMMARY.md")
        print()

    def quick_health_check(self) -> Dict[str, Any]:
        """
        Run quick health check.

        Returns:
            Health check results
        """
        results = {
            'status': 'healthy',
            'checks': {}
        }

        # Check configuration
        config_file = Path.home() / '.rag_config' / 'default.json'
        results['checks']['configuration'] = config_file.exists()

        # Check index
        db_path = self.project_root / 'chroma_db'
        results['checks']['index'] = db_path.exists()

        # Check Git hooks
        hooks_dir = self.project_root / '.git' / 'hooks'
        if hooks_dir.exists():
            results['checks']['git_hooks'] = (hooks_dir / 'post-commit').exists()

        # Overall status
        if not all(results['checks'].values()):
            results['status'] = 'degraded'

        return results


class DeploymentHelper:
    """
    Helper for deploying RAG system.
    """

    @staticmethod
    def generate_startup_script(output_path: Path):
        """
        Generate startup script.

        Args:
            output_path: Output script path
        """
        script_content = """#!/bin/bash
# RAG-MAF Plugin Startup Script

echo "Starting RAG-MAF Plugin..."

# Activate virtual environment (if exists)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start MCP server
echo "Starting MCP server..."
python -m src.mcp.enhanced_server &
MCP_PID=$!

echo "MCP server started (PID: $MCP_PID)"
echo "Server running at http://0.0.0.0:8000"
echo "API docs at http://0.0.0.0:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

# Wait for interrupt
trap "kill $MCP_PID; exit" INT
wait $MCP_PID
"""

        output_path.write_text(script_content)
        output_path.chmod(0o755)

        print(f"✅ Startup script generated: {output_path}")

    @staticmethod
    def generate_systemd_service(
        output_path: Path,
        project_path: Path,
        user: str = "ubuntu"
    ):
        """
        Generate systemd service file.

        Args:
            output_path: Output service file path
            project_path: Project path
            user: System user
        """
        service_content = f"""[Unit]
Description=RAG-MAF Plugin MCP Server
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={project_path}
ExecStart={sys.executable} -m src.mcp.enhanced_server
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
"""

        output_path.write_text(service_content)

        print(f"✅ Systemd service generated: {output_path}")
        print(f"\n   To install:")
        print(f"   sudo cp {output_path} /etc/systemd/system/")
        print(f"   sudo systemctl daemon-reload")
        print(f"   sudo systemctl enable rag-maf.service")
        print(f"   sudo systemctl start rag-maf.service")


def run_setup(skip_dependencies: bool = False) -> bool:
    """
    Run setup process.

    Args:
        skip_dependencies: Skip dependency installation

    Returns:
        True if successful
    """
    setup_manager = SetupManager()
    return setup_manager.run_full_setup(skip_dependencies=skip_dependencies)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG-MAF Plugin Setup")
    parser.add_argument('--skip-deps', action='store_true', help="Skip dependency installation")

    args = parser.parse_args()

    success = run_setup(skip_dependencies=args.skip_deps)

    sys.exit(0 if success else 1)
