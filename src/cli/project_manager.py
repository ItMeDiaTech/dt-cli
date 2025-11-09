"""
CLI commands for project and workspace management.

Provides commands to:
- List current projects
- Add new project folders
- Remove project folders
- Switch LLM providers
- View workspace status
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.advanced_rag_config import get_advanced_config, AdvancedRAGConfig

console = Console()


def list_projects(config: AdvancedRAGConfig):
    """List all projects in workspace."""
    projects = config.list_projects()

    if not projects:
        console.print("[yellow]No projects in workspace[/yellow]")
        return

    table = Table(title="Workspace Projects", box=box.ROUNDED, border_style="cyan")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="white")
    table.add_column("Indexed", style="green")
    table.add_column("Files", justify="right")

    for project in projects:
        indexed = "Yes" if project['indexed'] else "No"
        table.add_row(
            project['name'],
            project['path'],
            indexed,
            str(project['file_count'])
        )

    console.print(table)
    console.print(f"\n[dim]Total projects: {len(projects)}[/dim]")


def add_project(config: AdvancedRAGConfig, path: str, name: str = None):
    """Add a new project folder."""
    if not os.path.exists(path):
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        return False

    if not os.path.isdir(path):
        console.print(f"[red]Error: Path is not a directory: {path}[/red]")
        return False

    success = config.add_project_folder(path, name)

    if success:
        console.print(f"[green]Added project: {name or os.path.basename(path)}[/green]")
        console.print(f"[dim]Path: {path}[/dim]")
        return True
    else:
        console.print(f"[red]Failed to add project[/red]")
        return False


def remove_project(config: AdvancedRAGConfig, name_or_path: str):
    """Remove a project folder."""
    success = config.remove_project_folder(name_or_path)

    if success:
        console.print(f"[green]Removed project: {name_or_path}[/green]")
        return True
    else:
        console.print(f"[yellow]Project not found: {name_or_path}[/yellow]")
        return False


def show_workspace_status(config: AdvancedRAGConfig):
    """Show workspace status."""
    workspace_info = f"""
# Workspace Status

**Current Directory:** {config.get_current_directory()}

**Active LLM Provider:** {config.workspace.active_llm_provider}

**Projects:** {len(config.workspace.projects)}

**Advanced Features:**
- Self-RAG: {config.self_rag_enabled}
- HyDE: {config.hyde_enabled}
- Query Rewriting: {config.query_rewriting_enabled}
"""

    console.print(Panel(workspace_info, border_style="blue", box=box.DOUBLE))

    # Show LLM config
    llm_config = config.get_llm_config()
    console.print("\n[bold]LLM Configuration:[/bold]")
    console.print(f"  Provider: {llm_config.get('provider', 'unknown')}")
    console.print(f"  Model: {llm_config.get('model_name', 'unknown')}")
    console.print(f"  Temperature: {llm_config.get('temperature', 0.1)}")

    # Show RAG config
    rag_config = config.get_rag_config()
    console.print("\n[bold]RAG Configuration:[/bold]")
    console.print(f"  Chunk Size: {rag_config.get('chunk_size', 1000)}")
    console.print(f"  Max Results: {rag_config.get('max_results', 5)}")
    console.print(f"  Embedding Model: {rag_config.get('embedding_model', 'unknown')}")


def switch_provider(config: AdvancedRAGConfig, provider: str):
    """Switch LLM provider."""
    valid_providers = ['ollama', 'vllm', 'claude', 'openai']

    if provider not in valid_providers:
        console.print(f"[red]Invalid provider: {provider}[/red]")
        console.print(f"[yellow]Valid providers: {', '.join(valid_providers)}[/yellow]")
        return False

    success = config.switch_llm_provider(provider)

    if success:
        console.print(f"[green]Switched to {provider}[/green]")
        console.print(f"[dim]Restart server for changes to take effect[/dim]")
        return True
    else:
        console.print(f"[red]Failed to switch provider[/red]")
        return False


def main():
    """CLI entry point for project management."""
    import argparse

    parser = argparse.ArgumentParser(description="dt-cli Project Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List projects
    subparsers.add_parser('list', help='List all projects')

    # Add project
    add_parser = subparsers.add_parser('add', help='Add a project folder')
    add_parser.add_argument('path', help='Path to project directory')
    add_parser.add_argument('--name', help='Project name (optional)')

    # Remove project
    remove_parser = subparsers.add_parser('remove', help='Remove a project')
    remove_parser.add_argument('name_or_path', help='Project name or path')

    # Status
    subparsers.add_parser('status', help='Show workspace status')

    # Switch provider
    switch_parser = subparsers.add_parser('switch-llm', help='Switch LLM provider')
    switch_parser.add_argument('provider', choices=['ollama', 'vllm', 'claude', 'openai'])

    args = parser.parse_args()

    # Load config
    config = get_advanced_config()

    # Execute command
    if args.command == 'list':
        list_projects(config)
    elif args.command == 'add':
        add_project(config, args.path, args.name)
    elif args.command == 'remove':
        remove_project(config, args.name_or_path)
    elif args.command == 'status':
        show_workspace_status(config)
    elif args.command == 'switch-llm':
        switch_provider(config, args.provider)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
