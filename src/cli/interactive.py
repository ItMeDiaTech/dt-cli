"""
Interactive Terminal Interface for dt-cli.

Provides a beautiful, user-friendly TUI for interacting with the RAG/MAF system
without needing to know API endpoints or curl commands.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.layout import Layout
from rich import box
from typing import Optional, Dict, Any, List
import requests
import sys
import os

console = Console()


class DTCliInteractive:
    """
    Interactive terminal interface for dt-cli.
    """

    def __init__(self, base_url: str = "http://localhost:8765"):
        """
        Initialize interactive CLI.

        Args:
            base_url: Base URL for the dt-cli server
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def check_server(self) -> bool:
        """Check if server is running."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def show_welcome(self):
        """Display welcome message."""
        welcome_text = """
# dt-cli Interactive Terminal

Welcome to the **100% Open Source** RAG/MAF/LLM System!

## Available Features:
- 🔍 **RAG Queries** - Ask questions about your codebase
- 🐛 **Debug Errors** - Analyze errors and get fix suggestions
- ✅ **Code Review** - Automated code quality and security checks
- 🕸️  **Knowledge Graph** - Explore code dependencies and relationships
- 📊 **Evaluation** - Measure RAG quality with RAGAS metrics
- 🔎 **Hybrid Search** - Semantic + keyword search

Type `help` for available commands or choose from the menu below.
"""
        console.print(Panel(Markdown(welcome_text), border_style="blue", box=box.DOUBLE))

    def show_menu(self):
        """Display main menu."""
        table = Table(title="Main Menu", box=box.ROUNDED, border_style="cyan")
        table.add_column("Option", style="cyan", justify="center")
        table.add_column("Description", style="white")

        options = [
            ("1", "🔍 Ask a Question (RAG Query)"),
            ("2", "🐛 Debug an Error"),
            ("3", "✅ Review Code"),
            ("4", "🕸️  Explore Knowledge Graph"),
            ("5", "📊 Evaluate RAG Quality"),
            ("6", "🔎 Hybrid Search"),
            ("7", "📈 View Statistics"),
            ("8", "⚙️  Settings"),
            ("9", "❓ Help"),
            ("0", "🚪 Exit")
        ]

        for opt, desc in options:
            table.add_row(opt, desc)

        console.print(table)

    def ask_question(self):
        """Handle RAG query."""
        console.print("\n[bold cyan]Ask a Question[/bold cyan]", style="bold")
        console.print("━" * 60)

        query = Prompt.ask("💬 [bold]Your question[/bold]")

        if not query.strip():
            console.print("[yellow]⚠️  Empty query![/yellow]")
            return

        # Ask for context files
        add_files = Confirm.ask("📁 Add specific files to context?", default=False)
        context_files = []

        if add_files:
            while True:
                file_path = Prompt.ask("  File path (or press Enter to finish)")
                if not file_path:
                    break
                context_files.append(file_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🔍 Searching codebase...", total=None)

            try:
                response = self.session.post(
                    f"{self.base_url}/query",
                    json={
                        "query": query,
                        "auto_trigger": True,
                        "context_files": context_files if context_files else None
                    },
                    timeout=30
                )

                progress.update(task, completed=True)

                if response.status_code == 200:
                    result = response.json()

                    # Show results
                    console.print("\n[bold green]✓ Answer:[/bold green]")
                    console.print(Panel(result['response'], border_style="green"))

                    # Show metadata
                    if 'auto_trigger' in result:
                        trigger_info = result['auto_trigger']
                        console.print(f"\n[dim]Intent: {trigger_info['intent']} (confidence: {trigger_info['confidence']:.0%})[/dim]")
                        console.print(f"[dim]Actions: {', '.join(trigger_info['actions'])}[/dim]")

                    if 'context_used' in result:
                        console.print(f"[dim]Contexts used: {result['context_used']}[/dim]")

                else:
                    console.print(f"[red]❌ Error: {response.status_code}[/red]")

            except requests.exceptions.Timeout:
                console.print("[red]❌ Request timed out[/red]")
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")

    def debug_error(self):
        """Handle error debugging."""
        console.print("\n[bold red]Debug an Error[/bold red]", style="bold")
        console.print("━" * 60)

        console.print("[yellow]Paste your error output (press Ctrl+D or Ctrl+Z when done):[/yellow]")

        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        error_output = "\n".join(lines)

        if not error_output.strip():
            console.print("[yellow]⚠️  No error provided![/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🐛 Analyzing error...", total=None)

            try:
                response = self.session.post(
                    f"{self.base_url}/debug",
                    json={
                        "error_output": error_output,
                        "auto_extract_code": True
                    },
                    timeout=30
                )

                progress.update(task, completed=True)

                if response.status_code == 200:
                    result = response.json()

                    # Show error context
                    console.print("\n[bold]📍 Error Location:[/bold]")
                    error_ctx = result['error_context']
                    console.print(f"  Type: {error_ctx['error_type']}")
                    if error_ctx.get('file_path'):
                        console.print(f"  File: {error_ctx['file_path']}:{error_ctx.get('line_number', '?')}")
                    console.print(f"  Message: {error_ctx['error_message']}")

                    # Show code snippet
                    if error_ctx.get('code_snippet'):
                        console.print("\n[bold]📄 Code Context:[/bold]")
                        syntax = Syntax(error_ctx['code_snippet'], "python", line_numbers=True)
                        console.print(syntax)

                    # Show root cause
                    console.print(f"\n[bold red]🔍 Root Cause:[/bold red]")
                    console.print(Panel(result['root_cause'], border_style="red"))

                    # Show explanation
                    console.print(f"\n[bold]💡 Explanation:[/bold]")
                    console.print(result['explanation'])

                    # Show fixes
                    console.print(f"\n[bold green]🔧 Suggested Fixes:[/bold green]")
                    for i, fix in enumerate(result['suggested_fixes'], 1):
                        console.print(f"  {i}. {fix}")

                    console.print(f"\n[dim]Confidence: {result['confidence']:.0%}[/dim]")

                else:
                    console.print(f"[red]❌ Error: {response.status_code}[/red]")

            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")

    def review_code(self):
        """Handle code review."""
        console.print("\n[bold magenta]Review Code[/bold magenta]", style="bold")
        console.print("━" * 60)

        # Get code input
        file_path = Prompt.ask("📁 [bold]File path to review[/bold] (or press Enter to paste code)")

        if file_path.strip():
            # Read from file
            try:
                with open(file_path, 'r') as f:
                    code = f.read()
            except Exception as e:
                console.print(f"[red]❌ Could not read file: {e}[/red]")
                return
        else:
            # Paste code
            console.print("[yellow]Paste your code (press Ctrl+D or Ctrl+Z when done):[/yellow]")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            code = "\n".join(lines)
            file_path = "pasted_code.py"

        if not code.strip():
            console.print("[yellow]⚠️  No code provided![/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("✅ Reviewing code...", total=None)

            try:
                response = self.session.post(
                    f"{self.base_url}/review",
                    json={
                        "code": code,
                        "file_path": file_path,
                        "language": "python"
                    },
                    timeout=30
                )

                progress.update(task, completed=True)

                if response.status_code == 200:
                    result = response.json()

                    # Show score
                    score = result['overall_score']
                    score_color = "green" if score >= 7 else "yellow" if score >= 5 else "red"

                    console.print(f"\n[bold]📊 Quality Score: [{score_color}]{score:.1f}/10[/{score_color}][/bold]")
                    console.print(f"[dim]{result['summary']}[/dim]")

                    # Show issues
                    if result['issues']:
                        console.print(f"\n[bold]⚠️  Issues Found ({len(result['issues'])}):[/bold]")

                        # Group by severity
                        by_severity = {}
                        for issue in result['issues']:
                            sev = issue['severity']
                            if sev not in by_severity:
                                by_severity[sev] = []
                            by_severity[sev].append(issue)

                        # Show critical first
                        for severity in ['critical', 'high', 'medium', 'low', 'info']:
                            if severity in by_severity:
                                sev_color = {
                                    'critical': 'red',
                                    'high': 'red',
                                    'medium': 'yellow',
                                    'low': 'blue',
                                    'info': 'cyan'
                                }.get(severity, 'white')

                                console.print(f"\n[bold {sev_color}]{severity.upper()} ({len(by_severity[severity])})[/bold {sev_color}]")

                                for issue in by_severity[severity][:5]:  # Show first 5
                                    console.print(f"  • {issue['title']}")
                                    if issue.get('line_number'):
                                        console.print(f"    Line {issue['line_number']}: {issue['description']}")
                                    else:
                                        console.print(f"    {issue['description']}")
                                    if issue.get('suggestion'):
                                        console.print(f"    [dim]Fix: {issue['suggestion']}[/dim]")

                                if len(by_severity[severity]) > 5:
                                    console.print(f"  [dim]... and {len(by_severity[severity]) - 5} more[/dim]")

                    else:
                        console.print("\n[bold green]✓ No issues found![/bold green]")

                    # Show metrics
                    console.print(f"\n[bold]📈 Metrics:[/bold]")
                    metrics = result['metrics']
                    console.print(f"  Total lines: {metrics['total_lines']}")
                    console.print(f"  Code lines: {metrics['code_lines']}")
                    console.print(f"  Comment lines: {metrics['comment_lines']}")
                    console.print(f"  Issues: {metrics['total_issues']}")

                else:
                    console.print(f"[red]❌ Error: {response.status_code}[/red]")

            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")

    def explore_graph(self):
        """Handle knowledge graph exploration."""
        console.print("\n[bold blue]Explore Knowledge Graph[/bold blue]", style="bold")
        console.print("━" * 60)

        # Check if graph is built
        try:
            stats_response = self.session.get(f"{self.base_url}/graph/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                if stats['total_entities'] == 0:
                    console.print("[yellow]⚠️  Knowledge graph is empty![/yellow]")
                    if Confirm.ask("Build knowledge graph now?", default=True):
                        self.build_graph()
                        return
                    else:
                        return
        except:
            console.print("[red]❌ Could not check graph status[/red]")
            return

        # Choose query type
        console.print("\n[bold]Query Types:[/bold]")
        console.print("  1. Dependencies (what does X depend on?)")
        console.print("  2. Dependents (what depends on X?)")
        console.print("  3. Usages (where is X used?)")
        console.print("  4. Impact Analysis (what breaks if I change X?)")

        choice = Prompt.ask("Choose query type", choices=["1", "2", "3", "4"])

        query_types = {
            "1": "dependencies",
            "2": "dependents",
            "3": "usages",
            "4": "impact"
        }
        query_type = query_types[choice]

        # Get entity
        entity_name = Prompt.ask("Entity name (e.g., function or class name)")
        entity_type = Prompt.ask(
            "Entity type (optional)",
            choices=["", "function", "class", "method", "module"],
            default=""
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🕸️  Querying knowledge graph...", total=None)

            try:
                response = self.session.post(
                    f"{self.base_url}/graph/query",
                    json={
                        "entity_name": entity_name,
                        "entity_type": entity_type if entity_type else None,
                        "query_type": query_type
                    },
                    timeout=30
                )

                progress.update(task, completed=True)

                if response.status_code == 200:
                    result = response.json()

                    if query_type == "impact":
                        # Impact analysis
                        console.print(f"\n[bold]📊 Impact Analysis for '{entity_name}':[/bold]")
                        console.print(f"  Direct impact: {result['direct_impact']} entities")
                        console.print(f"  Total impact: {result['total_impact']} entities")

                        if result.get('affected_by_type'):
                            console.print(f"\n[bold]By Type:[/bold]")
                            for etype, count in result['affected_by_type'].items():
                                console.print(f"  {etype}: {count}")

                        if result.get('affected_by_file'):
                            console.print(f"\n[bold]By File:[/bold]")
                            for file, count in sorted(result['affected_by_file'].items(), key=lambda x: -x[1])[:10]:
                                console.print(f"  {file}: {count}")

                        if result.get('affected_entities'):
                            console.print(f"\n[bold]Affected Entities (top 10):[/bold]")
                            for entity in result['affected_entities'][:10]:
                                console.print(f"  • {entity['name']} ({entity['type']}) in {entity['file']}")

                    else:
                        # Other queries
                        console.print(f"\n[bold]Results for '{entity_name}' ({query_type}):[/bold]")

                        if 'results' in result:
                            results = result['results']
                            if results:
                                for item in results[:20]:  # Show first 20
                                    if 'name' in item:
                                        console.print(f"  • {item['name']} ({item.get('type', 'unknown')}) in {item.get('file', 'unknown')}")
                                    elif 'used_by' in item:
                                        console.print(f"  • {item['used_by']} ({item.get('type', 'unknown')}) in {item.get('file', 'unknown')}:{item.get('line', '?')}")

                                if len(results) > 20:
                                    console.print(f"  [dim]... and {len(results) - 20} more[/dim]")
                            else:
                                console.print("  [yellow]No results found[/yellow]")

                elif response.status_code == 404:
                    console.print(f"[yellow]⚠️  Entity '{entity_name}' not found in graph[/yellow]")
                else:
                    console.print(f"[red]❌ Error: {response.status_code}[/red]")

            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")

    def build_graph(self):
        """Build knowledge graph."""
        path = Prompt.ask("📁 [bold]Path to analyze[/bold]", default="src/")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🕸️  Building knowledge graph...", total=None)

            try:
                response = self.session.post(
                    f"{self.base_url}/graph/build",
                    json={"path": path},
                    timeout=120
                )

                progress.update(task, completed=True)

                if response.status_code == 200:
                    result = response.json()
                    stats = result['stats']

                    console.print("\n[bold green]✓ Knowledge graph built successfully![/bold green]")
                    console.print(f"\n[bold]📊 Statistics:[/bold]")
                    console.print(f"  Entities: {stats['total_entities']}")
                    console.print(f"  Relationships: {stats['total_relationships']}")

                    if 'entities_by_type' in stats:
                        console.print(f"\n[bold]By Type:[/bold]")
                        for etype, count in stats['entities_by_type'].items():
                            console.print(f"  {etype}: {count}")

                else:
                    console.print(f"[red]❌ Error: {response.status_code}[/red]")

            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")

    def view_stats(self):
        """View system statistics."""
        console.print("\n[bold]📈 System Statistics[/bold]", style="bold")
        console.print("━" * 60)

        try:
            # Get server info
            info_response = self.session.get(f"{self.base_url}/info", timeout=5)
            if info_response.status_code == 200:
                info = info_response.json()

                # LLM provider
                console.print("\n[bold cyan]LLM Provider:[/bold cyan]")
                llm_info = info.get('llm', {})
                console.print(f"  Provider: {llm_info.get('provider', 'unknown')}")
                console.print(f"  Model: {llm_info.get('model', 'unknown')}")

                # RAG stats
                console.print("\n[bold cyan]RAG System:[/bold cyan]")
                rag_info = info.get('rag', {})
                console.print(f"  Status: {rag_info.get('status', 'unknown')}")

                # Auto-trigger stats
                trigger_response = self.session.get(f"{self.base_url}/auto-trigger/stats", timeout=5)
                if trigger_response.status_code == 200:
                    trigger_stats = trigger_response.json()
                    if trigger_stats.get('total_queries', 0) > 0:
                        console.print("\n[bold cyan]Auto-Trigger:[/bold cyan]")
                        console.print(f"  Total queries: {trigger_stats.get('total_queries', 0)}")
                        console.print(f"  Avg confidence: {trigger_stats.get('average_confidence', 0):.0%}")

                # Graph stats
                graph_response = self.session.get(f"{self.base_url}/graph/stats", timeout=5)
                if graph_response.status_code == 200:
                    graph_stats = graph_response.json()
                    console.print("\n[bold cyan]Knowledge Graph:[/bold cyan]")
                    console.print(f"  Entities: {graph_stats.get('total_entities', 0)}")
                    console.print(f"  Relationships: {graph_stats.get('total_relationships', 0)}")

        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

    def show_help(self):
        """Show help information."""
        help_text = """
# dt-cli Help

## Commands

- **Ask Question**: Query your codebase using RAG
- **Debug Error**: Analyze errors and get automated fix suggestions
- **Review Code**: Get automated code quality and security analysis
- **Explore Graph**: Navigate code dependencies and relationships
- **Evaluate**: Measure RAG quality using RAGAS metrics
- **Hybrid Search**: Combine semantic and keyword search
- **View Stats**: See system statistics
- **Settings**: Configure system settings

## Tips

- Use **auto-triggering** for seamless queries (no need to specify /rag-query)
- Build the **knowledge graph** first for dependency analysis
- Use **code review** before committing to catch issues early
- Run **debug** on test failures for instant analysis

## Configuration

Server URL: `http://localhost:8765`
Config file: `llm-config.yaml`

## Documentation

See README.md and PHASE*.md files for detailed documentation.
"""
        console.print(Panel(Markdown(help_text), border_style="blue"))

    def run(self):
        """Run the interactive CLI."""
        # Check server
        if not self.check_server():
            console.print("[bold red]❌ Server is not running![/bold red]")
            console.print("\nPlease start the server first:")
            console.print("  [cyan]python src/mcp_server/standalone_server.py[/cyan]")
            return

        # Show welcome
        self.show_welcome()

        # Main loop
        while True:
            console.print()
            self.show_menu()

            try:
                choice = Prompt.ask("\n[bold]Choose an option[/bold]", default="1")

                if choice == "1":
                    self.ask_question()
                elif choice == "2":
                    self.debug_error()
                elif choice == "3":
                    self.review_code()
                elif choice == "4":
                    self.explore_graph()
                elif choice == "5":
                    console.print("[yellow]Evaluation feature - use /evaluate endpoint[/yellow]")
                elif choice == "6":
                    console.print("[yellow]Hybrid search feature - use /hybrid-search endpoint[/yellow]")
                elif choice == "7":
                    self.view_stats()
                elif choice == "8":
                    console.print("[yellow]Settings coming soon![/yellow]")
                elif choice == "9":
                    self.show_help()
                elif choice == "0":
                    console.print("\n[bold]👋 Goodbye![/bold]")
                    break
                else:
                    console.print("[yellow]Invalid option![/yellow]")

            except KeyboardInterrupt:
                console.print("\n\n[bold]👋 Goodbye![/bold]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="dt-cli Interactive Terminal")
    parser.add_argument(
        "--server",
        default="http://localhost:8765",
        help="Server URL (default: http://localhost:8765)"
    )

    args = parser.parse_args()

    cli = DTCliInteractive(base_url=args.server)
    cli.run()


if __name__ == "__main__":
    main()
