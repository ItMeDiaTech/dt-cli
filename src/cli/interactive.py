"""
Interactive Terminal Interface for dt-cli.

Provides a beautiful, user-friendly TUI for interacting with the RAG/MAF system
without needing to know API endpoints or curl commands.

Features:
- Command history with up/down arrow navigation
- Intelligent mode (bypass menu, go directly to RAG)
- Auto-start server if not running
- No emojis in interface
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
import subprocess
import time

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

console = Console()


class DTCliInteractive:
    """
    Interactive terminal interface for dt-cli.
    """

    def __init__(self, base_url: str = "http://localhost:8765", auto_start_server: bool = True):
        """
        Initialize interactive CLI.

        Args:
            base_url: Base URL for the dt-cli server
            auto_start_server: Whether to auto-start server if not running
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.auto_start_server = auto_start_server
        self.server_process = None

        # Setup command history
        if PROMPT_TOOLKIT_AVAILABLE:
            history_file = os.path.expanduser("~/.dt_cli_history")
            self.prompt_session = PromptSession(history=FileHistory(history_file))
        else:
            self.prompt_session = None

    def check_server(self) -> bool:
        """Check if server is running."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def start_server(self) -> bool:
        """Start the server if not running."""
        console.print("[yellow]Server not running. Starting server...[/yellow]")

        try:
            # Find the server script
            server_script = os.path.join(os.path.dirname(__file__), "..", "mcp_server", "standalone_server.py")

            if not os.path.exists(server_script):
                server_script = "src/mcp_server/standalone_server.py"

            if not os.path.exists(server_script):
                console.print("[red]Server script not found![/red]")
                return False

            # Start server as background process
            self.server_process = subprocess.Popen(
                [sys.executable, server_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start (max 10 seconds)
            console.print("[cyan]Waiting for server to start...[/cyan]")
            for i in range(10):
                time.sleep(1)
                if self.check_server():
                    console.print("[green]Server started successfully![/green]")
                    return True

            console.print("[red]Server failed to start in time[/red]")
            return False

        except Exception as e:
            console.print(f"[red]Failed to start server: {e}[/red]")
            return False

    def show_welcome(self):
        """Display welcome message."""
        welcome_text = """
# dt-cli Interactive Terminal

Welcome to the **100% Open Source** RAG/MAF/LLM System!

## Available Features:
- **RAG Queries** - Ask questions about your codebase
- **Debug Errors** - Analyze errors and get fix suggestions
- **Code Review** - Automated code quality and security checks
- **Knowledge Graph** - Explore code dependencies and relationships
- **Evaluation** - Measure RAG quality with RAGAS metrics
- **Hybrid Search** - Semantic + keyword search

Type `help` for available commands or choose from the menu below.
"""
        console.print(Panel(Markdown(welcome_text), border_style="blue", box=box.DOUBLE))

    def show_menu(self):
        """Display main menu."""
        table = Table(title="Main Menu", box=box.ROUNDED, border_style="cyan")
        table.add_column("Option", style="cyan", justify="center")
        table.add_column("Description", style="white")

        options = [
            ("1", "Ask a Question (RAG Query)"),
            ("2", "Debug an Error"),
            ("3", "Review Code"),
            ("4", "Explore Knowledge Graph"),
            ("5", "Evaluate RAG Quality"),
            ("6", "Hybrid Search"),
            ("7", "View Statistics"),
            ("8", "Settings"),
            ("9", "Help"),
            ("0", "Exit")
        ]

        for opt, desc in options:
            table.add_row(opt, desc)

        console.print(table)

    def get_input_with_history(self, prompt_text: str, default: str = "") -> str:
        """Get user input with command history support."""
        if self.prompt_session:
            try:
                return self.prompt_session.prompt(prompt_text + ": ", default=default)
            except (KeyboardInterrupt, EOFError):
                raise
        else:
            if default:
                return input(f"{prompt_text} [{default}]: ") or default
            else:
                return input(f"{prompt_text}: ")

    def ask_question(self, query: Optional[str] = None):
        """Handle RAG query."""
        console.print("\n[bold cyan]Ask a Question[/bold cyan]", style="bold")
        console.print("=" * 60)

        if not query:
            if self.prompt_session:
                query = self.get_input_with_history("[bold]Your question[/bold]")
            else:
                query = Prompt.ask("[bold]Your question[/bold]")

        if not query.strip():
            console.print("[yellow]Empty query![/yellow]")
            return

        # Ask for context files
        add_files = Confirm.ask("Add specific files to context?", default=False)
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
            task = progress.add_task("Searching codebase...", total=None)

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
                    console.print("\n[bold green]Answer:[/bold green]")
                    console.print(Panel(result['response'], border_style="green"))

                    # Show metadata
                    if 'auto_trigger' in result:
                        trigger_info = result['auto_trigger']
                        console.print(f"\n[dim]Intent: {trigger_info['intent']} (confidence: {trigger_info['confidence']:.0%})[/dim]")
                        console.print(f"[dim]Actions: {', '.join(trigger_info['actions'])}[/dim]")

                    if 'context_used' in result:
                        console.print(f"[dim]Contexts used: {result['context_used']}[/dim]")

                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")
                    if response.status_code == 404:
                        console.print("[yellow]The /query endpoint was not found. Make sure the server is running correctly.[/yellow]")

            except requests.exceptions.Timeout:
                console.print("[red]Request timed out[/red]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def debug_error(self):
        """Handle error debugging."""
        console.print("\n[bold red]Debug an Error[/bold red]", style="bold")
        console.print("=" * 60)

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
            console.print("[yellow]No error provided![/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing error...", total=None)

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
                    console.print("\n[bold]Error Location:[/bold]")
                    error_ctx = result['error_context']
                    console.print(f"  Type: {error_ctx['error_type']}")
                    if error_ctx.get('file_path'):
                        console.print(f"  File: {error_ctx['file_path']}:{error_ctx.get('line_number', '?')}")
                    console.print(f"  Message: {error_ctx['error_message']}")

                    # Show code snippet
                    if error_ctx.get('code_snippet'):
                        console.print("\n[bold]Code Context:[/bold]")
                        syntax = Syntax(error_ctx['code_snippet'], "python", line_numbers=True)
                        console.print(syntax)

                    # Show root cause
                    console.print(f"\n[bold red]Root Cause:[/bold red]")
                    console.print(Panel(result['root_cause'], border_style="red"))

                    # Show explanation
                    console.print(f"\n[bold]Explanation:[/bold]")
                    console.print(result['explanation'])

                    # Show fixes
                    console.print(f"\n[bold green]Suggested Fixes:[/bold green]")
                    for i, fix in enumerate(result['suggested_fixes'], 1):
                        console.print(f"  {i}. {fix}")

                    console.print(f"\n[dim]Confidence: {result['confidence']:.0%}[/dim]")

                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def review_code(self):
        """Handle code review."""
        console.print("\n[bold magenta]Review Code[/bold magenta]", style="bold")
        console.print("=" * 60)

        # Get code input
        file_path = Prompt.ask("[bold]File path to review[/bold] (or press Enter to paste code)")

        if file_path.strip():
            # Read from file
            try:
                with open(file_path, 'r') as f:
                    code = f.read()
            except Exception as e:
                console.print(f"[red]Could not read file: {e}[/red]")
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
            console.print("[yellow]No code provided![/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Reviewing code...", total=None)

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

                    console.print(f"\n[bold]Quality Score: [{score_color}]{score:.1f}/10[/{score_color}][/bold]")
                    console.print(f"[dim]{result['summary']}[/dim]")

                    # Show issues
                    if result['issues']:
                        console.print(f"\n[bold]Issues Found ({len(result['issues'])}):[/bold]")

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
                                    console.print(f"  - {issue['title']}")
                                    if issue.get('line_number'):
                                        console.print(f"    Line {issue['line_number']}: {issue['description']}")
                                    else:
                                        console.print(f"    {issue['description']}")
                                    if issue.get('suggestion'):
                                        console.print(f"    [dim]Fix: {issue['suggestion']}[/dim]")

                                if len(by_severity[severity]) > 5:
                                    console.print(f"  [dim]... and {len(by_severity[severity]) - 5} more[/dim]")

                    else:
                        console.print("\n[bold green]No issues found![/bold green]")

                    # Show metrics
                    console.print(f"\n[bold]Metrics:[/bold]")
                    metrics = result['metrics']
                    console.print(f"  Total lines: {metrics['total_lines']}")
                    console.print(f"  Code lines: {metrics['code_lines']}")
                    console.print(f"  Comment lines: {metrics['comment_lines']}")
                    console.print(f"  Issues: {metrics['total_issues']}")

                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

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

    def intelligent_mode(self, user_input: str):
        """
        Intelligent mode: analyze user input and route to appropriate function.
        Bypasses menu and goes directly to RAG/MAF.
        """
        user_input_lower = user_input.lower().strip()

        # Directly execute RAG query
        self.ask_question(query=user_input)

    def run(self, intelligent: bool = False):
        """
        Run the interactive CLI.

        Args:
            intelligent: If True, skip menu and go straight to intelligent mode
        """
        # Check server and auto-start if needed
        if not self.check_server():
            if self.auto_start_server:
                if not self.start_server():
                    console.print("[bold red]Server is not running and could not be started![/bold red]")
                    console.print("\nPlease start the server manually:")
                    console.print("  [cyan]python src/mcp_server/standalone_server.py[/cyan]")
                    return
            else:
                console.print("[bold red]Server is not running![/bold red]")
                console.print("\nPlease start the server first:")
                console.print("  [cyan]python src/mcp_server/standalone_server.py[/cyan]")
                return

        # Show welcome
        self.show_welcome()

        # Intelligent mode: get user input and route directly
        if intelligent:
            console.print("\n[bold cyan]Intelligent Mode[/bold cyan]")
            console.print("[dim]Enter your question and I'll automatically use RAG/MAF as needed.[/dim]\n")

            while True:
                try:
                    if self.prompt_session:
                        user_input = self.get_input_with_history("\n[bold]Your question (or 'exit' to quit)[/bold]")
                    else:
                        user_input = Prompt.ask("\n[bold]Your question (or 'exit' to quit)[/bold]")

                    if user_input.lower() in ['exit', 'quit', 'q']:
                        console.print("\n[bold]Goodbye![/bold]")
                        break

                    if user_input.strip():
                        self.intelligent_mode(user_input)

                except KeyboardInterrupt:
                    console.print("\n\n[bold]Goodbye![/bold]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

            return

        # Regular menu mode
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
                    console.print("[yellow]Knowledge Graph feature - use server API directly[/yellow]")
                elif choice == "5":
                    console.print("[yellow]Evaluation feature - use server API directly[/yellow]")
                elif choice == "6":
                    console.print("[yellow]Hybrid Search feature - use server API directly[/yellow]")
                elif choice == "7":
                    console.print("[yellow]Statistics feature - use server API directly[/yellow]")
                elif choice == "8":
                    console.print("[yellow]Settings coming soon![/yellow]")
                elif choice == "9":
                    self.show_help()
                elif choice == "0":
                    console.print("\n[bold]Goodbye![/bold]")
                    break
                else:
                    console.print("[yellow]Invalid option![/yellow]")

            except KeyboardInterrupt:
                console.print("\n\n[bold]Goodbye![/bold]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def __del__(self):
        """Cleanup: stop server if we started it."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                pass


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="dt-cli Interactive Terminal")
    parser.add_argument(
        "--server",
        default="http://localhost:8765",
        help="Server URL (default: http://localhost:8765)"
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't auto-start server if not running"
    )
    parser.add_argument(
        "--intelligent",
        "-i",
        action="store_true",
        help="Use intelligent mode (skip menu, go directly to RAG)"
    )

    args = parser.parse_args()

    cli = DTCliInteractive(
        base_url=args.server,
        auto_start_server=not args.no_auto_start
    )
    cli.run(intelligent=args.intelligent)


if __name__ == "__main__":
    main()
