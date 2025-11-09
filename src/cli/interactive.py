"""
Interactive Terminal Interface for dt-cli.

Provides a beautiful, user-friendly TUI for interacting with the RAG/MAF system
without needing to know API endpoints or curl commands.

Features:
- Intelligent intent classification (auto-detects debug, code, review, questions)
- Project folder selection on startup
- Planning mode with user approval
- Verbosity control via slash commands
- Summary display with diff tracking
- Command history with up/down arrow navigation
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
from rich.text import Text
from typing import Optional, Dict, Any, List, Tuple
import requests
import sys
import os
import subprocess
import time
import socket
import re
from pathlib import Path
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from datetime import datetime
import glob as glob_module

# Import session history manager
try:
    from .session_history import SessionHistoryManager
    SESSION_HISTORY_AVAILABLE = True
except ImportError:
    SESSION_HISTORY_AVAILABLE = False
    SessionHistoryManager = None

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

console = Console()


class VerbosityLevel(IntEnum):
    """Verbosity levels for output control."""
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2

    @classmethod
    def from_string(cls, value: str) -> 'VerbosityLevel':
        """Convert string to VerbosityLevel."""
        mapping = {
            "quiet": cls.QUIET,
            "normal": cls.NORMAL,
            "verbose": cls.VERBOSE
        }
        return mapping.get(value.lower(), cls.NORMAL)

    def to_string(self) -> str:
        """Convert VerbosityLevel to string name."""
        mapping = {
            self.QUIET: "quiet",
            self.NORMAL: "normal",
            self.VERBOSE: "verbose"
        }
        return mapping.get(self, "normal")


class IntentType(Enum):
    """Types of user intents."""
    DEBUG = "debug"
    CODE = "code"
    REVIEW = "review"
    QUESTION = "question"
    PLAN = "plan"
    UNKNOWN = "unknown"


@dataclass
class FileChange:
    """Track a file change for diff display."""
    file_path: str
    lines_added: int = 0
    lines_removed: int = 0
    before_snippet: Optional[str] = None
    after_snippet: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationContext:
    """
    Tracks conversation context for enhanced RAG/MAF utilization.
    Mirrors the server-side ConversationContext for consistency.
    """
    turn_count: int = 0
    files_in_context: List[str] = field(default_factory=list)
    last_intent: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    project_files_cache: Optional[List[str]] = None
    max_history: int = 10
    max_context_files: int = 20

    def add_turn(self, user_input: str, intent: str, response_summary: str):
        """Add a conversation turn."""
        self.turn_count += 1
        self.last_intent = intent
        self.conversation_history.append({
            'turn': self.turn_count,
            'user_input': user_input,
            'intent': intent,
            'response_summary': response_summary,
            'timestamp': datetime.now().isoformat()
        })
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def add_file_to_context(self, file_path: str):
        """Add a file to the current context."""
        file_path = str(file_path)
        if file_path not in self.files_in_context:
            self.files_in_context.append(file_path)
            # Keep only recent files
            if len(self.files_in_context) > self.max_context_files:
                self.files_in_context = self.files_in_context[-self.max_context_files:]

    def get_relevant_files(self, query: str, project_folder: Path) -> List[str]:
        """
        Get relevant files based on query keywords.
        Returns file paths that might be relevant to the query.
        """
        if not self.project_files_cache:
            return []

        # Extract potential file/module names from query
        words = re.findall(r'\b\w+\b', query.lower())

        relevant = []
        for file_path in self.project_files_cache[:100]:  # Limit search
            file_lower = file_path.lower()
            # Check if any query word matches part of filename
            for word in words:
                if len(word) > 3 and word in file_lower:
                    relevant.append(file_path)
                    break
            if len(relevant) >= 10:  # Limit to top 10
                break

        return relevant

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for server context."""
        return {
            'turn_count': self.turn_count,
            'files_in_context': self.files_in_context,
            'last_intent': self.last_intent,
            'is_followup': self.turn_count > 0,
            'conversation_history': self.conversation_history[-3:]  # Last 3 turns
        }


class IntentClassifier:
    """Classifies user intent from natural language input."""

    # Keywords for each intent type
    PATTERNS = {
        IntentType.DEBUG: [
            r'\berror\b', r'\bexception\b', r'\bfail(ed|ing|ure)?\b',
            r'\bbug\b', r'\bfix\b', r'\bcrash\b', r'\btraceback\b',
            r'\bstack trace\b', r'\bbroken\b', r'\bissue\b', r'\bdebug\b'
        ],
        IntentType.CODE: [
            r'\bimplement\b', r'\bcreate\b', r'\bbuild\b', r'\bwrite\b',
            r'\badd (a )?feature\b', r'\bgenerate\b', r'\bcode\b',
            r'\bmake\b', r'\bdevelop\b', r'\brefactor\b'
        ],
        IntentType.REVIEW: [
            r'\breview\b', r'\bcheck\b', r'\banalyze\b', r'\binspect\b',
            r'\bquality\b', r'\bsecurity\b', r'\bvulnerabilit(y|ies)\b',
            r'\bcode smell\b', r'\blint\b'
        ],
        IntentType.PLAN: [
            r'\bplan\b', r'\bdesign\b', r'\barchitecture\b',
            r'\bstrategy\b', r'\bapproach\b', r'\bhow should (i|we)\b',
            r"\bwhat['\u2019]s the best way\b", r'\bsteps? (to|for)\b'
        ],
        IntentType.QUESTION: [
            r'^\s*(what|how|where|when|why|who|which)\b',
            r'\bexplain\b', r'\bdescribe\b', r'\btell me\b',
            r'\bshow me\b', r'\bfind\b', r'\bsearch\b'
        ]
    }

    @classmethod
    def classify(cls, user_input: str) -> Tuple[IntentType, float]:
        """
        Classify user intent from input text.

        Args:
            user_input: User's natural language input

        Returns:
            Tuple of (intent_type, confidence_score)
        """
        user_input_lower = user_input.lower().strip()

        # Score each intent type
        scores = {intent: 0 for intent in IntentType}

        for intent_type, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    scores[intent_type] += 1

        # Find highest scoring intent
        if max(scores.values()) == 0:
            # Default to QUESTION if no patterns match
            return IntentType.QUESTION, 0.5

        best_intent = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_intent[1] / 3.0, 1.0)  # Normalize to 0-1

        return best_intent[0], confidence


class SummaryTracker:
    """Tracks actions and file changes for summary display."""

    def __init__(self):
        self.actions: List[str] = []
        self.file_changes: List[FileChange] = []
        self.start_time = datetime.now()

    def add_action(self, action: str):
        """Add an action to the tracker."""
        self.actions.append(action)

    def add_file_change(self, change: FileChange):
        """Add a file change to the tracker."""
        self.file_changes.append(change)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked changes."""
        total_added = sum(fc.lines_added for fc in self.file_changes)
        total_removed = sum(fc.lines_removed for fc in self.file_changes)

        return {
            'actions': self.actions,
            'files_changed': len(self.file_changes),
            'lines_added': total_added,
            'lines_removed': total_removed,
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'changes': self.file_changes
        }

    def display_summary(self, verbosity: VerbosityLevel = VerbosityLevel.NORMAL):
        """Display summary with appropriate verbosity level."""
        summary = self.get_summary()

        if verbosity == VerbosityLevel.QUIET:
            # Only show if there were changes
            if summary['files_changed'] > 0:
                console.print(f"[dim]Modified {summary['files_changed']} file(s)[/dim]")
            return

        # Normal and verbose modes
        console.print("\n[bold cyan]Summary[/bold cyan]")
        console.print("=" * 60)

        # Actions performed
        if summary['actions']:
            console.print("\n[bold]Actions performed:[/bold]")
            for action in summary['actions']:
                console.print(f"  - {action}")

        # File changes
        if summary['files_changed'] > 0:
            console.print(f"\n[bold]Files changed:[/bold] {summary['files_changed']}")
            console.print(f"  [green]+{summary['lines_added']}[/green] lines added")
            console.print(f"  [red]-{summary['lines_removed']}[/red] lines removed")

            if verbosity == VerbosityLevel.VERBOSE:
                # Show detailed changes
                console.print("\n[bold]Detailed changes:[/bold]")
                for change in summary['changes']:
                    console.print(f"\n  [cyan]{change.file_path}[/cyan]")
                    console.print(f"    [green]+{change.lines_added}[/green] / [red]-{change.lines_removed}[/red]")

                    if change.before_snippet and change.after_snippet:
                        console.print("\n    [dim]Before:[/dim]")
                        console.print(Panel(
                            Syntax(change.before_snippet, "python", theme="monokai", line_numbers=True),
                            border_style="red"
                        ))
                        console.print("\n    [dim]After:[/dim]")
                        console.print(Panel(
                            Syntax(change.after_snippet, "python", theme="monokai", line_numbers=True),
                            border_style="green"
                        ))

        console.print(f"\n[dim]Duration: {summary['duration']:.2f}s[/dim]")


def is_port_available(host: str, port: int) -> bool:
    """
    Check if a port is available.

    Args:
        host: Host address
        port: Port number

    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(host: str, preferred_port: int, max_attempts: int = 10) -> int:
    """
    Find an available port, starting with preferred port.

    Args:
        host: Host address
        preferred_port: Preferred port to try first
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number

    Raises:
        RuntimeError: If no available port found
    """
    # Try preferred port first
    if is_port_available(host, preferred_port):
        return preferred_port

    # Try adjacent ports
    for offset in range(1, max_attempts):
        port = preferred_port + offset
        if port > 65535:
            break
        if is_port_available(host, port):
            return port

    raise RuntimeError(f"Could not find an available port near {preferred_port}")


class DTCliInteractive:
    """
    Interactive terminal interface for dt-cli.
    """

    def __init__(self, base_url: str = "http://localhost:58432", auto_start_server: bool = True):
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
        self.project_folder: Optional[Path] = None
        self.verbosity: VerbosityLevel = VerbosityLevel.NORMAL
        self.tracker = SummaryTracker()
        self.conversation_context = ConversationContext()

        # Initialize session history manager
        if SESSION_HISTORY_AVAILABLE:
            self.session_history = SessionHistoryManager()
        else:
            self.session_history = None

        # Setup command history
        if PROMPT_TOOLKIT_AVAILABLE:
            history_file = os.path.expanduser("~/.dt_cli_history")
            self.prompt_session = PromptSession(history=FileHistory(history_file))
        else:
            self.prompt_session = None

    def check_server(self) -> bool:
        """Check if server is running and properly initialized."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            if response.status_code != 200:
                return False

            # Check if server is fully initialized with all endpoints
            health_data = response.json()
            if health_data.get('status') == 'degraded':
                # Server is running but not fully healthy
                # This is OK - the server can still function
                if self.verbosity >= VerbosityLevel.NORMAL:
                    console.print("[yellow]Server is running but degraded:[/yellow]")
                    if 'endpoints' in health_data:
                        missing = [k for k, v in health_data['endpoints'].items() if not v]
                        if missing:
                            console.print(f"  Missing endpoints: {', '.join(missing)}")
                    if health_data.get('llm') == 'unhealthy':
                        console.print("  [yellow]LLM provider is unhealthy (Ollama may not be running)[/yellow]")
                        console.print("  [dim]To fix: Start Ollama with 'ollama serve' or install it from https://ollama.ai[/dim]")
                # Return True - server is running even if degraded
                return True

            return True
        except:
            return False

    def start_server(self) -> bool:
        """Start the server if not running."""
        console.print("[yellow]Server not running. Starting server...[/yellow]")

        try:
            # Check for required dependencies first
            missing_deps = []
            required_modules = ['fastapi', 'uvicorn', 'chromadb', 'langchain', 'pydantic']

            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_deps.append(module)

            if missing_deps:
                console.print(f"[red]Missing required dependencies: {', '.join(missing_deps)}[/red]")
                console.print("\n[yellow]Please install dependencies first:[/yellow]")
                console.print("  [cyan]pip3 install -r requirements.txt[/cyan]")
                console.print("\nOr install missing packages individually:")
                console.print(f"  [cyan]pip3 install {' '.join(missing_deps)}[/cyan]")
                return False

            # Find the server script
            server_script = os.path.join(os.path.dirname(__file__), "..", "mcp_server", "standalone_server.py")

            if not os.path.exists(server_script):
                server_script = "src/mcp_server/standalone_server.py"

            if not os.path.exists(server_script):
                console.print("[red]Server script not found![/red]")
                return False

            # Extract host and port from base_url
            # base_url format: http://host:port
            url_parts = self.base_url.replace("http://", "").replace("https://", "").split(":")
            host = url_parts[0]
            preferred_port = int(url_parts[1]) if len(url_parts) > 1 else 58432

            # Find available port
            try:
                actual_port = find_available_port(host, preferred_port)
                if actual_port != preferred_port:
                    console.print(f"[yellow]Port {preferred_port} in use, using port {actual_port}[/yellow]")
                    # Update base_url to use the actual port
                    self.base_url = f"http://{host}:{actual_port}"
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
                return False

            # Start server as background process with the chosen port
            self.server_process = subprocess.Popen(
                [sys.executable, server_script, "--port", str(actual_port), "--host", host],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start (max 10 seconds)
            console.print(f"[cyan]Waiting for server to start on port {actual_port}...[/cyan]")
            for i in range(10):
                time.sleep(1)
                if self.check_server():
                    console.print(f"[green]Server started successfully on {self.base_url}![/green]")
                    return True

            console.print("[red]Server failed to start in time[/red]")
            return False

        except Exception as e:
            console.print(f"[red]Failed to start server: {e}[/red]")
            return False

    def _discover_project_files(self):
        """Discover and cache project files for context-aware queries."""
        if not self.project_folder:
            return

        try:
            # Common code file extensions
            extensions = ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.java', '*.go', '*.rs', '*.cpp', '*.c', '*.h']

            files = []
            for ext in extensions:
                pattern = str(self.project_folder / '**' / ext)
                found = glob_module.glob(pattern, recursive=True)
                files.extend(found)

            # Make paths relative to project folder for cleaner display
            relative_files = []
            for f in files:
                try:
                    rel = os.path.relpath(f, self.project_folder)
                    # Skip common exclusions
                    if not any(skip in rel for skip in ['.git/', 'node_modules/', '__pycache__/', '.venv/', 'venv/', 'dist/', 'build/']):
                        relative_files.append(rel)
                except ValueError:
                    continue

            self.conversation_context.project_files_cache = relative_files
            if self.verbosity == VerbosityLevel.VERBOSE:
                console.print(f"[dim]Discovered {len(relative_files)} project files[/dim]")

        except Exception as e:
            if self.verbosity == VerbosityLevel.VERBOSE:
                console.print(f"[dim]Could not discover project files: {e}[/dim]")

    def select_project_folder(self):
        """Prompt user to select project folder."""
        console.print("\n[bold cyan]Project Folder Selection[/bold cyan]")
        console.print("=" * 60)

        current_dir = Path.cwd()
        console.print(f"\nCurrent directory: [cyan]{current_dir}[/cyan]")

        use_current = Confirm.ask(
            "Is this the base folder for your project?",
            default=True
        )

        if use_current:
            self.project_folder = current_dir
        else:
            while True:
                folder_path = Prompt.ask("\nEnter project directory path")
                path = Path(folder_path).expanduser().resolve()

                if path.exists() and path.is_dir():
                    self.project_folder = path
                    break
                else:
                    console.print(f"[red]Directory not found: {path}[/red]")
                    retry = Confirm.ask("Try again?", default=True)
                    if not retry:
                        self.project_folder = current_dir
                        break

        console.print(f"\n[green]Project folder set to: {self.project_folder}[/green]")

        # Discover project files for context
        if self.verbosity >= VerbosityLevel.NORMAL:
            console.print("[cyan]Discovering project files for enhanced context...[/cyan]")
        self._discover_project_files()

        # Start session history for this project
        if self.session_history:
            session_id = self.session_history.start_session(str(self.project_folder))
            if self.verbosity == VerbosityLevel.VERBOSE:
                console.print(f"[dim]Started session: {session_id}[/dim]")

    def handle_slash_command(self, command: str) -> bool:
        """
        Handle slash commands.

        Args:
            command: Command string (e.g., "/verbosity normal")

        Returns:
            True if command was handled, False otherwise
        """
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/verbosity":
            if len(parts) < 2:
                console.print(f"[yellow]Current verbosity: {self.verbosity.to_string()}[/yellow]")
                console.print("Usage: /verbosity <quiet|normal|verbose>")
            else:
                level = parts[1].lower()
                if level in ["quiet", "normal", "verbose"]:
                    self.verbosity = VerbosityLevel.from_string(level)
                    console.print(f"[green]Verbosity set to: {level}[/green]")
                else:
                    console.print("[red]Invalid verbosity level. Use: quiet, normal, or verbose[/red]")
            return True

        elif cmd == "/help":
            self.show_help()
            return True

        elif cmd in ["/exit", "/quit"]:
            return False

        elif cmd == "/folder":
            console.print(f"[cyan]Current project folder: {self.project_folder}[/cyan]")
            change = Confirm.ask("Change folder?", default=False)
            if change:
                self.select_project_folder()
            return True

        elif cmd == "/history":
            self.show_session_history()
            return True

        elif cmd == "/sessions":
            self.show_all_sessions()
            return True

        elif cmd == "/clearsession":
            if Confirm.ask("[bold red]Clear ALL session history?[/bold red] This cannot be undone!", default=False):
                if self.session_history:
                    self.session_history.clear_all_history()
                    console.print("[green]Session history cleared[/green]")
                else:
                    console.print("[yellow]Session history not available[/yellow]")
            return True

        elif cmd == "/stats":
            self.show_session_stats()
            return True

        else:
            console.print(f"[red]Unknown command: {cmd}[/red]")
            console.print("Available commands: /verbosity, /help, /folder, /history, /sessions, /stats, /clearsession, /exit")
            return True

    def _calculate_importance_score(self, intent: IntentType, confidence: float) -> float:
        """
        Calculate importance score for a conversation turn.

        Used for selective retention in hierarchical memory.

        Args:
            intent: Detected intent
            confidence: Confidence score

        Returns:
            Importance score (0.0-1.0)
        """
        # Base score from confidence
        score = confidence

        # Boost certain intents
        importance_boost = {
            IntentType.DEBUG: 0.2,   # Debugging sessions are important
            IntentType.CODE: 0.15,   # Code changes are important
            IntentType.PLAN: 0.15,   # Plans are important
            IntentType.REVIEW: 0.1,  # Reviews are moderately important
        }

        boost = importance_boost.get(intent, 0.0)
        score = min(1.0, score + boost)

        return score

    def _build_enriched_query_payload(self, query: str, intent: Optional[str] = None) -> Dict[str, Any]:
        """
        Build an enriched query payload with full context for better RAG/MAF utilization.

        Args:
            query: User's query
            intent: Detected intent (optional)

        Returns:
            Enriched query payload dictionary
        """
        # Get relevant files based on query keywords
        relevant_files = self.conversation_context.get_relevant_files(query, self.project_folder)

        # Combine with files already in context
        all_context_files = list(set(self.conversation_context.files_in_context + relevant_files))

        # Build enriched query with project context
        enriched_query = query
        if self.project_folder and intent != 'plan':  # Don't modify planning queries
            # Add project context hint for RAG
            project_name = self.project_folder.name
            enriched_query = f"[Project: {project_name}] {query}"

        payload = {
            "query": enriched_query,
            "auto_trigger": True,
            "context_files": all_context_files[:20] if all_context_files else None  # Limit to 20 files
        }

        if self.verbosity == VerbosityLevel.VERBOSE and all_context_files:
            console.print(f"[dim]Sending {len(all_context_files)} context files to enhance query[/dim]")

        return payload

    def process_user_input(self, user_input: str):
        """
        Process user input with intelligent intent classification.

        Args:
            user_input: User's natural language input
        """
        # Reset tracker for new request
        self.tracker = SummaryTracker()

        # Classify intent
        intent, confidence = IntentClassifier.classify(user_input)

        if self.verbosity == VerbosityLevel.VERBOSE:
            console.print(f"\n[dim]Intent detected: {intent.value} (confidence: {confidence:.0%})[/dim]")
            if self.conversation_context.turn_count > 0:
                console.print(f"[dim]Conversation turn: {self.conversation_context.turn_count + 1}[/dim]")

        # Route to appropriate handler
        result_summary = ""
        if intent == IntentType.PLAN:
            self.handle_planning_request(user_input)
            result_summary = "Generated implementation plan"
        elif intent == IntentType.DEBUG:
            self.handle_debug_request(user_input)
            result_summary = "Analyzed error and provided fixes"
        elif intent == IntentType.CODE:
            self.handle_code_request(user_input)
            result_summary = "Implemented code changes"
        elif intent == IntentType.REVIEW:
            self.handle_review_request(user_input)
            result_summary = "Reviewed code for issues"
        else:  # QUESTION or UNKNOWN
            self.handle_question_request(user_input)
            result_summary = "Answered question"

        # Update conversation context
        self.conversation_context.add_turn(user_input, intent.value, result_summary)

        # Add to session history with hierarchical memory
        if self.session_history:
            # Calculate importance score based on intent
            importance_score = self._calculate_importance_score(intent, confidence)
            self.session_history.add_turn(user_input, intent.value, result_summary, importance_score)

        # Display summary
        if self.verbosity != VerbosityLevel.QUIET:
            self.tracker.display_summary(self.verbosity)

    def handle_planning_request(self, user_input: str):
        """Handle planning requests with user approval."""
        console.print("\n[bold magenta]Planning Mode[/bold magenta]")
        console.print("=" * 60)

        self.tracker.add_action("Analyzing request and creating plan")

        if self.verbosity >= VerbosityLevel.NORMAL:
            console.print("[cyan]Creating implementation plan...[/cyan]")

        # Generate plan using RAG
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating plan...", total=None)

            try:
                response = self.session.post(
                    f"{self.base_url}/query",
                    json={
                        "query": f"Create a detailed implementation plan for: {user_input}",
                        "auto_trigger": True
                    },
                    timeout=30
                )

                progress.update(task, completed=True)

                if response.status_code == 200:
                    result = response.json()
                    plan = result['response']

                    # Display plan
                    console.print("\n[bold]Implementation Plan:[/bold]")
                    console.print(Panel(Markdown(plan), border_style="magenta"))

                    # Ask for approval
                    approved = Confirm.ask("\nProceed with this plan?", default=True)

                    if approved:
                        console.print("[green]Plan approved. Proceeding with implementation...[/green]")
                        self.tracker.add_action("Plan approved by user")
                        # Now execute as a code request
                        self.handle_code_request(user_input)
                    else:
                        console.print("[yellow]Plan cancelled by user[/yellow]")
                        self.tracker.add_action("Plan cancelled by user")

                else:
                    console.print(f"[red]Error generating plan: {response.status_code}[/red]")

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def handle_debug_request(self, user_input: str):
        """Handle debug requests."""
        console.print("\n[bold red]Debug Mode[/bold red]")
        console.print("=" * 60)

        self.tracker.add_action("Analyzing error/bug")

        # Check if error output is in the input
        if len(user_input) > 200:  # Likely contains error output
            error_output = user_input
        else:
            console.print("[yellow]Please paste your error output (press Ctrl+D when done):[/yellow]")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            error_output = "\n".join(lines) if lines else user_input

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

                    # Show root cause
                    console.print(f"\n[bold red]Root Cause:[/bold red]")
                    console.print(Panel(result['root_cause'], border_style="red"))

                    # Show fixes
                    console.print(f"\n[bold green]Suggested Fixes:[/bold green]")
                    for i, fix in enumerate(result['suggested_fixes'], 1):
                        console.print(f"  {i}. {fix}")

                    self.tracker.add_action("Error analyzed and fixes suggested")

                elif response.status_code == 404:
                    console.print("[red]Error: The /debug endpoint was not found.[/red]")
                    console.print("\n[yellow]This could mean:[/yellow]")
                    console.print("  1. The server is not running properly")
                    console.print("  2. The server failed to initialize completely")
                    console.print(f"\n[cyan]Current server URL: {self.base_url}[/cyan]")
                    console.print("\n[yellow]Try restarting the server:[/yellow]")
                    console.print("  [cyan]python3 src/mcp_server/standalone_server.py[/cyan]")
                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")
                    try:
                        error_detail = response.json()
                        if 'detail' in error_detail:
                            console.print(f"[red]Details: {error_detail['detail']}[/red]")
                    except:
                        pass

            except requests.exceptions.ConnectionError:
                console.print(f"[red]Connection Error: Could not connect to server at {self.base_url}[/red]")
                console.print("[yellow]The server may not be running. Try starting it with:[/yellow]")
                console.print("  [cyan]python3 src/mcp_server/standalone_server.py[/cyan]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def handle_code_request(self, user_input: str):
        """Handle code implementation requests."""
        console.print("\n[bold green]Code Mode[/bold green]")
        console.print("=" * 60)

        self.tracker.add_action("Implementing code changes")

        # Use RAG to help with implementation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Implementing...", total=None)

            try:
                # Use enriched query payload with context
                payload = self._build_enriched_query_payload(user_input, intent='code')

                response = self.session.post(
                    f"{self.base_url}/query",
                    json=payload,
                    timeout=30
                )

                progress.update(task, completed=True)

                if response.status_code == 200:
                    result = response.json()

                    console.print("\n[bold green]Implementation:[/bold green]")
                    console.print(Panel(Markdown(result['response']), border_style="green"))

                    self.tracker.add_action("Code implementation completed")

                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def handle_review_request(self, user_input: str):
        """Handle code review requests."""
        self.review_code(user_input=user_input)

    def handle_question_request(self, user_input: str):
        """Handle question/query requests."""
        self.ask_question(query=user_input)

    def show_welcome(self):
        """Display welcome message."""
        # Get session info
        session_info = ""
        if self.session_history:
            stats = self.session_history.get_statistics()
            if stats['current_session_active']:
                session_info = f"\n**Session:** Active ({stats['current_session_turns']} turns this session)"
                if stats['archived_sessions'] > 0:
                    session_info += f" | {stats['archived_sessions']} archived sessions"

        welcome_text = f"""
# dt-cli Intelligent Interactive CLI

Welcome to the **100% Open Source** RAG/MAF/LLM System with AI-powered memory!

**Project:** `{self.project_folder}`
**Verbosity:** `{self.verbosity.to_string()}`{session_info}

## New Intelligent Features

**Hierarchical Session Memory** - Your conversations persist across sessions
- Remember context from days/weeks ago
- Automatic compression of older conversations
- Important discussions never forgotten

**Context-Aware Queries** - Automatically includes relevant files
- Smart file discovery from your project
- Keyword-based relevance matching
- Up to 20 context files sent per query

**Natural Language Interface** - Just type what you need
- No menu navigation required
- Intent auto-detection (debug, code, review, question)
- Follow-up questions understand context

## What can I help you with?

**Ask Questions:**
- "Where is authentication handled?"
- "How does the caching system work?"

**Debug Errors:**
- "Debug this ImportError"
- "Fix the failing tests"

**Review & Implement:**
- "Review codebase and find any errors"
- "Add logging to the API endpoints"

**Plan Features:**
- "Plan how to add rate limiting"
- "Design a caching strategy"

## Power User Commands

**Memory & Session:**
- `/history` - View current session with full context
- `/sessions` - List all sessions (current + archived)
- `/stats` - Show memory usage statistics
- `/clearsession` - Clear all history

**Settings:**
- `/verbosity <quiet|normal|verbose>` - Control output detail
- `/folder` - Change project folder
- `/help` - Comprehensive help
- `/exit` - Save session and exit

**Tip:** Your first query automatically discovers project files for better context!
"""
        console.print(Panel(Markdown(welcome_text), border_style="cyan", box=box.DOUBLE))


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
        if self.verbosity >= VerbosityLevel.NORMAL:
            console.print("\n[bold cyan]Question Mode[/bold cyan]")
            console.print("=" * 60)

        self.tracker.add_action("Querying codebase")

        if not query:
            if self.prompt_session:
                query = self.get_input_with_history("[bold]Your question[/bold]")
            else:
                query = Prompt.ask("[bold]Your question[/bold]")

        if not query.strip():
            console.print("[yellow]Empty query![/yellow]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching codebase...", total=None)

            try:
                # Use enriched query payload with context
                payload = self._build_enriched_query_payload(query, intent='question')

                response = self.session.post(
                    f"{self.base_url}/query",
                    json=payload,
                    timeout=30
                )

                progress.update(task, completed=True)

                if response.status_code == 200:
                    result = response.json()

                    # Show results
                    console.print("\n[bold green]Answer:[/bold green]")
                    console.print(Panel(result['response'], border_style="green"))

                    # Show metadata in verbose mode
                    if self.verbosity == VerbosityLevel.VERBOSE:
                        if 'auto_trigger' in result:
                            trigger_info = result['auto_trigger']
                            console.print(f"\n[dim]Intent: {trigger_info['intent']} (confidence: {trigger_info['confidence']:.0%})[/dim]")
                            console.print(f"[dim]Actions: {', '.join(trigger_info['actions'])}[/dim]")

                        if 'context_used' in result:
                            console.print(f"[dim]Contexts used: {result['context_used']}[/dim]")

                    self.tracker.add_action("Query answered successfully")

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

    def handle_codebase_review(self, user_input: str):
        """Handle review of entire codebase using RAG."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing codebase...", total=None)

            try:
                # Use RAG query to analyze the entire codebase
                query = f"Review the entire codebase and find any errors, issues, security vulnerabilities, or code quality problems. {user_input}"

                # Use enriched query payload with full project context
                payload = self._build_enriched_query_payload(query, intent='review')
                payload['timeout'] = 60  # Longer timeout for codebase review

                response = self.session.post(
                    f"{self.base_url}/query",
                    json=payload,
                    timeout=60
                )

                progress.update(task, completed=True)

                if response.status_code == 200:
                    result = response.json()

                    console.print("\n[bold magenta]Codebase Review Results:[/bold magenta]")
                    console.print(Panel(Markdown(result['response']), border_style="magenta"))

                    self.tracker.add_action("Codebase review completed")

                elif response.status_code == 404:
                    console.print("[red]Error: The /query endpoint was not found.[/red]")
                    console.print("\n[yellow]This could mean:[/yellow]")
                    console.print("  1. The server is not running properly")
                    console.print("  2. The server failed to initialize completely")
                    console.print("  3. You're connected to the wrong server/port")
                    console.print(f"\n[cyan]Current server URL: {self.base_url}[/cyan]")
                    console.print("\n[yellow]Troubleshooting steps:[/yellow]")
                    console.print("  1. Check if server is running:")
                    console.print("     [cyan]curl {}/health[/cyan]".format(self.base_url))
                    console.print("  2. Restart the server:")
                    console.print("     [cyan]python3 src/mcp_server/standalone_server.py[/cyan]")
                    console.print("  3. Check server logs for initialization errors")
                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")
                    try:
                        error_detail = response.json()
                        if 'detail' in error_detail:
                            console.print(f"[red]Details: {error_detail['detail']}[/red]")
                    except:
                        console.print(f"[red]Response: {response.text[:200]}[/red]")

            except requests.exceptions.Timeout:
                console.print("[red]Request timed out. Codebase review may take longer for large projects.[/red]")
            except requests.exceptions.ConnectionError:
                console.print(f"[red]Connection Error: Could not connect to server at {self.base_url}[/red]")
                console.print("[yellow]The server may not be running. Try starting it with:[/yellow]")
                console.print("  [cyan]python3 src/mcp_server/standalone_server.py[/cyan]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def review_code(self, user_input: Optional[str] = None):
        """Handle code review."""
        if self.verbosity >= VerbosityLevel.NORMAL:
            console.print("\n[bold magenta]Review Code[/bold magenta]", style="bold")
            console.print("=" * 60)

        self.tracker.add_action("Reviewing code")

        # Determine what to review based on user input
        file_path = None
        if user_input:
            # Check if user wants to review entire codebase
            codebase_keywords = [r'\bcodebase\b', r'\bentire (project|code)\b', r'\ball (files|code)\b', r'\bproject\b']
            is_codebase_review = any(re.search(pattern, user_input.lower()) for pattern in codebase_keywords)

            if is_codebase_review and self.project_folder:
                # Use RAG query to review the entire codebase
                console.print(f"[cyan]Reviewing entire codebase in: {self.project_folder}[/cyan]")
                self.handle_codebase_review(user_input)
                return

            # Try to extract file path from user input
            # Look for patterns like "review auth.py", "check src/api.py for errors"
            file_patterns = [
                r'(?:review|check|analyze|inspect)\s+([^\s]+\.py)',
                r'(?:in|file)\s+([^\s]+\.py)',
                r'([^\s]+\.py)'
            ]
            for pattern in file_patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    file_path = match.group(1)
                    # If it's a relative path, make it relative to project_folder
                    if not os.path.isabs(file_path) and self.project_folder:
                        potential_path = self.project_folder / file_path
                        if potential_path.exists():
                            file_path = str(potential_path)
                    break

        # Get code input if not determined from user input
        if not file_path:
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
                    if self.verbosity >= VerbosityLevel.NORMAL:
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

                                # In quiet mode, only show critical and high
                                if self.verbosity == VerbosityLevel.QUIET and severity not in ['critical', 'high']:
                                    continue

                                max_show = 5 if self.verbosity >= VerbosityLevel.NORMAL else 3
                                for issue in by_severity[severity][:max_show]:
                                    console.print(f"  - {issue['title']}")
                                    if self.verbosity >= VerbosityLevel.NORMAL:
                                        if issue.get('line_number'):
                                            console.print(f"    Line {issue['line_number']}: {issue['description']}")
                                        else:
                                            console.print(f"    {issue['description']}")
                                        if issue.get('suggestion'):
                                            console.print(f"    [dim]Fix: {issue['suggestion']}[/dim]")

                                if len(by_severity[severity]) > max_show:
                                    console.print(f"  [dim]... and {len(by_severity[severity]) - max_show} more[/dim]")

                    else:
                        console.print("\n[bold green]No issues found![/bold green]")

                    # Show metrics in normal/verbose mode
                    if self.verbosity >= VerbosityLevel.NORMAL:
                        console.print(f"\n[bold]Metrics:[/bold]")
                        metrics = result['metrics']
                        console.print(f"  Total lines: {metrics['total_lines']}")
                        console.print(f"  Code lines: {metrics['code_lines']}")
                        console.print(f"  Comment lines: {metrics['comment_lines']}")
                        console.print(f"  Issues: {metrics['total_issues']}")

                    self.tracker.add_action("Code review completed")

                elif response.status_code == 404:
                    console.print("[red]Error: The /review endpoint was not found.[/red]")
                    console.print("\n[yellow]This could mean:[/yellow]")
                    console.print("  1. The server is not running properly")
                    console.print("  2. The server failed to initialize completely")
                    console.print("  3. You're connected to the wrong server/port")
                    console.print(f"\n[cyan]Current server URL: {self.base_url}[/cyan]")
                    console.print("\n[yellow]Troubleshooting steps:[/yellow]")
                    console.print("  1. Check if server is running:")
                    console.print("     [cyan]curl {}/health[/cyan]".format(self.base_url))
                    console.print("  2. Restart the server:")
                    console.print("     [cyan]python3 src/mcp_server/standalone_server.py[/cyan]")
                    console.print("  3. Check server logs for initialization errors")
                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")
                    try:
                        error_detail = response.json()
                        if 'detail' in error_detail:
                            console.print(f"[red]Details: {error_detail['detail']}[/red]")
                    except:
                        console.print(f"[red]Response: {response.text[:200]}[/red]")

            except requests.exceptions.ConnectionError:
                console.print(f"[red]Connection Error: Could not connect to server at {self.base_url}[/red]")
                console.print("[yellow]The server may not be running. Try starting it with:[/yellow]")
                console.print("  [cyan]python3 src/mcp_server/standalone_server.py[/cyan]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def show_session_history(self):
        """Display current session history."""
        if not self.session_history:
            console.print("[yellow]Session history not available[/yellow]")
            return

        console.print("\n[bold cyan]Current Session History[/bold cyan]")
        console.print("=" * 60)

        context = self.session_history.get_full_context_for_llm(include_summarized=True)

        if not context:
            console.print("[dim]No conversation history yet[/dim]")
            return

        console.print(Panel(context, border_style="cyan", title="Session Context"))

    def show_all_sessions(self):
        """Display all sessions (current and archived)."""
        if not self.session_history:
            console.print("[yellow]Session history not available[/yellow]")
            return

        console.print("\n[bold cyan]All Sessions[/bold cyan]")
        console.print("=" * 60)

        # Show current session
        if self.session_history.current_session:
            session = self.session_history.current_session
            console.print(f"\n[bold green]Current Session (Active)[/bold green]")
            console.print(f"  ID: {session.session_id}")
            console.print(f"  Project: {session.project_folder}")
            console.print(f"  Started: {session.start_time}")
            console.print(f"  Turns: {session.total_turns}")
            console.print(f"  Last Activity: {session.last_activity}")

        # Show archived sessions
        if self.session_history.archived_sessions:
            console.print(f"\n[bold]Archived Sessions ({len(self.session_history.archived_sessions)})[/bold]")

            for session in reversed(self.session_history.archived_sessions[-5:]):  # Show last 5
                console.print(f"\n  Session: {session.session_id}")
                console.print(f"    Project: {session.project_folder}")
                console.print(f"    Duration: {session.start_time} → {session.last_activity}")
                console.print(f"    Turns: {session.total_turns}")

                if session.session_summary:
                    console.print(f"    Summary: {session.session_summary.summary}")
                    if session.session_summary.key_topics:
                        console.print(f"    Topics: {', '.join(session.session_summary.key_topics)}")
        else:
            console.print("\n[dim]No archived sessions[/dim]")

    def show_session_stats(self):
        """Display session statistics."""
        if not self.session_history:
            console.print("[yellow]Session history not available[/yellow]")
            return

        stats = self.session_history.get_statistics()

        console.print("\n[bold cyan]Session Statistics[/bold cyan]")
        console.print("=" * 60)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Current Session Active", "Yes" if stats['current_session_active'] else "No")
        table.add_row("Current Session Turns", str(stats['current_session_turns']))
        table.add_row("Archived Sessions", str(stats['archived_sessions']))
        table.add_row("Total Archived Turns", str(stats['total_archived_turns']))
        table.add_row("Total All Turns", str(stats['total_all_turns']))
        table.add_row("Storage File", stats['storage_file'])

        console.print(table)

    def show_help(self):
        """Show help information."""
        help_text = f"""
# dt-cli Help

## Intelligent Interface

dt-cli automatically detects what you want to do based on your natural language input.
No need to select options or specify commands!

## What You Can Do

### Ask Questions
- "What does the authentication module do?"
- "Where is user validation handled?"
- "Explain how the caching system works"

### Debug Errors
- "Fix this error: ImportError: No module named 'requests'"
- "Debug the failing test in test_api.py"
- Paste any error/traceback and I'll analyze it

### Implement Code
- "Add a new endpoint for user logout"
- "Implement caching for database queries"
- "Refactor the authentication function to use OAuth"

### Review Code
- "Review the code in auth.py"
- "Check api_handler.py for security issues"
- "Analyze code quality in my latest changes"

### Plan Features
- "Plan how to add rate limiting"
- "Design a caching strategy"
- "What's the best approach to add webhooks?"

## Slash Commands

- **`/verbosity <level>`** - Set output detail level
  - `quiet`: Minimal output
  - `normal`: Standard output (default)
  - `verbose`: Detailed output with explanations

- **`/folder`** - View or change project folder
- **`/history`** - Show current session history (hierarchical memory)
- **`/sessions`** - View all sessions (current and archived)
- **`/stats`** - Show session statistics
- **`/clearsession`** - Clear ALL session history (irreversible!)
- **`/help`** - Show this help message
- **`/exit`** - Exit the program

## Settings

- **Project Folder:** `{self.project_folder}`
- **Verbosity:** `{self.verbosity.value}`
- **Server URL:** `{self.base_url}`

## Tips

- Just type naturally - the AI figures out what you need
- Use `/verbosity verbose` to see detailed intent classification
- Planning mode asks for approval before executing
- All changes are tracked and summarized

## Configuration

Config file: `llm-config.yaml`
History file: `~/.dt_cli_history`

## Documentation

See README.md and PHASE*.md files for detailed documentation.
"""
        console.print(Panel(Markdown(help_text), border_style="blue"))

    def run(self):
        """
        Run the interactive CLI with intelligent processing.
        """
        # Check server and auto-start if needed
        if not self.check_server():
            if self.auto_start_server:
                if not self.start_server():
                    console.print("[bold red]Server is not running and could not be started![/bold red]")
                    console.print("\nPlease start the server manually:")
                    console.print("  [cyan]python3 src/mcp_server/standalone_server.py[/cyan]")
                    return
            else:
                console.print("[bold red]Server is not running![/bold red]")
                console.print("\nPlease start the server first:")
                console.print("  [cyan]python3 src/mcp_server/standalone_server.py[/cyan]")
                return

        # Select project folder
        self.select_project_folder()

        # Show welcome
        self.show_welcome()

        # Main loop with intelligent processing
        while True:
            try:
                # Get user input
                if self.prompt_session:
                    user_input = self.get_input_with_history("\n>")
                else:
                    user_input = input("\n> ").strip()

                if not user_input:
                    continue

                # Check for slash commands
                if user_input.startswith('/'):
                    should_continue = self.handle_slash_command(user_input)
                    if not should_continue:
                        console.print("\n[bold]Goodbye![/bold]")
                        break
                    continue

                # Check for exit keywords
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("\n[bold]Goodbye![/bold]")
                    break

                # Process with intelligent routing
                self.process_user_input(user_input)

            except KeyboardInterrupt:
                console.print("\n\n[bold]Goodbye![/bold]")
                break
            except EOFError:
                console.print("\n\n[bold]Goodbye![/bold]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if self.verbosity == VerbosityLevel.VERBOSE:
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")

    def __del__(self):
        """Cleanup: stop server and close session."""
        # Close session history
        if self.session_history:
            try:
                self.session_history.close_session(generate_summary=True)
            except:
                pass

        # Stop server if we started it
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                pass


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="dt-cli Interactive Terminal - Intelligent AI Assistant for Codebases"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:58432",
        help="Server URL (default: http://localhost:58432)"
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't auto-start server if not running"
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        choices=["quiet", "normal", "verbose"],
        default="normal",
        help="Set verbosity level (default: normal)"
    )

    args = parser.parse_args()

    cli = DTCliInteractive(
        base_url=args.server,
        auto_start_server=not args.no_auto_start
    )
    cli.verbosity = VerbosityLevel.from_string(args.verbosity)
    cli.run()


if __name__ == "__main__":
    main()
