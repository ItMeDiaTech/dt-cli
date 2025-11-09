"""
Session History Management with Hierarchical Summarization.

Implements best practices from 2024-2025 research on LLM conversation memory:
- Hierarchical summarization for long-term memory
- Sliding window with recent messages in full detail
- Automatic compression when thresholds exceeded
- Importance scoring for conversation weighting
- Persistence across CLI sessions

Based on research:
- Recursively Summarizing Enables Long-Term Dialogue Memory (arXiv:2308.15022)
- LLM Chat History Summarization Guide 2025 (mem0.ai)
- Dynamic Tree Memory Representation for LLMs (arXiv:2410.14052)
"""

import json
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import logging
import hashlib
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in conversation."""
    turn_number: int
    user_input: str
    intent: str
    response_summary: str
    timestamp: str
    importance_score: float = 1.0  # 0.0-1.0, used for selective retention

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SessionSummary:
    """Summary of a conversation session."""
    session_id: str
    project_folder: str
    start_time: str
    end_time: str
    total_turns: int
    summary: str  # LLM-generated or rule-based summary
    key_topics: List[str] = field(default_factory=list)
    important_turns: List[int] = field(default_factory=list)  # Turn numbers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionSummary':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Session:
    """Complete session with hierarchical memory structure."""
    session_id: str
    project_folder: str
    start_time: str
    last_activity: str

    # Hierarchical memory levels
    recent_turns: List[ConversationTurn] = field(default_factory=list)  # Full detail
    summarized_context: Optional[str] = None  # Compressed older conversations
    session_summary: Optional[SessionSummary] = None  # Final summary when closed

    # Metadata
    total_turns: int = 0
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert ConversationTurn objects
        data['recent_turns'] = [turn.to_dict() if hasattr(turn, 'to_dict') else turn
                                for turn in self.recent_turns]
        if self.session_summary:
            data['session_summary'] = (self.session_summary.to_dict()
                                      if hasattr(self.session_summary, 'to_dict')
                                      else self.session_summary)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create from dictionary."""
        # Convert dicts back to ConversationTurn objects
        if 'recent_turns' in data:
            data['recent_turns'] = [
                ConversationTurn.from_dict(turn) if isinstance(turn, dict) else turn
                for turn in data['recent_turns']
            ]
        if 'session_summary' in data and data['session_summary']:
            if isinstance(data['session_summary'], dict):
                data['session_summary'] = SessionSummary.from_dict(data['session_summary'])
        return cls(**data)


class SessionHistoryManager:
    """
    Manages conversation history with hierarchical memory structure.

    Memory Hierarchy (based on 2024-2025 best practices):

    Level 1: Working Memory (Last 20 turns)
    - Full detail, all information preserved
    - Used for immediate context

    Level 2: Summarized Context (Turns 21-100)
    - Compressed using rule-based or LLM summarization
    - Key information extracted and preserved

    Level 3: Session Summary (When session closes)
    - High-level overview of entire session
    - Key topics and important moments

    Level 4: Archived Sessions (Old sessions)
    - Stored with summaries only
    - Retrievable for reference
    """

    # Configuration constants (based on research best practices)
    WORKING_MEMORY_SIZE = 20  # Recent turns kept in full detail
    SUMMARIZATION_THRESHOLD = 50  # Trigger compression at this many turns
    MAX_RECENT_SESSIONS = 10  # Keep this many recent sessions
    SESSION_TIMEOUT_HOURS = 24  # Auto-close sessions after inactivity
    IMPORTANCE_THRESHOLD = 0.7  # Keep high-importance turns even when old

    def __init__(self, storage_file: Optional[Path] = None, llm_provider: Optional[Any] = None):
        """
        Initialize session history manager.

        Args:
            storage_file: Path to storage file (default: ~/.dt_cli_sessions.json)
            llm_provider: Optional LLM for intelligent summarization
        """
        self.storage_file = storage_file or Path.home() / '.dt_cli_sessions.json'
        self.llm_provider = llm_provider

        self.current_session: Optional[Session] = None
        self.archived_sessions: List[Session] = []

        # Thread safety
        self._lock = threading.RLock()

        # Load existing sessions
        self._load_sessions()

        logger.info(f"SessionHistoryManager initialized (storage: {self.storage_file})")

    def start_session(self, project_folder: str) -> str:
        """
        Start a new session.

        Args:
            project_folder: Project folder path

        Returns:
            Session ID
        """
        with self._lock:
            # Close any existing active session
            if self.current_session and self.current_session.is_active:
                self.close_session()

            # Create new session
            session_id = self._generate_session_id(project_folder)
            now = datetime.now().isoformat()

            self.current_session = Session(
                session_id=session_id,
                project_folder=project_folder,
                start_time=now,
                last_activity=now,
                is_active=True
            )

            logger.info(f"Started new session: {session_id}")
            return session_id

    def add_turn(self, user_input: str, intent: str, response_summary: str,
                 importance_score: float = 1.0):
        """
        Add a conversation turn to current session.

        Implements sliding window with automatic summarization.

        Args:
            user_input: User's input
            intent: Detected intent
            response_summary: Summary of response
            importance_score: Importance (0.0-1.0), affects retention
        """
        with self._lock:
            if not self.current_session:
                raise ValueError("No active session. Call start_session() first.")

            # Create turn
            turn = ConversationTurn(
                turn_number=self.current_session.total_turns + 1,
                user_input=user_input,
                intent=intent,
                response_summary=response_summary,
                timestamp=datetime.now().isoformat(),
                importance_score=importance_score
            )

            # Add to recent turns
            self.current_session.recent_turns.append(turn)
            self.current_session.total_turns += 1
            self.current_session.last_activity = datetime.now().isoformat()

            # Apply sliding window + summarization when threshold exceeded
            if len(self.current_session.recent_turns) > self.SUMMARIZATION_THRESHOLD:
                self._apply_hierarchical_compression()

            # Persist
            self._save_sessions()

            logger.debug(f"Added turn {turn.turn_number} to session {self.current_session.session_id}")

    def _apply_hierarchical_compression(self):
        """
        Apply hierarchical compression to maintain memory efficiency.

        Implements sliding window: keep recent turns in full, compress older ones.
        Based on: "Recursively Summarizing Enables Long-Term Dialogue Memory"
        """
        if not self.current_session:
            return

        recent_turns = self.current_session.recent_turns

        # Separate into recent (keep full) and old (compress)
        old_turns = recent_turns[:-self.WORKING_MEMORY_SIZE]
        recent = recent_turns[-self.WORKING_MEMORY_SIZE:]

        # Filter old turns: keep high-importance ones in full
        high_importance_turns = [t for t in old_turns
                                if t.importance_score >= self.IMPORTANCE_THRESHOLD]

        # Compress low-importance old turns
        low_importance_turns = [t for t in old_turns
                               if t.importance_score < self.IMPORTANCE_THRESHOLD]

        if low_importance_turns:
            # Create summary of compressed turns
            summary = self._summarize_turns(low_importance_turns)

            # Update summarized context
            if self.current_session.summarized_context:
                # Recursively summarize: combine old summary with new
                self.current_session.summarized_context = (
                    f"{self.current_session.summarized_context}\n\n"
                    f"Additional context:\n{summary}"
                )
            else:
                self.current_session.summarized_context = summary

        # Keep: recent turns + high-importance older turns
        self.current_session.recent_turns = high_importance_turns + recent

        logger.info(f"Compressed {len(low_importance_turns)} turns, "
                   f"kept {len(high_importance_turns)} important + {len(recent)} recent")

    def _summarize_turns(self, turns: List[ConversationTurn]) -> str:
        """
        Summarize a list of conversation turns.

        Uses LLM if available, otherwise rule-based summarization.

        Args:
            turns: Turns to summarize

        Returns:
            Summary text
        """
        if not turns:
            return ""

        # LLM-based summarization (if provider available)
        if self.llm_provider:
            return self._llm_summarize(turns)

        # Fallback: Rule-based summarization
        return self._rule_based_summarize(turns)

    def _llm_summarize(self, turns: List[ConversationTurn]) -> str:
        """LLM-based summarization (to be implemented with provider)."""
        # Placeholder for LLM summarization
        # Would use self.llm_provider to generate intelligent summary
        return self._rule_based_summarize(turns)

    def _rule_based_summarize(self, turns: List[ConversationTurn]) -> str:
        """
        Rule-based summarization as fallback.

        Extracts key information using simple heuristics.
        """
        if not turns:
            return ""

        # Group by intent
        by_intent: Dict[str, List[ConversationTurn]] = {}
        for turn in turns:
            if turn.intent not in by_intent:
                by_intent[turn.intent] = []
            by_intent[turn.intent].append(turn)

        # Build summary
        summary_parts = [
            f"Conversation context ({turns[0].turn_number}-{turns[-1].turn_number}):"
        ]

        for intent, intent_turns in by_intent.items():
            count = len(intent_turns)
            summary_parts.append(
                f"- {count} {intent} interaction{'s' if count > 1 else ''}"
            )
            # Include first of each intent type as example
            if intent_turns:
                example = intent_turns[0]
                summary_parts.append(f"  Example: \"{example.user_input[:100]}...\"")

        return "\n".join(summary_parts)

    def close_session(self, generate_summary: bool = True):
        """
        Close current session and archive it.

        Args:
            generate_summary: Whether to generate session summary
        """
        with self._lock:
            if not self.current_session:
                return

            self.current_session.is_active = False
            self.current_session.last_activity = datetime.now().isoformat()

            # Generate session summary
            if generate_summary and self.current_session.total_turns > 0:
                self.current_session.session_summary = self._generate_session_summary()

            # Archive session
            self.archived_sessions.append(self.current_session)

            # Limit archived sessions
            if len(self.archived_sessions) > self.MAX_RECENT_SESSIONS:
                # Keep most recent sessions
                self.archived_sessions = self.archived_sessions[-self.MAX_RECENT_SESSIONS:]

            session_id = self.current_session.session_id
            self.current_session = None

            # Persist
            self._save_sessions()

            logger.info(f"Closed session: {session_id}")

    def _generate_session_summary(self) -> SessionSummary:
        """Generate summary of completed session."""
        if not self.current_session:
            raise ValueError("No active session")

        # Extract key topics (simple: most common intents)
        intent_counts: Dict[str, int] = {}
        for turn in self.current_session.recent_turns:
            intent_counts[turn.intent] = intent_counts.get(turn.intent, 0) + 1

        key_topics = sorted(intent_counts.keys(),
                          key=lambda x: intent_counts[x],
                          reverse=True)[:3]

        # Find important turns
        important_turns = [
            turn.turn_number
            for turn in self.current_session.recent_turns
            if turn.importance_score >= self.IMPORTANCE_THRESHOLD
        ]

        # Generate summary text
        summary_text = self._generate_summary_text()

        return SessionSummary(
            session_id=self.current_session.session_id,
            project_folder=self.current_session.project_folder,
            start_time=self.current_session.start_time,
            end_time=datetime.now().isoformat(),
            total_turns=self.current_session.total_turns,
            summary=summary_text,
            key_topics=key_topics,
            important_turns=important_turns
        )

    def _generate_summary_text(self) -> str:
        """Generate human-readable summary of session."""
        if not self.current_session:
            return ""

        turns = self.current_session.total_turns
        recent = self.current_session.recent_turns

        if not recent:
            return f"Session with {turns} turn{'s' if turns != 1 else ''}"

        # Simple summary
        intents = [turn.intent for turn in recent]
        unique_intents = list(set(intents))

        return (
            f"Session covered: {', '.join(unique_intents)}. "
            f"Total {turns} interactions."
        )

    def get_relevant_history(self, query: str, max_turns: int = 5) -> List[ConversationTurn]:
        """
        Get relevant history for a query (semantic retrieval).

        Simple keyword matching for now, could be enhanced with embeddings.

        Args:
            query: Current query
            max_turns: Maximum turns to return

        Returns:
            Relevant conversation turns
        """
        with self._lock:
            if not self.current_session:
                return []

            # Simple keyword matching
            query_lower = query.lower()
            words = set(query_lower.split())

            # Score turns by keyword overlap
            scored_turns = []
            for turn in self.current_session.recent_turns:
                turn_words = set(turn.user_input.lower().split())
                overlap = len(words & turn_words)
                if overlap > 0:
                    scored_turns.append((overlap, turn))

            # Return top matches
            scored_turns.sort(key=lambda x: x[0], reverse=True)
            return [turn for _, turn in scored_turns[:max_turns]]

    def get_session_context(self) -> Dict[str, Any]:
        """
        Get current session context for enriching queries.

        Returns:
            Context dictionary
        """
        with self._lock:
            if not self.current_session:
                return {}

            return {
                'session_id': self.current_session.session_id,
                'project_folder': self.current_session.project_folder,
                'turn_count': self.current_session.total_turns,
                'recent_turns': len(self.current_session.recent_turns),
                'has_summarized_context': bool(self.current_session.summarized_context),
                'last_activity': self.current_session.last_activity
            }

    def get_full_context_for_llm(self, include_summarized: bool = True) -> str:
        """
        Get full context formatted for LLM consumption.

        Returns hierarchical context: summarized + recent turns.

        Args:
            include_summarized: Include compressed older context

        Returns:
            Formatted context string
        """
        with self._lock:
            if not self.current_session:
                return ""

            parts = []

            # Add summarized older context
            if include_summarized and self.current_session.summarized_context:
                parts.append("=== Previous Context (Summarized) ===")
                parts.append(self.current_session.summarized_context)
                parts.append("")

            # Add recent turns in full detail
            if self.current_session.recent_turns:
                parts.append("=== Recent Conversation ===")
                for turn in self.current_session.recent_turns:
                    parts.append(f"Turn {turn.turn_number} ({turn.intent}):")
                    parts.append(f"  User: {turn.user_input}")
                    parts.append(f"  System: {turn.response_summary}")
                    parts.append("")

            return "\n".join(parts)

    def _generate_session_id(self, project_folder: str) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        unique_str = f"{project_folder}_{timestamp}_{uuid.uuid4().hex[:8]}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    def _load_sessions(self):
        """Load sessions from storage."""
        if not self.storage_file.exists():
            logger.info("No existing session history found")
            return

        try:
            with open(self.storage_file, 'r') as f:
                data = json.load(f)

            # Load current session
            if 'current_session' in data and data['current_session']:
                self.current_session = Session.from_dict(data['current_session'])
                # Check if session timed out
                self._check_session_timeout()

            # Load archived sessions
            if 'archived_sessions' in data:
                self.archived_sessions = [
                    Session.from_dict(s) for s in data['archived_sessions']
                ]

            logger.info(f"Loaded {len(self.archived_sessions)} archived sessions")

        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            # Reset to clean state
            self.current_session = None
            self.archived_sessions = []

    def _save_sessions(self):
        """Save sessions to storage."""
        try:
            data = {
                'current_session': self.current_session.to_dict() if self.current_session else None,
                'archived_sessions': [s.to_dict() for s in self.archived_sessions],
                'last_updated': datetime.now().isoformat()
            }

            # Atomic write
            temp_file = self.storage_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.storage_file)

            logger.debug(f"Saved session history to {self.storage_file}")

        except Exception as e:
            logger.error(f"Error saving sessions: {e}")

    def _check_session_timeout(self):
        """Check if current session has timed out and close if needed."""
        if not self.current_session or not self.current_session.is_active:
            return

        last_activity = datetime.fromisoformat(self.current_session.last_activity)
        timeout = timedelta(hours=self.SESSION_TIMEOUT_HOURS)

        if datetime.now() - last_activity > timeout:
            logger.info(f"Session {self.current_session.session_id} timed out")
            self.close_session()

    def get_statistics(self) -> Dict[str, Any]:
        """Get session history statistics."""
        with self._lock:
            total_archived = len(self.archived_sessions)
            total_turns_archived = sum(s.total_turns for s in self.archived_sessions)

            current_turns = self.current_session.total_turns if self.current_session else 0

            return {
                'current_session_active': bool(self.current_session and self.current_session.is_active),
                'current_session_turns': current_turns,
                'archived_sessions': total_archived,
                'total_archived_turns': total_turns_archived,
                'total_all_turns': current_turns + total_turns_archived,
                'storage_file': str(self.storage_file)
            }

    def clear_all_history(self):
        """Clear all session history (irreversible)."""
        with self._lock:
            self.current_session = None
            self.archived_sessions = []

            if self.storage_file.exists():
                self.storage_file.unlink()

            logger.warning("All session history cleared")
