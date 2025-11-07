"""
Team workspace and collaboration features.

Features:
- Team workspaces for shared knowledge
- Shared searches across team members
- Shared code snippets
- Collaboration permissions and roles
- Activity tracking and history
- Team analytics
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User role in workspace."""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(Enum):
    """Workspace permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MANAGE_MEMBERS = "manage_members"
    MANAGE_SETTINGS = "manage_settings"


# Role permissions mapping
ROLE_PERMISSIONS = {
    UserRole.OWNER: [Permission.READ, Permission.WRITE, Permission.DELETE,
                     Permission.MANAGE_MEMBERS, Permission.MANAGE_SETTINGS],
    UserRole.ADMIN: [Permission.READ, Permission.WRITE, Permission.DELETE,
                     Permission.MANAGE_MEMBERS],
    UserRole.MEMBER: [Permission.READ, Permission.WRITE],
    UserRole.VIEWER: [Permission.READ]
}


@dataclass
class WorkspaceMember:
    """Workspace member information."""

    user_id: str
    username: str
    role: UserRole
    joined_at: str
    last_active: Optional[str] = None
    activity_count: int = 0


@dataclass
class SharedSearch:
    """Shared search query."""

    id: str
    query: str
    created_by: str
    created_at: str
    shared_with: List[str] = field(default_factory=list)  # User IDs
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    use_count: int = 0
    last_used: Optional[str] = None


@dataclass
class SharedSnippet:
    """Shared code snippet."""

    id: str
    title: str
    code: str
    language: str
    created_by: str
    created_at: str
    shared_with: List[str] = field(default_factory=list)  # User IDs
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    source_file: Optional[str] = None
    use_count: int = 0


@dataclass
class ActivityEntry:
    """Activity log entry."""

    id: str
    workspace_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workspace:
    """Team workspace."""

    id: str
    name: str
    description: str
    owner_id: str
    created_at: str
    members: Dict[str, WorkspaceMember] = field(default_factory=dict)
    shared_searches: Dict[str, SharedSearch] = field(default_factory=dict)
    shared_snippets: Dict[str, SharedSnippet] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class WorkspaceManager:
    """
    Manages team workspaces and collaboration.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize workspace manager.

        Args:
            storage_path: Path to workspace storage
        """
        self.storage_path = storage_path or Path.home() / '.rag_workspaces'
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.workspaces: Dict[str, Workspace] = {}
        self.activity_log: List[ActivityEntry] = []

        # Load workspaces and activity log
        self._load_workspaces()
        self._load_activity_log()

    def create_workspace(
        self,
        name: str,
        owner_id: str,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> Workspace:
        """
        Create new workspace.

        Args:
            name: Workspace name
            owner_id: Owner user ID
            description: Workspace description
            tags: Workspace tags

        Returns:
            Created workspace
        """
        workspace_id = self._generate_id(name)

        # Create owner member
        owner = WorkspaceMember(
            user_id=owner_id,
            username=owner_id,  # Would be replaced with actual username
            role=UserRole.OWNER,
            joined_at=datetime.now().isoformat()
        )

        workspace = Workspace(
            id=workspace_id,
            name=name,
            description=description,
            owner_id=owner_id,
            created_at=datetime.now().isoformat(),
            members={owner_id: owner},
            tags=tags or []
        )

        self.workspaces[workspace_id] = workspace
        self._save_workspace(workspace)

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=owner_id,
            action="create",
            resource_type="workspace",
            resource_id=workspace_id
        )

        logger.info(f"Created workspace: {name} ({workspace_id})")
        return workspace

    def add_member(
        self,
        workspace_id: str,
        user_id: str,
        username: str,
        role: UserRole,
        requesting_user_id: str
    ) -> bool:
        """
        Add member to workspace.

        Args:
            workspace_id: Workspace ID
            user_id: User ID to add
            username: Username
            role: User role
            requesting_user_id: ID of user making request

        Returns:
            True if added
        """
        workspace = self.workspaces.get(workspace_id)

        if not workspace:
            logger.error(f"Workspace not found: {workspace_id}")
            return False

        # Check permissions
        if not self._has_permission(workspace, requesting_user_id, Permission.MANAGE_MEMBERS):
            logger.error(f"User {requesting_user_id} lacks permission to add members")
            return False

        # Add member
        member = WorkspaceMember(
            user_id=user_id,
            username=username,
            role=role,
            joined_at=datetime.now().isoformat()
        )

        workspace.members[user_id] = member
        self._save_workspace(workspace)

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=requesting_user_id,
            action="add_member",
            resource_type="member",
            resource_id=user_id,
            details={'role': role.value}
        )

        logger.info(f"Added member {username} to workspace {workspace_id}")
        return True

    def remove_member(
        self,
        workspace_id: str,
        user_id: str,
        requesting_user_id: str
    ) -> bool:
        """
        Remove member from workspace.

        Args:
            workspace_id: Workspace ID
            user_id: User ID to remove
            requesting_user_id: ID of user making request

        Returns:
            True if removed
        """
        workspace = self.workspaces.get(workspace_id)

        if not workspace:
            return False

        # Check permissions
        if not self._has_permission(workspace, requesting_user_id, Permission.MANAGE_MEMBERS):
            return False

        # Can't remove owner
        if user_id == workspace.owner_id:
            logger.error("Cannot remove workspace owner")
            return False

        # Remove member
        if user_id in workspace.members:
            del workspace.members[user_id]
            self._save_workspace(workspace)

            # Log activity
            self._log_activity(
                workspace_id=workspace_id,
                user_id=requesting_user_id,
                action="remove_member",
                resource_type="member",
                resource_id=user_id
            )

            return True

        return False

    def share_search(
        self,
        workspace_id: str,
        query: str,
        user_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        share_with: Optional[List[str]] = None
    ) -> Optional[SharedSearch]:
        """
        Share search query with workspace.

        Args:
            workspace_id: Workspace ID
            query: Search query
            user_id: User sharing
            description: Search description
            tags: Search tags
            share_with: Specific users to share with (None = all members)

        Returns:
            Shared search or None
        """
        workspace = self.workspaces.get(workspace_id)

        if not workspace:
            return None

        # Check permissions
        if not self._has_permission(workspace, user_id, Permission.WRITE):
            logger.error(f"User {user_id} lacks permission to share searches")
            return None

        search_id = self._generate_id(f"{query}_{user_id}")

        shared_search = SharedSearch(
            id=search_id,
            query=query,
            created_by=user_id,
            created_at=datetime.now().isoformat(),
            description=description,
            tags=tags or [],
            shared_with=share_with or list(workspace.members.keys())
        )

        workspace.shared_searches[search_id] = shared_search
        self._save_workspace(workspace)

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=user_id,
            action="share_search",
            resource_type="search",
            resource_id=search_id
        )

        logger.info(f"Shared search in workspace {workspace_id}")
        return shared_search

    def share_snippet(
        self,
        workspace_id: str,
        title: str,
        code: str,
        language: str,
        user_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_file: Optional[str] = None,
        share_with: Optional[List[str]] = None
    ) -> Optional[SharedSnippet]:
        """
        Share code snippet with workspace.

        Args:
            workspace_id: Workspace ID
            title: Snippet title
            code: Code content
            language: Programming language
            user_id: User sharing
            description: Snippet description
            tags: Snippet tags
            source_file: Source file path
            share_with: Specific users to share with

        Returns:
            Shared snippet or None
        """
        workspace = self.workspaces.get(workspace_id)

        if not workspace:
            return None

        # Check permissions
        if not self._has_permission(workspace, user_id, Permission.WRITE):
            return None

        snippet_id = self._generate_id(f"{title}_{user_id}")

        shared_snippet = SharedSnippet(
            id=snippet_id,
            title=title,
            code=code,
            language=language,
            created_by=user_id,
            created_at=datetime.now().isoformat(),
            description=description,
            tags=tags or [],
            source_file=source_file,
            shared_with=share_with or list(workspace.members.keys())
        )

        workspace.shared_snippets[snippet_id] = shared_snippet
        self._save_workspace(workspace)

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=user_id,
            action="share_snippet",
            resource_type="snippet",
            resource_id=snippet_id
        )

        logger.info(f"Shared snippet in workspace {workspace_id}")
        return shared_snippet

    def get_shared_searches(
        self,
        workspace_id: str,
        user_id: str,
        tags: Optional[List[str]] = None
    ) -> List[SharedSearch]:
        """
        Get shared searches accessible to user.

        Args:
            workspace_id: Workspace ID
            user_id: User ID
            tags: Filter by tags

        Returns:
            List of shared searches
        """
        workspace = self.workspaces.get(workspace_id)

        if not workspace:
            return []

        # Check permissions
        if not self._has_permission(workspace, user_id, Permission.READ):
            return []

        searches = []

        for search in workspace.shared_searches.values():
            # SECURITY FIX: Check if shared with this user
            # Empty shared_with means all workspace members, not "nobody"
            is_shared = False
            if search.shared_with:
                # Explicit list: check if user is in it
                is_shared = user_id in search.shared_with
            else:
                # Empty list means shared with all workspace members
                is_shared = user_id in workspace.members

            if not is_shared:
                continue

            # Filter by tags
            if tags and not any(tag in search.tags for tag in tags):
                continue

            searches.append(search)

        return searches

    def get_shared_snippets(
        self,
        workspace_id: str,
        user_id: str,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> List[SharedSnippet]:
        """
        Get shared snippets accessible to user.

        Args:
            workspace_id: Workspace ID
            user_id: User ID
            tags: Filter by tags
            language: Filter by language

        Returns:
            List of shared snippets
        """
        workspace = self.workspaces.get(workspace_id)

        if not workspace:
            return []

        # Check permissions
        if not self._has_permission(workspace, user_id, Permission.READ):
            return []

        snippets = []

        for snippet in workspace.shared_snippets.values():
            # SECURITY FIX: Check if shared with this user
            # Empty shared_with means all workspace members, not "nobody"
            is_shared = False
            if snippet.shared_with:
                # Explicit list: check if user is in it
                is_shared = user_id in snippet.shared_with
            else:
                # Empty list means shared with all workspace members
                is_shared = user_id in workspace.members

            if not is_shared:
                continue

            # Filter by tags
            if tags and not any(tag in snippet.tags for tag in tags):
                continue

            # Filter by language
            if language and snippet.language != language:
                continue

            snippets.append(snippet)

        return snippets

    def get_workspace_activity(
        self,
        workspace_id: str,
        user_id: str,
        limit: int = 50,
        action_filter: Optional[str] = None
    ) -> List[ActivityEntry]:
        """
        Get workspace activity log.

        Args:
            workspace_id: Workspace ID
            user_id: Requesting user ID
            limit: Maximum entries to return
            action_filter: Filter by action type

        Returns:
            List of activity entries
        """
        workspace = self.workspaces.get(workspace_id)

        if not workspace:
            return []

        # Check permissions
        if not self._has_permission(workspace, user_id, Permission.READ):
            return []

        # Filter activity
        activities = [
            entry for entry in self.activity_log
            if entry.workspace_id == workspace_id
        ]

        # Filter by action
        if action_filter:
            activities = [a for a in activities if a.action == action_filter]

        # Sort by timestamp (newest first)
        activities.sort(key=lambda x: x.timestamp, reverse=True)

        return activities[:limit]

    def get_workspace_analytics(
        self,
        workspace_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get workspace analytics.

        Args:
            workspace_id: Workspace ID
            user_id: Requesting user ID

        Returns:
            Analytics data
        """
        workspace = self.workspaces.get(workspace_id)

        if not workspace:
            return {}

        # Check permissions
        if not self._has_permission(workspace, user_id, Permission.READ):
            return {}

        # Calculate analytics
        total_searches = len(workspace.shared_searches)
        total_snippets = len(workspace.shared_snippets)
        total_members = len(workspace.members)

        # Active members (activity in last 7 days)
        now = datetime.now()
        active_members = sum(
            1 for member in workspace.members.values()
            if member.last_active and
            (now - datetime.fromisoformat(member.last_active)).days <= 7
        )

        # Most active users
        member_activity = {}
        for entry in self.activity_log:
            if entry.workspace_id == workspace_id:
                member_activity[entry.user_id] = member_activity.get(entry.user_id, 0) + 1

        top_contributors = sorted(
            member_activity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Popular tags
        tag_counts = {}
        for search in workspace.shared_searches.values():
            for tag in search.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        for snippet in workspace.shared_snippets.values():
            for tag in snippet.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'workspace_id': workspace_id,
            'workspace_name': workspace.name,
            'total_members': total_members,
            'active_members': active_members,
            'total_shared_searches': total_searches,
            'total_shared_snippets': total_snippets,
            'top_contributors': [
                {'user_id': user_id, 'activity_count': count}
                for user_id, count in top_contributors
            ],
            'popular_tags': [
                {'tag': tag, 'count': count}
                for tag, count in popular_tags
            ],
            'created_at': workspace.created_at
        }

    def list_workspaces(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List workspaces accessible to user.

        Args:
            user_id: User ID

        Returns:
            List of workspace information
        """
        workspaces = []

        for workspace in self.workspaces.values():
            # Check if user is a member
            if user_id in workspace.members:
                member = workspace.members[user_id]

                workspaces.append({
                    'id': workspace.id,
                    'name': workspace.name,
                    'description': workspace.description,
                    'role': member.role.value,
                    'member_count': len(workspace.members),
                    'shared_search_count': len(workspace.shared_searches),
                    'shared_snippet_count': len(workspace.shared_snippets),
                    'created_at': workspace.created_at
                })

        return workspaces

    def _has_permission(
        self,
        workspace: Workspace,
        user_id: str,
        permission: Permission
    ) -> bool:
        """
        Check if user has permission.

        Args:
            workspace: Workspace
            user_id: User ID
            permission: Required permission

        Returns:
            True if has permission
        """
        member = workspace.members.get(user_id)

        if not member:
            return False

        role_perms = ROLE_PERMISSIONS.get(member.role, [])
        return permission in role_perms

    def _log_activity(
        self,
        workspace_id: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log activity entry and persist to disk."""
        entry = ActivityEntry(
            id=self._generate_id(f"{workspace_id}_{user_id}_{action}"),
            workspace_id=workspace_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            timestamp=datetime.now().isoformat(),
            details=details or {}
        )

        self.activity_log.append(entry)

        # CRITICAL FIX: Persist activity to disk immediately
        self._save_activity_entry(entry)

        # Update member last active
        workspace = self.workspaces.get(workspace_id)
        if workspace and user_id in workspace.members:
            workspace.members[user_id].last_active = entry.timestamp
            workspace.members[user_id].activity_count += 1
            # Save workspace with updated member info
            self._save_workspace(workspace)

    def _generate_id(self, base: str) -> str:
        """Generate unique ID."""
        import hashlib
        return hashlib.md5(f"{base}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    def _save_workspace(self, workspace: Workspace):
        """Save workspace to storage."""
        workspace_file = self.storage_path / f"{workspace.id}.json"

        data = asdict(workspace)
        # Convert enums to values
        for member_id, member in data['members'].items():
            member['role'] = member['role'].value

        workspace_file.write_text(json.dumps(data, indent=2))

    def _save_activity_entry(self, entry: ActivityEntry):
        """
        Save single activity entry to JSONL file.

        Uses append-only JSONL format for efficiency and crash safety.
        """
        try:
            activity_file = self.storage_path / 'activity.jsonl'

            # Append entry as single JSON line
            with open(activity_file, 'a') as f:
                json.dump(asdict(entry), f)
                f.write('\n')
                f.flush()  # Ensure written to disk

        except Exception as e:
            logger.error(f"Failed to save activity entry: {e}")

    def _load_activity_log(self):
        """Load activity log from JSONL file."""
        try:
            activity_file = self.storage_path / 'activity.jsonl'

            if not activity_file.exists():
                return

            with open(activity_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        entry = ActivityEntry(**data)
                        self.activity_log.append(entry)

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON at line {line_num}: {e}")
                    except Exception as e:
                        logger.error(f"Failed to parse activity entry at line {line_num}: {e}")

            logger.info(f"Loaded {len(self.activity_log)} activity entries")

        except Exception as e:
            logger.error(f"Failed to load activity log: {e}")

    def _load_workspaces(self):
        """Load workspaces from storage."""
        for workspace_file in self.storage_path.glob('*.json'):
            # Skip activity log file
            if workspace_file.name == 'activity.jsonl':
                continue

            try:
                data = json.loads(workspace_file.read_text())

                # Convert role strings back to enums
                members = {}
                for member_id, member_data in data.get('members', {}).items():
                    member_data['role'] = UserRole(member_data['role'])
                    members[member_id] = WorkspaceMember(**member_data)

                # Convert shared searches
                shared_searches = {
                    search_id: SharedSearch(**search_data)
                    for search_id, search_data in data.get('shared_searches', {}).items()
                }

                # Convert shared snippets
                shared_snippets = {
                    snippet_id: SharedSnippet(**snippet_data)
                    for snippet_id, snippet_data in data.get('shared_snippets', {}).items()
                }

                workspace = Workspace(
                    id=data['id'],
                    name=data['name'],
                    description=data['description'],
                    owner_id=data['owner_id'],
                    created_at=data['created_at'],
                    members=members,
                    shared_searches=shared_searches,
                    shared_snippets=shared_snippets,
                    settings=data.get('settings', {}),
                    tags=data.get('tags', [])
                )

                self.workspaces[workspace.id] = workspace

            except Exception as e:
                logger.error(f"Failed to load workspace from {workspace_file}: {e}")


# Global instance
workspace_manager = WorkspaceManager()
