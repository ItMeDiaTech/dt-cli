"""
Query templates for common code search patterns.

Provides pre-built query templates for frequent use cases:
- "How does X work?"
- "Find all uses of X"
- "Dependencies of X"
- "Examples of X"
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import re
import logging
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QueryTemplate:
    """Represents a query template."""

    id: str
    name: str
    description: str
    pattern: str
    variables: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    n_results: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryTemplate':
        """Create from dictionary."""
        return cls(**data)

    def format(self, **kwargs) -> str:
        """
        Format template with variables.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted query
        """
        query = self.pattern

        for var in self.variables:
            value = kwargs.get(var, '')
            query = query.replace(f'{{{var}}}', value)

        return query

    def validate_variables(self, **kwargs) -> bool:
        """
        Validate that all required variables are provided.

        Args:
            **kwargs: Template variables

        Returns:
            True if valid
        """
        for var in self.variables:
            if var not in kwargs:
                return False

        return True


# Pre-defined templates
BUILTIN_TEMPLATES = [
    QueryTemplate(
        id='how_does_it_work',
        name='How does it work?',
        description='Understand how a feature or component works',
        pattern='how does {feature} work? explain {feature} implementation',
        variables=['feature'],
        examples=[
            'how does authentication work?',
            'how does caching work?'
        ],
        tags=['understanding', 'explanation']
    ),

    QueryTemplate(
        id='find_uses',
        name='Find all uses',
        description='Find all places where something is used',
        pattern='find all uses of {name} where {name} is called or imported',
        variables=['name'],
        examples=[
            'find all uses of UserManager',
            'find all uses of authenticate'
        ],
        tags=['usage', 'references']
    ),

    QueryTemplate(
        id='find_dependencies',
        name='Find dependencies',
        description='Find dependencies of a component',
        pattern='what does {component} depend on? imports used by {component}',
        variables=['component'],
        examples=[
            'what does UserService depend on?',
            'imports used by AuthController'
        ],
        tags=['dependencies', 'imports']
    ),

    QueryTemplate(
        id='find_examples',
        name='Find examples',
        description='Find usage examples',
        pattern='show examples of {concept} how to use {concept}',
        variables=['concept'],
        examples=[
            'show examples of database queries',
            'how to use JWT tokens'
        ],
        tags=['examples', 'how-to']
    ),

    QueryTemplate(
        id='error_handling',
        name='Error handling',
        description='Find error handling for a feature',
        pattern='error handling in {feature} exception handling {feature}',
        variables=['feature'],
        examples=[
            'error handling in authentication',
            'exception handling in file upload'
        ],
        tags=['errors', 'exceptions']
    ),

    QueryTemplate(
        id='find_tests',
        name='Find tests',
        description='Find tests for a component',
        pattern='tests for {component} test cases {component}',
        variables=['component'],
        examples=[
            'tests for UserManager',
            'test cases for login function'
        ],
        tags=['testing', 'test-cases']
    ),

    QueryTemplate(
        id='api_endpoints',
        name='API endpoints',
        description='Find API endpoints related to a feature',
        pattern='API endpoints for {feature} routes {feature}',
        variables=['feature'],
        examples=[
            'API endpoints for users',
            'routes for authentication'
        ],
        tags=['api', 'endpoints', 'routes']
    ),

    QueryTemplate(
        id='config_settings',
        name='Configuration settings',
        description='Find configuration for a feature',
        pattern='configuration for {feature} settings {feature}',
        variables=['feature'],
        examples=[
            'configuration for database',
            'settings for email'
        ],
        tags=['configuration', 'settings']
    ),

    QueryTemplate(
        id='security_checks',
        name='Security checks',
        description='Find security-related code',
        pattern='security checks {aspect} authentication authorization {aspect}',
        variables=['aspect'],
        examples=[
            'security checks for user input',
            'authentication for API'
        ],
        tags=['security', 'auth']
    ),

    QueryTemplate(
        id='performance_optimizations',
        name='Performance optimizations',
        description='Find performance-related code',
        pattern='performance optimization {area} caching {area}',
        variables=['area'],
        examples=[
            'performance optimization for queries',
            'caching for API responses'
        ],
        tags=['performance', 'optimization']
    ),

    QueryTemplate(
        id='data_models',
        name='Data models',
        description='Find data models and schemas',
        pattern='data model for {entity} schema {entity}',
        variables=['entity'],
        examples=[
            'data model for User',
            'schema for Product'
        ],
        tags=['models', 'schema', 'database']
    ),

    QueryTemplate(
        id='compare_implementations',
        name='Compare implementations',
        description='Compare different implementations',
        pattern='{concept1} vs {concept2} comparison differences',
        variables=['concept1', 'concept2'],
        examples=[
            'REST vs GraphQL',
            'SQL vs NoSQL implementation'
        ],
        tags=['comparison', 'alternatives']
    ),
]


class QueryTemplateManager:
    """
    Manages query templates.

    MEDIUM PRIORITY FIX: Add thread safety and persistence.
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize template manager.

        MEDIUM PRIORITY FIX: Add persistence support.

        Args:
            persist_path: Path to persist custom templates
        """
        self.templates: Dict[str, QueryTemplate] = {}

        # MEDIUM PRIORITY FIX: Add thread safety
        self._lock = threading.RLock()

        # MEDIUM PRIORITY FIX: Add persistence
        self.persist_path = Path(persist_path) if persist_path else Path('.rag_data/custom_templates.json')

        # Load builtin templates
        for template in BUILTIN_TEMPLATES:
            self.templates[template.id] = template

        # Load custom templates from disk
        self._load_custom_templates()

        logger.info(f"Loaded {len(self.templates)} query templates")

    def _load_custom_templates(self):
        """
        MEDIUM PRIORITY FIX: Load custom templates from disk.
        """
        if not self.persist_path.exists():
            return

        try:
            with self._lock:
                data = json.loads(self.persist_path.read_text())
                for template_data in data.get('templates', []):
                    try:
                        template = QueryTemplate.from_dict(template_data)
                        # Only load if not a builtin template
                        is_builtin = any(t.id == template.id for t in BUILTIN_TEMPLATES)
                        if not is_builtin:
                            self.templates[template.id] = template
                            logger.debug(f"Loaded custom template: {template.id}")
                    except Exception as e:
                        logger.error(f"Error loading template: {e}")

                logger.info(f"Loaded {len(data.get('templates', []))} custom templates from disk")

        except Exception as e:
            logger.error(f"Error loading custom templates: {e}")

    def _save_custom_templates(self):
        """
        MEDIUM PRIORITY FIX: Save custom templates to disk.
        """
        try:
            # Ensure directory exists
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            # Get only custom templates (not builtins)
            builtin_ids = {t.id for t in BUILTIN_TEMPLATES}
            custom_templates = [
                t.to_dict() for t_id, t in self.templates.items()
                if t_id not in builtin_ids
            ]

            # Write atomically
            import tempfile
            import os

            fd, temp_path = tempfile.mkstemp(
                dir=str(self.persist_path.parent),
                prefix='.templates.tmp.',
                suffix='.json'
            )

            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump({'templates': custom_templates}, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(temp_path, str(self.persist_path))
                logger.debug(f"Saved {len(custom_templates)} custom templates to disk")

            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            logger.error(f"Error saving custom templates: {e}")

    def get_template(self, template_id: str) -> Optional[QueryTemplate]:
        """
        Get template by ID.

        MEDIUM PRIORITY FIX: Add thread safety.

        Args:
            template_id: Template ID

        Returns:
            QueryTemplate or None
        """
        with self._lock:
            return self.templates.get(template_id)

    def list_templates(
        self,
        tags: Optional[List[str]] = None
    ) -> List[QueryTemplate]:
        """
        List available templates.

        MEDIUM PRIORITY FIX: Add thread safety.

        Args:
            tags: Filter by tags

        Returns:
            List of templates
        """
        with self._lock:
            templates = list(self.templates.values())

            if tags:
                tag_set = set(tags)
                templates = [
                    t for t in templates
                    if tag_set & set(t.tags)
                ]

            return templates

    def format_template(
        self,
        template_id: str,
        **variables
    ) -> Optional[str]:
        """
        Format a template with variables.

        Args:
            template_id: Template ID
            **variables: Template variables

        Returns:
            Formatted query or None
        """
        template = self.get_template(template_id)

        if not template:
            logger.error(f"Template not found: {template_id}")
            return None

        if not template.validate_variables(**variables):
            logger.error(f"Missing required variables for template {template_id}")
            return None

        return template.format(**variables)

    def execute_template(
        self,
        template_id: str,
        query_engine,
        **variables
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a template query.

        Args:
            template_id: Template ID
            query_engine: Query engine instance
            **variables: Template variables

        Returns:
            Query results or None
        """
        query = self.format_template(template_id, **variables)

        if not query:
            return None

        template = self.get_template(template_id)

        try:
            results = query_engine.query(
                query,
                n_results=template.n_results
            )

            return {
                'template_id': template_id,
                'template_name': template.name,
                'query': query,
                'results': results
            }

        except Exception as e:
            logger.error(f"Template execution failed: {e}")
            return None

    def add_template(self, template: QueryTemplate) -> bool:
        """
        Add custom template.

        MEDIUM PRIORITY FIX: Add persistence.

        Args:
            template: QueryTemplate instance

        Returns:
            True if added
        """
        with self._lock:
            if template.id in self.templates:
                logger.warning(f"Template {template.id} already exists")
                return False

            self.templates[template.id] = template
            logger.info(f"Added template: {template.id}")

            # MEDIUM PRIORITY FIX: Persist to disk
            self._save_custom_templates()

            return True

    def remove_template(self, template_id: str) -> bool:
        """
        Remove template.

        MEDIUM PRIORITY FIX: Add persistence and prevent removing builtins.

        Args:
            template_id: Template ID

        Returns:
            True if removed
        """
        with self._lock:
            # MEDIUM PRIORITY FIX: Prevent removing builtin templates
            builtin_ids = {t.id for t in BUILTIN_TEMPLATES}
            if template_id in builtin_ids:
                logger.error(f"Cannot remove builtin template: {template_id}")
                return False

            if template_id in self.templates:
                del self.templates[template_id]
                logger.info(f"Removed template: {template_id}")

                # MEDIUM PRIORITY FIX: Persist to disk
                self._save_custom_templates()

                return True

            return False

    def search_templates(self, query: str) -> List[QueryTemplate]:
        """
        Search templates by name or description.

        Args:
            query: Search query

        Returns:
            Matching templates
        """
        query_lower = query.lower()

        matches = []

        for template in self.templates.values():
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower()):
                matches.append(template)

        return matches

    def suggest_template(self, query: str) -> Optional[QueryTemplate]:
        """
        Suggest best template for a query.

        Args:
            query: User query

        Returns:
            Suggested template or None
        """
        query_lower = query.lower()

        # Pattern matching for suggestions
        if 'how does' in query_lower or 'how to' in query_lower:
            return self.get_template('how_does_it_work')

        if 'find uses' in query_lower or 'where is used' in query_lower:
            return self.get_template('find_uses')

        if 'dependencies' in query_lower or 'depends on' in query_lower:
            return self.get_template('find_dependencies')

        if 'example' in query_lower:
            return self.get_template('find_examples')

        if 'error' in query_lower or 'exception' in query_lower:
            return self.get_template('error_handling')

        if 'test' in query_lower:
            return self.get_template('find_tests')

        if 'api' in query_lower or 'endpoint' in query_lower or 'route' in query_lower:
            return self.get_template('api_endpoints')

        if 'config' in query_lower or 'setting' in query_lower:
            return self.get_template('config_settings')

        if 'security' in query_lower or 'auth' in query_lower:
            return self.get_template('security_checks')

        if 'performance' in query_lower or 'optimization' in query_lower:
            return self.get_template('performance_optimizations')

        if 'model' in query_lower or 'schema' in query_lower:
            return self.get_template('data_models')

        if ' vs ' in query_lower or 'compare' in query_lower:
            return self.get_template('compare_implementations')

        return None

    def extract_variables(self, template_id: str, query: str) -> Optional[Dict[str, str]]:
        """
        Extract variables from a query using template pattern.

        HIGH PRIORITY FIX: Support multi-word variable extraction.
        Original only matched single words, now handles multi-word concepts.

        Args:
            template_id: Template ID
            query: User query

        Returns:
            Extracted variables or None
        """
        template = self.get_template(template_id)

        if not template:
            return None

        # HIGH PRIORITY FIX: Improved extraction with multi-word support
        variables = {}

        # For now, just extract the main concept
        # This is a simplified version (can be improved with NLP)
        for var in template.variables:
            # Try to extract based on common patterns
            if var == 'feature' or var == 'component' or var == 'name':
                # HIGH PRIORITY FIX: Extended pattern to capture multi-word concepts
                # Original: r'(?:how does|find|for|in|of)\s+(\w+(?:\s+\w+)?)'
                # New: Captures up to 4 words to handle complex names like "user authentication system"
                patterns = [
                    # Pattern 1: "how does <concept> work"
                    r'(?:how does|how do)\s+([\w\s]+?)\s+(?:work|function|operate)',
                    # Pattern 2: "find <concept>"
                    r'(?:find|search for|locate)\s+([\w\s]+?)(?:\s+in|\s+for|\?|$)',
                    # Pattern 3: "for <concept>"
                    r'for\s+([\w\s]+?)(?:\s+in|\?|$)',
                    # Pattern 4: "in <concept>"
                    r'in\s+([\w\s]+?)(?:\?|$)',
                    # Pattern 5: "of <concept>"
                    r'of\s+([\w\s]+?)(?:\?|$)',
                    # Pattern 6: General catch-all (captures 1-4 words)
                    r'(?:how does|find|for|in|of)\s+([\w]+(?:\s+[\w]+){0,3})'
                ]

                for pattern in patterns:
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        # Clean up extracted text (remove extra spaces, trailing words)
                        concept = match.group(1).strip()
                        # Remove common trailing words that aren't part of the concept
                        concept = re.sub(r'\s+(?:work|function|operate|in|for|the|a|an)$', '', concept, flags=re.IGNORECASE)

                        # HIGH PRIORITY FIX: Validate extracted variable
                        if concept and self._validate_variable(concept, var):
                            variables[var] = concept
                            break  # Use first successful match

        return variables if variables else None

    def _validate_variable(self, value: str, var_name: str) -> bool:
        """
        HIGH PRIORITY FIX: Validate extracted variable.

        Args:
            value: Variable value to validate
            var_name: Variable name

        Returns:
            True if valid
        """
        if not value or not isinstance(value, str):
            return False

        # Remove extra whitespace
        value = value.strip()

        # Minimum length check (at least 2 characters)
        if len(value) < 2:
            return False

        # Maximum length check (reasonable concept name)
        if len(value) > 100:
            logger.warning(f"Variable '{var_name}' too long: {len(value)} chars")
            return False

        # Check for only valid characters (alphanumeric, spaces, underscores, hyphens)
        if not re.match(r'^[\w\s\-]+$', value):
            logger.warning(f"Variable '{var_name}' contains invalid characters: {value}")
            return False

        # Check it's not just whitespace or special characters
        if not re.search(r'[a-zA-Z0-9]', value):
            return False

        return True


# Global instance
template_manager = QueryTemplateManager()
