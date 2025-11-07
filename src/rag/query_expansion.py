"""
Query expansion with synonyms and related terms.
"""

from typing import List, Set, Dict
import re
import logging

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand queries with synonyms and related terms."""

    # Programming term synonyms
    SYNONYMS: Dict[str, List[str]] = {
        'function': ['method', 'procedure', 'routine', 'func', 'def'],
        'class': ['object', 'type', 'interface', 'struct'],
        'variable': ['var', 'attribute', 'field', 'property', 'member'],
        'error': ['exception', 'bug', 'issue', 'failure', 'fault'],
        'test': ['unit test', 'spec', 'assertion', 'unittest'],
        'api': ['endpoint', 'route', 'service', 'interface'],
        'database': ['db', 'sql', 'storage', 'persistence', 'datastore'],
        'auth': ['authentication', 'login', 'credentials', 'signin'],
        'config': ['configuration', 'settings', 'options', 'preferences'],
        'docs': ['documentation', 'readme', 'guide', 'manual'],
        'import': ['include', 'require', 'using', 'from'],
        'return': ['returns', 'output', 'result', 'yield'],
        'parameter': ['param', 'argument', 'arg', 'input'],
        'initialize': ['init', 'setup', 'create', 'construct'],
        'delete': ['remove', 'destroy', 'drop', 'unlink'],
        'update': ['modify', 'change', 'edit', 'alter'],
        'create': ['add', 'new', 'make', 'insert'],
        'read': ['get', 'fetch', 'retrieve', 'select'],
        'write': ['save', 'store', 'persist', 'insert'],
        'async': ['asynchronous', 'await', 'promise', 'concurrent'],
        'sync': ['synchronous', 'blocking', 'sequential'],
    }

    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with synonyms.

        HIGH PRIORITY FIX: Improved word boundary handling for edge cases.

        Args:
            query: Original query
            max_expansions: Maximum number of synonym expansions per term

        Returns:
            List of expanded queries including original
        """
        expansions = [query]
        words = query.lower().split()

        for word in words:
            # HIGH PRIORITY FIX: Strip punctuation before lookup
            # Handle cases like "function," or "class."
            clean_word = word.strip('.,!?;:()[]{}"\'-')

            if clean_word in self.SYNONYMS:
                synonyms = self.SYNONYMS[clean_word][:max_expansions]

                for synonym in synonyms:
                    # HIGH PRIORITY FIX: Build pattern that handles word boundaries properly
                    # Account for punctuation and ensure whole-word match
                    # Pattern matches word with optional leading/trailing punctuation
                    pattern = r'([.,!?;:()[\]{}"\'\s-]|^)(' + re.escape(clean_word) + r')([.,!?;:()[\]{}"\'\s-]|$)'

                    def replacer(match):
                        # Preserve surrounding punctuation/whitespace
                        return match.group(1) + synonym + match.group(3)

                    expanded = re.sub(
                        pattern,
                        replacer,
                        query,
                        flags=re.IGNORECASE
                    )

                    if expanded != query and expanded not in expansions and len(expansions) < 10:
                        expansions.append(expanded)

        logger.debug(f"Expanded '{query}' to {len(expansions)} queries")
        return expansions

    def extract_technical_terms(self, query: str) -> Set[str]:
        """
        Extract technical terms from query.

        Args:
            query: Query string

        Returns:
            Set of technical terms
        """
        patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
            r'\b[a-z]+_[a-z_]+\b',                # snake_case
            r'\b[A-Z_]{2,}\b',                    # CONSTANTS
            r'\b\w+\(\)',                         # functions()
            r'\b[a-z]+\.[a-z]+\b',               # module.function
        ]

        terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, query)
            terms.update(matches)

        return terms

    def add_context_terms(self, query: str, file_type: str = None) -> str:
        """
        Add contextual terms based on file type.

        Args:
            query: Original query
            file_type: File type (e.g., '.py', '.js')

        Returns:
            Query with added context
        """
        if not file_type:
            return query

        context_terms = {
            '.py': ['python', 'def', 'class', 'import'],
            '.js': ['javascript', 'function', 'const', 'import'],
            '.ts': ['typescript', 'interface', 'type', 'import'],
            '.java': ['java', 'public', 'class', 'import'],
            '.go': ['golang', 'func', 'struct', 'package'],
            '.rs': ['rust', 'fn', 'impl', 'use'],
        }

        if file_type in context_terms:
            # Add most relevant context term if not already in query
            for term in context_terms[file_type]:
                if term.lower() not in query.lower():
                    return f"{query} {term}"

        return query

    def expand_with_technical_patterns(self, query: str) -> List[str]:
        """
        Expand query with common technical patterns.

        Args:
            query: Original query

        Returns:
            List of expanded queries
        """
        expansions = [query]

        # Check for common patterns
        if 'how to' in query.lower():
            # Add implementation-focused variation
            expanded = query.replace('how to', 'implementation of')
            expansions.append(expanded)

        if 'what is' in query.lower():
            # Add definition-focused variation
            expanded = query.replace('what is', 'definition of')
            expansions.append(expanded)

        if '?' in query:
            # Remove question mark for code search
            expansions.append(query.replace('?', ''))

        return list(set(expansions))
