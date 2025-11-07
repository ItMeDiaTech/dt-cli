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

        MEDIUM PRIORITY FIX: Add more expansion techniques.

        Args:
            query: Original query

        Returns:
            List of expanded queries
        """
        expansions = [query]

        # MEDIUM PRIORITY FIX: Expanded pattern matching
        query_lower = query.lower()

        # Pattern 1: "how to" variations
        if 'how to' in query_lower:
            expansions.append(query.replace('how to', 'implementation of'))
            expansions.append(query.replace('how to', 'example of'))

        # Pattern 2: "what is" variations
        if 'what is' in query_lower:
            expansions.append(query.replace('what is', 'definition of'))
            expansions.append(query.replace('what is', 'explanation of'))

        # Pattern 3: "where is" variations
        if 'where is' in query_lower:
            expansions.append(query.replace('where is', 'location of'))
            expansions.append(query.replace('where is', 'find'))

        # Pattern 4: "why does" variations
        if 'why does' in query_lower:
            expansions.append(query.replace('why does', 'reason for'))
            expansions.append(query.replace('why does', 'explanation why'))

        # Pattern 5: Remove question marks
        if '?' in query:
            expansions.append(query.replace('?', ''))

        # MEDIUM PRIORITY FIX: Add code-specific expansions
        # Pattern 6: Add "code" suffix for implementation queries
        if any(word in query_lower for word in ['implement', 'create', 'build', 'develop']):
            if 'code' not in query_lower:
                expansions.append(f"{query} code")
                expansions.append(f"{query} implementation")

        # Pattern 7: Add "example" for tutorial queries
        if any(word in query_lower for word in ['how', 'usage', 'use']):
            if 'example' not in query_lower:
                expansions.append(f"{query} example")

        # Pattern 8: Add "test" for testing queries
        if any(word in query_lower for word in ['test', 'testing', 'unittest']):
            if 'test' in query_lower and 'test case' not in query_lower:
                expansions.append(query.replace('test', 'test case'))

        # Pattern 9: Add language-agnostic variations
        # Remove specific language names to find general concepts
        language_patterns = {
            'python': 'programming',
            'javascript': 'programming',
            'java': 'programming',
            'typescript': 'programming',
        }
        for lang, generic in language_patterns.items():
            if lang in query_lower:
                expansions.append(query.replace(lang, generic))

        return list(set(expansions))[:10]  # Limit to top 10 expansions
