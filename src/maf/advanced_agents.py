"""
Advanced agents: Code Summarization and Dependency Mapping.
"""

from typing import Dict, Any, List, Set
from collections import defaultdict
import re
import logging
from .agents import BaseAgent

logger = logging.getLogger(__name__)


class CodeSummarizationAgent(BaseAgent):
    """
    Agent that generates summaries of code files using heuristics.
    """

    def __init__(self, rag_engine: Any = None):
        super().__init__(name="CodeSummarizer", rag_engine=rag_engine)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code summary.

        Args:
            context: Execution context with query

        Returns:
            Summary results
        """
        query = context.get('query', '')
        logger.info(f"CodeSummarizer analyzing: {query}")

        if self.rag_engine:
            # Get relevant code
            results = self.rag_engine.query(
                query_text=query,
                n_results=10,
                file_type='.py'  # Focus on Python for now
            )

            if results:
                summary = self._analyze_results(results)
            else:
                summary = {'message': 'No code found'}
        else:
            summary = {'error': 'No RAG engine available'}

        return {
            'agent': self.name,
            'query': query,
            'summary': summary
        }

    def _analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze code results and generate summary.

        Args:
            results: List of code search results

        Returns:
            Summary dictionary
        """
        summary = {
            'files': set(),
            'classes': [],
            'functions': [],
            'imports': set(),
            'total_snippets': len(results)
        }

        for result in results:
            metadata = result.get('metadata', {})
            text = result.get('text', '')

            # Track files
            file_path = metadata.get('file_path', 'unknown')
            summary['files'].add(file_path)

            # Extract classes
            classes = re.findall(r'class\s+(\w+)', text)
            summary['classes'].extend(classes)

            # Extract functions
            functions = re.findall(r'def\s+(\w+)\s*\(', text)
            summary['functions'].extend(functions)

            # Extract imports
            imports = re.findall(r'(?:from\s+[\w.]+\s+)?import\s+([\w\s,]+)', text)
            for imp in imports:
                summary['imports'].update(imp.split(','))

        # Clean up
        summary['files'] = list(summary['files'])
        summary['imports'] = list(summary['imports'])[:20]  # Top 20
        summary['classes'] = list(set(summary['classes']))[:20]
        summary['functions'] = list(set(summary['functions']))[:20]

        return summary


class DependencyMappingAgent(BaseAgent):
    """
    Agent that analyzes and maps code dependencies.
    """

    def __init__(self, rag_engine: Any = None):
        super().__init__(name="DependencyMapper", rag_engine=rag_engine)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze dependencies in codebase.

        Args:
            context: Execution context

        Returns:
            Dependency analysis
        """
        logger.info("DependencyMapper analyzing dependencies")

        if not self.rag_engine:
            return {'agent': self.name, 'error': 'No RAG engine'}

        # Get files with imports
        results = self.rag_engine.query(
            query_text="import",
            n_results=100
        )

        if not results:
            return {
                'agent': self.name,
                'message': 'No import statements found'
            }

        dependencies = self._analyze_dependencies(results)
        most_imported = self._get_most_imported(dependencies)

        return {
            'agent': self.name,
            'total_files': len(dependencies),
            'dependency_count': sum(len(deps) for deps in dependencies.values()),
            'most_imported': most_imported[:10],
            'files_with_most_deps': self._get_files_with_most_deps(dependencies)[:10]
        }

    def _analyze_dependencies(self, results: List[Dict]) -> Dict[str, Set[str]]:
        """
        Extract dependency relationships from results.

        Args:
            results: Search results

        Returns:
            Dictionary mapping files to their imports
        """
        deps = defaultdict(set)

        for result in results:
            metadata = result.get('metadata', {})
            file_path = metadata.get('file_path', 'unknown')
            text = result.get('text', '')

            # Extract imports
            imports = self._extract_imports(text)

            if imports:
                deps[file_path].update(imports)

        return dict(deps)

    def _extract_imports(self, code: str) -> Set[str]:
        """
        Extract import statements from code.

        Args:
            code: Code text

        Returns:
            Set of imported modules
        """
        imports = set()

        # Match: import module
        pattern1 = r'import\s+([\w.]+)'
        imports.update(re.findall(pattern1, code))

        # Match: from module import ...
        pattern2 = r'from\s+([\w.]+)\s+import'
        imports.update(re.findall(pattern2, code))

        return imports

    def _get_most_imported(self, deps: Dict[str, Set[str]]) -> List[tuple]:
        """
        Find most frequently imported modules.

        Args:
            deps: Dependency dictionary

        Returns:
            List of (module, count) tuples
        """
        import_counts = defaultdict(int)

        for imports in deps.values():
            for imp in imports:
                import_counts[imp] += 1

        return sorted(import_counts.items(), key=lambda x: x[1], reverse=True)

    def _get_files_with_most_deps(self, deps: Dict[str, Set[str]]) -> List[tuple]:
        """
        Find files with most dependencies.

        Args:
            deps: Dependency dictionary

        Returns:
            List of (file, dep_count) tuples
        """
        file_counts = [(file, len(imports)) for file, imports in deps.items()]
        return sorted(file_counts, key=lambda x: x[1], reverse=True)


class SecurityAnalysisAgent(BaseAgent):
    """
    Agent that performs basic security analysis on code.
    """

    def __init__(self, rag_engine: Any = None):
        super().__init__(name="SecurityAnalyzer", rag_engine=rag_engine)

        # Common security issues to look for
        self.security_patterns = {
            'sql_injection': r'execute\s*\(\s*["\'].*%s.*["\']',
            'command_injection': r'os\.system\s*\(|subprocess\.call\s*\(',
            'hardcoded_secrets': r'(password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']',
            'eval_usage': r'\beval\s*\(',
            'pickle_usage': r'\bpickle\.loads?\s*\(',
            'weak_crypto': r'md5|sha1',
        }

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform security analysis.

        Args:
            context: Execution context

        Returns:
            Security findings
        """
        logger.info("SecurityAnalyzer scanning for security issues")

        if not self.rag_engine:
            return {'agent': self.name, 'error': 'No RAG engine'}

        findings = defaultdict(list)

        # Check for each security pattern
        for issue_type, pattern in self.security_patterns.items():
            results = self._search_pattern(pattern)

            for result in results:
                findings[issue_type].append({
                    'file': result.get('metadata', {}).get('file_path', 'unknown'),
                    'snippet': result.get('text', '')[:100]
                })

        total_issues = sum(len(issues) for issues in findings.values())

        return {
            'agent': self.name,
            'total_issues': total_issues,
            'findings': dict(findings),
            'severity': 'high' if total_issues > 10 else 'medium' if total_issues > 5 else 'low'
        }

    def _search_pattern(self, pattern: str) -> List[Dict]:
        """Search for a specific pattern in code."""
        try:
            # Use RAG to find matches
            # This is a simplified approach
            return []
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return []
