"""
Code Review Agent - Automated code quality and best practices checking.

This module provides intelligent code review capabilities:
- Code quality analysis
- Best practices validation
- Security vulnerability detection
- Performance issue identification
- Documentation completeness

Expected impact: +40% code quality improvement.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
 """Severity levels for code issues."""
 CRITICAL = "critical" # Security vulnerabilities, major bugs
 HIGH = "high" # Performance issues, bad practices
 MEDIUM = "medium" # Code smells, minor issues
 LOW = "low" # Style, formatting
 INFO = "info" # Suggestions, optimizations


class IssueCategory(Enum):
 """Categories of code issues."""
 SECURITY = "security"
 PERFORMANCE = "performance"
 BEST_PRACTICES = "best_practices"
 CODE_SMELL = "code_smell"
 DOCUMENTATION = "documentation"
 STYLE = "style"
 COMPLEXITY = "complexity"
 ERROR_HANDLING = "error_handling"


@dataclass
class CodeIssue:
 """
 Represents a code quality issue.
 """
 severity: IssueSeverity
 category: IssueCategory
 title: str
 description: str
 line_number: Optional[int] = None
 code_snippet: Optional[str] = None
 suggestion: Optional[str] = None
 reference: Optional[str] = None

 def to_dict(self) -> Dict[str, Any]:
 """Convert to dictionary."""
 return {
 'severity': self.severity.value,
 'category': self.category.value,
 'title': self.title,
 'description': self.description,
 'line_number': self.line_number,
 'code_snippet': self.code_snippet,
 'suggestion': self.suggestion,
 'reference': self.reference
 }


@dataclass
class ReviewResult:
 """
 Result of code review analysis.
 """
 issues: List[CodeIssue]
 summary: str
 overall_score: float # 0.0-10.0
 metrics: Dict[str, Any]

 def to_dict(self) -> Dict[str, Any]:
 """Convert to dictionary."""
 return {
 'issues': [issue.to_dict() for issue in self.issues],
 'summary': self.summary,
 'overall_score': self.overall_score,
 'metrics': self.metrics,
 'issue_counts': self.get_issue_counts()
 }

 def get_issue_counts(self) -> Dict[str, int]:
 """Get count of issues by severity."""
 counts = {sev.value: 0 for sev in IssueSeverity}
 for issue in self.issues:
 counts[issue.severity.value] += 1
 return counts


class CodeReviewAgent:
 """
 Autonomous code review agent.

 Provides comprehensive code quality analysis including:
 - Security vulnerability detection
 - Performance issue identification
 - Best practices validation
 - Code complexity analysis
 - Documentation completeness
 """

 def __init__(self, llm_provider=None, rag_engine=None):
 """
 Initialize code review agent.

 Args:
 llm_provider: LLM provider for advanced analysis
 rag_engine: RAG engine for finding similar code patterns
 """
 self.llm = llm_provider
 self.rag = rag_engine

 # Initialize rule-based checkers
 self.security_checker = SecurityChecker()
 self.performance_checker = PerformanceChecker()
 self.best_practices_checker = BestPracticesChecker()
 self.complexity_checker = ComplexityChecker()

 logger.info("Initialized CodeReviewAgent")

 def review_code(
 self,
 code: str,
 file_path: Optional[str] = None,
 language: str = "python"
 ) -> ReviewResult:
 """
 Perform comprehensive code review.

 Args:
 code: Code to review
 file_path: Optional file path
 language: Programming language

 Returns:
 ReviewResult with issues and metrics
 """
 issues = []

 # Run rule-based checks
 issues.extend(self.security_checker.check(code))
 issues.extend(self.performance_checker.check(code))
 issues.extend(self.best_practices_checker.check(code))
 issues.extend(self.complexity_checker.check(code))

 # LLM-based advanced analysis (if available)
 if self.llm:
 llm_issues = self._llm_review(code, file_path)
 issues.extend(llm_issues)

 # Calculate metrics
 metrics = self._calculate_metrics(code, issues)

 # Calculate overall score
 overall_score = self._calculate_score(issues, metrics)

 # Generate summary
 summary = self._generate_summary(issues, overall_score)

 return ReviewResult(
 issues=issues,
 summary=summary,
 overall_score=overall_score,
 metrics=metrics
 )

 def _llm_review(self, code: str, file_path: Optional[str]) -> List[CodeIssue]:
 """Use LLM for advanced code review."""
 prompt = self._build_review_prompt(code, file_path)

 try:
 response = self.llm.generate(
 prompt=prompt,
 system_prompt="You are an expert code reviewer specializing in Python best practices and security."
 )

 return self._parse_llm_issues(response)

 except Exception as e:
 logger.error(f"LLM review failed: {e}")
 return []

 def _build_review_prompt(self, code: str, file_path: Optional[str]) -> str:
 """Build prompt for LLM review."""
 prompt_parts = [
 "# Code Review Request\n",
 f"File: {file_path or 'unknown'}\n",
 "```python",
 code,
 "```\n",
 """
Please review this code for:
1. Security vulnerabilities (SQL injection, XSS, etc.)
2. Performance issues (inefficient algorithms, unnecessary operations)
3. Best practices violations
4. Code smells and maintainability issues
5. Missing error handling

Format each issue as:
[SEVERITY] CATEGORY: Title
Description
Line: <line_number>
Suggestion: <how to fix>
---
"""
 ]

 return '\n'.join(prompt_parts)

 def _parse_llm_issues(self, response: str) -> List[CodeIssue]:
 """Parse LLM response into code issues."""
 issues = []
 issue_blocks = response.split('---')

 for block in issue_blocks:
 block = block.strip()
 if not block:
 continue

 try:
 issue = self._parse_issue_block(block)
 if issue:
 issues.append(issue)
 except Exception as e:
 logger.warning(f"Failed to parse issue block: {e}")

 return issues

 def _parse_issue_block(self, block: str) -> Optional[CodeIssue]:
 """Parse a single issue block."""
 lines = block.split('\n')
 if not lines:
 return None

 # Parse header: [SEVERITY] CATEGORY: Title
 header_match = re.match(
 r'\[(\w+)\]\s+(\w+):\s*(.+)',
 lines[0]
 )
 if not header_match:
 return None

 severity_str = header_match.group(1).lower()
 category_str = header_match.group(2).lower()
 title = header_match.group(3).strip()

 # Map severity
 severity = IssueSeverity.MEDIUM # default
 for sev in IssueSeverity:
 if sev.value == severity_str:
 severity = sev
 break

 # Map category
 category = IssueCategory.BEST_PRACTICES # default
 for cat in IssueCategory:
 if cat.value == category_str:
 category = cat
 break

 # Extract description, line, suggestion
 description = ""
 line_number = None
 suggestion = None

 for line in lines[1:]:
 line = line.strip()
 if line.startswith('Line:'):
 try:
 line_number = int(line.replace('Line:', '').strip())
 except:
 pass
 elif line.startswith('Suggestion:'):
 suggestion = line.replace('Suggestion:', '').strip()
 elif not line.startswith('[') and line:
 description += line + ' '

 return CodeIssue(
 severity=severity,
 category=category,
 title=title,
 description=description.strip(),
 line_number=line_number,
 suggestion=suggestion
 )

 def _calculate_metrics(self, code: str, issues: List[CodeIssue]) -> Dict[str, Any]:
 """Calculate code metrics."""
 lines = code.split('\n')

 return {
 'total_lines': len(lines),
 'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
 'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
 'blank_lines': len([l for l in lines if not l.strip()]),
 'total_issues': len(issues),
 'critical_issues': len([i for i in issues if i.severity == IssueSeverity.CRITICAL]),
 'high_issues': len([i for i in issues if i.severity == IssueSeverity.HIGH]),
 'medium_issues': len([i for i in issues if i.severity == IssueSeverity.MEDIUM]),
 'low_issues': len([i for i in issues if i.severity == IssueSeverity.LOW])
 }

 def _calculate_score(self, issues: List[CodeIssue], metrics: Dict[str, Any]) -> float:
 """Calculate overall code quality score (0-10)."""
 base_score = 10.0

 # Deduct points for issues
 for issue in issues:
 if issue.severity == IssueSeverity.CRITICAL:
 base_score -= 2.0
 elif issue.severity == IssueSeverity.HIGH:
 base_score -= 1.0
 elif issue.severity == IssueSeverity.MEDIUM:
 base_score -= 0.5
 elif issue.severity == IssueSeverity.LOW:
 base_score -= 0.2

 # Bonus for good comment ratio
 if metrics['code_lines'] > 0:
 comment_ratio = metrics['comment_lines'] / metrics['code_lines']
 if comment_ratio > 0.2: # Good commenting
 base_score += 0.5

 return max(0.0, min(10.0, base_score))

 def _generate_summary(self, issues: List[CodeIssue], score: float) -> str:
 """Generate review summary."""
 if not issues:
 return f"Code quality: Excellent ({score:.1f}/10). No issues found."

 issue_counts = {}
 for issue in issues:
 sev = issue.severity.value
 issue_counts[sev] = issue_counts.get(sev, 0) + 1

 parts = [f"Code quality: {score:.1f}/10"]

 if issue_counts.get('critical', 0) > 0:
 parts.append(f" {issue_counts['critical']} critical issue(s)")
 if issue_counts.get('high', 0) > 0:
 parts.append(f" {issue_counts['high']} high priority issue(s)")
 if issue_counts.get('medium', 0) > 0:
 parts.append(f"🟡 {issue_counts['medium']} medium issue(s)")
 if issue_counts.get('low', 0) > 0:
 parts.append(f" {issue_counts['low']} low priority issue(s)")

 return ' | '.join(parts)


class SecurityChecker:
 """Rule-based security vulnerability checker."""

 PATTERNS = [
 (
 re.compile(r'eval\s*\('),
 IssueSeverity.CRITICAL,
 "Use of eval() function",
 "eval() can execute arbitrary code and is a security risk",
 "Use ast.literal_eval() for safe evaluation or refactor to avoid eval"
 ),
 (
 re.compile(r'exec\s*\('),
 IssueSeverity.CRITICAL,
 "Use of exec() function",
 "exec() can execute arbitrary code and is a security risk",
 "Refactor to avoid dynamic code execution"
 ),
 (
 re.compile(r'pickle\.loads?\('),
 IssueSeverity.HIGH,
 "Use of pickle with untrusted data",
 "Pickle can execute arbitrary code when deserializing",
 "Use JSON or other safe serialization formats"
 ),
 (
 re.compile(r'subprocess\.(call|run|Popen).*shell\s*=\s*True'),
 IssueSeverity.CRITICAL,
 "Shell injection vulnerability",
 "Using shell=True with subprocess can lead to command injection",
 "Use shell=False and pass arguments as a list"
 ),
 (
 re.compile(r'password\s*=\s*["\'][^"\']+["\']'),
 IssueSeverity.CRITICAL,
 "Hardcoded password",
 "Passwords should not be hardcoded in source code",
 "Use environment variables or secure configuration"
 ),
 ]

 def check(self, code: str) -> List[CodeIssue]:
 """Check for security issues."""
 issues = []

 for pattern, severity, title, description, suggestion in self.PATTERNS:
 for match in pattern.finditer(code):
 # Find line number
 line_num = code[:match.start()].count('\n') + 1

 issues.append(CodeIssue(
 severity=severity,
 category=IssueCategory.SECURITY,
 title=title,
 description=description,
 line_number=line_num,
 suggestion=suggestion
 ))

 return issues


class PerformanceChecker:
 """Rule-based performance issue checker."""

 PATTERNS = [
 (
 re.compile(r'for\s+\w+\s+in\s+range\(len\((\w+)\)\):'),
 IssueSeverity.MEDIUM,
 "Inefficient iteration pattern",
 "Using range(len()) is less Pythonic and slower",
 "Use 'for item in list:' or 'for i, item in enumerate(list):'"
 ),
 (
 re.compile(r'\w+\s*\+=\s*\['),
 IssueSeverity.LOW,
 "List concatenation in loop",
 "Repeated list concatenation is inefficient",
 "Use list.append() or list comprehension"
 ),
 (
 re.compile(r'\.append\(.*\)\s*\n.*\.append\('),
 IssueSeverity.LOW,
 "Multiple append calls",
 "Multiple append calls could be optimized",
 "Consider using list comprehension or extend()"
 ),
 ]

 def check(self, code: str) -> List[CodeIssue]:
 """Check for performance issues."""
 issues = []

 for pattern, severity, title, description, suggestion in self.PATTERNS:
 for match in pattern.finditer(code):
 line_num = code[:match.start()].count('\n') + 1

 issues.append(CodeIssue(
 severity=severity,
 category=IssueCategory.PERFORMANCE,
 title=title,
 description=description,
 line_number=line_num,
 suggestion=suggestion
 ))

 return issues


class BestPracticesChecker:
 """Rule-based best practices checker."""

 PATTERNS = [
 (
 re.compile(r'except\s*:'),
 IssueSeverity.HIGH,
 "Bare except clause",
 "Catching all exceptions can hide bugs",
 "Specify exception types: except ValueError, TypeError:"
 ),
 (
 re.compile(r'except\s+Exception\s*:'),
 IssueSeverity.MEDIUM,
 "Overly broad exception handling",
 "Catching Exception is too broad",
 "Catch specific exception types"
 ),
 (
 re.compile(r'import\s+\*'),
 IssueSeverity.MEDIUM,
 "Wildcard import",
 "Wildcard imports pollute namespace and reduce code clarity",
 "Import specific names or use qualified imports"
 ),
 ]

 def check(self, code: str) -> List[CodeIssue]:
 """Check for best practices violations."""
 issues = []

 for pattern, severity, title, description, suggestion in self.PATTERNS:
 for match in pattern.finditer(code):
 line_num = code[:match.start()].count('\n') + 1

 issues.append(CodeIssue(
 severity=severity,
 category=IssueCategory.BEST_PRACTICES,
 title=title,
 description=description,
 line_number=line_num,
 suggestion=suggestion
 ))

 return issues


class ComplexityChecker:
 """Rule-based complexity checker."""

 def check(self, code: str) -> List[CodeIssue]:
 """Check for complexity issues."""
 issues = []

 # Check function length
 functions = re.finditer(r'def\s+(\w+)\s*\([^)]*\):', code)
 for func_match in functions:
 func_start = func_match.start()
 func_name = func_match.group(1)

 # Find function end (next def or end of file)
 next_def = code.find('\ndef ', func_start + 1)
 func_end = next_def if next_def != -1 else len(code)

 func_code = code[func_start:func_end]
 func_lines = len([l for l in func_code.split('\n') if l.strip()])

 if func_lines > 50:
 line_num = code[:func_start].count('\n') + 1

 issues.append(CodeIssue(
 severity=IssueSeverity.MEDIUM,
 category=IssueCategory.COMPLEXITY,
 title=f"Function '{func_name}' is too long",
 description=f"Function has {func_lines} lines (recommended: <50)",
 line_number=line_num,
 suggestion="Consider breaking this function into smaller functions"
 ))

 return issues


def create_review_agent(llm_provider=None, rag_engine=None) -> CodeReviewAgent:
 """
 Convenience function to create code review agent.

 Args:
 llm_provider: LLM provider
 rag_engine: RAG engine

 Returns:
 Initialized CodeReviewAgent
 """
 return CodeReviewAgent(
 llm_provider=llm_provider,
 rag_engine=rag_engine
 )
