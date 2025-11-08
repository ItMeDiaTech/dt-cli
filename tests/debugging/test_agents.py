"""
Tests for debugging and code review agents.
"""

import pytest
from src.debugging import (
    DebugAgent,
    ErrorContext,
    ErrorParser,
    ErrorType,
    CodeReviewAgent,
    CodeIssue,
    IssueSeverity,
    IssueCategory
)


class TestErrorParser:
    """Test error parsing functionality."""

    def test_parse_python_error(self):
        """Test parsing standard Python error."""
        error_output = """
Traceback (most recent call last):
  File "test.py", line 10, in main
    result = divide(a, b)
  File "test.py", line 5, in divide
    return a / b
ZeroDivisionError: division by zero
"""
        context = ErrorParser.parse_error(error_output)

        assert context.error_type == ErrorType.RUNTIME_ERROR or context.error_type == ErrorType.UNKNOWN
        assert "division by zero" in context.error_message
        assert context.file_path == "test.py"
        assert context.line_number == 5
        assert context.function_name == "divide"

    def test_parse_import_error(self):
        """Test parsing import error."""
        error_output = """
Traceback (most recent call last):
  File "main.py", line 1, in <module>
    import missing_module
ImportError: No module named 'missing_module'
"""
        context = ErrorParser.parse_error(error_output)

        assert context.error_type == ErrorType.IMPORT_ERROR
        assert "missing_module" in context.error_message
        assert context.file_path == "main.py"

    def test_parse_type_error(self):
        """Test parsing type error."""
        error_output = """
Traceback (most recent call last):
  File "script.py", line 15, in process
    result = "text" + 42
TypeError: can only concatenate str (not "int") to str
"""
        context = ErrorParser.parse_error(error_output)

        assert context.error_type == ErrorType.TYPE_ERROR
        assert "concatenate" in context.error_message.lower()

    def test_parse_attribute_error(self):
        """Test parsing attribute error."""
        error_output = """
AttributeError: 'NoneType' object has no attribute 'upper'
"""
        context = ErrorParser.parse_error(error_output)

        assert context.error_type == ErrorType.ATTRIBUTE_ERROR
        assert "NoneType" in context.error_message

    def test_extract_code_snippet(self):
        """Test code snippet extraction."""
        # Create a temporary test file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("line 1\n")
            f.write("line 2\n")
            f.write("line 3 - error here\n")
            f.write("line 4\n")
            f.write("line 5\n")
            temp_path = f.name

        try:
            snippet = ErrorParser.extract_code_snippet(temp_path, 3, context_lines=2)
            assert "line 3" in snippet
            assert ">>>" in snippet  # Error marker
            assert "line 1" in snippet
            assert "line 5" in snippet
        finally:
            os.unlink(temp_path)


class TestDebugAgent:
    """Test debug agent functionality."""

    def test_init(self):
        """Test agent initialization."""
        agent = DebugAgent()
        assert agent is not None
        assert isinstance(agent.parser, ErrorParser)

    def test_analyze_simple_error(self):
        """Test analyzing a simple error without LLM."""
        agent = DebugAgent()

        error_output = """
ImportError: No module named 'missing_package'
"""

        analysis = agent.analyze_error(error_output, auto_extract_code=False)

        assert analysis is not None
        assert analysis.error_context.error_type == ErrorType.IMPORT_ERROR
        assert len(analysis.root_cause) > 0
        assert len(analysis.suggested_fixes) > 0
        assert 0.0 <= analysis.confidence <= 1.0

    def test_analyze_type_error(self):
        """Test analyzing type error."""
        agent = DebugAgent()

        error_output = """
TypeError: unsupported operand type(s) for +: 'int' and 'str'
"""

        analysis = agent.analyze_error(error_output, auto_extract_code=False)

        assert analysis.error_context.error_type == ErrorType.TYPE_ERROR
        assert "type" in analysis.root_cause.lower() or "type" in analysis.explanation.lower()
        assert len(analysis.suggested_fixes) > 0

    def test_analyze_attribute_error(self):
        """Test analyzing attribute error."""
        agent = DebugAgent()

        error_output = """
AttributeError: 'list' object has no attribute 'append_all'
"""

        analysis = agent.analyze_error(error_output, auto_extract_code=False)

        assert analysis.error_context.error_type == ErrorType.ATTRIBUTE_ERROR
        assert len(analysis.suggested_fixes) > 0

    def test_analyze_name_error(self):
        """Test analyzing name error."""
        agent = DebugAgent()

        error_output = """
NameError: name 'undefined_variable' is not defined
"""

        analysis = agent.analyze_error(error_output, auto_extract_code=False)

        assert analysis.error_context.error_type == ErrorType.NAME_ERROR
        assert "undefined" in analysis.error_context.error_message
        assert len(analysis.suggested_fixes) > 0

    def test_analysis_to_dict(self):
        """Test converting analysis to dictionary."""
        agent = DebugAgent()

        error_output = "ValueError: invalid literal for int()"

        analysis = agent.analyze_error(error_output, auto_extract_code=False)
        result_dict = analysis.to_dict()

        assert 'error_context' in result_dict
        assert 'root_cause' in result_dict
        assert 'suggested_fixes' in result_dict
        assert 'confidence' in result_dict


class TestCodeReviewAgent:
    """Test code review agent functionality."""

    def test_init(self):
        """Test agent initialization."""
        agent = CodeReviewAgent()
        assert agent is not None

    def test_review_clean_code(self):
        """Test reviewing clean code."""
        agent = CodeReviewAgent()

        code = """
def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers together.\"\"\"
    return a + b

def main():
    result = add_numbers(5, 3)
    print(result)
"""

        review = agent.review_code(code)

        assert review is not None
        assert review.overall_score >= 7.0  # Should score high for clean code
        assert isinstance(review.issues, list)

    def test_review_security_issue_eval(self):
        """Test detecting eval() security issue."""
        agent = CodeReviewAgent()

        code = """
def dangerous_function(user_input):
    result = eval(user_input)
    return result
"""

        review = agent.review_code(code)

        # Should detect eval() as critical security issue
        critical_issues = [i for i in review.issues if i.severity == IssueSeverity.CRITICAL]
        assert len(critical_issues) > 0
        assert any("eval" in issue.title.lower() for issue in critical_issues)

    def test_review_security_issue_exec(self):
        """Test detecting exec() security issue."""
        agent = CodeReviewAgent()

        code = """
def run_code(code_string):
    exec(code_string)
"""

        review = agent.review_code(code)

        critical_issues = [i for i in review.issues if i.severity == IssueSeverity.CRITICAL]
        assert len(critical_issues) > 0
        assert any("exec" in issue.title.lower() for issue in critical_issues)

    def test_review_security_hardcoded_password(self):
        """Test detecting hardcoded passwords."""
        agent = CodeReviewAgent()

        code = """
def connect_db():
    password = "secret123"
    # connect to database
"""

        review = agent.review_code(code)

        critical_issues = [i for i in review.issues if i.severity == IssueSeverity.CRITICAL]
        assert len(critical_issues) > 0
        assert any("password" in issue.title.lower() for issue in critical_issues)

    def test_review_best_practice_bare_except(self):
        """Test detecting bare except clauses."""
        agent = CodeReviewAgent()

        code = """
def risky_function():
    try:
        dangerous_operation()
    except:
        pass
"""

        review = agent.review_code(code)

        # Should detect bare except
        high_issues = [i for i in review.issues if i.severity == IssueSeverity.HIGH]
        assert len(high_issues) > 0
        assert any("except" in issue.title.lower() for issue in high_issues)

    def test_review_best_practice_wildcard_import(self):
        """Test detecting wildcard imports."""
        agent = CodeReviewAgent()

        code = """
from os import *
from sys import *
"""

        review = agent.review_code(code)

        # Should detect wildcard import
        medium_issues = [i for i in review.issues if i.severity == IssueSeverity.MEDIUM]
        assert len(medium_issues) > 0
        assert any("import" in issue.title.lower() for issue in medium_issues)

    def test_review_complexity_long_function(self):
        """Test detecting long functions."""
        agent = CodeReviewAgent()

        # Create a function with >50 lines
        code = "def long_function():\n"
        code += "    pass\n" * 60

        review = agent.review_code(code)

        # Should detect complexity issue
        complexity_issues = [i for i in review.issues if i.category == IssueCategory.COMPLEXITY]
        assert len(complexity_issues) > 0
        assert any("long" in issue.title.lower() for issue in complexity_issues)

    def test_review_performance_range_len(self):
        """Test detecting inefficient iteration."""
        agent = CodeReviewAgent()

        code = """
def process_items(items):
    for i in range(len(items)):
        print(items[i])
"""

        review = agent.review_code(code)

        # Should detect inefficient iteration
        perf_issues = [i for i in review.issues if i.category == IssueCategory.PERFORMANCE]
        assert len(perf_issues) > 0

    def test_review_multiple_issues(self):
        """Test detecting multiple issues."""
        agent = CodeReviewAgent()

        code = """
password = "hardcoded123"

def bad_function(data):
    try:
        result = eval(data)
        return result
    except:
        pass
"""

        review = agent.review_code(code)

        # Should have multiple issues
        assert len(review.issues) >= 3
        assert review.overall_score < 5.0  # Should score low

    def test_review_metrics(self):
        """Test code metrics calculation."""
        agent = CodeReviewAgent()

        code = """
# This is a comment
def example():
    # Another comment
    return 42

# More comments
"""

        review = agent.review_code(code)

        metrics = review.metrics
        assert 'total_lines' in metrics
        assert 'code_lines' in metrics
        assert 'comment_lines' in metrics
        assert 'blank_lines' in metrics
        assert metrics['comment_lines'] >= 3

    def test_review_to_dict(self):
        """Test converting review to dictionary."""
        agent = CodeReviewAgent()

        code = "def test(): pass"

        review = agent.review_code(code)
        result_dict = review.to_dict()

        assert 'issues' in result_dict
        assert 'summary' in result_dict
        assert 'overall_score' in result_dict
        assert 'metrics' in result_dict
        assert 'issue_counts' in result_dict

    def test_issue_to_dict(self):
        """Test converting issue to dictionary."""
        issue = CodeIssue(
            severity=IssueSeverity.HIGH,
            category=IssueCategory.SECURITY,
            title="Test Issue",
            description="This is a test",
            line_number=10,
            suggestion="Fix it"
        )

        issue_dict = issue.to_dict()

        assert issue_dict['severity'] == 'high'
        assert issue_dict['category'] == 'security'
        assert issue_dict['title'] == "Test Issue"
        assert issue_dict['line_number'] == 10


class TestIntegration:
    """Integration tests for agents working together."""

    def test_debug_and_review_same_code(self):
        """Test both agents on the same problematic code."""
        debug_agent = DebugAgent()
        review_agent = CodeReviewAgent()

        # Code with error
        code_with_error = """
def buggy_function():
    result = eval("1 + 1")  # Security issue
    return result
"""

        # Review should find security issue
        review = review_agent.review_code(code_with_error)
        assert len(review.issues) > 0

        # If we get an error from running this code, debug agent should analyze it
        simulated_error = "SecurityError: eval() is not allowed"
        analysis = debug_agent.analyze_error(simulated_error, auto_extract_code=False)
        assert analysis is not None

    def test_agents_without_llm(self):
        """Test that agents work without LLM provider."""
        debug_agent = DebugAgent(llm_provider=None)
        review_agent = CodeReviewAgent(llm_provider=None)

        # Both should still work with rule-based analysis
        error = "ValueError: invalid value"
        analysis = debug_agent.analyze_error(error, auto_extract_code=False)
        assert analysis is not None

        code = "def test(): pass"
        review = review_agent.review_code(code)
        assert review is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
