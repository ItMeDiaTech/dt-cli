"""
Tests for AST-based code chunking.
"""

import pytest
from src.rag.ast_chunker import ASTChunker, chunk_file, CodeChunk
from src.rag.parsers import is_supported


# Sample Python code for testing
PYTHON_CODE = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

class Calculator:
    """Simple calculator class."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def subtract(self, a, b):
        """Subtract two numbers."""
        return a - b

def main():
    """Main function."""
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
'''

# Sample JavaScript code for testing
JAVASCRIPT_CODE = '''
function greet(name) {
    return `Hello, ${name}!`;
}

class Person {
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    sayHello() {
        console.log(greet(this.name));
    }
}

const person = new Person("Alice", 30);
person.sayHello();
'''


class TestASTChunker:
    """Test AST-based chunking."""

    def test_python_chunking(self):
        """Test chunking Python code."""
        chunker = ASTChunker()
        chunks = chunker.chunk_code(PYTHON_CODE, "test.py")

        # Should extract functions and class
        assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"

        # Check chunk types
        chunk_types = {chunk.chunk_type for chunk in chunks}
        assert 'function' in chunk_types or 'method' in chunk_types
        assert 'class' in chunk_types

        # Verify chunks have metadata
        for chunk in chunks:
            assert chunk.metadata['file_path'] == "test.py"
            assert chunk.metadata['language'] == 'python'
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line

    def test_javascript_chunking(self):
        """Test chunking JavaScript code."""
        chunker = ASTChunker()
        chunks = chunker.chunk_code(JAVASCRIPT_CODE, "test.js")

        # Should extract function and class
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

        # Verify metadata
        for chunk in chunks:
            assert chunk.metadata['file_path'] == "test.js"
            assert chunk.metadata['language'] == 'javascript'

    def test_context_headers(self):
        """Test that context headers are added."""
        chunker = ASTChunker(add_context_headers=True)
        chunks = chunker.chunk_code(PYTHON_CODE, "test.py")

        # Check that chunks have context headers
        for chunk in chunks:
            assert "# File:" in chunk.content
            assert "test.py" in chunk.content

    def test_no_context_headers(self):
        """Test chunking without context headers."""
        chunker = ASTChunker(add_context_headers=False)
        chunks = chunker.chunk_code(PYTHON_CODE, "test.py")

        # Check that chunks don't have context headers
        for chunk in chunks:
            if chunk.chunk_type != 'text_chunk':
                assert "# File:" not in chunk.content

    def test_chunk_metadata(self):
        """Test chunk metadata extraction."""
        chunker = ASTChunker()
        chunks = chunker.chunk_code(PYTHON_CODE, "test.py")

        # Find the Calculator class chunk
        calc_chunks = [c for c in chunks if c.metadata.get('name') == 'Calculator']
        assert len(calc_chunks) > 0, "Should find Calculator class"

        calc_chunk = calc_chunks[0]
        assert calc_chunk.chunk_type == 'class'
        assert calc_chunk.metadata['type'] == 'class'

        # Find method chunks
        method_chunks = [c for c in chunks if c.chunk_type == 'method']
        for method in method_chunks:
            assert 'class' in method.metadata

    def test_fallback_chunking(self):
        """Test fallback to text chunking for unsupported files."""
        chunker = ASTChunker()

        # Test with unsupported file type
        code = "Some random text\n" * 100
        chunks = chunker.chunk_code(code, "test.txt")

        # Should fall back to text chunking
        assert len(chunks) > 0
        assert all(c.metadata.get('fallback') == True for c in chunks)

    def test_empty_code(self):
        """Test handling of empty code."""
        chunker = ASTChunker()
        chunks = chunker.chunk_code("", "test.py")

        # Should return empty list or fallback chunks
        assert isinstance(chunks, list)

    def test_large_function(self):
        """Test handling of very large functions."""
        large_code = "def large_function():\n" + "    x = 1\n" * 2000

        chunker = ASTChunker(max_chunk_size=1000)
        chunks = chunker.chunk_code(large_code, "test.py")

        # Should either skip or handle large function gracefully
        assert isinstance(chunks, list)

    def test_syntax_error_fallback(self):
        """Test fallback when code has syntax errors."""
        invalid_code = "def invalid(\n    print('missing closing paren'"

        chunker = ASTChunker()
        # Should not crash, should fall back to text chunking
        chunks = chunker.chunk_code(invalid_code, "test.py")

        assert isinstance(chunks, list)


class TestParserSupport:
    """Test parser support detection."""

    def test_supported_extensions(self):
        """Test supported file extensions."""
        assert is_supported("test.py")
        assert is_supported("test.js")
        assert is_supported("test.jsx")
        assert is_supported("test.ts")
        assert is_supported("test.tsx")

    def test_unsupported_extensions(self):
        """Test unsupported file extensions."""
        assert not is_supported("test.txt")
        assert not is_supported("test.md")
        assert not is_supported("test.json")


class TestQualityImprovement:
    """Test quality improvements from AST chunking."""

    def test_no_broken_functions(self):
        """Test that functions are never broken across chunks."""
        chunker = ASTChunker()
        chunks = chunker.chunk_code(PYTHON_CODE, "test.py")

        for chunk in chunks:
            # Each chunk should be syntactically valid
            # (we can't fully validate without parsing again,
            # but we can check basic structure)
            if chunk.chunk_type == 'function':
                assert 'def ' in chunk.content
                # Function should have complete signature and body

    def test_semantic_units(self):
        """Test that chunks represent semantic units."""
        chunker = ASTChunker()
        chunks = chunker.chunk_code(PYTHON_CODE, "test.py")

        # Each chunk should be a complete unit
        for chunk in chunks:
            if chunk.chunk_type in ['function', 'class', 'method']:
                # Should have a name
                assert chunk.metadata['name']
                # Should have valid line range
                assert chunk.start_line <= chunk.end_line


@pytest.mark.integration
class TestIntegration:
    """Integration tests."""

    def test_chunk_real_file(self, tmp_path):
        """Test chunking a real file."""
        # Create a temporary Python file
        test_file = tmp_path / "test.py"
        test_file.write_text(PYTHON_CODE)

        # Chunk it
        chunks = chunk_file(str(test_file))

        assert len(chunks) > 0
        assert all(isinstance(c, CodeChunk) for c in chunks)

    def test_quality_vs_text_chunking(self):
        """
        Test quality improvement vs text-based chunking.

        AST chunking should produce more meaningful chunks.
        """
        ast_chunker = ASTChunker()
        ast_chunks = ast_chunker.chunk_code(PYTHON_CODE, "test.py")

        # AST chunks should be semantic units
        ast_has_complete_functions = any(
            'def ' in chunk.content and chunk.chunk_type == 'function'
            for chunk in ast_chunks
        )
        assert ast_has_complete_functions, "AST chunking should produce complete functions"

        # Text chunking would arbitrarily split at character boundaries
        # and likely break function definitions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
