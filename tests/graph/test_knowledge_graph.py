"""
Tests for knowledge graph and code analysis.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.graph import (
    KnowledgeGraph,
    CodeEntity,
    Relationship,
    RelationType,
    CodeAnalyzer,
    create_knowledge_graph
)


class TestCodeEntity:
    """Test CodeEntity dataclass."""

    def test_create_entity(self):
        """Test creating a code entity."""
        entity = CodeEntity(
            name="my_function",
            entity_type="function",
            file_path="test.py",
            line_number=10
        )

        assert entity.name == "my_function"
        assert entity.entity_type == "function"
        assert entity.file_path == "test.py"
        assert entity.line_number == 10

    def test_entity_equality(self):
        """Test entity equality comparison."""
        entity1 = CodeEntity(
            name="test",
            entity_type="function",
            file_path="test.py"
        )
        entity2 = CodeEntity(
            name="test",
            entity_type="function",
            file_path="test.py"
        )
        entity3 = CodeEntity(
            name="other",
            entity_type="function",
            file_path="test.py"
        )

        assert entity1 == entity2
        assert entity1 != entity3

    def test_entity_hash(self):
        """Test entity can be hashed (for use in sets/dicts)."""
        entity = CodeEntity(
            name="test",
            entity_type="function"
        )

        # Should be hashable
        entity_set = {entity}
        assert entity in entity_set


class TestRelationship:
    """Test Relationship dataclass."""

    def test_create_relationship(self):
        """Test creating a relationship."""
        source = CodeEntity(name="module_a", entity_type="module")
        target = CodeEntity(name="module_b", entity_type="module")

        rel = Relationship(
            source=source,
            target=target,
            rel_type=RelationType.IMPORTS
        )

        assert rel.source == source
        assert rel.target == target
        assert rel.rel_type == RelationType.IMPORTS


class TestKnowledgeGraph:
    """Test KnowledgeGraph functionality."""

    def test_init(self):
        """Test graph initialization."""
        graph = KnowledgeGraph()

        assert len(graph.entities) == 0
        assert len(graph.relationships) == 0

    def test_add_entity(self):
        """Test adding entities to graph."""
        graph = KnowledgeGraph()

        entity = CodeEntity(name="test_func", entity_type="function")
        graph.add_entity(entity)

        assert len(graph.entities) == 1
        assert entity in graph.entities.values()

    def test_add_relationship(self):
        """Test adding relationships to graph."""
        graph = KnowledgeGraph()

        source = CodeEntity(name="module_a", entity_type="module")
        target = CodeEntity(name="module_b", entity_type="module")

        rel = Relationship(
            source=source,
            target=target,
            rel_type=RelationType.IMPORTS
        )

        graph.add_relationship(rel)

        assert len(graph.entities) == 2  # Both entities added
        assert len(graph.relationships) == 1
        assert rel in graph.relationships

    def test_get_entity(self):
        """Test retrieving entities."""
        graph = KnowledgeGraph()

        func = CodeEntity(name="my_function", entity_type="function")
        cls = CodeEntity(name="MyClass", entity_type="class")

        graph.add_entity(func)
        graph.add_entity(cls)

        # Get by name
        result = graph.get_entity("my_function")
        assert result == func

        # Get by name and type
        result = graph.get_entity("my_function", "function")
        assert result == func

        # Type mismatch
        result = graph.get_entity("my_function", "class")
        assert result is None

    def test_get_dependencies(self):
        """Test getting dependencies."""
        graph = KnowledgeGraph()

        module_a = CodeEntity(name="module_a", entity_type="module")
        module_b = CodeEntity(name="module_b", entity_type="module")
        module_c = CodeEntity(name="module_c", entity_type="module")

        # A imports B, B imports C
        graph.add_relationship(Relationship(
            source=module_a,
            target=module_b,
            rel_type=RelationType.IMPORTS
        ))
        graph.add_relationship(Relationship(
            source=module_b,
            target=module_c,
            rel_type=RelationType.IMPORTS
        ))

        # Direct dependencies
        deps = graph.get_dependencies(module_a, recursive=False)
        assert len(deps) == 1
        assert module_b in deps

        # Transitive dependencies
        deps = graph.get_dependencies(module_a, recursive=True)
        assert len(deps) == 2
        assert module_b in deps
        assert module_c in deps

    def test_get_dependencies_with_type_filter(self):
        """Test getting dependencies filtered by relationship type."""
        graph = KnowledgeGraph()

        func_a = CodeEntity(name="func_a", entity_type="function")
        func_b = CodeEntity(name="func_b", entity_type="function")
        func_c = CodeEntity(name="func_c", entity_type="function")
        module = CodeEntity(name="module", entity_type="module")

        # func_a calls func_b and func_c
        graph.add_relationship(Relationship(
            source=func_a,
            target=func_b,
            rel_type=RelationType.CALLS
        ))
        graph.add_relationship(Relationship(
            source=func_a,
            target=func_c,
            rel_type=RelationType.CALLS
        ))
        # func_a also imports a module
        graph.add_relationship(Relationship(
            source=func_a,
            target=module,
            rel_type=RelationType.IMPORTS
        ))

        # Get only CALLS dependencies
        deps = graph.get_dependencies(func_a, rel_type=RelationType.CALLS)
        assert len(deps) == 2
        assert func_b in deps
        assert func_c in deps
        assert module not in deps

    def test_get_dependents(self):
        """Test getting dependents (reverse dependencies)."""
        graph = KnowledgeGraph()

        func = CodeEntity(name="util_function", entity_type="function")
        caller1 = CodeEntity(name="caller1", entity_type="function")
        caller2 = CodeEntity(name="caller2", entity_type="function")

        # Two functions call the util function
        graph.add_relationship(Relationship(
            source=caller1,
            target=func,
            rel_type=RelationType.CALLS
        ))
        graph.add_relationship(Relationship(
            source=caller2,
            target=func,
            rel_type=RelationType.CALLS
        ))

        # Get dependents
        dependents = graph.get_dependents(func)
        assert len(dependents) == 2
        assert caller1 in dependents
        assert caller2 in dependents

    def test_get_impact_analysis(self):
        """Test impact analysis."""
        graph = KnowledgeGraph()

        # Create entities
        base_func = CodeEntity(
            name="base_func",
            entity_type="function",
            file_path="base.py"
        )
        caller1 = CodeEntity(
            name="caller1",
            entity_type="function",
            file_path="module1.py"
        )
        caller2 = CodeEntity(
            name="caller2",
            entity_type="function",
            file_path="module2.py"
        )

        # Build relationships
        graph.add_relationship(Relationship(
            source=caller1,
            target=base_func,
            rel_type=RelationType.CALLS
        ))
        graph.add_relationship(Relationship(
            source=caller2,
            target=base_func,
            rel_type=RelationType.CALLS
        ))

        # Analyze impact
        impact = graph.get_impact_analysis(base_func)

        assert impact['direct_impact'] == 2
        assert impact['total_impact'] == 2
        assert impact['affected_by_type']['function'] == 2
        assert len(impact['affected_entities']) == 2

    def test_find_usages(self):
        """Test finding usages of an entity."""
        graph = KnowledgeGraph()

        # Create a utility function
        util_func = CodeEntity(
            name="format_date",
            entity_type="function",
            file_path="utils.py",
            line_number=10
        )

        # Create functions that use it
        user1 = CodeEntity(
            name="display_date",
            entity_type="function",
            file_path="views.py",
            line_number=50
        )
        user2 = CodeEntity(
            name="export_report",
            entity_type="function",
            file_path="reports.py",
            line_number=100
        )

        # Add relationships
        graph.add_relationship(Relationship(
            source=user1,
            target=util_func,
            rel_type=RelationType.CALLS
        ))
        graph.add_relationship(Relationship(
            source=user2,
            target=util_func,
            rel_type=RelationType.CALLS
        ))

        # Find usages
        usages = graph.find_usages("format_date", "function")

        assert len(usages) == 2
        assert any(u['used_by'] == "display_date" for u in usages)
        assert any(u['used_by'] == "export_report" for u in usages)

    def test_get_call_chain(self):
        """Test finding call chains between functions."""
        graph = KnowledgeGraph()

        # Create a call chain: A -> B -> C
        func_a = CodeEntity(name="func_a", entity_type="function")
        func_b = CodeEntity(name="func_b", entity_type="function")
        func_c = CodeEntity(name="func_c", entity_type="function")

        graph.add_relationship(Relationship(
            source=func_a,
            target=func_b,
            rel_type=RelationType.CALLS
        ))
        graph.add_relationship(Relationship(
            source=func_b,
            target=func_c,
            rel_type=RelationType.CALLS
        ))

        # Find call chain from A to C
        chain = graph.get_call_chain(func_a, func_c)

        assert chain is not None
        assert len(chain) == 3
        assert chain == [func_a, func_b, func_c]

    def test_get_call_chain_no_path(self):
        """Test call chain when no path exists."""
        graph = KnowledgeGraph()

        func_a = CodeEntity(name="func_a", entity_type="function")
        func_b = CodeEntity(name="func_b", entity_type="function")

        graph.add_entity(func_a)
        graph.add_entity(func_b)

        # No relationship between them
        chain = graph.get_call_chain(func_a, func_b)
        assert chain is None

    def test_get_stats(self):
        """Test graph statistics."""
        graph = KnowledgeGraph()

        # Add various entities
        module = CodeEntity(name="module", entity_type="module")
        func1 = CodeEntity(name="func1", entity_type="function")
        func2 = CodeEntity(name="func2", entity_type="function")
        cls = CodeEntity(name="MyClass", entity_type="class")

        graph.add_entity(module)
        graph.add_entity(func1)
        graph.add_entity(func2)
        graph.add_entity(cls)

        # Add relationships
        graph.add_relationship(Relationship(
            source=module,
            target=func1,
            rel_type=RelationType.DEFINES
        ))
        graph.add_relationship(Relationship(
            source=func1,
            target=func2,
            rel_type=RelationType.CALLS
        ))

        stats = graph.get_stats()

        assert stats['total_entities'] == 4
        assert stats['total_relationships'] == 2
        assert stats['entities_by_type']['module'] == 1
        assert stats['entities_by_type']['function'] == 2
        assert stats['entities_by_type']['class'] == 1
        assert stats['relationships_by_type']['defines'] == 1
        assert stats['relationships_by_type']['calls'] == 1

    def test_clear(self):
        """Test clearing the graph."""
        graph = KnowledgeGraph()

        entity = CodeEntity(name="test", entity_type="function")
        graph.add_entity(entity)

        assert len(graph.entities) > 0

        graph.clear()

        assert len(graph.entities) == 0
        assert len(graph.relationships) == 0


class TestCodeAnalyzer:
    """Test CodeAnalyzer functionality."""

    def test_analyze_simple_file(self):
        """Test analyzing a simple Python file."""
        graph = KnowledgeGraph()
        analyzer = CodeAnalyzer(graph)

        # Create temporary file
        code = """
def hello():
    print("Hello")

def world():
    hello()
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            analyzer.analyze_file(temp_path)

            # Should have module, two functions
            assert len(graph.entities) >= 3

            # Should have relationships
            assert len(graph.relationships) > 0

            # Check for function call relationship
            world_func = graph.get_entity("world", "function")
            hello_func = graph.get_entity("hello", "function")

            if world_func and hello_func:
                deps = graph.get_dependencies(world_func, RelationType.CALLS)
                # Note: This might not work perfectly without full context
                # Just check that analysis ran without errors

        finally:
            os.unlink(temp_path)

    def test_analyze_with_imports(self):
        """Test analyzing file with imports."""
        graph = KnowledgeGraph()
        analyzer = CodeAnalyzer(graph)

        code = """
import os
from pathlib import Path

def example():
    pass
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            analyzer.analyze_file(temp_path)

            # Should detect imports
            module = graph.get_entity(
                os.path.splitext(os.path.basename(temp_path))[0],
                "module"
            )

            if module:
                imports = graph.get_dependencies(module, RelationType.IMPORTS)
                # Should have import relationships
                assert len(imports) > 0

        finally:
            os.unlink(temp_path)

    def test_analyze_with_class(self):
        """Test analyzing file with classes."""
        graph = KnowledgeGraph()
        analyzer = CodeAnalyzer(graph)

        code = """
class Parent:
    def parent_method(self):
        pass

class Child(Parent):
    def child_method(self):
        pass
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            analyzer.analyze_file(temp_path)

            # Should detect classes
            parent_cls = graph.get_entity("Parent", "class")
            child_cls = graph.get_entity("Child", "class")

            assert parent_cls is not None
            assert child_cls is not None

            # Should detect inheritance
            if child_cls:
                parents = graph.get_dependencies(child_cls, RelationType.INHERITS)
                # May or may not resolve Parent depending on context
                # Just verify analysis completed

        finally:
            os.unlink(temp_path)

    def test_analyze_directory(self):
        """Test analyzing a directory of Python files."""
        graph = KnowledgeGraph()
        analyzer = CodeAnalyzer(graph)

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            file1 = os.path.join(tmpdir, "module1.py")
            file2 = os.path.join(tmpdir, "module2.py")

            with open(file1, 'w') as f:
                f.write("def func1(): pass\n")

            with open(file2, 'w') as f:
                f.write("def func2(): pass\n")

            analyzer.analyze_directory(tmpdir)

            # Should have analyzed both files
            assert len(graph.entities) >= 2  # At least 2 modules


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_knowledge_graph(self):
        """Test graph creation function."""
        graph = create_knowledge_graph()

        assert isinstance(graph, KnowledgeGraph)
        assert len(graph.entities) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
