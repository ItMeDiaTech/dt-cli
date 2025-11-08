"""
Knowledge graph builder for code relationships.

Creates a graph database to understand:
- Classes -> Functions -> Imports
- Call relationships
- Dependency chains
- Inheritance hierarchies
"""

import networkx as nx
from typing import List, Dict, Any, Set, Optional
from pathlib import Path
import json
import logging

from ..entity_extraction.extractor import EntityExtractor, CodeEntity

logger = logging.getLogger(__name__)


class CodeKnowledgeGraph:
    """
    Builds and queries a knowledge graph of code relationships.
    """

    def __init__(self):
        """Initialize knowledge graph."""
        self.graph = nx.DiGraph()
        self.entity_extractor = EntityExtractor()

    def build_from_directory(
        self,
        root_path: Path,
        ignore_dirs: Set[str] = None
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from directory.

        Args:
            root_path: Root directory path
            ignore_dirs: Directories to ignore

        Returns:
            Build statistics
        """
        logger.info(f"Building knowledge graph from {root_path}")

        # Extract entities
        entities = self.entity_extractor.extract_from_directory(
            root_path,
            ignore_dirs
        )

        # Add entities as nodes
        for entity in entities:
            self._add_entity_node(entity)

        # Build relationships
        self._build_relationships(entities)

        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_count': len(entities)
        }

        logger.info(f"Knowledge graph built: {stats}")
        return stats

    def _add_entity_node(self, entity: CodeEntity):
        """
        Add entity as node to graph.

        Args:
            entity: Code entity
        """
        node_id = self._get_node_id(entity)

        self.graph.add_node(
            node_id,
            name=entity.name,
            type=entity.entity_type,
            file_path=entity.file_path,
            line_number=entity.line_number,
            metadata=entity.metadata
        )

    def _get_node_id(self, entity: CodeEntity) -> str:
        """
        Get unique node ID for entity.

        Args:
            entity: Code entity

        Returns:
            Unique node ID
        """
        return f"{entity.file_path}:{entity.entity_type}:{entity.name}"

    def _build_relationships(self, entities: List[CodeEntity]):
        """
        Build relationships between entities.

        Args:
            entities: List of entities
        """
        # Build lookup maps
        classes_by_file = {}
        functions_by_file = {}
        imports_by_file = {}

        for entity in entities:
            file_path = entity.file_path

            if entity.entity_type == 'class':
                if file_path not in classes_by_file:
                    classes_by_file[file_path] = []
                classes_by_file[file_path].append(entity)

            elif entity.entity_type == 'function':
                if file_path not in functions_by_file:
                    functions_by_file[file_path] = []
                functions_by_file[file_path].append(entity)

            elif entity.entity_type == 'import':
                if file_path not in imports_by_file:
                    imports_by_file[file_path] = []
                imports_by_file[file_path].append(entity)

        # Build relationships within files
        for file_path in set(list(classes_by_file.keys()) + list(functions_by_file.keys())):
            classes = classes_by_file.get(file_path, [])
            functions = functions_by_file.get(file_path, [])
            imports = imports_by_file.get(file_path, [])

            # File contains classes/functions
            for cls in classes:
                for func in functions:
                    # Function likely belongs to class if nearby
                    if abs(func.line_number - cls.line_number) < 100:
                        self._add_relationship(
                            cls,
                            func,
                            'contains',
                            {'context': 'class_method'}
                        )

            # Imports used by file
            for imp in imports:
                for cls in classes:
                    self._add_relationship(
                        cls,
                        imp,
                        'imports',
                        {'context': 'dependency'}
                    )

                for func in functions:
                    self._add_relationship(
                        func,
                        imp,
                        'imports',
                        {'context': 'dependency'}
                    )

        # Build inheritance relationships
        for entity in entities:
            if entity.entity_type == 'class' and 'bases' in entity.metadata:
                for base_class in entity.metadata['bases']:
                    # Find base class entity
                    base_entity = self._find_entity_by_name(base_class, 'class')
                    if base_entity:
                        self._add_relationship(
                            entity,
                            base_entity,
                            'inherits',
                            {'context': 'inheritance'}
                        )

    def _add_relationship(
        self,
        from_entity: CodeEntity,
        to_entity: CodeEntity,
        relationship_type: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add relationship edge between entities.

        Args:
            from_entity: Source entity
            to_entity: Target entity
            relationship_type: Type of relationship
            metadata: Optional metadata
        """
        from_id = self._get_node_id(from_entity)
        to_id = self._get_node_id(to_entity)

        self.graph.add_edge(
            from_id,
            to_id,
            type=relationship_type,
            metadata=metadata or {}
        )

    def _find_entity_by_name(
        self,
        name: str,
        entity_type: str
    ) -> Optional[CodeEntity]:
        """
        Find entity by name and type.

        Args:
            name: Entity name
            entity_type: Entity type

        Returns:
            Entity or None
        """
        for node_id, data in self.graph.nodes(data=True):
            if data.get('name') == name and data.get('type') == entity_type:
                # Reconstruct entity
                return CodeEntity(
                    name=data['name'],
                    entity_type=data['type'],
                    file_path=data['file_path'],
                    line_number=data['line_number'],
                    metadata=data['metadata']
                )

        return None

    def find_related_entities(
        self,
        entity_name: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to given entity.

        Args:
            entity_name: Entity name to search for
            max_depth: Maximum relationship depth
            relationship_types: Filter by relationship types

        Returns:
            List of related entities with relationships
        """
        # Find starting nodes
        start_nodes = [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get('name') == entity_name
        ]

        if not start_nodes:
            return []

        related = []

        for start_node in start_nodes:
            # Get neighbors within max_depth
            for target in nx.single_source_shortest_path_length(
                self.graph,
                start_node,
                cutoff=max_depth
            ):
                if target == start_node:
                    continue

                # Get relationship path
                try:
                    path = nx.shortest_path(self.graph, start_node, target)

                    # Check relationship types
                    valid_path = True
                    if relationship_types:
                        for i in range(len(path) - 1):
                            edge_data = self.graph.edges[path[i], path[i + 1]]
                            if edge_data.get('type') not in relationship_types:
                                valid_path = False
                                break

                    if valid_path:
                        node_data = self.graph.nodes[target]
                        related.append({
                            'name': node_data.get('name'),
                            'type': node_data.get('type'),
                            'file_path': node_data.get('file_path'),
                            'line_number': node_data.get('line_number'),
                            'relationship_path': [
                                self.graph.edges[path[i], path[i + 1]].get('type')
                                for i in range(len(path) - 1)
                            ],
                            'depth': len(path) - 1
                        })

                except nx.NetworkXNoPath:
                    continue

        return related

    def find_dependencies(
        self,
        entity_name: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find all dependencies for an entity.

        Args:
            entity_name: Entity name
            max_depth: Maximum dependency depth

        Returns:
            List of dependencies
        """
        return self.find_related_entities(
            entity_name,
            max_depth=max_depth,
            relationship_types=['imports', 'inherits']
        )

    def find_dependents(
        self,
        entity_name: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find all entities that depend on given entity.

        Args:
            entity_name: Entity name
            max_depth: Maximum dependency depth

        Returns:
            List of dependent entities
        """
        # Find nodes for entity
        target_nodes = [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get('name') == entity_name
        ]

        if not target_nodes:
            return []

        dependents = []

        for target_node in target_nodes:
            # Find all nodes that have paths TO this node
            for source in self.graph.nodes():
                if source == target_node:
                    continue

                try:
                    path = nx.shortest_path(self.graph, source, target_node)

                    if len(path) - 1 <= max_depth:
                        node_data = self.graph.nodes[source]
                        dependents.append({
                            'name': node_data.get('name'),
                            'type': node_data.get('type'),
                            'file_path': node_data.get('file_path'),
                            'line_number': node_data.get('line_number'),
                            'relationship_path': [
                                self.graph.edges[path[i], path[i + 1]].get('type')
                                for i in range(len(path) - 1)
                            ],
                            'depth': len(path) - 1
                        })

                except nx.NetworkXNoPath:
                    continue

        return dependents

    def get_entity_context(
        self,
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive context for an entity.

        Args:
            entity_name: Entity name

        Returns:
            Entity context including relationships
        """
        # Find entity node
        entity_node = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get('name') == entity_name:
                entity_node = node_id
                break

        if not entity_node:
            return {}

        node_data = self.graph.nodes[entity_node]

        # Get immediate neighbors
        predecessors = list(self.graph.predecessors(entity_node))
        successors = list(self.graph.successors(entity_node))

        return {
            'entity': {
                'name': node_data.get('name'),
                'type': node_data.get('type'),
                'file_path': node_data.get('file_path'),
                'line_number': node_data.get('line_number'),
                'metadata': node_data.get('metadata', {})
            },
            'used_by': [
                {
                    'name': self.graph.nodes[pred].get('name'),
                    'type': self.graph.nodes[pred].get('type'),
                    'relationship': self.graph.edges[pred, entity_node].get('type')
                }
                for pred in predecessors
            ],
            'uses': [
                {
                    'name': self.graph.nodes[succ].get('name'),
                    'type': self.graph.nodes[succ].get('type'),
                    'relationship': self.graph.edges[entity_node, succ].get('type')
                }
                for succ in successors
            ],
            'related_entities': self.find_related_entities(entity_name, max_depth=2)
        }

    def export_graph(self, output_path: Path):
        """
        Export graph to file.

        Args:
            output_path: Output file path
        """
        # Convert to JSON-serializable format
        graph_data = {
            'nodes': [
                {
                    'id': node_id,
                    **data
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    **data
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }

        output_path.write_text(json.dumps(graph_data, indent=2))
        logger.info(f"Graph exported to {output_path}")

    def import_graph(self, input_path: Path):
        """
        Import graph from file.

        Args:
            input_path: Input file path
        """
        graph_data = json.loads(input_path.read_text())

        # Clear existing graph
        self.graph.clear()

        # Add nodes
        for node in graph_data['nodes']:
            node_id = node.pop('id')
            self.graph.add_node(node_id, **node)

        # Add edges
        for edge in graph_data['edges']:
            source = edge.pop('source')
            target = edge.pop('target')
            self.graph.add_edge(source, target, **edge)

        logger.info(f"Graph imported from {input_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Statistics dictionary
        """
        # Count entity types
        entity_types = {}
        for node_id, data in self.graph.nodes(data=True):
            entity_type = data.get('type', 'unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        # Count relationship types
        relationship_types = {}
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('type', 'unknown')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_types': entity_types,
            'relationship_types': relationship_types,
            'is_connected': nx.is_weakly_connected(self.graph),
            'number_of_components': nx.number_weakly_connected_components(self.graph)
        }
