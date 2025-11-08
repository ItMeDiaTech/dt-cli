---
description: Query knowledge graph for code relationships
---

# Knowledge Graph Query Command

Query the knowledge graph to understand code relationships and dependencies.

## Usage

**Entity Name**: {{args}}

## Instructions

Find an entity (class, function, etc.) in the knowledge graph and show its relationships.

## Implementation

```python
import httpx
import json

entity = "{{args}}"

if not entity or entity == "{{args}}":
    print("Usage: /rag-graph <entity_name>")
    print("\nExamples:")
    print("  /rag-graph UserManager")
    print("  /rag-graph authenticate_user")
else:
    try:
        # Get entity context
        response = httpx.get(
            f"http://127.0.0.1:8765/knowledge-graph/entity/{entity}",
            timeout=10.0
        )

        if response.status_code == 200:
            context = response.json()

            entity_info = context.get("entity", {})
            used_by = context.get("used_by", [])
            uses = context.get("uses", [])
            related = context.get("related_entities", [])

            print(f"\n[PKG] Entity: {entity_info.get('name', entity)}")
            print(f"   Type: {entity_info.get('type', 'unknown')}")
            print(f"   File: {entity_info.get('file_path', 'unknown')}")

            line_num = entity_info.get('line_number')
            if line_num:
                print(f"   Line: {line_num}")

            print()

            # Used by (dependencies)
            if used_by:
                print(f"📥 USED BY ({len(used_by)}):")
                for dep in used_by[:5]:
                    name = dep.get('name', '')
                    dep_type = dep.get('type', '')
                    rel = dep.get('relationship', '')
                    print(f"   • {name} ({dep_type}) - {rel}")
                if len(used_by) > 5:
                    print(f"   ... and {len(used_by) - 5} more")
                print()

            # Uses (what this entity depends on)
            if uses:
                print(f"📤 USES ({len(uses)}):")
                for dep in uses[:5]:
                    name = dep.get('name', '')
                    dep_type = dep.get('type', '')
                    rel = dep.get('relationship', '')
                    print(f"   • {name} ({dep_type}) - {rel}")
                if len(uses) > 5:
                    print(f"   ... and {len(uses) - 5} more")
                print()

            # Related entities
            if related:
                print(f"[LINK] RELATED ENTITIES ({len(related)}):")
                for rel_entity in related[:5]:
                    name = rel_entity.get('name', '')
                    rel_type = rel_entity.get('type', '')
                    depth = rel_entity.get('depth', 0)
                    print(f"   • {name} ({rel_type}) - depth {depth}")
                if len(related) > 5:
                    print(f"   ... and {len(related) - 5} more")

        elif response.status_code == 404:
            print(f"[X] Entity not found: {entity}")
            print("\n[i] Make sure the knowledge graph is built.")
            print("   Re-index with: /rag-index")
        else:
            print(f"[X] Server error: {response.status_code}")

    except Exception as e:
        print(f"[X] Error: {e}")
        print("Make sure the MCP server is running on port 8000.")
```

Execute this code to query the knowledge graph.
