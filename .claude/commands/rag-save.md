---
description: Save a search query for quick access later
---

# Save Search Command

Save a frequently used query for quick access.

## Usage

**Arguments**: {{args}}

## Instructions

Save a search query with optional name, description, and tags.

Format: `name | query | description | tags`

Example: `/rag-save auth | authentication flow | Find auth code | auth,security`

## Implementation

```python
import httpx
import json

args = "{{args}}"

if not args or args == "{{args}}":
    print("Usage: /rag-save <name> | <query> | [description] | [tags]")
    print("\nExamples:")
    print("  /rag-save auth | authentication flow")
    print("  /rag-save auth | authentication flow | Find auth code | auth,security")
    print("\nTo list saved searches: /rag-searches")
else:
    try:
        # Parse arguments
        parts = [p.strip() for p in args.split('|')]

        if len(parts) < 2:
            print("[X] Error: Need at least name and query")
            print("Format: <name> | <query> | [description] | [tags]")
        else:
            name = parts[0]
            query = parts[1]
            description = parts[2] if len(parts) > 2 else ""
            tags = parts[3].split(',') if len(parts) > 3 else []
            tags = [t.strip() for t in tags if t.strip()]

            # Save search
            response = httpx.post(
                "http://127.0.0.1:8765/searches",
                json={
                    "name": name,
                    "query": query,
                    "description": description,
                    "tags": tags,
                    "n_results": 5
                },
                timeout=10.0
            )

            if response.status_code == 200:
                result = response.json()
                search = result.get("search", {})

                print(f"[OK] Saved search '{name}'")
                print(f"   Query: {query}")
                if description:
                    print(f"   Description: {description}")
                if tags:
                    print(f"   Tags: {', '.join(tags)}")

                print(f"\n[i] Execute with: /rag-exec {search.get('id', '')}")

            else:
                print(f"[X] Server error: {response.status_code}")

    except Exception as e:
        print(f"[X] Error: {e}")
        print("Make sure the MCP server is running on port 8000.")
```

Execute this code to save a search query.
