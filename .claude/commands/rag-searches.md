---
description: List and manage saved searches
---

# Saved Searches Command

List all saved searches with filtering options.

## Usage

**Filter**: {{args}} (optional tag filter)

## Instructions

List all saved searches, optionally filtered by tag.

## Implementation

```python
import httpx
import json

filter_tags = "{{args}}"
if filter_tags == "{{args}}":
    filter_tags = None

try:
    # Get saved searches
    params = {}
    if filter_tags:
        params['tags'] = filter_tags

    response = httpx.get(
        "http://127.0.0.1:8765/searches",
        params=params,
        timeout=10.0
    )

    if response.status_code == 200:
        result = response.json()
        searches = result.get("searches", [])
        total = result.get("total", 0)

        if total == 0:
            print("📭 No saved searches found.")
            print("\n[i] Save a search with: /rag-save <name> | <query>")
        else:
            print(f"[#] Saved Searches ({total}):")
            print()

            for search in searches:
                name = search.get("name", "")
                query = search.get("query", "")
                search_id = search.get("id", "")
                tags = search.get("tags", [])
                use_count = search.get("use_count", 0)

                print(f"  • {name}")
                print(f"    Query: {query}")

                if tags:
                    print(f"    Tags: {', '.join(tags)}")

                print(f"    Used: {use_count} times")
                print(f"    Execute: /rag-exec {search_id}")
                print()

    else:
        print(f"[X] Server error: {response.status_code}")

except Exception as e:
    print(f"[X] Error: {e}")
    print("Make sure the MCP server is running on port 8000.")
```

Execute this code to list saved searches.
