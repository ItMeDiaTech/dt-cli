---
description: Execute a saved search by ID or name
---

# Execute Saved Search Command

Execute a previously saved search query.

## Usage

**Search ID or Name**: {{args}}

## Instructions

Execute a saved search using its ID or name.

## Implementation

```python
import httpx
import json

search_ref = "{{args}}"

if not search_ref or search_ref == "{{args}}":
    print("Usage: /rag-exec <search_id_or_name>")
    print("\nList saved searches with: /rag-searches")
else:
    try:
        # First, try to find by name
        response = httpx.get(
            "http://127.0.0.1:8765/searches",
            timeout=10.0
        )

        search_id = None

        if response.status_code == 200:
            result = response.json()
            searches = result.get("searches", [])

            # Look for exact name match
            for search in searches:
                if search.get("name") == search_ref or search.get("id") == search_ref:
                    search_id = search.get("id")
                    break

        if not search_id:
            search_id = search_ref  # Assume it's an ID

        # Execute the search
        exec_response = httpx.post(
            f"http://127.0.0.1:8765/searches/{search_id}/execute",
            timeout=30.0
        )

        if exec_response.status_code == 200:
            result = exec_response.json()
            search_info = result.get("search", {})
            results = result.get("results", [])

            print(f"[?] Executing: {search_info.get('name', '')}")
            print(f"   Query: {search_info.get('query', '')}")
            print(f"   Results: {len(results)}")
            print()

            for i, item in enumerate(results, 1):
                meta = item.get("metadata", {})
                file_path = meta.get("file_path", "unknown")
                score = item.get("score", 0)

                print(f"{i}. {file_path}")
                print(f"   Relevance: {score:.2%}")

                content = item.get("content", "")
                if len(content) > 150:
                    content = content[:147] + "..."
                print(f"   {content}")
                print()

        elif exec_response.status_code == 404:
            print(f"[X] Search not found: {search_ref}")
            print("List available searches with: /rag-searches")
        else:
            print(f"[X] Server error: {exec_response.status_code}")

    except Exception as e:
        print(f"[X] Error: {e}")
        print("Make sure the MCP server is running on port 8000.")
```

Execute this code to run a saved search.
