---
description: Query the RAG system for relevant code and documentation
---

# RAG Query Command

Query the local RAG (Retrieval-Augmented Generation) system to find relevant code and documentation.

## Usage

The user is asking you to query the RAG system. Use the MCP server to perform the query.

**User Query**: {{args}}

## Instructions

1. Parse the user's query from the arguments
2. Use the RAG system via HTTP request to the MCP server at http://127.0.0.1:8765
3. Execute a POST request to `/rag/query` with the query
4. Format and display the results to the user

## Implementation

```python
import httpx
import json

query = "{{args}}"

if not query or query == "{{args}}":
    print("Usage: /rag-query <your query>")
    print("Example: /rag-query how does authentication work?")
else:
    try:
        response = httpx.post(
            "http://127.0.0.1:8765/rag/query",
            json={"query": query},
            timeout=30.0
        )

        if response.status_code == 200:
            result = response.json()

            if result.get("success"):
                results_list = result.get("results", [])

                print(f"\n🔍 Found {len(results_list)} relevant results:\n")

                for i, item in enumerate(results_list[:5], 1):
                    metadata = item.get("metadata", {})
                    file_path = metadata.get("file_path", "unknown")
                    score = 1 - item.get("distance", 1)

                    print(f"{i}. {file_path} (relevance: {score:.2%})")
                    print(f"   {item.get('text', '')[:150]}...")
                    print()
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"MCP Server error: {response.status_code}")

    except Exception as e:
        print(f"Error querying RAG system: {e}")
        print("Make sure the MCP server is running.")
```

Execute this Python code to query the RAG system and display results.
