---
description: Index or re-index the codebase for RAG search
---

# RAG Index Command

Index or re-index the codebase to update the RAG system's knowledge.

## Usage

The user wants to index the codebase. This will:
- Scan all code files in the repository
- Generate embeddings for code chunks
- Store them in the local vector database

## Instructions

1. Trigger the indexing process via the MCP server
2. Show progress to the user
3. Display completion status

## Implementation

```python
import httpx
import json

print("[#] Starting codebase indexing...")
print("This may take a few minutes depending on codebase size.\n")

try:
    response = httpx.post(
        "http://127.0.0.1:8765/execute",
        json={
            "category": "rag",
            "tool_name": "rag_index",
            "parameters": {"root_path": "."}
        },
        timeout=300.0  # 5 minutes timeout
    )

    if response.status_code == 200:
        result = response.json()

        if result.get("success"):
            status = result.get("status", {})

            print("[OK] Indexing complete!\n")
            print(f"[=] Status:")
            print(f"   Indexed chunks: {status.get('indexed_chunks', 0)}")
            print(f"   Embedding model: {status.get('embedding_model', 'unknown')}")
            print(f"   Status: {status.get('status', 'unknown')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print(f"MCP Server error: {response.status_code}")

except Exception as e:
    print(f"Error indexing codebase: {e}")
    print("Make sure the MCP server is running.")
```

Execute this Python code to index the codebase.
