---
description: Check the status of the RAG and MAF systems
---

# RAG Status Command

Check the current status of the RAG and Multi-Agent Framework systems.

## Usage

Display system status including:
- RAG system information
- MAF agent status
- MCP server status

## Implementation

```python
import httpx
import json

print("📊 RAG-MAF System Status\n")

try:
    response = httpx.get(
        "http://127.0.0.1:8765/status",
        timeout=10.0
    )

    if response.status_code == 200:
        status = response.json()

        # RAG Status
        rag = status.get("rag", {})
        print("🔍 RAG System:")
        print(f"   Indexed chunks: {rag.get('indexed_chunks', 0)}")
        print(f"   Embedding model: {rag.get('embedding_model', 'unknown')}")
        print(f"   Embedding dimension: {rag.get('embedding_dimension', 0)}")
        print(f"   Status: {rag.get('status', 'unknown')}")
        print()

        # MAF Status
        maf = status.get("maf", {})
        print("🤖 Multi-Agent Framework:")
        agents = maf.get('agents', [])
        print(f"   Available agents: {', '.join(agents)}")
        print(f"   Active contexts: {maf.get('active_contexts', 0)}")
        print(f"   RAG enabled: {'Yes' if maf.get('rag_enabled') else 'No'}")
        print()

        # Server Status
        server = status.get("server", {})
        print("📡 MCP Server:")
        print(f"   Host: {server.get('host', 'unknown')}")
        print(f"   Port: {server.get('port', 'unknown')}")
        print(f"   Status: {server.get('status', 'unknown')}")

    else:
        print(f"❌ MCP Server error: {response.status_code}")

except httpx.ConnectError:
    print("❌ Cannot connect to MCP server")
    print("   The server may not be running.")
    print("   Try restarting your Claude Code session.")
except Exception as e:
    print(f"❌ Error: {e}")
```

Execute this Python code to display system status.
