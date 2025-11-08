#!/bin/bash
# SessionStart Hook for RAG-MAF Plugin
# Automatically initializes the RAG and MAF systems when Claude Code session starts

echo "[*] Initializing RAG-MAF Plugin..."

# Get the plugin directory
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Check if MCP server is already running
MCP_PID=$(pgrep -f "mcp_server/server.py")

if [ -z "$MCP_PID" ]; then
    echo "[+] Starting MCP Server..."

    # Start MCP server in background
    cd "$PLUGIN_DIR"
    python3 -m src.mcp_server.server > /tmp/rag-maf-mcp.log 2>&1 &

    # Wait for server to start
    sleep 2

    # Check if server started successfully
    if pgrep -f "mcp_server/server.py" > /dev/null; then
        echo "[OK] MCP Server started successfully"
    else
        echo "[WARNING] MCP Server failed to start. Check /tmp/rag-maf-mcp.log for details"
    fi
else
    echo "[OK] MCP Server already running (PID: $MCP_PID)"
fi

# Check if codebase is indexed
if [ ! -d "$PLUGIN_DIR/.rag_data" ]; then
    echo "[#] First run detected. Indexing codebase..."
    echo "    (This may take a few minutes)"

    # Trigger indexing in background
    python3 -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR/src')
from rag import QueryEngine
engine = QueryEngine()
engine.index_codebase('.')
print('[OK] Codebase indexed successfully')
" &

    echo "[...] Indexing in progress (running in background)..."
else
    echo "[OK] Codebase already indexed"
fi

echo ""
echo "[**] RAG-MAF Plugin ready!"
echo ""
echo "Available commands:"
echo "  /rag-query <query>  - Query the RAG system"
echo "  /rag-index          - Re-index the codebase"
echo "  /rag-status         - Check system status"
echo ""
echo "The plugin will automatically provide context to Claude as needed."
echo ""
