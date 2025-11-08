#!/bin/bash
# Installation script for RAG-MAF Plugin for Claude Code

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   RAG-MAF Plugin Installer for Claude Code                    ║"
echo "║   Local RAG + Multi-Agent Framework                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where the script is located
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check Python version
echo "[?] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "[X] Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "[OK] Found Python $PYTHON_VERSION"

# Check if Python version is >= 3.8
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "[OK] Python version is compatible"
else
    echo "[X] Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment (optional but recommended)
echo ""
echo "[~] Setting up Python virtual environment..."
if [ ! -d "$PLUGIN_DIR/venv" ]; then
    python3 -m venv "$PLUGIN_DIR/venv"
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

# Activate virtual environment
source "$PLUGIN_DIR/venv/bin/activate"

# Upgrade pip
echo ""
echo "[PKG] Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo ""
echo "[PKG] Installing dependencies..."
echo "   (This may take several minutes on first install)"
pip install -r "$PLUGIN_DIR/requirements.txt" --quiet

if [ $? -eq 0 ]; then
    echo "[OK] Dependencies installed successfully"
else
    echo "[X] Failed to install dependencies"
    exit 1
fi

# Download embedding model
echo ""
echo "[AI] Downloading embedding model (all-MiniLM-L6-v2)..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('   Downloading model...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('   [OK] Model downloaded successfully')
"

# Create necessary directories
echo ""
echo "[FOLDER] Creating directories..."
mkdir -p "$PLUGIN_DIR/.rag_data"
mkdir -p "$PLUGIN_DIR/logs"
echo "[OK] Directories created"

# Make hook executable
echo ""
echo "[~] Configuring Claude Code integration..."
chmod +x "$PLUGIN_DIR/.claude/hooks/SessionStart.sh"
echo "[OK] Hook configured"

# Test MCP server
echo ""
echo "🧪 Testing MCP server..."
python3 -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR/src')
from mcp_server import MCPServer
print('   [OK] MCP server module loaded successfully')
"

# Create systemd service file (optional)
if command -v systemctl &> /dev/null; then
    echo ""
    echo "[NOTE] Creating systemd service (optional)..."
    cat > /tmp/rag-maf-mcp.service << EOF
[Unit]
Description=RAG-MAF MCP Server for Claude Code
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PLUGIN_DIR
ExecStart=$PLUGIN_DIR/venv/bin/python3 -m src.mcp_server.server
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    echo "   Service file created at: /tmp/rag-maf-mcp.service"
    echo "   To enable auto-start, run:"
    echo "   sudo cp /tmp/rag-maf-mcp.service /etc/systemd/system/"
    echo "   sudo systemctl enable rag-maf-mcp"
    echo "   sudo systemctl start rag-maf-mcp"
fi

# Create CLI wrapper
echo ""
echo "[~] Creating CLI wrapper..."
cat > "$PLUGIN_DIR/rag-maf" << 'EOF'
#!/bin/bash
# RAG-MAF CLI wrapper

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
if [ ! -d "$PLUGIN_DIR/venv" ]; then
    echo "Error: Virtual environment not found at $PLUGIN_DIR/venv"
    echo "Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
source "$PLUGIN_DIR/venv/bin/activate"

case "$1" in
    start)
        echo "Starting MCP server..."
        cd "$PLUGIN_DIR"
        # Use direct execution to avoid module import warning
        PYTHONPATH="$PLUGIN_DIR/src" python3 "$PLUGIN_DIR/src/mcp_server/server.py" &
        SERVER_PID=$!
        echo $SERVER_PID > "$PLUGIN_DIR/.mcp_server.pid"
        echo "MCP server started (PID: $SERVER_PID)"
        echo "Server running on http://127.0.0.1:8765"
        ;;
    stop)
        echo "Stopping MCP server..."
        if [ -f "$PLUGIN_DIR/.mcp_server.pid" ]; then
            PID=$(cat "$PLUGIN_DIR/.mcp_server.pid")
            kill $PID 2>/dev/null && echo "MCP server stopped (PID: $PID)" || echo "Process not found"
            rm "$PLUGIN_DIR/.mcp_server.pid"
        else
            # Fallback: kill by process name
            pkill -f "mcp_server/server.py" && echo "MCP server stopped" || echo "MCP server not running"
        fi
        ;;
    status)
        if [ -f "$PLUGIN_DIR/.mcp_server.pid" ]; then
            PID=$(cat "$PLUGIN_DIR/.mcp_server.pid")
            if ps -p $PID > /dev/null 2>&1; then
                echo "MCP server is running (PID: $PID)"
                echo "Server URL: http://127.0.0.1:8765"
                exit 0
            else
                echo "MCP server is not running (stale PID file)"
                rm "$PLUGIN_DIR/.mcp_server.pid"
                exit 1
            fi
        else
            if pgrep -f "mcp_server/server.py" > /dev/null; then
                echo "MCP server is running (no PID file)"
                exit 0
            else
                echo "MCP server is not running"
                exit 1
            fi
        fi
        ;;
    index)
        echo "Indexing codebase..."
        cd "$PLUGIN_DIR"
        PYTHONPATH="$PLUGIN_DIR/src" python3 -c "
import sys
from rag import QueryEngine
engine = QueryEngine()
engine.index_codebase('.')
print('Indexing complete!')
"
        ;;
    test)
        echo "Testing RAG system..."
        cd "$PLUGIN_DIR"
        PYTHONPATH="$PLUGIN_DIR/src" python3 -c "
import sys
from rag import QueryEngine
engine = QueryEngine()
print('RAG system initialized successfully!')
print(f'Vector store location: {engine.vector_store.persist_directory}')
"
        ;;
    logs)
        echo "Showing MCP server logs..."
        if [ -f "$PLUGIN_DIR/logs/mcp_server.log" ]; then
            tail -f "$PLUGIN_DIR/logs/mcp_server.log"
        else
            echo "No log file found at $PLUGIN_DIR/logs/mcp_server.log"
        fi
        ;;
    restart)
        echo "Restarting MCP server..."
        $0 stop
        sleep 2
        $0 start
        ;;
    *)
        echo "RAG-MAF Plugin Control Script"
        echo ""
        echo "Usage: $0 {start|stop|status|restart|index|test|logs}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the MCP server"
        echo "  stop     - Stop the MCP server"
        echo "  status   - Check if MCP server is running"
        echo "  restart  - Restart the MCP server"
        echo "  index    - Index the current codebase"
        echo "  test     - Test RAG system initialization"
        echo "  logs     - Show MCP server logs"
        echo ""
        exit 1
        ;;
esac
EOFWRAPPER

chmod +x "$PLUGIN_DIR/rag-maf"
echo "[OK] CLI wrapper created"

# Installation complete
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   [OK] Installation Complete!                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "[**] The RAG-MAF plugin has been successfully installed!"
echo ""
echo "[BOOK] Next Steps:"
echo "   1. Start a Claude Code session in your project"
echo "   2. The plugin will auto-initialize via SessionStart hook"
echo "   3. Use slash commands: /rag-query, /rag-index, /rag-status"
echo ""
echo "[~] Manual Control:"
echo "   Start server:  ./rag-maf start"
echo "   Stop server:   ./rag-maf stop"
echo "   Check status:  ./rag-maf status"
echo "   Restart:       ./rag-maf restart"
echo "   Index code:    ./rag-maf index"
echo "   Test RAG:      ./rag-maf test"
echo "   View logs:     ./rag-maf logs"
echo ""
echo "[#] Documentation: $PLUGIN_DIR/README.md"
echo ""
echo "Happy coding with RAG-powered context awareness! [*]"
echo ""
