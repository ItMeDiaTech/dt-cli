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
source "$PLUGIN_DIR/venv/bin/activate"

case "$1" in
    start)
        echo "Starting MCP server..."
        python3 -m src.mcp_server.server
        ;;
    stop)
        echo "Stopping MCP server..."
        pkill -f "mcp_server/server.py"
        ;;
    status)
        if pgrep -f "mcp_server/server.py" > /dev/null; then
            echo "MCP server is running"
        else
            echo "MCP server is not running"
        fi
        ;;
    index)
        echo "Indexing codebase..."
        python3 -c "
import sys
sys.path.insert(0, '$PLUGIN_DIR/src')
from rag import QueryEngine
engine = QueryEngine()
engine.index_codebase('.')
"
        ;;
    *)
        echo "Usage: $0 {start|stop|status|index}"
        exit 1
        ;;
esac
EOF

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
echo "   Index code:    ./rag-maf index"
echo ""
echo "[#] Documentation: $PLUGIN_DIR/README.md"
echo ""
echo "Happy coding with RAG-powered context awareness! [*]"
echo ""
