#!/bin/bash
# Comprehensive Installation Script for Claude Code + dt-cli RAG Plugin
# For Ubuntu Server (18.04+)
# Supports Claude Code Max plan authentication

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║  Claude Code + dt-cli RAG Plugin - Ubuntu Installation        ║
║  Complete setup for Ubuntu Server                             ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: Running as root. Consider running as a regular user.${NC}"
    echo -e "${YELLOW}   The script will use sudo when needed.${NC}"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Step 1: System Prerequisites                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt update -qq
sudo apt upgrade -y -qq
print_status "System packages updated"

# Install basic prerequisites
echo ""
echo "📦 Installing system prerequisites..."
sudo apt install -y \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    unzip \
    > /dev/null 2>&1

print_status "System prerequisites installed"

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Step 2: Node.js Installation                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Node.js is already installed
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_info "Node.js already installed: $NODE_VERSION"

    # Check if version is >= 18
    if node -e "process.exit(parseInt(process.version.slice(1)) >= 18 ? 0 : 1)"; then
        print_status "Node.js version is compatible"
    else
        print_warning "Node.js version < 18. Upgrading..."
        sudo apt remove -y nodejs > /dev/null 2>&1 || true
    fi
fi

# Install Node.js 20.x if not installed or upgraded
if ! node -e "process.exit(parseInt(process.version.slice(1)) >= 18 ? 0 : 1)" 2>/dev/null; then
    echo "📦 Installing Node.js 20.x..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - > /dev/null 2>&1
    sudo apt install -y nodejs > /dev/null 2>&1
    print_status "Node.js installed: $(node --version)"
fi

# Verify npm
if command -v npm &> /dev/null; then
    print_status "npm installed: $(npm --version)"
else
    print_error "npm installation failed"
    exit 1
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Step 3: Python Environment Setup                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_info "Found Python $PYTHON_VERSION"

# Check if Python version is >= 3.8
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    print_status "Python version is compatible (>= 3.8)"
else
    print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Step 4: Claude Code Installation                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Claude Code is already installed
if command -v claude-code &> /dev/null; then
    CLAUDE_VERSION=$(claude-code --version 2>/dev/null || echo "unknown")
    print_info "Claude Code already installed: $CLAUDE_VERSION"
    read -p "Reinstall Claude Code? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Reinstalling Claude Code..."
        sudo npm uninstall -g @anthropic-ai/claude-code > /dev/null 2>&1 || true
        sudo npm install -g @anthropic-ai/claude-code
        print_status "Claude Code reinstalled"
    fi
else
    echo "📦 Installing Claude Code CLI..."
    sudo npm install -g @anthropic-ai/claude-code
    print_status "Claude Code installed"
fi

# Verify installation
if command -v claude-code &> /dev/null; then
    CLAUDE_VERSION=$(claude-code --version 2>/dev/null || echo "installed")
    print_status "Claude Code verified: $CLAUDE_VERSION"
else
    print_error "Claude Code installation failed"
    exit 1
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Step 5: Claude Code Authentication                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

print_info "Claude Code Max Plan Authentication"
echo ""
echo "Choose authentication method:"
echo "  1) Interactive authentication (will open browser/provide URL)"
echo "  2) API key authentication (manual setup)"
echo "  3) Skip authentication (configure later)"
echo ""
read -p "Select option (1-3): " -n 1 -r AUTH_CHOICE
echo ""
echo ""

case $AUTH_CHOICE in
    1)
        print_info "Starting interactive authentication..."
        echo ""
        print_warning "For headless servers, you'll receive a URL to authenticate"
        print_info "Copy the URL and open it on your local machine"
        print_info "After authentication, return to this terminal"
        echo ""
        read -p "Press Enter to continue..."

        # Try to authenticate
        if claude-code auth login 2>/dev/null; then
            print_status "Authentication successful!"
        else
            print_warning "Interactive auth may require additional steps"
            print_info "Run 'claude-code auth login' manually if needed"
        fi
        ;;
    2)
        print_info "Manual API Key Setup"
        echo ""
        echo "To set up API key authentication:"
        echo "  1. Get your API key from: https://console.anthropic.com/settings/keys"
        echo "  2. Run: export ANTHROPIC_API_KEY='your-api-key-here'"
        echo "  3. Add to ~/.bashrc for persistence:"
        echo "     echo 'export ANTHROPIC_API_KEY=\"your-api-key-here\"' >> ~/.bashrc"
        echo ""
        read -p "Do you have your API key ready? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            read -p "Enter your API key: " API_KEY
            export ANTHROPIC_API_KEY="$API_KEY"
            echo "export ANTHROPIC_API_KEY=\"$API_KEY\"" >> ~/.bashrc
            print_status "API key configured and saved to ~/.bashrc"
        else
            print_info "You can configure the API key later"
        fi
        ;;
    3)
        print_info "Skipping authentication. Configure later with:"
        echo "  - claude-code auth login (interactive)"
        echo "  - export ANTHROPIC_API_KEY='your-key' (API key)"
        ;;
    *)
        print_warning "Invalid option. Skipping authentication."
        ;;
esac

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Step 6: dt-cli RAG Plugin Installation                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create virtual environment
echo "🔧 Setting up Python virtual environment..."
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    python3 -m venv "$SCRIPT_DIR/venv"
    print_status "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_status "pip upgraded"

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
print_info "This may take several minutes on first install..."

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
    if [ $? -eq 0 ]; then
        print_status "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
else
    print_error "requirements.txt not found in $SCRIPT_DIR"
    exit 1
fi

# Download embedding model
echo ""
echo "🤖 Downloading embedding model (all-MiniLM-L6-v2)..."
python3 -c "
from sentence_transformers import SentenceTransformer
print('   Downloading model...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('   ✅ Model downloaded successfully')
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_status "Embedding model downloaded"
else
    print_error "Failed to download embedding model"
    exit 1
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p "$SCRIPT_DIR/.rag_data"
mkdir -p "$SCRIPT_DIR/logs"
print_status "Directories created"

# Make scripts executable
echo ""
echo "🔧 Configuring scripts and hooks..."
if [ -f "$SCRIPT_DIR/.claude/hooks/SessionStart.sh" ]; then
    chmod +x "$SCRIPT_DIR/.claude/hooks/SessionStart.sh"
    print_status "SessionStart hook configured"
fi

if [ -f "$SCRIPT_DIR/install.sh" ]; then
    chmod +x "$SCRIPT_DIR/install.sh"
fi

if [ -f "$SCRIPT_DIR/rag-maf" ]; then
    chmod +x "$SCRIPT_DIR/rag-maf"
    print_status "CLI wrapper configured"
fi

# Test MCP server
echo ""
echo "🧪 Testing MCP server..."
python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/src')
try:
    from mcp_server import MCPServer
    print('   ✅ MCP server module loaded successfully')
except Exception as e:
    print(f'   ⚠️  Warning: {e}')
    sys.exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    print_status "MCP server verified"
else
    print_warning "MCP server test had issues (may work anyway)"
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Step 7: Optional Services Setup                               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Systemd service
if command -v systemctl &> /dev/null; then
    read -p "Create systemd service for auto-start? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📝 Creating systemd service..."
        cat > /tmp/rag-maf-mcp.service << EOF
[Unit]
Description=RAG-MAF MCP Server for Claude Code
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/venv/bin/python3 -m src.mcp_server.server
Restart=on-failure
RestartSec=5
Environment="PATH=$SCRIPT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=multi-user.target
EOF

        sudo cp /tmp/rag-maf-mcp.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable rag-maf-mcp
        print_status "Systemd service created and enabled"

        read -p "Start the service now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo systemctl start rag-maf-mcp
            print_status "Service started"
        fi
    fi
fi

# Create CLI wrapper if it doesn't exist
if [ ! -f "$SCRIPT_DIR/rag-maf" ]; then
    echo ""
    echo "🔧 Creating CLI wrapper..."
    cat > "$SCRIPT_DIR/rag-maf" << 'EOFWRAPPER'
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
EOFWRAPPER

    chmod +x "$SCRIPT_DIR/rag-maf"
    print_status "CLI wrapper created"
fi

# Add to PATH
echo ""
read -p "Add dt-cli to PATH? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ! grep -q "$SCRIPT_DIR" ~/.bashrc; then
        echo "export PATH=\"\$PATH:$SCRIPT_DIR\"" >> ~/.bashrc
        print_status "Added to PATH in ~/.bashrc"
        export PATH="$PATH:$SCRIPT_DIR"
    else
        print_info "Already in PATH"
    fi
fi

# Installation complete
echo ""
echo -e "${GREEN}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════╗
║   ✅ Installation Complete!                                    ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

print_status "Claude Code + dt-cli RAG Plugin successfully installed!"
echo ""

# Print next steps
echo -e "${BLUE}📖 Next Steps:${NC}"
echo ""
echo "1️⃣  Reload your shell configuration:"
echo "   ${YELLOW}source ~/.bashrc${NC}"
echo ""
echo "2️⃣  Verify Claude Code installation:"
echo "   ${YELLOW}claude-code --version${NC}"
echo ""
echo "3️⃣  Start a Claude Code session:"
echo "   ${YELLOW}cd /path/to/your/project${NC}"
echo "   ${YELLOW}claude-code${NC}"
echo ""
echo "4️⃣  Use RAG slash commands in Claude Code:"
echo "   ${YELLOW}/rag-query how does authentication work?${NC}"
echo "   ${YELLOW}/rag-index${NC}"
echo "   ${YELLOW}/rag-status${NC}"
echo ""
echo -e "${BLUE}🔧 Manual Control:${NC}"
echo "   ${YELLOW}$SCRIPT_DIR/rag-maf start${NC}   - Start MCP server"
echo "   ${YELLOW}$SCRIPT_DIR/rag-maf stop${NC}    - Stop MCP server"
echo "   ${YELLOW}$SCRIPT_DIR/rag-maf status${NC}  - Check status"
echo "   ${YELLOW}$SCRIPT_DIR/rag-maf index${NC}   - Index codebase"
echo ""

if command -v systemctl &> /dev/null && systemctl is-enabled rag-maf-mcp &>/dev/null; then
    echo -e "${BLUE}🚀 Systemd Service:${NC}"
    echo "   ${YELLOW}sudo systemctl status rag-maf-mcp${NC}  - Check service"
    echo "   ${YELLOW}sudo systemctl restart rag-maf-mcp${NC} - Restart service"
    echo ""
fi

echo -e "${BLUE}📚 Documentation:${NC}"
echo "   README: $SCRIPT_DIR/README.md"
echo "   Architecture: $SCRIPT_DIR/ARCHITECTURE.md"
echo ""

print_status "Happy coding with RAG-powered Claude Code! 🚀"
echo ""

# Deactivate virtual environment
deactivate 2>/dev/null || true
