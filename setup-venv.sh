#!/bin/bash

# setup-venv.sh - Virtual Environment Setup Script for dt-cli
# This script creates a Python virtual environment and installs all dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  dt-cli Virtual Environment Setup${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python version
echo -e "${YELLOW}[1/5]${NC} Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
echo ""

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}[2/5]${NC} Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment"
        SKIP_CREATE=true
    fi
fi

# Create virtual environment
if [ "$SKIP_CREATE" != "true" ]; then
    echo -e "${YELLOW}[2/5]${NC} Creating virtual environment..."

    # Check if python3-venv is installed (needed on some Linux distros)
    if ! python3 -m venv --help &> /dev/null; then
        echo -e "${RED}Error: python3-venv is not installed${NC}"
        echo ""
        echo "On Ubuntu/Debian, install it with:"
        echo "  sudo apt install python3-venv python3-full"
        echo ""
        echo "On other systems, consult your package manager."
        exit 1
    fi

    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Using existing virtual environment"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}[3/5]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

# Upgrade pip
echo -e "${YELLOW}[4/5]${NC} Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓${NC} pip upgraded"
echo ""

# Install requirements
echo -e "${YELLOW}[5/5]${NC} Installing dependencies from requirements.txt..."
echo "This may take a few minutes (downloading and installing packages)..."
echo ""

if pip install -r requirements.txt; then
    echo ""
    echo -e "${GREEN}✓${NC} All dependencies installed successfully!"
else
    echo ""
    echo -e "${RED}✗${NC} Failed to install some dependencies"
    echo "Please check the error messages above"
    exit 1
fi

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}     Setup Complete! 🎉${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Virtual environment created at: ${SCRIPT_DIR}/venv"
echo ""
echo "To activate the virtual environment:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "To deactivate when done:"
echo -e "  ${YELLOW}deactivate${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment (command above)"
echo "  2. Try running: python3 src/cli/interactive.py"
echo "  3. Or start the MCP server: python3 -m src.mcp_server.server"
echo ""
echo "For more information, see: docs/guides/INSTALLATION.md"
echo ""
