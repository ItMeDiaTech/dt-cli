"""
Main entry point for running mcp_server as a module.

This allows running the server with:
    python -m src.mcp_server.standalone_server

or with the parent directory in PYTHONPATH:
    python -m mcp_server.standalone_server
"""

import sys
import os

# Ensure parent directory is in path for imports
# This handles both direct execution and module execution
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from mcp_server.standalone_server import main

if __name__ == "__main__":
    main()
