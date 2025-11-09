"""
MCP Server implementation using FastAPI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import QueryEngine
from maf import AgentOrchestrator
from .tools import RAGTools, MAFTools
from .bridge import ClaudeCodeBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ToolRequest(BaseModel):
    """Request model for tool execution."""
    category: str
    tool_name: str
    parameters: Dict[str, Any]


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str


class MCPServer:
    """
    MCP Server for Claude Code integration.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 58432):
        """
        Initialize the MCP server.

        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.app = FastAPI(title="RAG-MAF MCP Server", version="1.0.0")

        # Initialize RAG and MAF systems
        logger.info("Initializing RAG and MAF systems...")
        self.rag_engine = QueryEngine()
        self.orchestrator = AgentOrchestrator(rag_engine=self.rag_engine)

        # Initialize tools
        self.rag_tools = RAGTools(self.rag_engine)
        self.maf_tools = MAFTools(self.orchestrator)

        # Initialize bridge
        self.bridge = ClaudeCodeBridge(self.rag_tools, self.maf_tools)

        # Setup routes
        self._setup_routes()

        logger.info("MCP Server initialized")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "RAG-MAF MCP Server",
                "version": "1.0.0",
                "status": "running"
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy"}

        @self.app.get("/tools")
        async def get_tools():
            """Get all available tools."""
            return self.bridge.get_all_tools()

        @self.app.post("/execute")
        async def execute_tool(request: ToolRequest):
            """Execute a tool."""
            try:
                result = self.bridge.execute_tool(
                    category=request.category,
                    tool_name=request.tool_name,
                    parameters=request.parameters
                )
                return result
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/context")
        async def get_context(request: QueryRequest):
            """Get context for a query."""
            try:
                context = self.bridge.get_context_for_claude(request.query)
                return {"context": context}
            except Exception as e:
                logger.error(f"Context retrieval error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/rag/query")
        async def rag_query(request: QueryRequest):
            """Quick RAG query endpoint."""
            try:
                result = self.rag_tools.execute_tool(
                    "rag_query",
                    {"query": request.query, "n_results": 5}
                )
                return result
            except Exception as e:
                logger.error(f"RAG query error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/maf/orchestrate")
        async def maf_orchestrate(request: QueryRequest):
            """Quick MAF orchestration endpoint."""
            try:
                result = self.maf_tools.execute_tool(
                    "maf_orchestrate",
                    {"query": request.query}
                )
                return result
            except Exception as e:
                logger.error(f"MAF orchestration error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/status")
        async def get_status():
            """Get system status."""
            rag_status = self.rag_engine.get_status()
            maf_status = self.orchestrator.get_status()

            return {
                "rag": rag_status,
                "maf": maf_status,
                "server": {
                    "host": self.host,
                    "port": self.port,
                    "status": "running"
                }
            }

    def run(self):
        """Run the MCP server."""
        logger.info(f"Starting MCP server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


def main():
    """Main entry point."""
    server = MCPServer()
    server.run()


if __name__ == "__main__":
    main()
