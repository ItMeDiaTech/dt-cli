"""
Standalone MCP Server with integrated LLM.

This server can run independently without Claude Code,
using open source LLMs via Ollama or vLLM.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import QueryEngine
from maf import AgentOrchestrator
from llm import LLMProviderFactory
from config.llm_config import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str
    use_rag: bool = True
    stream: bool = False


class GenerateRequest(BaseModel):
    """Request model for direct generation."""
    prompt: str
    context: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    stream: bool = False


class StandaloneMCPServer:
    """
    Standalone MCP Server with integrated LLM provider.

    This server includes:
    - RAG system (ChromaDB + embeddings)
    - MAF system (LangGraph agents)
    - LLM provider (Ollama/vLLM/Claude)
    - Complete standalone operation
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        config_path: Optional[str] = None
    ):
        """
        Initialize the standalone server.

        Args:
            host: Server host
            port: Server port
            config_path: Path to config file
        """
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="dt-cli Standalone Server",
            version="2.0.0",
            description="100% Open Source RAG/MAF/LLM Server"
        )

        # Load configuration
        logger.info("Loading configuration...")
        self.config = LLMConfig(config_path)

        # Initialize LLM provider
        logger.info("Initializing LLM provider...")
        llm_config = self.config.get_llm_config()
        self.llm = LLMProviderFactory.create_from_config(llm_config)

        logger.info(f"Using LLM provider: {self.llm}")

        # Initialize RAG system
        logger.info("Initializing RAG system...")
        self.rag_engine = QueryEngine()

        # Initialize MAF system
        logger.info("Initializing MAF system...")
        self.orchestrator = AgentOrchestrator(rag_engine=self.rag_engine)

        # Setup routes
        self._setup_routes()

        logger.info("Standalone MCP Server initialized successfully")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint with server info."""
            return {
                "name": "dt-cli Standalone Server",
                "version": "2.0.0",
                "description": "100% Open Source RAG/MAF/LLM Server",
                "status": "running",
                "provider": self.llm.get_info()
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            llm_healthy = self.llm.check_health()
            rag_healthy = True  # Could add RAG health check

            return {
                "status": "healthy" if llm_healthy and rag_healthy else "degraded",
                "llm": "healthy" if llm_healthy else "unhealthy",
                "rag": "healthy" if rag_healthy else "unhealthy"
            }

        @self.app.get("/info")
        async def get_info():
            """Get detailed server information."""
            return {
                "llm": self.llm.get_info(),
                "rag": self.rag_engine.get_status(),
                "maf": self.orchestrator.get_status(),
                "config": {
                    "provider": self.config.get_provider_type(),
                    "auto_trigger": self.config.get_auto_trigger_config()
                }
            }

        @self.app.post("/query")
        async def query(request: QueryRequest):
            """
            Process a query with optional RAG.

            This is the main endpoint for asking questions.
            """
            try:
                # Retrieve context if RAG enabled
                context = None
                if request.use_rag:
                    logger.info(f"Retrieving context for: {request.query}")
                    rag_results = self.rag_engine.query(
                        request.query,
                        n_results=self.config.get_rag_config().get('max_results', 5)
                    )

                    if rag_results:
                        context = [
                            f"From {r['metadata'].get('file_path', 'unknown')}:\n{r['text']}"
                            for r in rag_results
                        ]

                # Generate response
                if request.stream:
                    return StreamingResponse(
                        self._stream_response(request.query, context),
                        media_type="text/event-stream"
                    )
                else:
                    response = self.llm.generate(
                        prompt=request.query,
                        context=context,
                        system_prompt="You are an expert coding assistant with access to the codebase."
                    )

                    return {
                        "response": response,
                        "context_used": len(context) if context else 0,
                        "provider": str(self.llm)
                    }

            except Exception as e:
                logger.error(f"Query error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/generate")
        async def generate(request: GenerateRequest):
            """
            Generate response directly without RAG.

            For when you just want LLM generation.
            """
            try:
                if request.stream:
                    return StreamingResponse(
                        self._stream_response(
                            request.prompt,
                            request.context,
                            request.system_prompt
                        ),
                        media_type="text/event-stream"
                    )
                else:
                    response = self.llm.generate(
                        prompt=request.prompt,
                        context=request.context,
                        system_prompt=request.system_prompt
                    )

                    return {
                        "response": response,
                        "provider": str(self.llm)
                    }

            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/rag/index")
        async def rag_index(path: Optional[str] = None):
            """Index or re-index the codebase."""
            try:
                # TODO: Implement indexing
                return {"status": "indexing", "path": path or "."}
            except Exception as e:
                logger.error(f"Indexing error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/rag/search")
        async def rag_search(request: QueryRequest):
            """Search RAG system without LLM generation."""
            try:
                results = self.rag_engine.query(
                    request.query,
                    n_results=10
                )

                return {
                    "results": results,
                    "count": len(results)
                }

            except Exception as e:
                logger.error(f"RAG search error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/config")
        async def get_config():
            """Get current configuration."""
            return {
                "provider": self.config.get_provider_type(),
                "llm": self.config.get_llm_config(),
                "rag": self.config.get_rag_config(),
                "maf": self.config.get_maf_config(),
                "auto_trigger": self.config.get_auto_trigger_config()
            }

        @self.app.post("/config/reload")
        async def reload_config():
            """Reload configuration from file."""
            try:
                self.config.reload()
                return {"status": "reloaded", "config": str(self.config)}
            except Exception as e:
                logger.error(f"Config reload error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _stream_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Stream response generator.

        Args:
            prompt: User prompt
            context: Optional context
            system_prompt: Optional system prompt

        Yields:
            Server-sent events
        """
        try:
            for chunk in self.llm.generate_streaming(
                prompt=prompt,
                context=context,
                system_prompt=system_prompt or "You are an expert coding assistant."
            ):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    def run(self):
        """Run the standalone server."""
        logger.info(f"Starting standalone server on {self.host}:{self.port}")
        logger.info(f"Provider: {self.llm}")
        logger.info("Server is 100% open source - no proprietary dependencies!")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="dt-cli Standalone Server - 100% Open Source"
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Server host (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='Server port (default: 8765)'
    )
    parser.add_argument(
        '--config',
        help='Path to config file (default: llm-config.yaml)'
    )

    args = parser.parse_args()

    server = StandaloneMCPServer(
        host=args.host,
        port=args.port,
        config_path=args.config
    )
    server.run()


if __name__ == "__main__":
    main()
