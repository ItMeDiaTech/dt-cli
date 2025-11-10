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
import socket

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import QueryEngine
from rag.auto_trigger import AutoTrigger, TriggerStats, TriggerAction
from maf import AgentOrchestrator
from llm import LLMProviderFactory
from config.llm_config import LLMConfig
from debugging import DebugAgent, CodeReviewAgent
from graph import KnowledgeGraph, CodeAnalyzer
from evaluation import RAGASEvaluator, ABTester, HybridSearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_port_available(host: str, port: int) -> bool:
    """
    Check if a port is available for binding.

    Args:
        host: Host address
        port: Port number

    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(host: str, preferred_port: int, max_attempts: int = 10) -> Optional[int]:
    """
    Find an available port, starting with preferred port.

    Args:
        host: Host address
        preferred_port: Preferred port to try first
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number, or None if no port found
    """
    # Try preferred port first
    if is_port_available(host, preferred_port):
        return preferred_port

    # Try adjacent ports
    for offset in range(1, max_attempts):
        port = preferred_port + offset
        if port > 65535:
            break
        if is_port_available(host, port):
            logger.info(f"Port {preferred_port} is in use, using port {port} instead")
            return port

    return None


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str
    use_rag: Optional[bool] = None  # None = auto-trigger, True/False = manual override
    auto_trigger: bool = True  # Enable auto-triggering
    stream: bool = False
    context_files: Optional[List[str]] = None  # Files in user's context


class GenerateRequest(BaseModel):
    """Request model for direct generation."""
    prompt: str
    context: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    stream: bool = False


class DebugRequest(BaseModel):
    """Request model for error debugging."""
    error_output: str
    auto_extract_code: bool = True


class ReviewRequest(BaseModel):
    """Request model for code review."""
    code: str
    file_path: Optional[str] = None
    language: str = "python"


class GraphBuildRequest(BaseModel):
    """Request model for building knowledge graph."""
    path: str  # Directory or file to analyze


class GraphQueryRequest(BaseModel):
    """Request model for graph queries."""
    entity_name: str
    entity_type: Optional[str] = None
    query_type: str = "dependencies"  # dependencies, dependents, usages, impact


class EvaluateRequest(BaseModel):
    """Request model for RAG evaluation."""
    query: str
    retrieved_contexts: List[str]
    generated_answer: str
    ground_truth: Optional[str] = None


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""
    query: str
    documents: List[str]
    metadata: Optional[List[Dict]] = None
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    top_k: int = 5


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
        port: int = 58432,
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

        try:
            self.llm = LLMProviderFactory.create_from_config(llm_config)

            # Check if LLM is actually healthy
            if self.llm and self.llm.check_health():
                logger.info(f"Using LLM provider: {self.llm}")
                self.llm_available = True
            else:
                provider_type = llm_config.get('provider', 'unknown')
                logger.warning(
                    f"LLM provider '{provider_type}' initialized but health check failed. "
                    f"The system will operate with limited functionality (rule-based checks only)."
                )
                self.llm_available = False
        except Exception as e:
            provider_type = llm_config.get('provider', 'unknown')
            logger.error(f"Failed to initialize LLM provider '{provider_type}': {e}")
            logger.warning(
                f"Continuing without LLM support. "
                f"Rule-based code review and debugging will still work. "
                f"To enable full LLM features, install and start {provider_type}."
            )
            self.llm = None
            self.llm_available = False

        # CRITICAL: Setup routes FIRST to ensure server has endpoints even if components fail
        logger.info("Setting up API routes (priority initialization)...")
        self._setup_routes()

        # Verify routes were registered
        route_count = len([r for r in self.app.routes if hasattr(r, 'path')])
        logger.info(f"Registered {route_count} API routes")

        # Log key endpoints
        key_endpoints = ['/health', '/query', '/review', '/debug']
        registered_paths = [r.path for r in self.app.routes if hasattr(r, 'path')]
        for endpoint in key_endpoints:
            status = "✓ registered" if endpoint in registered_paths else "✗ MISSING"
            logger.info(f"  {endpoint}: {status}")

        # Initialize components with error handling - server will work with degraded functionality if these fail
        try:
            logger.info("Initializing RAG system...")
            self.rag_engine = QueryEngine()
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}. RAG features will be unavailable.")
            self.rag_engine = None

        try:
            logger.info("Initializing MAF system...")
            self.orchestrator = AgentOrchestrator(rag_engine=self.rag_engine) if self.rag_engine else None
        except Exception as e:
            logger.error(f"Failed to initialize MAF system: {e}. Agent orchestration will be unavailable.")
            self.orchestrator = None

        try:
            logger.info("Initializing auto-trigger system...")
            auto_trigger_config = self.config.get_auto_trigger_config()
            self.auto_trigger = AutoTrigger(
                confidence_threshold=auto_trigger_config.get('threshold', 0.7),
                show_activity=auto_trigger_config.get('show_activity', True)
            )
            self.trigger_stats = TriggerStats()
        except Exception as e:
            logger.error(f"Failed to initialize auto-trigger system: {e}. Auto-trigger will be unavailable.")
            self.auto_trigger = None
            self.trigger_stats = TriggerStats()  # At least have basic stats

        try:
            logger.info("Initializing debugging agents...")
            self.debug_agent = DebugAgent(
                llm_provider=self.llm,
                rag_engine=self.rag_engine
            )
            self.review_agent = CodeReviewAgent(
                llm_provider=self.llm,
                rag_engine=self.rag_engine
            )
        except Exception as e:
            logger.error(f"Failed to initialize debugging agents: {e}. Debug/review features may have limited functionality.")
            # Create basic agents anyway
            self.debug_agent = DebugAgent()
            self.review_agent = CodeReviewAgent()

        try:
            logger.info("Initializing knowledge graph...")
            self.knowledge_graph = KnowledgeGraph()
            self.code_analyzer = CodeAnalyzer(self.knowledge_graph)
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {e}. Graph features will be unavailable.")
            self.knowledge_graph = None
            self.code_analyzer = None

        try:
            logger.info("Initializing evaluation system...")
            self.ragas_evaluator = RAGASEvaluator(llm_provider=self.llm)
            self.ab_tester = ABTester(self.ragas_evaluator)
            self.hybrid_search = HybridSearch()
        except Exception as e:
            logger.error(f"Failed to initialize evaluation system: {e}. Evaluation features will be unavailable.")
            self.ragas_evaluator = None
            self.ab_tester = None
            self.hybrid_search = None

        logger.info("Standalone MCP Server initialized successfully")

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint with server info."""
            provider_info = self.llm.get_info() if self.llm and self.llm_available else {
                "provider": "none",
                "status": "unavailable",
                "message": "LLM provider not available. Rule-based checks still functional."
            }
            return {
                "name": "dt-cli Standalone Server",
                "version": "2.0.0",
                "description": "100% Open Source RAG/MAF/LLM Server",
                "status": "running",
                "llm_available": self.llm_available,
                "provider": provider_info
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            llm_healthy = self.llm.check_health() if self.llm else False
            rag_healthy = True  # Could add RAG health check

            # Check if key endpoints are registered
            registered_paths = [r.path for r in self.app.routes if hasattr(r, 'path')]
            key_endpoints = {
                'query': '/query' in registered_paths,
                'review': '/review' in registered_paths,
                'debug': '/debug' in registered_paths
            }
            endpoints_healthy = all(key_endpoints.values())

            # Determine overall status
            # Server is "healthy" if endpoints work, even without LLM
            # Server is "degraded" if LLM is unavailable but endpoints work
            # Server is "unhealthy" if endpoints are missing
            if not endpoints_healthy:
                status = "unhealthy"
            elif not llm_healthy:
                status = "degraded"
            else:
                status = "healthy"

            return {
                "status": status,
                "llm": "healthy" if llm_healthy else "unhealthy",
                "llm_available": self.llm_available,
                "rag": "healthy" if rag_healthy else "unhealthy",
                "endpoints": key_endpoints
            }

        @self.app.get("/info")
        async def get_info():
            """Get detailed server information."""
            return {
                "llm": self.llm.get_info() if self.llm else {"status": "unavailable"},
                "rag": self.rag_engine.get_status() if self.rag_engine else {"status": "unavailable"},
                "maf": self.orchestrator.get_status() if self.orchestrator else {"status": "unavailable"},
                "config": {
                    "provider": self.config.get_provider_type(),
                    "auto_trigger": self.config.get_auto_trigger_config()
                }
            }

        @self.app.post("/query")
        async def query(request: QueryRequest):
            """
            Process a query with intelligent auto-triggering.

            This endpoint automatically determines whether to use RAG, agents,
            or direct LLM based on query intent classification.
            """
            try:
                # Check if LLM is available
                if not self.llm or not self.llm_available:
                    raise HTTPException(
                        status_code=503,
                        detail="LLM provider is not available. Please configure an LLM provider to use the query endpoint."
                    )

                # Update context with files if provided
                if request.context_files and self.auto_trigger:
                    for file_path in request.context_files:
                        self.auto_trigger.add_file_to_context(file_path)

                # Decide what to trigger
                use_rag = request.use_rag
                decision = None
                activity_message = None

                if request.auto_trigger and use_rag is None and self.auto_trigger:
                    # Auto-trigger: let the system decide
                    decision = self.auto_trigger.decide(request.query)
                    use_rag = decision.should_use_rag()

                    # Record statistics
                    if self.trigger_stats:
                        self.trigger_stats.record(decision)

                    # Get activity message
                    primary_action = decision.primary_action()
                    if self.auto_trigger.should_show_activity(primary_action):
                        activity_message = self.auto_trigger.get_activity_message(primary_action)

                    logger.info(
                        f"Auto-trigger decision: {decision.reasoning}"
                    )
                elif use_rag is None:
                    # Default to RAG if auto-trigger disabled and no manual override
                    # But only if RAG is available
                    use_rag = True if self.rag_engine else False

                # Retrieve context if RAG enabled and available
                context = None
                if use_rag and self.rag_engine:
                    logger.info(f"Retrieving context for: {request.query}")
                    try:
                        rag_results = self.rag_engine.query(
                            request.query,
                            n_results=self.config.get_rag_config().get('max_results', 5)
                        )

                        if rag_results:
                            context = [
                                f"From {r['metadata'].get('file_path', 'unknown')}:\n{r['text']}"
                                for r in rag_results
                            ]
                    except Exception as e:
                        logger.warning(f"RAG query failed: {e}. Continuing without context.")
                        context = None

                # Generate response
                if request.stream:
                    return StreamingResponse(
                        self._stream_response(request.query, context, activity_message),
                        media_type="text/event-stream"
                    )
                else:
                    response = self.llm.generate(
                        prompt=request.query,
                        context=context,
                        system_prompt="You are an expert coding assistant with access to the codebase."
                    )

                    result = {
                        "response": response,
                        "context_used": len(context) if context else 0,
                        "provider": str(self.llm)
                    }

                    # Add auto-trigger info if used
                    if decision:
                        result["auto_trigger"] = {
                            "intent": decision.intent,
                            "confidence": decision.confidence,
                            "actions": [a.value for a in decision.actions],
                            "reasoning": decision.reasoning
                        }

                    if activity_message:
                        result["activity"] = activity_message

                    return result

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Query error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

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
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Generation error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

        @self.app.post("/rag/index")
        async def rag_index(path: Optional[str] = None):
            """Index or re-index the codebase."""
            try:
                # TODO: Implement indexing
                return {"status": "indexing", "path": path or "."}
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Indexing error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

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
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"RAG search error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

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
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Config reload error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

        @self.app.get("/auto-trigger/stats")
        async def get_trigger_stats():
            """Get auto-trigger statistics."""
            if not self.trigger_stats:
                return {"message": "Auto-trigger statistics not available"}
            return self.trigger_stats.get_summary()

        @self.app.get("/auto-trigger/context")
        async def get_trigger_context():
            """Get current auto-trigger context."""
            if not self.auto_trigger:
                raise HTTPException(
                    status_code=503,
                    detail="Auto-trigger system is not available. Server may have initialization issues."
                )
            return self.auto_trigger.get_context_summary()

        @self.app.post("/auto-trigger/context/clear")
        async def clear_trigger_context():
            """Clear auto-trigger context."""
            if not self.auto_trigger:
                raise HTTPException(
                    status_code=503,
                    detail="Auto-trigger system is not available. Server may have initialization issues."
                )
            self.auto_trigger.clear_context()
            return {"status": "cleared"}

        @self.app.post("/auto-trigger/context/add-file")
        async def add_context_file(file_path: str):
            """Add a file to auto-trigger context."""
            if not self.auto_trigger:
                raise HTTPException(
                    status_code=503,
                    detail="Auto-trigger system is not available. Server may have initialization issues."
                )
            self.auto_trigger.add_file_to_context(file_path)
            return {
                "status": "added",
                "file": file_path,
                "context": self.auto_trigger.get_context_summary()
            }

        @self.app.post("/auto-trigger/context/remove-file")
        async def remove_context_file(file_path: str):
            """Remove a file from auto-trigger context."""
            if not self.auto_trigger:
                raise HTTPException(
                    status_code=503,
                    detail="Auto-trigger system is not available. Server may have initialization issues."
                )
            self.auto_trigger.remove_file_from_context(file_path)
            return {
                "status": "removed",
                "file": file_path,
                "context": self.auto_trigger.get_context_summary()
            }

        @self.app.post("/debug")
        async def debug_error(request: DebugRequest):
            """
            Analyze an error and provide debugging insights.

            This endpoint uses the debug agent to:
            - Parse error messages and stack traces
            - Identify root causes
            - Suggest fixes
            - Find similar historical errors
            """
            try:
                analysis = self.debug_agent.analyze_error(
                    request.error_output,
                    auto_extract_code=request.auto_extract_code
                )

                return analysis.to_dict()

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Debug error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

        @self.app.post("/review")
        async def review_code(request: ReviewRequest):
            """
            Perform comprehensive code review.

            This endpoint uses the review agent to check for:
            - Security vulnerabilities
            - Performance issues
            - Best practices violations
            - Code complexity
            - Documentation completeness
            """
            try:
                review = self.review_agent.review_code(
                    request.code,
                    file_path=request.file_path,
                    language=request.language
                )

                result = review.to_dict()

                # Add LLM availability warning if applicable
                if not self.llm_available:
                    result['warning'] = (
                        "LLM provider is not available. "
                        "Review includes rule-based checks only. "
                        "For advanced AI-powered analysis, please install and configure an LLM provider."
                    )
                    result['llm_used'] = False
                else:
                    result['llm_used'] = True

                return result

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Review error: {e}\n{error_trace}")

                # Provide more helpful error message
                error_msg = f"{type(e).__name__}: {str(e)}"
                if not self.llm_available and "llm" in str(e).lower():
                    error_msg += " (Note: LLM provider is not available)"

                raise HTTPException(
                    status_code=500,
                    detail=error_msg
                )

        @self.app.post("/graph/build")
        async def build_graph(request: GraphBuildRequest):
            """
            Build or rebuild the knowledge graph.

            Analyzes code files to extract:
            - Import dependencies
            - Function calls
            - Class inheritance
            - Code relationships
            """
            if not self.knowledge_graph or not self.code_analyzer:
                raise HTTPException(
                    status_code=503,
                    detail="Knowledge graph system is not available. Server may have initialization issues."
                )

            try:
                # Clear existing graph
                self.knowledge_graph.clear()

                # Analyze path
                if os.path.isdir(request.path):
                    self.code_analyzer.analyze_directory(request.path)
                elif os.path.isfile(request.path):
                    self.code_analyzer.analyze_file(request.path)
                else:
                    raise HTTPException(status_code=400, detail="Invalid path")

                stats = self.knowledge_graph.get_stats()
                return {
                    "status": "built",
                    "path": request.path,
                    "stats": stats
                }

            except HTTPException:
                raise
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Graph build error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

        @self.app.post("/graph/query")
        async def query_graph(request: GraphQueryRequest):
            """
            Query the knowledge graph.

            Supported query types:
            - dependencies: What does this entity depend on?
            - dependents: What depends on this entity?
            - usages: Where is this entity used?
            - impact: What's the impact of changing this entity?
            """
            if not self.knowledge_graph:
                raise HTTPException(
                    status_code=503,
                    detail="Knowledge graph system is not available. Server may have initialization issues."
                )

            try:
                entity = self.knowledge_graph.get_entity(
                    request.entity_name,
                    request.entity_type
                )

                if not entity:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Entity '{request.entity_name}' not found"
                    )

                if request.query_type == "dependencies":
                    deps = self.knowledge_graph.get_dependencies(entity)
                    return {
                        "entity": request.entity_name,
                        "query_type": "dependencies",
                        "results": [
                            {
                                "name": dep.name,
                                "type": dep.entity_type,
                                "file": dep.file_path
                            }
                            for dep in deps
                        ]
                    }

                elif request.query_type == "dependents":
                    deps = self.knowledge_graph.get_dependents(entity)
                    return {
                        "entity": request.entity_name,
                        "query_type": "dependents",
                        "results": [
                            {
                                "name": dep.name,
                                "type": dep.entity_type,
                                "file": dep.file_path
                            }
                            for dep in deps
                        ]
                    }

                elif request.query_type == "usages":
                    usages = self.knowledge_graph.find_usages(
                        request.entity_name,
                        request.entity_type
                    )
                    return {
                        "entity": request.entity_name,
                        "query_type": "usages",
                        "results": usages
                    }

                elif request.query_type == "impact":
                    impact = self.knowledge_graph.get_impact_analysis(entity)
                    return impact

                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown query type: {request.query_type}"
                    )

            except HTTPException:
                raise
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Graph query error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

        @self.app.get("/graph/stats")
        async def get_graph_stats():
            """Get knowledge graph statistics."""
            if not self.knowledge_graph:
                raise HTTPException(
                    status_code=503,
                    detail="Knowledge graph system is not available. Server may have initialization issues."
                )
            return self.knowledge_graph.get_stats()

        @self.app.post("/evaluate")
        async def evaluate_rag(request: EvaluateRequest):
            """
            Evaluate RAG quality using RAGAS metrics.

            Returns:
            - Context relevance
            - Answer faithfulness
            - Answer relevance
            - Context precision (if ground truth provided)
            - Context recall (if ground truth provided)
            - Overall score
            """
            if not self.ragas_evaluator:
                raise HTTPException(
                    status_code=503,
                    detail="Evaluation system is not available. Server may have initialization issues."
                )

            try:
                evaluation = self.ragas_evaluator.evaluate(
                    query=request.query,
                    retrieved_contexts=request.retrieved_contexts,
                    generated_answer=request.generated_answer,
                    ground_truth=request.ground_truth
                )

                return evaluation.to_dict()

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Evaluation error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

        @self.app.post("/hybrid-search")
        async def hybrid_search(request: HybridSearchRequest):
            """
            Perform hybrid search combining semantic and keyword search.

            Combines:
            - Semantic similarity (embeddings)
            - Keyword relevance (BM25)

            Returns ranked results with individual and combined scores.
            """
            try:
                # Index documents
                self.hybrid_search = HybridSearch(
                    semantic_weight=request.semantic_weight,
                    keyword_weight=request.keyword_weight
                )
                self.hybrid_search.index_documents(
                    request.documents,
                    request.metadata
                )

                # Perform search (semantic scores would come from embeddings)
                # For now, using None to rely on keyword search
                results = self.hybrid_search.search(
                    request.query,
                    semantic_scores=None,
                    top_k=request.top_k
                )

                return {
                    "query": request.query,
                    "results": [r.to_dict() for r in results],
                    "weights": {
                        "semantic": request.semantic_weight,
                        "keyword": request.keyword_weight
                    }
                }

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Hybrid search error: {e}\n{error_trace}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(e).__name__}: {str(e)}"
                )

        @self.app.get("/evaluation/stats")
        async def get_evaluation_stats():
            """Get evaluation statistics from A/B testing."""
            if not self.ab_tester.experiments:
                return {
                    "experiments": {},
                    "total_experiments": 0
                }

            stats = {}
            for name, evaluations in self.ab_tester.experiments.items():
                metrics = self.ragas_evaluator.aggregate_metrics(evaluations)
                stats[name] = metrics

            return {
                "experiments": stats,
                "total_experiments": len(self.ab_tester.experiments)
            }

    async def _stream_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        activity_message: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Stream response generator.

        Args:
            prompt: User prompt
            context: Optional context
            activity_message: Optional activity indicator
            system_prompt: Optional system prompt

        Yields:
            Server-sent events
        """
        try:
            # Send activity message first if present
            if activity_message:
                yield f"data: {activity_message}\n\n"

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

        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"Port {self.port} is already in use!")
                logger.error("Try using a different port with --port option")
            raise


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
        default=58432,
        help='Server port (default: 58432)'
    )
    parser.add_argument(
        '--config',
        help='Path to config file (default: llm-config.yaml)'
    )
    parser.add_argument(
        '--auto-port',
        action='store_true',
        help='Automatically find an available port if default is in use'
    )

    args = parser.parse_args()

    # Find available port if auto-port is enabled
    port = args.port
    if args.auto_port:
        available_port = find_available_port(args.host, args.port)
        if available_port is None:
            logger.error(f"Could not find an available port near {args.port}")
            sys.exit(1)
        port = available_port
    elif not is_port_available(args.host, args.port):
        logger.error(f"Port {args.port} is already in use!")
        logger.error("Use --auto-port to automatically find an available port")
        logger.error("Or specify a different port with --port")
        sys.exit(1)

    server = StandaloneMCPServer(
        host=args.host,
        port=port,
        config_path=args.config
    )
    server.run()


if __name__ == "__main__":
    main()
