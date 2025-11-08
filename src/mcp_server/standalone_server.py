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

        # Initialize auto-trigger system
        logger.info("Initializing auto-trigger system...")
        auto_trigger_config = self.config.get_auto_trigger_config()
        self.auto_trigger = AutoTrigger(
            confidence_threshold=auto_trigger_config.get('threshold', 0.7),
            show_activity=auto_trigger_config.get('show_activity', True)
        )
        self.trigger_stats = TriggerStats()

        # Initialize debugging agents
        logger.info("Initializing debugging agents...")
        self.debug_agent = DebugAgent(
            llm_provider=self.llm,
            rag_engine=self.rag_engine
        )
        self.review_agent = CodeReviewAgent(
            llm_provider=self.llm,
            rag_engine=self.rag_engine
        )

        # Initialize knowledge graph
        logger.info("Initializing knowledge graph...")
        self.knowledge_graph = KnowledgeGraph()
        self.code_analyzer = CodeAnalyzer(self.knowledge_graph)

        # Initialize evaluation system
        logger.info("Initializing evaluation system...")
        self.ragas_evaluator = RAGASEvaluator(llm_provider=self.llm)
        self.ab_tester = ABTester(self.ragas_evaluator)
        self.hybrid_search = HybridSearch()

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
            Process a query with intelligent auto-triggering.

            This endpoint automatically determines whether to use RAG, agents,
            or direct LLM based on query intent classification.
            """
            try:
                # Update context with files if provided
                if request.context_files:
                    for file_path in request.context_files:
                        self.auto_trigger.add_file_to_context(file_path)

                # Decide what to trigger
                use_rag = request.use_rag
                decision = None
                activity_message = None

                if request.auto_trigger and use_rag is None:
                    # Auto-trigger: let the system decide
                    decision = self.auto_trigger.decide(request.query)
                    use_rag = decision.should_use_rag()

                    # Record statistics
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
                    use_rag = True

                # Retrieve context if RAG enabled
                context = None
                if use_rag:
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

        @self.app.get("/auto-trigger/stats")
        async def get_trigger_stats():
            """Get auto-trigger statistics."""
            return self.trigger_stats.get_summary()

        @self.app.get("/auto-trigger/context")
        async def get_trigger_context():
            """Get current auto-trigger context."""
            return self.auto_trigger.get_context_summary()

        @self.app.post("/auto-trigger/context/clear")
        async def clear_trigger_context():
            """Clear auto-trigger context."""
            self.auto_trigger.clear_context()
            return {"status": "cleared"}

        @self.app.post("/auto-trigger/context/add-file")
        async def add_context_file(file_path: str):
            """Add a file to auto-trigger context."""
            self.auto_trigger.add_file_to_context(file_path)
            return {
                "status": "added",
                "file": file_path,
                "context": self.auto_trigger.get_context_summary()
            }

        @self.app.post("/auto-trigger/context/remove-file")
        async def remove_context_file(file_path: str):
            """Remove a file from auto-trigger context."""
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
                logger.error(f"Debug error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

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

                return review.to_dict()

            except Exception as e:
                logger.error(f"Review error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

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

            except Exception as e:
                logger.error(f"Graph build error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

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
                logger.error(f"Graph query error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/graph/stats")
        async def get_graph_stats():
            """Get knowledge graph statistics."""
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
            try:
                evaluation = self.ragas_evaluator.evaluate(
                    query=request.query,
                    retrieved_contexts=request.retrieved_contexts,
                    generated_answer=request.generated_answer,
                    ground_truth=request.ground_truth
                )

                return evaluation.to_dict()

            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

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
                logger.error(f"Hybrid search error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

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
