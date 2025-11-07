"""
Enhanced MCP (Model Context Protocol) server for RAG system.

Provides comprehensive REST API endpoints for:
- Query execution
- Index management
- System monitoring
- Search management
- Configuration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System MCP Server",
    description="Model Context Protocol server for RAG-MAF plugin",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="Search query")
    n_results: int = Field(5, description="Number of results")
    use_hybrid: bool = Field(True, description="Use hybrid search")
    use_reranking: bool = Field(True, description="Use reranking")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")


class QueryResponse(BaseModel):
    """Query response model."""
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    correlation_id: Optional[str] = None


class IndexRequest(BaseModel):
    """Index request model."""
    incremental: bool = Field(True, description="Incremental indexing")
    use_git: bool = Field(True, description="Use git diff")
    background: bool = Field(False, description="Run in background")


class IndexResponse(BaseModel):
    """Index response model."""
    status: str
    stats: Dict[str, Any]
    task_id: Optional[str] = None


class SavedSearchRequest(BaseModel):
    """Saved search request model."""
    name: str
    query: str
    description: str = ""
    tags: List[str] = []
    n_results: int = 5


class HealthResponse(BaseModel):
    """Health response model."""
    status: str
    timestamp: str
    metrics: Dict[str, Any]


# Global instances (to be initialized)
query_engine = None
saved_search_manager = None
health_monitor = None
cache_manager = None
task_manager = None
query_learning_system = None
knowledge_graph = None


def initialize_server(
    qe=None,
    ssm=None,
    hm=None,
    cm=None,
    tm=None,
    qls=None,
    kg=None
):
    """
    Initialize server with component instances.

    Args:
        qe: Query engine
        ssm: Saved search manager
        hm: Health monitor
        cm: Cache manager
        tm: Task manager
        qls: Query learning system
        kg: Knowledge graph
    """
    global query_engine, saved_search_manager, health_monitor, cache_manager
    global task_manager, query_learning_system, knowledge_graph

    query_engine = qe
    saved_search_manager = ssm
    health_monitor = hm
    cache_manager = cm
    task_manager = tm
    query_learning_system = qls
    knowledge_graph = kg

    logger.info("MCP server initialized")


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "RAG System MCP Server",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Get system health status."""
    try:
        if health_monitor:
            metrics = health_monitor.get_metrics()
            status = metrics.get('status', 'unknown')
        else:
            metrics = {}
            status = 'unknown'

        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            metrics=metrics
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute search query."""
    if not query_engine:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    try:
        # Execute query
        results = query_engine.query(
            request.query,
            n_results=request.n_results
        )

        # Record in learning system
        if query_learning_system:
            query_learning_system.record_query(
                query=request.query,
                results_count=len(results),
                correlation_id=request.correlation_id
            )

        return QueryResponse(
            results=results,
            metadata={
                'query': request.query,
                'result_count': len(results),
                'timestamp': datetime.now().isoformat()
            },
            correlation_id=request.correlation_id
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index or re-index codebase."""
    if not query_engine:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    try:
        if request.background and task_manager:
            # Run in background
            def index_task():
                return query_engine.index_codebase(
                    incremental=request.incremental,
                    use_git=request.use_git
                )

            task_id = task_manager.submit_task(index_task)

            return IndexResponse(
                status="started",
                stats={},
                task_id=task_id
            )

        else:
            # Run synchronously
            stats = query_engine.index_codebase(
                incremental=request.incremental,
                use_git=request.use_git
            )

            return IndexResponse(
                status="completed",
                stats=stats or {}
            )

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/searches")
async def list_searches(tags: Optional[str] = None):
    """List saved searches."""
    if not saved_search_manager:
        raise HTTPException(status_code=503, detail="Saved search manager not initialized")

    try:
        tag_list = tags.split(',') if tags else None

        searches = saved_search_manager.list_searches(tags=tag_list)

        return {
            "searches": [s.to_dict() for s in searches],
            "total": len(searches)
        }

    except Exception as e:
        logger.error(f"List searches failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/searches")
async def create_search(request: SavedSearchRequest):
    """Create saved search."""
    if not saved_search_manager:
        raise HTTPException(status_code=503, detail="Saved search manager not initialized")

    try:
        search = saved_search_manager.save_search(
            name=request.name,
            query=request.query,
            description=request.description,
            tags=request.tags,
            n_results=request.n_results
        )

        return {
            "success": True,
            "search": search.to_dict()
        }

    except Exception as e:
        logger.error(f"Create search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/searches/{search_id}")
async def get_search(search_id: str):
    """Get saved search by ID."""
    if not saved_search_manager:
        raise HTTPException(status_code=503, detail="Saved search manager not initialized")

    try:
        search = saved_search_manager.get_search(search_id)

        if not search:
            raise HTTPException(status_code=404, detail="Search not found")

        return search.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/searches/{search_id}/execute")
async def execute_search(search_id: str):
    """Execute saved search."""
    if not saved_search_manager or not query_engine:
        raise HTTPException(status_code=503, detail="Services not initialized")

    try:
        result = saved_search_manager.execute_search(search_id, query_engine)

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Execute search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/searches/{search_id}")
async def delete_search(search_id: str):
    """Delete saved search."""
    if not saved_search_manager:
        raise HTTPException(status_code=503, detail="Saved search manager not initialized")

    try:
        deleted = saved_search_manager.delete_search(search_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Search not found")

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        metrics = {}

        # Health metrics
        if health_monitor:
            metrics['health'] = health_monitor.get_metrics()

        # Cache metrics
        if cache_manager:
            metrics['cache'] = cache_manager.get_statistics()

        # Query performance
        if query_learning_system:
            metrics['performance'] = query_learning_system.get_performance_metrics(days=7)

        return metrics

    except Exception as e:
        logger.error(f"Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get background task status."""
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")

    try:
        status = task_manager.get_task_status(task_id)

        if not status:
            raise HTTPException(status_code=404, detail="Task not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get task status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches."""
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache manager not initialized")

    try:
        cache_manager.clear()

        return {"success": True, "message": "Cache cleared"}

    except Exception as e:
        logger.error(f"Clear cache failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-graph/entity/{entity_name}")
async def get_entity_context(entity_name: str):
    """Get knowledge graph context for entity."""
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")

    try:
        context = knowledge_graph.get_entity_context(entity_name)

        if not context:
            raise HTTPException(status_code=404, detail="Entity not found")

        return context

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get entity context failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-graph/related/{entity_name}")
async def get_related_entities(
    entity_name: str,
    max_depth: int = 2
):
    """Get related entities from knowledge graph."""
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")

    try:
        related = knowledge_graph.find_related_entities(entity_name, max_depth=max_depth)

        return {
            "entity": entity_name,
            "related": related,
            "count": len(related)
        }

    except Exception as e:
        logger.error(f"Get related entities failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query-history")
async def get_query_history(days: int = 7):
    """Get query history insights."""
    if not query_learning_system:
        raise HTTPException(status_code=503, detail="Query learning not initialized")

    try:
        insights = query_learning_system.get_learning_insights()
        performance = query_learning_system.get_performance_metrics(days=days)
        popular = query_learning_system.get_popular_queries(days=days, top_k=10)

        return {
            "insights": insights,
            "performance": performance,
            "popular_queries": popular
        }

    except Exception as e:
        logger.error(f"Get query history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggestions")
async def get_suggestions(partial: str):
    """Get query suggestions."""
    if not query_learning_system:
        raise HTTPException(status_code=503, detail="Query learning not initialized")

    try:
        suggestions = query_learning_system.get_query_suggestions(
            partial,
            max_suggestions=10
        )

        return {
            "partial": partial,
            "suggestions": suggestions
        }

    except Exception as e:
        logger.error(f"Get suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start MCP server.

    Args:
        host: Host address
        port: Port number
    """
    import uvicorn

    logger.info(f"Starting MCP server on {host}:{port}")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
