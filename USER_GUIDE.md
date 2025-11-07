# RAG-MAF Plugin User Guide

**Version**: 2.0.0
**Last Updated**: 2025-11-07

Complete guide to using the RAG-MAF (Retrieval-Augmented Generation with Multi-Agent Framework) plugin for Claude Code.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Features](#basic-features)
3. [Advanced Features](#advanced-features)
4. [Slash Commands](#slash-commands)
5. [MCP Server API](#mcp-server-api)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Index your codebase (first time)
/rag-index

# Query the system
/rag-query how does authentication work?
```

### First Query

The simplest way to use the RAG system:

```bash
/rag-query <your question>
```

Example:
```bash
/rag-query how does user authentication work?
```

---

## Basic Features

### 1. Querying the Codebase

**Simple Query**:
```bash
/rag-query database connection setup
```

**Advanced Query** (with profiling and explanations):
```bash
/rag-query-advanced authentication flow
```

The advanced query provides:
- Performance profiling
- Result explanations (why each result was returned)
- Query suggestions
- Option to save the search

### 2. Indexing

**Full Re-index**:
```bash
/rag-index
```

**Incremental Indexing** (only changed files):
```bash
/rag-index --incremental
```

**Background Indexing** (non-blocking):
```bash
/rag-index --background
```

### 3. System Status

Check system health and metrics:

```bash
/rag-status
```

View detailed metrics dashboard:

```bash
/rag-metrics
```

Shows:
- System health status
- Query performance (latency, throughput)
- Cache statistics
- Recent activity
- Top queries

---

## Advanced Features

### Saved Searches

Save frequently used queries for quick access.

**Save a Search**:
```bash
/rag-save auth | authentication flow | Find auth code | auth,security
```

Format: `name | query | description | tags`

**List Saved Searches**:
```bash
/rag-searches
```

**Filter by Tags**:
```bash
/rag-searches security
```

**Execute Saved Search**:
```bash
/rag-exec auth
```

Or use the search ID:
```bash
/rag-exec abc123def456
```

### Knowledge Graph

Query the knowledge graph to understand code relationships.

**Find Entity Relationships**:
```bash
/rag-graph UserManager
```

Shows:
- What uses this entity (dependencies)
- What this entity uses (imports, etc.)
- Related entities
- File location and line number

**Find Function Relationships**:
```bash
/rag-graph authenticate_user
```

### Query Performance

**Profiling**:

Queries are automatically profiled with detailed metrics:
- Execution time per stage
- Memory usage
- Bottleneck identification

View profile in query results under `_profile` metadata.

**Optimization Tips**:
- Use incremental indexing to keep index fresh
- Leverage saved searches for common queries
- Enable query prefetching for predictive loading
- Monitor cache hit rates in `/rag-metrics`

### Auto-Indexing

Git hooks automatically re-index on code changes.

**Install Git Hooks**:
```python
from src.git_integration import install_git_hooks
install_git_hooks()
```

Hooks installed:
- `post-commit` - Incremental index after commit
- `post-merge` - Incremental index after merge/pull
- `post-checkout` - Full re-index on branch switch

All hooks run in background (non-blocking).

### Cache Management

The system uses intelligent caching with multiple invalidation strategies.

**Cache Invalidation Triggers**:
- TTL expiration (default: 1 hour)
- File modifications detected
- Git commits
- Memory pressure

**Manual Cache Operations** (via MCP server):
```bash
curl -X POST http://127.0.0.1:8000/cache/clear
```

### Query Learning

The system learns from your query patterns to improve suggestions.

**Features**:
- Query history tracking
- Result selection tracking
- Feedback collection
- Similar query suggestions
- Popular queries analysis

**View Learning Insights** (via MCP server):
```bash
curl http://127.0.0.1:8000/query-history
```

### Index Warming

Pre-load frequently accessed data for faster cold-start performance.

**Automatic Warming**:
```python
from src.rag.index_warming import IndexWarmer

warmer = IndexWarmer(query_engine, query_learning_system)
warmer.warm_on_startup(background=True)
```

Warms:
- ML models (embeddings, reranker)
- Popular queries (top 10)
- Frequently accessed files

**Manual Warming**:
```python
warmer.warm_all(max_time_seconds=30)
```

### Query Prefetching

Predictively prefetch likely next queries based on learned patterns.

**Enable Prefetching**:
```python
from src.rag.query_prefetching import QueryPrefetcher

prefetcher = QueryPrefetcher(query_engine, cache_manager, query_learning_system)
prefetcher.learn_from_history()
prefetcher.start_prefetching()
```

**How It Works**:
1. Learns query sequences (A → B → C)
2. Predicts next likely queries
3. Prefetches results in background
4. Results available instantly when queried

**Statistics**:
```python
stats = prefetcher.get_statistics()
# {'total_patterns': 145, 'high_confidence_patterns': 23, ...}
```

---

## Slash Commands Reference

### Query Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/rag-query` | Basic query | `/rag-query authentication` |
| `/rag-query-advanced` | Advanced query with profiling | `/rag-query-advanced user login flow` |

### Search Management

| Command | Description | Example |
|---------|-------------|---------|
| `/rag-save` | Save a search | `/rag-save auth \| authentication flow` |
| `/rag-searches` | List saved searches | `/rag-searches` or `/rag-searches security` |
| `/rag-exec` | Execute saved search | `/rag-exec auth` |

### System Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/rag-index` | Index codebase | `/rag-index` |
| `/rag-status` | System status | `/rag-status` |
| `/rag-metrics` | Metrics dashboard | `/rag-metrics` |
| `/rag-graph` | Knowledge graph query | `/rag-graph UserManager` |

---

## MCP Server API

The system exposes a comprehensive REST API via the MCP server.

### Starting the Server

```python
from src.mcp import start_server, initialize_server

# Initialize with components
initialize_server(
    qe=query_engine,
    ssm=saved_search_manager,
    hm=health_monitor,
    cm=cache_manager,
    tm=task_manager,
    qls=query_learning_system,
    kg=knowledge_graph
)

# Start server
start_server(host="0.0.0.0", port=8000)
```

### API Endpoints

#### Health & Metrics

**GET /health**
```bash
curl http://127.0.0.1:8000/health
```

**GET /metrics**
```bash
curl http://127.0.0.1:8000/metrics
```

#### Query

**POST /query**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication flow",
    "n_results": 5,
    "use_hybrid": true,
    "use_reranking": true
  }'
```

#### Indexing

**POST /index**
```bash
curl -X POST http://127.0.0.1:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "incremental": true,
    "use_git": true,
    "background": false
  }'
```

#### Saved Searches

**GET /searches**
```bash
curl http://127.0.0.1:8000/searches
```

**POST /searches**
```bash
curl -X POST http://127.0.0.1:8000/searches \
  -H "Content-Type: application/json" \
  -d '{
    "name": "auth",
    "query": "authentication flow",
    "description": "Find auth code",
    "tags": ["auth", "security"]
  }'
```

**POST /searches/{search_id}/execute**
```bash
curl -X POST http://127.0.0.1:8000/searches/abc123/execute
```

#### Knowledge Graph

**GET /knowledge-graph/entity/{entity_name}**
```bash
curl http://127.0.0.1:8000/knowledge-graph/entity/UserManager
```

**GET /knowledge-graph/related/{entity_name}**
```bash
curl "http://127.0.0.1:8000/knowledge-graph/related/UserManager?max_depth=2"
```

#### Query History

**GET /query-history**
```bash
curl "http://127.0.0.1:8000/query-history?days=7"
```

**GET /suggestions**
```bash
curl "http://127.0.0.1:8000/suggestions?partial=auth"
```

---

## Performance Optimization

### Best Practices

1. **Use Incremental Indexing**
   - 96x faster than full re-indexing
   - Automatically enabled with Git hooks

2. **Leverage Caching**
   - 10x faster query performance
   - Intelligent auto-invalidation
   - Monitor cache hit rate in metrics

3. **Enable Query Prefetching**
   - Reduces latency by pre-loading likely queries
   - Learn patterns from history
   - Background execution

4. **Use Saved Searches**
   - Instant execution of common queries
   - Organize with tags
   - Share across team

5. **Monitor Performance**
   - Check `/rag-metrics` regularly
   - Watch for bottlenecks
   - Optimize based on insights

### Performance Targets

- **Query Latency**: < 500ms (avg), < 1s (P95)
- **Indexing**: < 1s for incremental (< 10 files)
- **Cache Hit Rate**: > 50% for common queries
- **Memory Usage**: < 1GB under normal load

### Tuning Parameters

**Query Engine**:
```python
config = {
    'use_cache': True,
    'lazy_loading': True,
    'batch_size': 32,
    'n_results': 5
}
```

**Cache**:
```python
cache_manager = IntelligentCacheManager(
    ttl_seconds=3600,          # 1 hour TTL
    enable_file_tracking=True,
    enable_git_tracking=True,
    enable_memory_pressure=True
)
```

**Prefetching**:
```python
prefetcher = QueryPrefetcher(
    query_engine,
    min_confidence=0.3  # Lower = more aggressive prefetching
)
```

---

## Troubleshooting

### Common Issues

**1. "MCP Server not available"**

Solution:
```bash
# Start the MCP server
python -m src.mcp.enhanced_server
```

**2. "No results found"**

Solutions:
- Re-index: `/rag-index`
- Check if files are in `.gitignore`
- Verify codebase path in config

**3. "Slow queries"**

Solutions:
- Check cache hit rate: `/rag-metrics`
- Enable incremental indexing
- Reduce `n_results`
- Check for bottlenecks in profile

**4. "High memory usage"**

Solutions:
- Enable lazy loading
- Clear cache: `curl -X POST http://127.0.0.1:8000/cache/clear`
- Reduce cache TTL
- Use incremental indexing

**5. "Index out of date"**

Solutions:
- Install Git hooks for auto-indexing
- Run `/rag-index --incremental` manually
- Check hook installation: `ls .git/hooks/`

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-07T12:00:00Z",
  "metrics": {
    "error_rate": 0.01,
    "memory_usage_mb": 512.5,
    "avg_query_latency_ms": 234.5
  }
}
```

### Getting Help

1. Check system metrics: `/rag-metrics`
2. View health status: `/rag-status`
3. Check logs for errors
4. Review configuration
5. Verify all dependencies installed

---

## Advanced Configuration

### Custom Embedding Model

```python
config = {
    'embedding_model': 'all-mpnet-base-v2',  # More accurate, slower
    # OR
    'embedding_model': 'all-MiniLM-L6-v2',   # Faster, default
}
```

### Custom Ignore Patterns

```python
ignore_dirs = {
    '__pycache__',
    '.git',
    'node_modules',
    'venv',
    'build',
    'dist'
}
```

### Hybrid Search Tuning

```python
# Adjust semantic vs keyword balance
semantic_weight = 0.7  # 70% semantic
keyword_weight = 0.3   # 30% keyword (BM25)
```

---

## FAQ

**Q: How often should I re-index?**

A: With Git hooks installed, never manually. Hooks auto-index on commits.

**Q: Can I use this offline?**

A: Yes! 100% local, zero external API dependencies.

**Q: What languages are supported?**

A: Entity extraction supports Python (AST-based), JavaScript, TypeScript, Java. Semantic search works for all languages.

**Q: How much disk space is needed?**

A: ChromaDB index typically 10-20% of source code size.

**Q: Can I customize the agents?**

A: Yes, see `src/maf/enhanced_orchestrator.py` for agent definitions.

**Q: Is this production-ready?**

A: Yes, with comprehensive tests, monitoring, and error handling.

---

## Next Steps

1. **Set up auto-indexing**: Install Git hooks
2. **Save common queries**: Use `/rag-save`
3. **Enable prefetching**: Reduce latency
4. **Monitor performance**: Check `/rag-metrics` weekly
5. **Share saved searches**: Export and import

---

**Documentation Version**: 2.0.0
**Plugin Version**: 2.0.0
**Last Updated**: 2025-11-07

For more information, see:
- [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) - Feature overview
- [COMPREHENSIVE_ANALYSIS.md](./COMPREHENSIVE_ANALYSIS.md) - Technical deep-dive
- [README.md](./README.md) - Project overview
