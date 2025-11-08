# RAG-MAF Plugin Architecture

## Overview

The RAG-MAF plugin is a sophisticated system that combines Retrieval-Augmented Generation (RAG) with Multi-Agent Framework (MAF) orchestration to provide context-aware development assistance in Claude Code.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Claude Code CLI                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ MCP Protocol (HTTP/JSON-RPC)
                 │
┌────────────────┴────────────────────────────────────────────────┐
│                    MCP Server (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Claude Code Bridge                          │  │
│  │  - Tool routing                                          │  │
│  │  - Context formatting                                    │  │
│  │  - Zero-token overhead integration                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│           ┌─────────────────┴─────────────────┐                │
│           │                                   │                │
│  ┌────────┴────────┐              ┌──────────┴─────────┐       │
│  │   RAG Tools     │              │    MAF Tools       │       │
│  └────────┬────────┘              └──────────┬─────────┘       │
└───────────┼───────────────────────────────────┼─────────────────┘
            │                                   │
            │                                   │
┌───────────┴────────────┐          ┌──────────┴──────────────────┐
│     RAG System         │          │   Multi-Agent Framework     │
│                        │          │                             │
│  ┌─────────────────┐  │          │  ┌──────────────────────┐  │
│  │ Embedding       │  │          │  │ Agent Orchestrator   │  │
│  │ Engine          │  │          │  │ (LangGraph)          │  │
│  └─────────────────┘  │          │  └──────────────────────┘  │
│                        │          │                             │
│  ┌─────────────────┐  │          │  ┌──────────────────────┐  │
│  │ Vector Store    │  │          │  │ Code Analyzer        │  │
│  │ (ChromaDB)      │  │          │  │ Agent                │  │
│  └─────────────────┘  │          │  └──────────────────────┘  │
│                        │          │                             │
│  ┌─────────────────┐  │          │  ┌──────────────────────┐  │
│  │ Document        │  │          │  │ Documentation        │  │
│  │ Ingestion       │  │          │  │ Retriever Agent      │  │
│  └─────────────────┘  │          │  └──────────────────────┘  │
│                        │          │                             │
│  ┌─────────────────┐  │          │  ┌──────────────────────┐  │
│  │ Query Engine    │  │          │  │ Context Synthesizer  │  │
│  └─────────────────┘  │          │  │ Agent                │  │
│                        │          │  └──────────────────────┘  │
│                        │          │                             │
│                        │          │  ┌──────────────────────┐  │
│                        │          │  │ Suggestion Generator │  │
│                        │          │  │ Agent                │  │
│                        │          │  └──────────────────────┘  │
└────────────────────────┘          └─────────────────────────────┘
```

## Component Details

### 1. RAG System

#### Embedding Engine
- **Technology**: sentence-transformers
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Features**:
  - Local embedding generation
  - No API calls or token usage
  - Fast inference on CPU
  - Privacy-preserving

#### Vector Store
- **Technology**: ChromaDB
- **Features**:
  - Persistent local storage
  - Efficient similarity search
  - Metadata filtering
  - Automatic collection management

#### Document Ingestion
- **Process**:
  1. File discovery (respects .gitignore patterns)
  2. Content extraction
  3. Text chunking with overlap
  4. Metadata attachment
  5. Unique ID generation

#### Query Engine
- **Capabilities**:
  - Semantic search across codebase
  - File type filtering
  - Relevance scoring
  - Context ranking

### 2. Multi-Agent Framework (MAF)

#### Agent Orchestrator
- **Technology**: LangGraph
- **Workflow**:
  ```
  Start -> Code Analyzer -> Documentation Retriever
                            ↓
                        Synthesizer
                            ↓
                    Suggestion Generator -> End
  ```

#### Agents

##### Code Analyzer Agent
- Analyzes code structure and patterns
- Queries RAG for relevant code examples
- Identifies implementation patterns
- Provides relevance scoring

##### Documentation Retriever Agent
- Searches for relevant documentation
- Filters by file type (.md, .rst, .txt)
- Ranks documentation by relevance
- Extracts key information

##### Context Synthesizer Agent
- Combines results from multiple agents
- Creates unified context
- Generates summaries
- Identifies relationships

##### Suggestion Generator Agent
- Generates context-aware suggestions
- Identifies related files
- Provides next steps
- Creates actionable recommendations

### 3. MCP Server

#### Server Implementation
- **Framework**: FastAPI
- **Protocol**: HTTP/JSON-RPC
- **Port**: 8765 (configurable)
- **Features**:
  - RESTful API
  - Tool registration
  - Context management
  - Error handling

#### Bridge Layer
- Translates between Claude Code and internal systems
- Formats results for optimal display
- Provides zero-token context injection
- Manages tool routing

### 4. Claude Code Integration

#### Session Hook
- Auto-starts on session initialization
- Checks for existing MCP server
- Triggers initial indexing if needed
- Provides user feedback

#### Slash Commands
- `/rag-query`: Query the RAG system
- `/rag-index`: Index or re-index codebase
- `/rag-status`: Display system status

#### MCP Configuration
- Tool definitions
- Server endpoints
- Auto-start settings

## Data Flow

### Query Flow

```
1. User Query
   ↓
2. Claude Code CLI (via slash command or MCP tool)
   ↓
3. MCP Server receives request
   ↓
4. Bridge routes to appropriate system
   ↓
5a. RAG System:                    5b. MAF System:
    - Generate query embedding         - Create context
    - Search vector store              - Execute agents in parallel
    - Rank results                     - Synthesize results
    - Return matches                   - Generate suggestions
   ↓                                  ↓
6. Bridge formats results
   ↓
7. Return to Claude Code
   ↓
8. Display to user
```

### Indexing Flow

```
1. Trigger (manual or auto)
   ↓
2. File Discovery
   - Scan directory tree
   - Filter by extension
   - Exclude ignored directories
   ↓
3. Document Processing
   - Read file content
   - Chunk text
   - Add metadata
   ↓
4. Embedding Generation
   - Batch process chunks
   - Generate vectors
   ↓
5. Vector Store
   - Insert documents
   - Create indexes
   - Persist to disk
```

## Configuration

### RAG Configuration
```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "max_results": 5
}
```

### MAF Configuration
```json
{
  "enabled": true,
  "agents": {
    "code_analyzer": true,
    "doc_retriever": true,
    "synthesizer": true,
    "suggestion_generator": true
  }
}
```

### MCP Configuration
```json
{
  "host": "127.0.0.1",
  "port": 8765,
  "auto_start": true
}
```

## Performance Characteristics

### Embedding Generation
- Speed: ~1000 sentences/second on CPU
- Memory: ~500MB for model
- Latency: <100ms for single query

### Vector Search
- Speed: <10ms for similarity search
- Scalability: Handles 100k+ documents
- Memory: ~1GB for 50k documents

### Agent Orchestration
- Latency: 100-500ms depending on complexity
- Parallel execution where possible
- Cached results for repeated queries

## Security & Privacy

- **Fully Local**: No data leaves your machine
- **No API Keys**: No external services required
- **No Telemetry**: ChromaDB telemetry disabled
- **Privacy-First**: All processing is local

## Extensibility

### Adding New Agents
```python
class CustomAgent(BaseAgent):
    def execute(self, context):
        # Your logic here
        return results
```

### Custom Embeddings
```python
engine = EmbeddingEngine(model_name="your-model")
```

### Additional Tools
```python
class CustomTools:
    def get_tools(self):
        return [tool_definitions]
```

## Deployment

### Standalone Mode
```bash
./rag-maf start
```

### Service Mode
```bash
sudo systemctl enable rag-maf-mcp
sudo systemctl start rag-maf-mcp
```

### Development Mode
```bash
python -m src.mcp_server.server
```

## Monitoring

### Logs
- MCP Server: `/tmp/rag-maf-mcp.log`
- System logs: `./logs/`

### Status Endpoint
- Health: `http://127.0.0.1:8765/health`
- Status: `http://127.0.0.1:8765/status`

## Future Enhancements

1. **Incremental Indexing**: Index only changed files
2. **Multi-Language Support**: Language-specific embeddings
3. **Custom Agents**: Plugin system for custom agents
4. **Advanced Filtering**: More sophisticated query filters
5. **Caching**: Query result caching
6. **Metrics**: Detailed performance metrics
