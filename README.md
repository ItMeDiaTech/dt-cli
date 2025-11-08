# Local RAG Plugin for Claude Code

A comprehensive Retrieval-Augmented Generation (RAG) plugin for Claude Code that combines vector embeddings with intelligent information retrieval and Multi-Agent Framework (MAF) orchestration for context-aware development assistance.

## Features

- **Local Vector Embeddings**: Uses sentence-transformers for local, privacy-preserving embeddings
- **Vector Database**: ChromaDB for efficient similarity search
- **Multi-Agent Framework**: LangGraph-based orchestration for intelligent task handling
- **Zero Token Overhead**: Direct bridge to Claude Code CLI using MCP
- **Auto-Setup**: Automatic installation of hooks, MCP servers, and slash commands
- **Open Source**: 100% free and open source components

## Architecture

### RAG System
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Ingestion**: Automatic codebase indexing
- **Query**: Semantic search with context ranking

### Multi-Agent Framework
- **Code Analyzer Agent**: Analyzes code structure and patterns
- **Documentation Retriever Agent**: Finds relevant documentation
- **Context Synthesizer Agent**: Combines multiple sources
- **Suggestion Generator Agent**: Provides context-aware recommendations

### Claude Code Integration
- **MCP Server**: Model Context Protocol server for tool integration
- **Session Hook**: Auto-starts RAG system on session start
- **Slash Commands**: `/rag-query`, `/rag-index`, `/rag-status`

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Quick Install

1. **Clone the repository**:
```bash
git clone https://github.com/ItMeDiaTech/dt-cli.git
cd dt-cli
```

2. **Install Python dependencies**:
```bash
pip3 install -r requirements.txt
```

Or with a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python3 -c "import httpx, chromadb, sentence_transformers, fastapi; print('[OK] All dependencies installed')"
```

4. **Start using**:
The plugin will automatically activate when you start a Claude Code session in this directory. The SessionStart hook will:
- Launch the MCP server on port 8765
- Index your codebase (first run only)
- Make all `/rag-*` commands available

### Manual Installation

If you prefer to manually set up:

```bash
# Install dependencies
./install.sh

# Or manually start the MCP server
python3 -m src.mcp_server.server
```

### Troubleshooting

**Dependencies not installed:**
```bash
pip3 install -r requirements.txt
```

**MCP server won't start:**
- Check if port 8765 is available: `lsof -i :8765`
- Check logs: `tail -f /tmp/rag-maf-mcp.log`
- Verify dependencies are installed (see step 3 above)

**Commands not working:**
- Ensure MCP server is running: `/rag-status`
- Check that you're in the project directory
- Verify `.claude/` configuration files are present

For detailed troubleshooting, see [CODEBASE_AUDIT_REPORT.md](./CODEBASE_AUDIT_REPORT.md)

## Usage

### Slash Commands

**Query RAG System**:
```
/rag-query how does authentication work?
```

**Index Codebase**:
```
/rag-index
```

**Check Status**:
```
/rag-status
```

### Automatic Features

- **Auto-Indexing**: Codebase is automatically indexed on session start
- **Context Awareness**: RAG system provides context to Claude automatically
- **Multi-Agent Orchestration**: Complex queries trigger agent collaboration

## Configuration

Configuration is stored in `.claude/rag-config.json`:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "max_results": 5,
  "maf_enabled": true
}
```

## Components

### RAG System (`src/rag/`)
- `embeddings.py`: Embedding generation
- `vector_store.py`: ChromaDB interface
- `ingestion.py`: Document processing
- `query_engine.py`: Search and retrieval

### MAF System (`src/maf/`)
- `orchestrator.py`: Agent coordination
- `agents.py`: Agent implementations
- `context_manager.py`: Context handling

### MCP Server (`src/mcp_server/`)
- `server.py`: MCP protocol implementation
- `tools.py`: RAG and MAF tools
- `bridge.py`: Claude Code CLI bridge

## Requirements

- Python 3.8+
- ChromaDB
- sentence-transformers
- LangGraph
- FastAPI
- uvicorn

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please open issues or pull requests.
