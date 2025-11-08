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

### Ubuntu Server (Complete Setup)

For a complete installation on Ubuntu Server including Claude Code + dt-cli:

```bash
# Quick start
git clone <repository-url> dt-cli
cd dt-cli
chmod +x ubuntu-install.sh
./ubuntu-install.sh
```

📖 **Installation Guides:**
- 🚀 [Quick Start Guide](./QUICKSTART_UBUNTU.md) - Get started in 5 minutes
- 📋 [Complete Deployment Guide](./UBUNTU_DEPLOYMENT_GUIDE.md) - Full documentation with authentication, troubleshooting, and advanced scenarios

### Standard Installation

Simply install the plugin in Claude Code:

```bash
# The plugin will auto-install all dependencies and configure itself
claude install-plugin dt-cli
```

Or manually:

```bash
# Clone the repository
git clone <repository-url>
cd dt-cli

# Run installation script
./install.sh
```

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
