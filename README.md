# dt-cli: 100% Open Source RAG/MAF/LLM Development System

**A comprehensive development assistance system combining Retrieval-Augmented Generation (RAG), Multi-Agent Framework (MAF), and configurable LLM backends - completely open source and free.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open Source](https://img.shields.io/badge/Open%20Source-100%25-green.svg)](https://github.com/ItMeDiaTech/dt-cli)

---

## 🎯 Overview

**dt-cli** is a powerful development assistant that provides:
- **Intelligent code search** using semantic RAG with AST-based chunking
- **Automated debugging** with multi-agent error analysis
- **Code review** with security checks and quality scoring
- **Knowledge graph** for dependency tracking and impact analysis
- **Quality evaluation** using RAGAS metrics
- **Hybrid search** combining semantic and keyword algorithms
- **Three interaction modes**: Claude Code plugin, Interactive TUI, or REST API

**100% Free & Open Source** - No API keys required for local LLMs (Ollama, vLLM)

---

## ✨ Key Features

### 🔍 Advanced RAG System
- **AST-Based Chunking**: Intelligent code parsing using tree-sitter for Python, JavaScript, TypeScript
- **BGE Embeddings**: Instruction-aware embeddings for better code understanding
- **Auto-Trigger**: Automatic determination of when to use RAG vs. direct LLM
- **Intent Classification**: Semantic routing based on query intent

### 🤖 Agentic Debugging
- **Error Analysis**: Automatic root cause identification from stack traces
- **Fix Suggestions**: Multi-step reasoning for proposed fixes
- **Security Checks**: Detection of SQL injection, XSS, and OWASP Top 10 vulnerabilities
- **Code Review**: Quality scoring (0-10) with severity-categorized issues

### 🕸️ Knowledge Graph
- **Dependency Tracking**: What does this code depend on?
- **Impact Analysis**: What breaks if I change this?
- **Usage Finding**: Where is this function/class used?
- **Relationship Mapping**: Full code relationship graph

### 📊 Quality Evaluation
- **RAGAS Metrics**: Context relevance, answer faithfulness, answer relevance
- **Hybrid Search**: BM25 + semantic search with tunable weights
- **A/B Testing**: Compare different RAG configurations
- **Performance Metrics**: Query time, cache hit rate, confidence scores

### 🎨 Three Interaction Modes

**1. Claude Code Plugin (MCP)**
```bash
# Auto-configured via .claude/mcp-config.json
# Use dt-cli tools seamlessly in Claude Code conversations
```

**2. Intelligent Interactive CLI** ⭐ **ENHANCED**
```bash
python src/cli/interactive.py
# Natural language interface with intelligent context awareness
# Hierarchical session memory across CLI restarts
# Auto-discovers project files for enhanced context
# 10+ slash commands for power users
```

**NEW Features in Interactive CLI:**
- 🧠 **Session History with Hierarchical Memory** - Conversations persist across sessions with intelligent compression
- 🎯 **Context-Aware Queries** - Automatically includes relevant project files in queries
- 📁 **Smart File Discovery** - Indexes your project automatically for better context
- 💬 **Natural Language Input** - Just type what you need, no menu navigation required
- 📊 **Conversation Continuity** - Resume from where you left off, even days later
- ⚡ **Importance Scoring** - Critical conversations are never forgotten

**3. REST API**
```bash
# Start server
python src/mcp_server/standalone_server.py

# Use API
curl http://localhost:8765/query -X POST -d '{"query": "..."}'
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/ItMeDiaTech/dt-cli.git
cd dt-cli

# Install dependencies
pip install -r requirements.txt

# Start interactive TUI
python dt-cli.py
```

**That's it!** The system is ready to use.

### First Steps

**Option 1: Interactive TUI**
```bash
python dt-cli.py
# Choose from the menu:
# 1. Ask a Question
# 2. Debug an Error
# 3. Review Code
# etc.
```

**Option 2: Start Server for API/Claude Code**
```bash
# Start the server
python src/mcp_server/standalone_server.py

# Server runs on http://localhost:8765
# Claude Code will auto-detect via .claude/mcp-config.json
```

**Option 3: Use as Claude Code Plugin**
1. Ensure server is running
2. Claude Code auto-detects MCP configuration
3. Use dt-cli tools directly in conversations

---

## 📚 Usage

### Interactive CLI with Intelligent Features ⭐

The **new Interactive CLI** (`src/cli/interactive.py`) provides a natural language interface with production-grade conversation memory:

```bash
python src/cli/interactive.py
```

**Key Features:**

🧠 **Hierarchical Session Memory** (Based on 2024-2025 Research)
```
> Review codebase and find any errors
[Analyzing entire codebase in /home/user/dt-cli...]
[System remembers this conversation across sessions]

> (Next day) What errors did we discuss yesterday?
[Retrieves relevant history from hierarchical memory]
```

- **4-Level Memory Hierarchy:**
  - Level 1: Working Memory (last 20 turns, full detail)
  - Level 2: Summarized Context (automatic compression)
  - Level 3: Session Summary (when closed)
  - Level 4: Archived Sessions (retrievable history)

- **Automatic Compression:** ~90% memory reduction while preserving important information
- **Importance Scoring:** Critical conversations (debug, code changes) never forgotten
- **Persistent Storage:** `~/.dt_cli_sessions.json` survives CLI restarts

🎯 **Context-Aware Queries**
```
> Where is authentication handled?
[Automatically includes relevant auth files as context]
[Project: dt-cli] Where is authentication handled?
  Context files: src/auth/*.py (intelligently selected)
```

**Slash Commands:**
```
/history          - View current session with hierarchical memory
/sessions         - List all sessions (current + archived)
/stats            - Show memory usage and statistics
/clearsession     - Clear all history (with confirmation)
/verbosity <level> - Set output detail (quiet/normal/verbose)
/folder           - Change project folder
/help             - Show comprehensive help
/exit             - Exit and save session
```

**Natural Language Interaction:**
```
> Review codebase and find any errors
  ✓ Detects REVIEW intent
  ✓ Uses project folder automatically
  ✓ No redundant prompts!

> Debug this authentication error
  ✓ Detects DEBUG intent
  ✓ High importance score (0.95)
  ✓ Always kept in memory

> What did we just fix?
  ✓ Follows up using conversation history
  ✓ Context from previous turns
```

**Session Statistics Example:**
```
> /stats

Session Statistics
══════════════════════════════════════════════════
Metric                    | Value
──────────────────────────┼──────────────────────
Current Session Active    | Yes
Current Session Turns     | 45
Archived Sessions         | 3
Total Archived Turns      | 187
Total All Turns           | 232
Storage File              | ~/.dt_cli_sessions.json
```

### Traditional Menu Interface (dt-cli.py)

For users preferring a traditional menu:

```
┌─────────────────────────────────────────────┐
│      dt-cli - Interactive Terminal UI       │
│   RAG/MAF/LLM System - 100% Open Source     │
└─────────────────────────────────────────────┘

Main Menu:
  1. Ask a Question (RAG Query)      → Semantic code search
  2. Debug an Error                   → AI error analysis
  3. Review Code                      → Quality & security checks
  4. Explore Knowledge Graph          → Dependencies & impact
  5. Evaluate RAG Quality             → RAGAS metrics
  6. Hybrid Search                    → Semantic + keyword
  7. View Statistics                  → System health
  8. Settings                         → Configuration
  9. Help                            → Documentation
  0. Exit
```

### API Endpoints

**Query RAG System**
```bash
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work?",
    "auto_trigger": true
  }'
```

**Debug Error**
```bash
curl -X POST http://localhost:8765/debug \
  -H "Content-Type: application/json" \
  -d '{
    "error_output": "KeyError: value...",
    "auto_extract_code": true
  }'
```

**Review Code**
```bash
curl -X POST http://localhost:8765/review \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def login(user, pwd): ...",
    "language": "python"
  }'
```

**Build Knowledge Graph**
```bash
curl -X POST http://localhost:8765/graph/build \
  -H "Content-Type: application/json" \
  -d '{"path": "src/"}'
```

**Query Knowledge Graph**
```bash
curl -X POST http://localhost:8765/graph/query \
  -H "Content-Type: application/json" \
  -d '{
    "entity_name": "parse_code",
    "query_type": "dependencies"
  }'
```

**Evaluate RAG**
```bash
curl -X POST http://localhost:8765/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "retrieved_contexts": ["ctx1", "ctx2"],
    "generated_answer": "answer",
    "ground_truth": "expected"
  }'
```

**Hybrid Search**
```bash
curl -X POST http://localhost:8765/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication",
    "documents": ["doc1", "doc2"],
    "semantic_weight": 0.7,
    "keyword_weight": 0.3
  }'
```

**View Statistics**
```bash
curl http://localhost:8765/info
curl http://localhost:8765/graph/stats
curl http://localhost:8765/auto-trigger/stats
```

---

## ⚙️ Configuration

### LLM Configuration (`llm-config.yaml`)

```yaml
llm:
  provider: "openai"     # or "anthropic", "local", "ollama"
  model: "gpt-4"
  temperature: 0.7
  api_key_env: "OPENAI_API_KEY"  # Environment variable name

embedding:
  model: "BAAI/bge-base-en-v1.5"
  device: "cpu"  # or "cuda"
  instruction_prefix: "Represent this code for retrieval: "

auto_trigger:
  enabled: true
  similarity_threshold: 0.7
  intent_threshold: 0.6
  cache_ttl: 900  # 15 minutes

vector_store:
  collection_name: "dt_cli_code"
  persist_directory: "./chroma_db"
  chunk_size: 1000
  chunk_overlap: 200

hybrid_search:
  semantic_weight: 0.7
  keyword_weight: 0.3
  query_expansion: true

knowledge_graph:
  cache_size: 1000
  analysis_timeout: 300
```

### Environment Variables (`.env`)

```bash
# LLM API Keys (choose what you need)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Server Configuration
DT_CLI_HOST=0.0.0.0
DT_CLI_PORT=8765

# Logging
LOG_LEVEL=INFO
```

### Using Local LLMs (No API Keys!)

```yaml
# llm-config.yaml
llm:
  provider: "ollama"
  model: "codellama:7b"
  base_url: "http://localhost:11434"
  # No API key needed!
```

---

## 🏗️ Architecture

### System Components

```
dt-cli/
├── src/
│   ├── rag/                   # RAG System
│   │   ├── parsers.py         # Tree-sitter AST parsers
│   │   ├── ast_chunker.py     # Intelligent code chunking
│   │   ├── embeddings.py      # BGE embeddings
│   │   ├── intent_router.py   # Query intent classification
│   │   └── auto_trigger.py    # Auto-trigger orchestration
│   │
│   ├── debugging/             # Agentic Debugging
│   │   ├── debug_agent.py     # Error analysis agent
│   │   └── review_agent.py    # Code review agent
│   │
│   ├── graph/                 # Knowledge Graph
│   │   └── knowledge_graph.py # Dependency tracking
│   │
│   ├── evaluation/            # Quality Metrics
│   │   ├── ragas.py           # RAGAS evaluator
│   │   └── hybrid_search.py   # BM25 + semantic search
│   │
│   ├── mcp_server/            # MCP Server
│   │   └── standalone_server.py  # FastAPI server
│   │
│   └── cli/                   # Interactive TUI
│       └── interactive.py     # Rich-based interface
│
├── .claude/
│   └── mcp-config.json        # Claude Code integration
│
└── dt-cli.py                  # Entry point
```

### Data Flow

```
User Query
    ↓
Auto-Trigger (Intent Classification)
    ↓
┌───────────┬───────────┐
│    RAG    │  Direct   │
│  Search   │   LLM     │
└─────┬─────┴─────┬─────┘
      │           │
   Context    No Context
      ↓           ↓
  ┌─────────────────┐
  │   LLM Provider  │
  │ (OpenAI/Ollama) │
  └────────┬────────┘
           ↓
        Response
```

---

## 📖 Documentation

### Guides
- [Integration Guide](./INTEGRATION_GUIDE.md) - Complete integration documentation
- [Installation](./docs/guides/INSTALLATION.md) - Detailed installation instructions
- [Quick Start](./docs/guides/QUICKSTART.md) - Get started in 5 minutes
- [User Guide](./docs/guides/USER_GUIDE.md) - Comprehensive user documentation
- [Architecture](./docs/guides/ARCHITECTURE.md) - System architecture details

### Implementation Phases
- [Phase 1: AST Chunking & Auto-Trigger](./docs/phases/PHASE1_WEEK1_COMPLETE.md)
- [Phase 2: Agentic Debugging](./docs/phases/PHASE2_COMPLETE.md)
- [Phase 3: Knowledge Graph](./docs/phases/PHASE3_COMPLETE.md)
- [Phase 4: RAGAS & Hybrid Search](./docs/phases/PHASE4_COMPLETE.md)

### Reference
- [API Reference](./docs/API_REFERENCE.md) - Complete API documentation
- [Configuration Guide](./docs/CONFIGURATION.md) - All configuration options

---

## 🛠️ Development

### Project Structure

```
src/
├── rag/           # Retrieval-Augmented Generation
├── maf/           # Multi-Agent Framework
├── llm/           # LLM provider abstraction
├── config/        # Configuration management
├── debugging/     # Debug & review agents
├── graph/         # Knowledge graph system
├── evaluation/    # Quality evaluation
├── mcp_server/    # MCP server implementation
└── cli/           # Interactive TUI

tests/             # Comprehensive test suite
├── rag/
├── debugging/
├── graph/
├── evaluation/
└── cli/

docs/              # Documentation
├── guides/        # User guides
├── phases/        # Implementation phases
└── archive/       # Historical documentation
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run specific test suite
pytest tests/rag/
pytest tests/debugging/
pytest tests/cli/

# Run with coverage
pytest --cov=src tests/
```

### Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

---

## 🎓 Use Cases

### For Developers
- **Codebase Navigation**: Quickly understand large codebases
- **Bug Fixing**: Get automated error analysis and fix suggestions
- **Code Review**: Catch security issues before deployment
- **Refactoring**: Understand impact before making changes

### For Teams
- **Knowledge Sharing**: Build team knowledge graph
- **Quality Assurance**: Automated code quality checks
- **Documentation**: Generate context-aware documentation
- **Onboarding**: Help new developers understand code

### For Learning
- **Code Understanding**: Learn how code works through Q&A
- **Best Practices**: Get suggestions aligned with standards
- **Security**: Learn about common vulnerabilities
- **Patterns**: Discover architectural patterns in code

---

## 📊 Performance

### Benchmarks

| Operation | Avg Time | Cache Hit Rate |
|-----------|----------|----------------|
| RAG Query | 245ms | 67% |
| Error Debug | 1.2s | N/A |
| Code Review | 2.5s | N/A |
| Graph Build | 15s (1000 files) | N/A |
| Graph Query | 50ms | 85% |

### Optimization Tips

1. **Use Hybrid Search Weights Tuning**
   ```python
   from src.evaluation.hybrid_search import HybridSearch
   search = HybridSearch()
   search.tune_weights(queries, ground_truth, scores)
   ```

2. **Adjust Chunk Size for Your Codebase**
   - Smaller chunks (500-800): Better precision
   - Larger chunks (1500-2000): Better context

3. **Pre-build Knowledge Graph**
   ```bash
   curl -X POST http://localhost:8765/graph/build \
     -d '{"path": "src/"}'
   ```

4. **Use Auto-Trigger Threshold Tuning**
   - Higher (0.8+): More direct LLM calls, faster
   - Lower (0.6-): More RAG usage, better context

---

## 🔧 Troubleshooting

### Common Issues

**Server Won't Start**
```bash
# Check if port is in use
lsof -i :8765

# Use different port
python src/mcp_server/standalone_server.py --port 8766
```

**Import Errors**
```bash
# Ensure correct directory
cd dt-cli

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**Tree-sitter Errors**
```bash
# Reinstall parsers
rm -rf ~/.tree-sitter
python -c "from src.rag.parsers import ParserRegistry; ParserRegistry()"
```

**Low RAG Quality**
1. Tune hybrid search weights
2. Adjust chunk size in config
3. Use RAGAS evaluation to identify issues

**Claude Code Integration Issues**
1. Verify server is running: `curl http://localhost:8765/health`
2. Check `.claude/mcp-config.json` exists
3. Restart Claude Code
4. Check logs for errors

See [Integration Guide](./INTEGRATION_GUIDE.md) for detailed troubleshooting.

---

## 🌟 Features Roadmap

### ✅ Completed (v1.0)
- AST-based chunking with tree-sitter
- BGE embeddings with instruction prefix
- Auto-trigger with intent classification
- Debug agent with error analysis
- Code review agent with security checks
- Knowledge graph with dependency tracking
- RAGAS evaluation metrics
- Hybrid search (BM25 + semantic)
- Interactive TUI with Rich
- Claude Code MCP integration
- REST API server

### 🚧 In Progress
- Additional language support (Go, Rust, Java)
- Web UI dashboard
- VS Code extension
- Docker containerization

### 📋 Planned
- Conversation memory across sessions
- Custom agent creation framework
- Team collaboration features
- Integration with CI/CD pipelines
- Metrics dashboard
- Plugin marketplace

---

## 📄 License

MIT License - see [LICENSE](./LICENSE) file for details.

This project is 100% free and open source. You can:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

---

## 🙏 Acknowledgments

Built with these amazing open source projects:
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - Embeddings
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [FastAPI](https://github.com/tiangolo/fastapi) - REST API framework
- [Rich](https://github.com/Textualize/rich) - Terminal UI
- [tree-sitter](https://github.com/tree-sitter/tree-sitter) - Code parsing
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) - Keyword search

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ItMeDiaTech/dt-cli/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ItMeDiaTech/dt-cli/discussions)
- **Documentation**: [docs/](./docs/)

---

## 🎉 Quick Links

- [Installation Guide](./INTEGRATION_GUIDE.md)
- [Interactive TUI Demo](#-three-interaction-modes)
- [API Documentation](#-usage)
- [Configuration Options](#%EF%B8%8F-configuration)
- [Architecture Overview](#%EF%B8%8F-architecture)
- [Contributing Guidelines](#%EF%B8%8F-development)

---

**Made with ❤️ by the dt-cli team | 100% Open Source**
