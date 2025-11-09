# dt-cli Integration Guide

Complete guide for using dt-cli as a Claude Code plugin or standalone interactive TUI.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage Methods](#usage-methods)
   - [Method 1: Claude Code Plugin](#method-1-claude-code-plugin)
   - [Method 2: Standalone Interactive TUI](#method-2-standalone-interactive-tui)
   - [Method 3: Direct API](#method-3-direct-api)
4. [Features](#features)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

---

## Overview

dt-cli provides **three ways** to interact with the RAG/MAF/LLM system:

1. **Claude Code Plugin**: Use dt-cli from within Claude Code via MCP
2. **Interactive TUI**: Beautiful terminal interface for direct interaction
3. **REST API**: Direct HTTP API for custom integrations

All methods provide access to the same powerful features:
- ✅ RAG-powered code search with auto-triggering
- ✅ Agentic debugging with error analysis
- ✅ Code review with security checks
- ✅ Knowledge graph exploration
- ✅ RAGAS evaluation metrics
- ✅ Hybrid search (semantic + BM25)

---

## Installation

### Prerequisites

- Python 3.8+
- Git (for cloning)

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/your-org/dt-cli.git
cd dt-cli

# Install dependencies
pip install -r requirements.txt
```

### Install Tree-sitter Grammars

```bash
# The grammars will auto-install on first use, but you can pre-install:
python -c "from src.rag.parsers import ParserRegistry; ParserRegistry()"
```

---

## Usage Methods

### Method 1: Claude Code Plugin

Use dt-cli as an MCP plugin within Claude Code for seamless AI-assisted development.

#### Setup

1. **Start the dt-cli server**:

```bash
python src/mcp_server/standalone_server.py
```

The server starts on `http://localhost:8765` by default.

2. **Configure Claude Code**:

The MCP configuration is already provided in `.claude/mcp-config.json`:

```json
{
  "mcpServers": {
    "dt-cli": {
      "command": "python",
      "args": ["-m", "src.mcp_server.standalone_server", "--host", "127.0.0.1", "--port", "8765"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "description": "dt-cli RAG/MAF/LLM Server - 100% Open Source",
      "timeout": 30000
    }
  }
}
```

3. **Use with Claude Code**:

Claude Code will automatically detect the MCP server and make dt-cli features available as tools.

#### Features Available in Claude Code

- Ask questions about your codebase
- Debug errors with AI assistance
- Get code reviews with security analysis
- Explore code dependencies and relationships
- Evaluate RAG quality
- Perform hybrid searches

---

### Method 2: Standalone Interactive TUI

Use the beautiful Rich-based terminal interface for direct interaction.

#### Quick Start

```bash
# Option 1: Using the convenience script
python dt-cli.py

# Option 2: Using the module directly
python -m src.cli.interactive

# Option 3: Make it executable (Linux/Mac)
chmod +x dt-cli.py
./dt-cli.py
```

#### Interactive Menu

```
┌─────────────────────────────────────────────┐
│      dt-cli - Interactive Terminal UI       │
│   RAG/MAF/LLM System - 100% Open Source     │
└─────────────────────────────────────────────┘

Main Menu:
  1. Ask a Question (RAG Query)
  2. Debug an Error
  3. Review Code
  4. Explore Knowledge Graph
  5. Evaluate RAG Quality
  6. Hybrid Search
  7. View Statistics
  8. Settings
  9. Help
  0. Exit

Choose an option:
```

#### Feature Walkthroughs

**1. Ask a Question**
```
Enter your question about the codebase:
> How does authentication work in this project?

[Processing with auto-trigger...]

Answer:
Authentication uses JWT tokens with bcrypt password hashing...

Context Files:
  • src/auth/login.py
  • src/auth/token.py

Confidence: 95%
```

**2. Debug an Error**
```
Paste your error (empty line when done):
> Traceback (most recent call last):
>   File "main.py", line 42, in process_data
>     result = data['value']
> KeyError: 'value'
>

[Analyzing error...]

Root Cause:
  The 'value' key is missing from the data dictionary.

Suggested Fixes:
  1. Add validation: if 'value' in data: ...
  2. Use .get() method: data.get('value', default)
  3. Add error handling: try/except KeyError

Confidence: 92%
```

**3. Review Code**
```
Choose input method:
  1. From file
  2. Paste code

Choice: 1
Enter file path: src/api/handler.py

[Reviewing code...]

Quality Score: 7.8/10

Issues Found:
  🔴 Critical (1):
    • SQL injection vulnerability on line 42

  🟡 Medium (2):
    • Missing input validation on line 18
    • Unused import on line 5

  🟢 Low (3):
    • Inconsistent naming on line 34
    • Missing docstring on line 12
    • Line too long on line 56

Security Notes:
  ⚠️ Use parameterized queries to prevent SQL injection
```

**4. Explore Knowledge Graph**
```
Graph Operations:
  1. Get dependencies
  2. Get dependents
  3. Find usages
  4. Impact analysis

Choice: 1
Enter file path: src/utils/parser.py

Dependencies of src/utils/parser.py:
  → src/utils/tokenizer.py
  → src/utils/validator.py
  → external: re, json
```

**5. Evaluate RAG Quality**
```
Enter test query: What does the login function do?
Enter expected answer: Validates credentials and returns JWT

[Evaluating with RAGAS metrics...]

Metrics:
  • Context Relevance: 0.92
  • Answer Faithfulness: 0.88
  • Answer Relevance: 0.95
  • Overall Score: 0.92

Assessment: Excellent ✅
```

**6. Hybrid Search**
```
Enter search query: authentication error handling

[Searching with hybrid approach...]

Results (semantic + keyword):

1. src/auth/login.py (score: 0.94)
   Semantic: 0.91 | Keyword: 0.97
   "def handle_auth_error(error): ..."

2. src/auth/middleware.py (score: 0.87)
   Semantic: 0.88 | Keyword: 0.86
   "class AuthErrorHandler: ..."

3. src/utils/errors.py (score: 0.73)
   Semantic: 0.82 | Keyword: 0.64
   "AuthenticationError exception class ..."
```

**7. View Statistics**
```
System Statistics:

📊 Vector Store:
  • Total chunks: 2,543
  • Total files: 187
  • Storage size: 45.2 MB

🔍 Auto-Trigger:
  • Total queries: 1,247
  • RAG triggered: 892 (71.5%)
  • Direct LLM: 355 (28.5%)
  • Avg confidence: 0.84

📈 Knowledge Graph:
  • Total nodes: 187
  • Total edges: 543
  • Avg dependencies: 2.9

⚡ Performance:
  • Avg query time: 245ms
  • Cache hit rate: 67%
```

---

### Method 3: Direct API

Use the REST API for custom integrations.

#### Start Server

```bash
python src/mcp_server/standalone_server.py --host 0.0.0.0 --port 8765
```

#### API Endpoints

**Health Check**
```bash
curl http://localhost:8765/health
```

**Ask Question**
```bash
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work?",
    "use_auto_trigger": true
  }'
```

**Debug Error**
```bash
curl -X POST http://localhost:8765/debug \
  -H "Content-Type: application/json" \
  -d '{
    "error_message": "KeyError: value",
    "stack_trace": "..."
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
  -d '{
    "root_path": "/path/to/code"
  }'
```

**Query Knowledge Graph**
```bash
curl -X POST http://localhost:8765/graph/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_type": "dependencies",
    "file_path": "src/utils/parser.py"
  }'
```

**Evaluate RAG**
```bash
curl -X POST http://localhost:8765/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "contexts": ["context1", "context2"],
    "answer": "test answer"
  }'
```

**Hybrid Search**
```bash
curl -X POST http://localhost:8765/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication",
    "top_k": 5
  }'
```

**View Statistics**
```bash
curl http://localhost:8765/graph/stats
curl http://localhost:8765/info
curl http://localhost:8765/auto-trigger/stats
```

---

## Features

### 1. RAG-Powered Code Search

- **AST-based chunking** for semantic understanding
- **BGE embeddings** with instruction prefix
- **Auto-trigger** intelligence (70% accuracy threshold)
- **Context-aware** retrieval

### 2. Agentic Debugging

- **Error analysis** with root cause identification
- **Fix suggestions** with confidence scores
- **Multi-step reasoning** using LangGraph
- **Stack trace analysis**

### 3. Code Review

- **Security analysis** (SQL injection, XSS, etc.)
- **Quality scoring** (0-10 scale)
- **Best practice checks**
- **Severity categorization** (Critical, Medium, Low)

### 4. Knowledge Graph

- **Dependency tracking** (imports, requires)
- **Impact analysis** (what breaks if you change X?)
- **Usage finding** (where is this function used?)
- **Relationship mapping** (calls, inherits, implements)

### 5. RAGAS Evaluation

- **Context Relevance**: How relevant are retrieved docs?
- **Answer Faithfulness**: Is answer grounded in context?
- **Answer Relevance**: Does answer match query?
- **Precision/Recall**: With ground truth comparison

### 6. Hybrid Search

- **Semantic search**: Embedding similarity
- **Keyword search**: BM25 algorithm
- **Query expansion**: Synonym-based
- **Weight tuning**: Optimize for your codebase

---

## Configuration

### LLM Configuration

Edit `llm-config.yaml`:

```yaml
llm:
  provider: "openai"  # or "anthropic", "local"
  model: "gpt-4"
  temperature: 0.7

auto_trigger:
  enabled: true
  similarity_threshold: 0.7
  intent_threshold: 0.6

vector_store:
  collection_name: "dt_cli_code"
  chunk_size: 1000
  chunk_overlap: 200

hybrid_search:
  semantic_weight: 0.7
  keyword_weight: 0.3
```

### Environment Variables

Create `.env` file:

```bash
# LLM API Keys (choose what you need)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Server Configuration
DT_CLI_HOST=0.0.0.0
DT_CLI_PORT=58432

# Logging
LOG_LEVEL=INFO
```

---

## Troubleshooting

### Server Won't Start

**Issue**: `Address already in use`

**Solution**:
```bash
# Check if port 8765 is in use
lsof -i :8765

# Kill the process or use different port
python src/mcp_server/standalone_server.py --port 8766
```

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in the dt-cli directory
cd /path/to/dt-cli

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or use python -m
python -m src.mcp_server.standalone_server
```

### Tree-sitter Errors

**Issue**: `Language not found: python`

**Solution**:
```bash
# The parsers install automatically, but you can force reinstall:
rm -rf ~/.tree-sitter
python -c "from src.rag.parsers import ParserRegistry; ParserRegistry()"
```

### Low RAG Quality

**Issue**: Poor retrieval results

**Solutions**:
1. **Tune hybrid search weights**:
   ```python
   from src.evaluation.hybrid_search import HybridSearch
   search = HybridSearch()
   search.tune_weights(queries, ground_truth, semantic_scores)
   ```

2. **Adjust chunk size** in `llm-config.yaml`:
   ```yaml
   vector_store:
     chunk_size: 1500  # Increase for more context
     chunk_overlap: 300
   ```

3. **Use RAGAS evaluation** to identify issues:
   ```bash
   # In TUI: Option 5 - Evaluate RAG Quality
   # Check which metrics are low and adjust accordingly
   ```

### Claude Code Integration Issues

**Issue**: MCP server not detected

**Solution**:
1. Verify server is running: `curl http://localhost:8765/health`
2. Check `.claude/mcp-config.json` exists
3. Restart Claude Code
4. Check Claude Code logs for connection errors

### TUI Display Issues

**Issue**: Colors or formatting broken

**Solution**:
```bash
# Check terminal support
echo $TERM

# Use simple output if needed
export TERM=xterm-256color

# Or disable rich features
python dt-cli.py --simple  # (if implemented)
```

---

## Performance Tips

1. **Pre-build knowledge graph** for large codebases:
   ```bash
   curl -X POST http://localhost:8765/graph/build \
     -H "Content-Type: application/json" \
     -d '{"root_path": "/path/to/code"}'
   ```

2. **Use caching** (already enabled by default):
   - Query cache: 15 minutes
   - Embedding cache: 1 hour

3. **Tune auto-trigger threshold**:
   - Higher threshold (0.8+): More direct LLM, faster
   - Lower threshold (0.6-): More RAG, better context

4. **Optimize chunk size**:
   - Smaller chunks (500-800): Better precision
   - Larger chunks (1500-2000): Better context

---

## Next Steps

1. **Customize for your project**:
   - Edit `llm-config.yaml` with your preferences
   - Add custom slash commands in `.claude/commands/`
   - Tune hybrid search weights for your codebase

2. **Integrate with CI/CD**:
   - Use API for automated code review
   - Evaluate RAG quality on test sets
   - Monitor metrics over time

3. **Extend functionality**:
   - Add custom agents in `src/debugging/`
   - Implement new graph queries
   - Create custom evaluation metrics

4. **Contribute**:
   - Report issues on GitHub
   - Submit pull requests
   - Share your use cases

---

## Support

- **Documentation**: See `README.md` and phase completion docs
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

Built with ❤️ by the dt-cli team | 100% Open Source | MIT License
