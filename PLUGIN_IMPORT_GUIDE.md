# dt-cli Plugin Import Guide

This guide explains how to import and use **dt-cli** as a Claude Code CLI plugin, enabling you to utilize all RAG/MAF/LLM features directly within Claude Code sessions.

---

## Overview

**dt-cli** can be used in three ways:
1. **As a Claude Code Plugin** (Recommended for Claude Code users)
2. **As a Standalone Interactive CLI** (For terminal enthusiasts)
3. **As a REST API Server** (For custom integrations)

This guide focuses on **importing dt-cli as a Claude Code plugin**.

---

## What You Get as a Plugin

When imported as a plugin, dt-cli provides:

### ✅ **Slash Commands**
- `/rag-query` - Query the RAG system for code and documentation
- `/rag-index` - Index/re-index your codebase
- `/rag-status` - Check system status and metrics
- `/rag-save` - Save search queries for later use
- `/rag-searches` - List and manage saved searches
- `/rag-exec` - Execute saved searches
- `/rag-graph` - Query knowledge graph for code relationships
- `/rag-metrics` - View performance metrics
- `/rag-query-advanced` - Advanced RAG queries with profiling

### ✅ **Session Hooks**
- **SessionStart Hook** - Automatically initializes RAG/MAF systems when Claude Code starts
- Auto-starts MCP server if not running
- Auto-indexes codebase on first run
- Provides seamless integration without manual setup

### ✅ **MCP Server Integration**
- **5 Core Tools** exposed via MCP:
  - `rag_query` - Semantic code search
  - `rag_index` - Codebase indexing
  - `rag_status` - System status
  - `maf_orchestrate` - Multi-agent orchestration
  - `maf_status` - MAF system status

### ✅ **Core Features**
- **RAG System**: Semantic code search with AST-based chunking
- **Multi-Agent Framework**: Orchestrated agent workflows
- **Knowledge Graph**: Dependency tracking and impact analysis
- **Debugging Agent**: Automated error analysis
- **Code Review Agent**: Security and quality checks
- **RAGAS Evaluation**: Quality metrics for RAG performance
- **Hybrid Search**: BM25 + semantic search
- **100% Open Source**: Works with local LLMs (Ollama, vLLM)

---

## Installation Methods

### Method 1: Clone and Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/ItMeDiaTech/dt-cli.git
cd dt-cli

# Run the installation script
./install.sh

# The installer will:
# - Check Python version (3.8+ required)
# - Create virtual environment
# - Install dependencies
# - Download embedding model (all-MiniLM-L6-v2)
# - Configure Claude Code integration
# - Set up hooks and commands
```

### Method 2: Import as Claude Code Plugin

If Claude Code supports plugin import from repositories:

```bash
# In Claude Code, use the plugin import command:
claude-code plugin install https://github.com/ItMeDiaTech/dt-cli.git

# Or import from local directory:
claude-code plugin install /path/to/dt-cli
```

### Method 3: Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/ItMeDiaTech/dt-cli.git
cd dt-cli

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make hooks executable
chmod +x .claude/hooks/SessionStart.sh

# 5. Link to Claude Code plugins directory (if applicable)
ln -s $(pwd) ~/.claude-code/plugins/dt-cli-rag-maf
```

---

## Plugin Configuration

### Plugin Manifest Files

The plugin includes comprehensive manifest files:

**`plugin.json`** (Root manifest)
```json
{
  "name": "dt-cli-rag-maf",
  "version": "1.0.0",
  "description": "Local RAG plugin with Multi-Agent Framework orchestration",
  "requires": {
    "claude-code": ">=1.0.0",
    "python": ">=3.8"
  },
  "install": {
    "script": "./install.sh",
    "hooks": true,
    "mcp": true,
    "commands": true
  }
}
```

**`.claude-plugin/plugin.json`** (Detailed plugin configuration)
- References all slash commands
- Defines hooks configuration
- Lists plugin metadata

**`.claude-plugin/hooks.json`** (Hook configuration)
- SessionStart hook for auto-initialization
- Timeout: 60 seconds
- Triggers on: startup, resume, clear, compact

### MCP Server Configuration

**`.claude/mcp-config.json`**
```json
{
  "mcpServers": {
    "dt-cli": {
      "command": "python",
      "args": ["-m", "src.mcp_server.standalone_server", "--host", "127.0.0.1", "--port", "8765"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  }
}
```

### LLM Configuration

**`llm-config.yaml`**
```yaml
llm:
  provider: "ollama"       # or "openai", "anthropic", "local"
  model: "codellama:7b"    # or any Ollama model
  temperature: 0.7

embedding:
  model: "BAAI/bge-base-en-v1.5"
  device: "cpu"

auto_trigger:
  enabled: true
  similarity_threshold: 0.7
```

---

## Usage After Installation

### Automatic Initialization

When you start a Claude Code session in a project with dt-cli installed:

1. **SessionStart Hook** automatically runs
2. **MCP Server** starts (if not already running)
3. **Codebase Indexing** begins (first run only)
4. **Slash Commands** become available immediately

You'll see output like:
```
[*] Initializing RAG-MAF Plugin...
[+] Starting MCP Server...
[OK] MCP Server started successfully
[OK] Codebase already indexed
[**] RAG-MAF Plugin ready!

Available commands:
  /rag-query <query>  - Query the RAG system
  /rag-index          - Re-index the codebase
  /rag-status         - Check system status
```

### Manual Server Control

Use the `rag-maf` CLI wrapper:

```bash
# Start server
./rag-maf start

# Check status
./rag-maf status

# Stop server
./rag-maf stop

# Restart server
./rag-maf restart

# Index codebase
./rag-maf index

# Test RAG system
./rag-maf test

# View logs
./rag-maf logs
```

### Using Slash Commands

In any Claude Code session:

```bash
# Query for code
/rag-query How is authentication handled?

# Check system status
/rag-status

# Re-index after major changes
/rag-index

# Save a query for later
/rag-save "authentication flow" How does login work?

# List saved searches
/rag-searches

# Execute saved search by ID
/rag-exec 1

# Query knowledge graph
/rag-graph dependencies parse_code

# View metrics
/rag-metrics
```

### Using MCP Tools

Tools are automatically available in Claude Code conversations:

```
You: "Find all authentication-related code"
Claude: [Uses rag_query tool automatically]

You: "What would break if I change this function?"
Claude: [Uses knowledge graph via MAF tools]
```

---

## Features Available as Plugin

### 1. **RAG System**
- **AST-Based Chunking**: Intelligent code parsing
- **BGE Embeddings**: Instruction-aware embeddings
- **Auto-Trigger**: Decides when to use RAG vs direct LLM
- **Intent Classification**: Routes queries intelligently

### 2. **Multi-Agent Framework (MAF)**
- **Agent Orchestration**: Coordinated multi-agent workflows
- **Task Decomposition**: Breaks complex queries into sub-tasks
- **LangGraph Integration**: State-based agent coordination

### 3. **Knowledge Graph**
- **Dependency Tracking**: What does this code depend on?
- **Impact Analysis**: What breaks if I change this?
- **Usage Finding**: Where is this function used?
- **Relationship Mapping**: Full code relationship graph

### 4. **Debugging & Code Review**
- **Error Analysis**: Root cause identification
- **Fix Suggestions**: Multi-step reasoning for fixes
- **Security Checks**: OWASP Top 10 vulnerability detection
- **Quality Scoring**: 0-10 score with severity levels

### 5. **Quality Evaluation**
- **RAGAS Metrics**: Context relevance, answer faithfulness
- **A/B Testing**: Compare RAG configurations
- **Performance Metrics**: Query time, cache hit rate

---

## Plugin Directory Structure

```
dt-cli/
├── plugin.json                    # Root plugin manifest
├── .claude-plugin/
│   ├── plugin.json                # Detailed plugin config
│   ├── hooks.json                 # Hook definitions
│   └── marketplace.json           # Marketplace metadata
├── .claude/
│   ├── mcp-config.json            # MCP server config
│   ├── mcp-servers.json           # Server tools definition
│   ├── hooks/
│   │   └── SessionStart.sh        # Auto-initialization hook
│   └── commands/                  # Slash command definitions
│       ├── rag-query.md
│       ├── rag-index.md
│       ├── rag-status.md
│       └── ... (9 total commands)
├── src/
│   ├── mcp_server/
│   │   ├── __main__.py            # Module entry point (NEW)
│   │   ├── standalone_server.py   # Standalone MCP server
│   │   ├── server.py              # FastAPI MCP server
│   │   ├── tools.py               # RAG & MAF tools
│   │   └── bridge.py              # Claude Code bridge
│   ├── rag/                       # RAG system
│   ├── maf/                       # Multi-Agent Framework
│   ├── llm/                       # LLM provider abstraction
│   ├── debugging/                 # Debug & review agents
│   ├── graph/                     # Knowledge graph
│   └── evaluation/                # Quality metrics
├── install.sh                     # Installation script
├── rag-maf                        # CLI wrapper
└── llm-config.yaml                # LLM configuration
```

---

## Verifying Plugin Import

### Check Plugin Installation

```bash
# Check if hooks are executable
ls -la .claude/hooks/SessionStart.sh
# Should show: -rwxr-xr-x (executable)

# Test MCP server module import
python -c "from src.mcp_server import MCPServer; print('✓ MCP server imports OK')"

# Test RAG system
python -c "from src.rag import QueryEngine; print('✓ RAG system imports OK')"

# Check if virtual environment is active
which python
# Should point to: /path/to/dt-cli/venv/bin/python
```

### Test Server Startup

```bash
# Start server manually
./rag-maf start

# Check if server is running
./rag-maf status
# Should show: "MCP server is running (PID: XXXX)"

# Test health endpoint
curl http://127.0.0.1:8765/health
# Should return: {"status": "healthy", ...}

# Stop server
./rag-maf stop
```

### Test Plugin in Claude Code

1. Start a Claude Code session
2. Check for initialization message
3. Try a slash command: `/rag-status`
4. Expected output: System status with metrics

---

## Troubleshooting

### Plugin Not Loading

**Symptom**: Claude Code doesn't recognize the plugin

**Solution**:
```bash
# 1. Verify plugin.json exists and is valid
cat plugin.json | python -m json.tool

# 2. Check hooks are executable
chmod +x .claude/hooks/SessionStart.sh

# 3. Restart Claude Code
```

### MCP Server Not Starting

**Symptom**: SessionStart hook fails or server doesn't start

**Solution**:
```bash
# 1. Check logs
cat /tmp/rag-maf-mcp.log

# 2. Manually start server with debug output
python -m src.mcp_server.standalone_server --host 127.0.0.1 --port 8765

# 3. Check port availability
lsof -i :8765

# 4. Use auto-port mode
python -m src.mcp_server.standalone_server --auto-port
```

### Import Errors

**Symptom**: Python import errors when running server

**Solution**:
```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Reinstall dependencies
pip install -r requirements.txt

# 3. Set PYTHONPATH
export PYTHONPATH=/path/to/dt-cli/src:$PYTHONPATH

# 4. Test imports
python -c "from rag import QueryEngine; from maf import AgentOrchestrator"
```

### Slash Commands Not Working

**Symptom**: `/rag-query` and other commands not recognized

**Solution**:
```bash
# 1. Verify command files exist
ls -la .claude/commands/

# 2. Check plugin.json references commands
grep -A 10 "commands" plugin.json

# 3. Restart Claude Code session
```

### Indexing Fails

**Symptom**: Codebase indexing fails or hangs

**Solution**:
```bash
# 1. Manually trigger indexing with verbose output
./rag-maf index

# 2. Check for tree-sitter parser errors
python -c "from src.rag.parsers import ParserRegistry; ParserRegistry()"

# 3. Check disk space and permissions
df -h
ls -la .rag_data/
```

---

## Uninstalling the Plugin

To remove dt-cli plugin from Claude Code:

```bash
# 1. Stop MCP server
./rag-maf stop

# 2. Remove plugin directory (if linked)
rm ~/.claude-code/plugins/dt-cli-rag-maf

# 3. Remove systemd service (if installed)
sudo systemctl stop rag-maf-mcp
sudo systemctl disable rag-maf-mcp
sudo rm /etc/systemd/system/rag-maf-mcp.service

# 4. Remove repository (optional)
cd ..
rm -rf dt-cli
```

---

## Support and Contributing

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/ItMeDiaTech/dt-cli/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ItMeDiaTech/dt-cli/discussions)
- **Documentation**: [README.md](./README.md) | [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)

### Contributing

We welcome contributions! See our contribution guidelines:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Commit: `git commit -m "Add amazing feature"`
5. Push: `git push origin feature/amazing-feature`
6. Open Pull Request

---

## License

MIT License - 100% Free and Open Source

You can:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

See [LICENSE](./LICENSE) for full details.

---

## What's Next?

After successful plugin import:

1. **Index your codebase**: `/rag-index`
2. **Query for code**: `/rag-query How does X work?`
3. **Check knowledge graph**: `/rag-graph dependencies MyClass`
4. **Review code quality**: Use code review agent via MAF
5. **Debug errors**: Use debug agent for error analysis
6. **Tune performance**: Use `/rag-metrics` to optimize

**Happy coding with RAG-powered context awareness!** 🚀

---

**Made with ❤️ by ItMeDiaTech | 100% Open Source**
