# Installation Guide - RAG-MAF Plugin for Claude Code

Complete step-by-step installation guide for the dt-cli RAG-MAF plugin.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [First Run](#first-run)
5. [Troubleshooting](#troubleshooting)
6. [Uninstallation](#uninstallation)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 2GB available (4GB+ recommended)
- **Disk Space**: ~2GB for dependencies and model files

### Software Requirements

1. **Python 3.8+**

   Check your Python version:
   ```bash
   python3 --version
   ```

   If not installed, download from [python.org](https://www.python.org/downloads/)

2. **pip (Python package manager)**

   Usually comes with Python. Verify:
   ```bash
   pip3 --version
   ```

3. **Git**

   Verify:
   ```bash
   git --version
   ```

4. **Claude Code**

   This plugin requires Claude Code to be installed and configured.

---

## Installation Steps

### Step 1: Clone the Repository

```bash
cd ~  # or wherever you want to install
git clone https://github.com/ItMeDiaTech/dt-cli.git
cd dt-cli
```

### Step 2: Choose Installation Method

You have two options: **Virtual Environment** (recommended) or **Global Installation**.

#### Option A: Virtual Environment (Recommended)

This keeps dependencies isolated and prevents conflicts:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/macOS
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

**To activate in future sessions:**
```bash
cd ~/dt-cli
source venv/bin/activate  # Always activate before using
```

#### Option B: Global Installation

Install dependencies system-wide:

```bash
pip3 install -r requirements.txt
```

**Note**: May require `sudo` on some systems or cause conflicts with other Python projects.

### Step 3: Verify Installation

Run the verification script:

```bash
python3 -c "
import sys
packages = ['httpx', 'chromadb', 'sentence_transformers', 'fastapi', 'uvicorn', 'langchain', 'langgraph']
missing = []
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)

if missing:
    print('[X] Missing packages:', ', '.join(missing))
    sys.exit(1)
else:
    print('[OK] All required packages installed successfully!')
"
```

Expected output:
```
[OK] All required packages installed successfully!
```

### Step 4: Configure Claude Code

The `.claude/` directory is already configured with:
- SessionStart hook (`.claude/hooks/SessionStart.sh`)
- 9 slash commands (`.claude/commands/*.md`)
- MCP server configuration (`.claude/mcp-servers.json`)
- RAG configuration (`.claude/rag-config.json`)

**If using virtual environment**, update the SessionStart hook:

Edit `.claude/hooks/SessionStart.sh` and add these lines after line 8:

```bash
# Activate virtual environment if it exists
if [ -d "$PLUGIN_DIR/venv" ]; then
    source "$PLUGIN_DIR/venv/bin/activate"
fi
```

### Step 5: First Run Test

Start the MCP server manually to test:

```bash
# If using venv, make sure it's activated
python3 -m src.mcp_server.server
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765
```

**Press Ctrl+C to stop** (the SessionStart hook will auto-start this later)

---

## Verification

### Verify Each Component

1. **MCP Server**:
   ```bash
   # Start server
   python3 -m src.mcp_server.server &

   # Test endpoint
   curl http://127.0.0.1:8765/status

   # Stop server
   pkill -f "mcp_server/server.py"
   ```

2. **RAG System**:
   ```bash
   python3 -c "
   from src.rag.embeddings import EmbeddingGenerator
   gen = EmbeddingGenerator()
   print('[OK] RAG system working')
   "
   ```

3. **MAF System**:
   ```bash
   python3 -c "
   from src.maf.orchestrator import Orchestrator
   print('[OK] MAF system working')
   "
   ```

### Test in Claude Code

1. **Start a Claude Code session** in the dt-cli directory
2. **Wait for initialization** (you should see the SessionStart hook output)
3. **Test a command**:
   ```
   /rag-status
   ```
4. **Run a query**:
   ```
   /rag-query what is the RAG system
   ```

---

## First Run

### What Happens on First Run

When you first start Claude Code in the dt-cli directory:

1. **SessionStart hook executes** (`.claude/hooks/SessionStart.sh`)
2. **MCP server starts** on port 8765
3. **Codebase indexing begins** (may take 2-10 minutes depending on size)
4. **Embedding model downloads** (~100MB, one-time)
5. **Vector database created** (`.rag_data/` directory)

### Expected Output

```
[*] Initializing RAG-MAF Plugin...
[+] Starting MCP Server...
[OK] MCP Server started successfully
[#] First run detected. Indexing codebase...
   (This may take a few minutes)
[...] Indexing in progress (running in background)...

[**] RAG-MAF Plugin ready!

Available commands:
  /rag-query <query>  - Query the RAG system
  /rag-index          - Re-index the codebase
  /rag-status         - Check system status

The plugin will automatically provide context to Claude as needed.
```

### First Indexing

The initial indexing process:
- Scans all code files (`.py`, `.js`, `.ts`, etc.)
- Chunks code into 1000-token segments
- Generates embeddings for each chunk
- Stores in ChromaDB

**Estimated time**:
- Small project (< 100 files): 1-2 minutes
- Medium project (100-1000 files): 5-10 minutes
- Large project (1000+ files): 15-30 minutes

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'httpx'"

**Problem**: Dependencies not installed

**Solution**:
```bash
pip3 install -r requirements.txt
```

If using venv:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. "MCP Server failed to start"

**Problem**: Port 8765 already in use

**Check what's using the port**:
```bash
lsof -i :8765
# OR
netstat -an | grep 8765
```

**Solution**: Kill the process or change the port in `.claude/rag-config.json`

#### 3. "Connection refused" when running slash commands

**Problem**: MCP server not running

**Solution**:
```bash
# Start manually
python3 -m src.mcp_server.server &

# Check logs
tail -f /tmp/rag-maf-mcp.log
```

#### 4. SessionStart hook doesn't run

**Problem**: Hook not executable

**Solution**:
```bash
chmod +x .claude/hooks/SessionStart.sh
```

#### 5. "Import error: no module named 'sentence_transformers'"

**Problem**: Large dependencies not fully installed

**Solution**:
```bash
# Install with verbose output to see progress
pip3 install -v sentence-transformers torch
```

#### 6. Indexing is very slow

**Problem**: Large codebase

**Solutions**:
- **Exclude directories**: Edit `.claude/rag-config.json`, add to `ignore_directories`
- **Reduce file types**: Edit `file_extensions` to only include main languages
- **Use incremental indexing**: After first index, updates are 100-1000x faster

#### 7. High memory usage

**Problem**: Large embedding model

**Solutions**:
- Use a smaller model (edit `embedding_model` in config)
- Close other applications
- Increase system RAM

#### 8. Port 8000 vs 8765 confusion

**Problem**: Some commands use port 8000 instead of 8765

**Solution**: This is a known configuration issue. Update all slash commands to use port 8765:

```bash
# Fix all commands at once
cd .claude/commands
sed -i 's/127.0.0.1:8000/127.0.0.1:8765/g' rag-*.md
```

### Getting Help

1. **Check logs**:
   ```bash
   tail -f /tmp/rag-maf-mcp.log
   ```

2. **Verify system status**:
   ```
   /rag-status
   ```

3. **Review audit report**:
   See [CODEBASE_AUDIT_REPORT.md](./CODEBASE_AUDIT_REPORT.md) for detailed analysis

4. **GitHub Issues**:
   Report problems at [github.com/ItMeDiaTech/dt-cli/issues](https://github.com/ItMeDiaTech/dt-cli/issues)

---

## Uninstallation

### Remove the Plugin

```bash
# Stop MCP server
pkill -f "mcp_server/server.py"

# Remove directory
cd ~
rm -rf dt-cli

# If using virtual environment, it's already removed
# If using global install, optionally remove packages:
pip3 uninstall -r dt-cli/requirements.txt
```

### Clean Up Data

```bash
# Remove indexed data
rm -rf ~/dt-cli/.rag_data

# Remove logs
rm /tmp/rag-maf-mcp.log
```

---

## Advanced Configuration

### Custom Embedding Model

Edit `.claude/rag-config.json`:

```json
{
  "rag": {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",  // Larger, more accurate
    // OR
    "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",  // Smaller, faster
    ...
  }
}
```

### Adjust Chunk Size

For different programming languages:

```json
{
  "rag": {
    "chunk_size": 1500,      // Larger for verbose languages (Java, C#)
    "chunk_overlap": 300,    // More overlap for better context
    ...
  }
}
```

### Enable/Disable MAF Agents

```json
{
  "maf": {
    "enabled": true,
    "agents": {
      "code_analyzer": true,        // Code structure analysis
      "doc_retriever": false,       // Disable doc retrieval
      "synthesizer": true,          // Context synthesis
      "suggestion_generator": true  // Suggestions
    }
  }
}
```

### Change Server Port

If port 8765 conflicts:

```json
{
  "mcp": {
    "host": "127.0.0.1",
    "port": 9000,  // Use any available port
    "auto_start": true
  }
}
```

**Remember to update**:
- `.claude/mcp-servers.json` (change URL)
- All slash commands in `.claude/commands/*.md`

---

## Performance Optimization

### Speed Up First Indexing

```bash
# Use multiple workers (edit src/rag/ingestion.py)
# Or exclude large directories
```

### Reduce Memory Usage

1. Use a smaller embedding model
2. Reduce `max_results` in config
3. Close other applications
4. Enable lazy loading (already implemented)

### Improve Query Speed

- Use query caching (enabled by default)
- Use incremental indexing for updates
- Enable query prefetching (see config)

---

## Next Steps

After successful installation:

1. **Read the User Guide**: [USER_GUIDE.md](./USER_GUIDE.md)
2. **Explore slash commands**: Type `/rag-` and see available commands
3. **Review architecture**: [ARCHITECTURE.md](./ARCHITECTURE.md)
4. **Check implementation status**: [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)

---

**Installation Guide Version**: 1.0
**Last Updated**: 2025-11-08
**Plugin Version**: 1.0.0
