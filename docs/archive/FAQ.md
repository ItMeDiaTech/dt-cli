# Frequently Asked Questions (FAQ)

## Installation & Setup

### Q: Do I need to activate the virtual environment every time?

**A: No!** The `rag-maf` script automatically activates the virtual environment for you. You just use Claude normally:

```bash
# The script handles venv activation
~/dt-cli/rag-maf start

# Just use Claude - no venv needed!
claude
```

The virtual environment is only used internally by the Python/RAG backend.

---

### Q: Can I install this plugin globally for all my Claude projects?

**A: Sort of.** Claude doesn't support truly global plugins, but we've made it super easy to add to any project:

```bash
# One-time setup (during installation, add dt-cli to PATH)
# Then use anywhere:

cd /any/project
rag-plugin-global install
```

This copies the `.claude` configuration to your project. The MCP server runs globally and serves all projects.

---

### Q: Which Git branch should I use?

**A: Use `main` branch for stable releases:**

```bash
git clone https://github.com/ItMeDiaTech/dt-cli.git
cd dt-cli
./ubuntu-install.sh
```

For bleeding-edge features, use feature branches, but `main` is recommended for most users.

---

### Q: How do I update the plugin?

**A: Three methods:**

**Method 1: Global command** (if in PATH):
```bash
rag-plugin-global update
```

**Method 2: Manual**:
```bash
cd ~/dt-cli
git pull
./install.sh
./rag-maf restart
```

**Method 3: Re-clone**:
```bash
rm -rf ~/dt-cli
git clone https://github.com/ItMeDiaTech/dt-cli.git
cd dt-cli
./ubuntu-install.sh
```

---

## Usage

### Q: Do I need to start the MCP server every time I use Claude?

**A: Yes, but it's easy:**

```bash
# Start once (stays running in background)
~/dt-cli/rag-maf start

# Check if running
~/dt-cli/rag-maf status

# Use Claude anytime - server stays running!
claude
```

**Pro tip:** Add auto-start to `~/.bashrc`:
```bash
# Add this to ~/.bashrc
if ! pgrep -f "mcp_server/server.py" > /dev/null; then
    ~/dt-cli/rag-maf start > /dev/null 2>&1
fi
```

---

### Q: Can I use the plugin in multiple projects simultaneously?

**A: Yes!** The MCP server runs once and serves all projects:

```bash
# Start server once
~/dt-cli/rag-maf start

# Add plugin to multiple projects
cd ~/project1
rag-plugin-global install

cd ~/project2
rag-plugin-global install

# Use Claude in any project - same server!
```

---

### Q: Why don't I see the slash commands in Claude?

**A: Three common reasons:**

1. **No `.claude` directory in your project:**
   ```bash
   ls .claude/commands/  # Should show rag-*.md files
   ```
   **Fix:** `rag-plugin-global install`

2. **MCP server not running:**
   ```bash
   ~/dt-cli/rag-maf status
   ```
   **Fix:** `~/dt-cli/rag-maf start`

3. **Wrong directory:**
   ```bash
   pwd  # Should be in project with .claude/ directory
   ```
   **Fix:** `cd /path/to/project`

---

### Q: What's the difference between `rag-maf` and `rag-plugin-global`?

**A:**

- **`rag-maf`** - Controls the MCP server (start/stop/status/index)
- **`rag-plugin-global`** - Installs/removes plugin in projects

**Example workflow:**
```bash
# 1. Install plugin in project
cd ~/my-project
rag-plugin-global install

# 2. Start MCP server
~/dt-cli/rag-maf start

# 3. Use Claude
claude
```

---

## Troubleshooting

### Q: I'm getting "RuntimeWarning: 'src.mcp_server.server' found in sys.modules"

**A: This was fixed!** Update to the latest version:

```bash
cd ~/dt-cli
git pull
./install.sh
```

The new `rag-maf` script uses direct Python execution instead of `-m` flag.

---

### Q: The MCP server crashes or won't start

**A: Check logs and dependencies:**

```bash
# View logs
~/dt-cli/rag-maf logs

# Test RAG system
~/dt-cli/rag-maf test

# Reinstall dependencies
cd ~/dt-cli
source venv/bin/activate
pip install -r requirements.txt
deactivate

# Restart server
~/dt-cli/rag-maf restart
```

---

### Q: How much disk space does this use?

**A: Approximately 2-3GB:**

- Node.js + Claude Code: ~500MB
- Python dependencies: ~1GB
- Embedding models: ~500MB
- Vector database (varies by codebase): 100MB-1GB

---

### Q: Can I use this without Node.js/Claude Code?

**A: No.** This is specifically a Claude Code plugin. The RAG system is designed to integrate with Claude Code's slash commands and MCP protocol.

---

### Q: Does this work on Mac/Windows?

**A:**

- **Ubuntu/Debian:** ✅ Fully supported (uses `ubuntu-install.sh`)
- **Mac:** ⚠️ Should work, but use `install.sh` instead
- **Windows:** ⚠️ Use WSL2 with Ubuntu, then `ubuntu-install.sh`

---

### Q: Is my code sent to external servers?

**A: No!** Everything runs locally:

- ✅ Embeddings: Local (sentence-transformers)
- ✅ Vector store: Local (ChromaDB)
- ✅ MCP server: Local (127.0.0.1:8765)
- ✅ Code never leaves your machine

Only Claude API calls go to Anthropic (as normal with Claude).

---

### Q: Can I customize the embedding model or chunk size?

**A: Yes!** Edit `.claude/rag-config.json`:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "max_results": 5,
  "maf_enabled": true
}
```

After changes, restart:
```bash
~/dt-cli/rag-maf restart
```

---

### Q: How do I uninstall?

**A: Remove from individual projects or completely:**

**Remove from one project:**
```bash
cd ~/my-project
rag-plugin-global remove
```

**Complete uninstall:**
```bash
# Stop server
~/dt-cli/rag-maf stop

# Remove Claude Code
sudo npm uninstall -g @anthropic-ai/claude-code

# Remove dt-cli
rm -rf ~/dt-cli

# Remove from PATH (edit ~/.bashrc and remove dt-cli line)
nano ~/.bashrc
```

---

### Q: Can I contribute or report bugs?

**A: Yes!**

- Report issues: https://github.com/ItMeDiaTech/dt-cli/issues
- Pull requests welcome
- See [CONTRIBUTING.md](./CONTRIBUTING.md) (if exists)

---

### Q: What Claude Code plan do I need?

**A:** The plugin works with any Claude plan that has API access:

- ✅ Claude Pro
- ✅ Claude Max (recommended)
- ✅ API key access

You need either:
- Interactive authentication (`claude auth login`), OR
- API key (`export ANTHROPIC_API_KEY='your-key'`)

---

### Q: Performance - how fast is it?

**A:**

- **Initial indexing:** 1-5 minutes (depends on codebase size)
- **Query response:** <1 second (cached)
- **Re-indexing:** Only changed files (incremental)
- **Memory usage:** ~500MB-1GB RAM

For large codebases (>100k LOC), consider:
- Increasing chunk size
- Excluding test directories
- Using more powerful embedding models

---

### Q: Where is data stored?

**A:**

- Vector database: `~/dt-cli/.rag_data/`
- Logs: `~/dt-cli/logs/`
- Config: `~/dt-cli/.claude/rag-config.json`
- MCP PID: `~/dt-cli/.mcp_server.pid`

Safe to delete `.rag_data/` and re-index if needed.

---

## Advanced

### Q: Can I run multiple MCP servers for different projects?

**A: Not recommended.** One server can handle all projects. But if needed:

1. Clone dt-cli to different locations
2. Change port in each `.claude/mcp-servers.json`
3. Start each server separately

---

### Q: How do I integrate with CI/CD?

**A:** Example GitHub Actions workflow:

```yaml
- name: Setup dt-cli RAG
  run: |
    git clone https://github.com/ItMeDiaTech/dt-cli.git
    cd dt-cli
    ./install.sh
    ./rag-maf start
    ./rag-maf index
```

---

### Q: Can I use custom embeddings or LLMs?

**A:** Not out of the box, but you can modify:

- `src/rag/embeddings.py` - Change embedding model
- `src/maf/orchestrator.py` - Change LLM for agents

After modifications:
```bash
~/dt-cli/rag-maf restart
```

---

Still have questions? Check:
- [Plugin Usage Guide](./PLUGIN_USAGE.md)
- [Ubuntu Deployment Guide](./UBUNTU_DEPLOYMENT_GUIDE.md)
- [GitHub Issues](https://github.com/ItMeDiaTech/dt-cli/issues)
