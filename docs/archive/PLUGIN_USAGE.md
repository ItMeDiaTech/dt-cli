# dt-cli RAG Plugin - Usage Guide

Complete guide for using the dt-cli RAG plugin with Claude Code.

---

## 🚀 Quick Start

### 1. Run Installation (If Not Done)

```bash
cd ~/dt-cli
./ubuntu-install.sh
```

### 2. Test the Plugin

```bash
# Test RAG system
./rag-maf test

# Start MCP server
./rag-maf start

# Check server status
./rag-maf status
```

### 3. Use with Claude

```bash
# Navigate to dt-cli directory
cd ~/dt-cli

# Start Claude
claude

# Inside Claude, try:
/rag-status
```

---

## 📁 Plugin Installation Per-Project

The dt-cli RAG plugin works on a **per-project basis**. To use it in your own projects:

### Method 1: Copy Configuration (Recommended)

```bash
# Go to your project
cd /path/to/your/project

# Copy .claude directory from dt-cli
cp -r ~/dt-cli/.claude .

# Make hooks executable
chmod +x .claude/hooks/*.sh

# Start MCP server
~/dt-cli/rag-maf start

# Start Claude in your project
claude

# Use slash commands
/rag-index
/rag-query how does authentication work?
```

### Method 2: Symlink Configuration (Auto-updates)

```bash
# Go to your project
cd /path/to/your/project

# Create symbolic link
ln -s ~/dt-cli/.claude .claude

# Start MCP server
~/dt-cli/rag-maf start

# Start Claude
claude
```

### Method 3: Use Setup Script

Create a helper script:

```bash
#!/bin/bash
# ~/setup-rag.sh

PROJECT_DIR="$1"
DT_CLI_DIR="$HOME/dt-cli"

if [ -z "$PROJECT_DIR" ]; then
    echo "Usage: $0 /path/to/your/project"
    exit 1
fi

cd "$PROJECT_DIR" || exit 1

echo "Setting up dt-cli RAG plugin in $PROJECT_DIR..."

# Copy .claude configuration
cp -r "$DT_CLI_DIR/.claude" .
chmod +x .claude/hooks/*.sh

echo "✅ Plugin configured!"
echo ""
echo "Next steps:"
echo "  1. Start MCP server: $DT_CLI_DIR/rag-maf start"
echo "  2. Index this project: cd $PROJECT_DIR && $DT_CLI_DIR/rag-maf index"
echo "  3. Start Claude: claude"
echo "  4. Try: /rag-status"
```

**Usage:**

```bash
chmod +x ~/setup-rag.sh
~/setup-rag.sh /path/to/your/project
```

---

## 🎮 RAG-MAF Control Commands

The `rag-maf` script provides several commands:

### Start/Stop/Status

```bash
# Start MCP server (runs in background)
./rag-maf start
# Output: MCP server started (PID: 12345)
#         Server running on http://127.0.0.1:8765

# Stop MCP server
./rag-maf stop
# Output: MCP server stopped (PID: 12345)

# Check server status
./rag-maf status
# Output: MCP server is running (PID: 12345)
#         Server URL: http://127.0.0.1:8765

# Restart server
./rag-maf restart
```

### Indexing

```bash
# Index the current directory
cd /path/to/your/project
~/dt-cli/rag-maf index

# This will:
# - Scan all source files
# - Generate embeddings
# - Store in vector database
# - Make code searchable via /rag-query
```

### Testing

```bash
# Test RAG system initialization
./rag-maf test

# Output:
# RAG system initialized successfully!
# Vector store location: /home/user/dt-cli/.rag_data
```

### Logs

```bash
# View server logs (follows in real-time)
./rag-maf logs
```

---

## 📝 Available Slash Commands

Once the plugin is configured and MCP server is running, these commands are available in Claude:

### /rag-query

Query the RAG system for relevant code and documentation.

```
/rag-query how does authentication work?
/rag-query find all database queries
/rag-query what is the API structure?
```

### /rag-index

Index or re-index the current codebase.

```
/rag-index
```

### /rag-status

Check the status of the RAG and MAF systems.

```
/rag-status
```

### /rag-save

Save a search query for quick access later.

```
/rag-save authentication-code "how does authentication work?"
```

### /rag-searches

List all saved searches.

```
/rag-searches
```

### /rag-exec

Execute a saved search by ID or name.

```
/rag-exec authentication-code
```

### /rag-graph

Query the knowledge graph for code relationships.

```
/rag-graph show dependencies for UserController
```

### /rag-metrics

Display system metrics and performance dashboard.

```
/rag-metrics
```

### /rag-query-advanced

Advanced RAG query with profiling and explanations.

```
/rag-query-advanced --profile how does the login flow work?
```

---

## 🔧 Troubleshooting

### Plugin Not Visible in Claude

**Problem:** Slash commands don't appear when typing `/rag`

**Solutions:**

1. **Check you're in the right directory:**
   ```bash
   # Plugin only works if .claude directory exists here
   ls -la .claude/
   ```

2. **Verify .claude directory has slash commands:**
   ```bash
   ls -la .claude/commands/
   # Should show: rag-query.md, rag-index.md, etc.
   ```

3. **Copy .claude from dt-cli:**
   ```bash
   cp -r ~/dt-cli/.claude .
   ```

### MCP Server Not Running

**Problem:** `/rag-status` shows "MCP server not responding"

**Solutions:**

```bash
# Check if server is running
~/dt-cli/rag-maf status

# If not running, start it
~/dt-cli/rag-maf start

# Check logs for errors
~/dt-cli/rag-maf logs
```

### Python Module Warning

**Problem:** Seeing `RuntimeWarning: 'src.mcp_server.server' found in sys.modules`

**Solution:** This warning has been fixed in the latest version. Update your rag-maf script:

```bash
cd ~/dt-cli
git pull
./install.sh  # Re-run installation
```

Or manually recreate the rag-maf file with the updated version from the repository.

### Indexing Fails

**Problem:** `/rag-index` command fails or doesn't find files

**Solutions:**

1. **Check you're in the project directory:**
   ```bash
   pwd  # Should be in your project root
   ```

2. **Run index manually:**
   ```bash
   cd /path/to/your/project
   ~/dt-cli/rag-maf index
   ```

3. **Check permissions:**
   ```bash
   ls -la ~/dt-cli/.rag_data/
   ```

### Authentication Issues

**Problem:** Claude commands not working

**Solutions:**

```bash
# Check authentication
claude auth status

# Re-authenticate
claude auth login

# Or use API key
export ANTHROPIC_API_KEY='your-key-here'
```

---

## 🎯 Best Practices

### 1. Start MCP Server Before Claude

```bash
# Always start server first
~/dt-cli/rag-maf start

# Then start Claude
claude
```

### 2. Index After Code Changes

```bash
# After making significant code changes, re-index
~/dt-cli/rag-maf index
```

### 3. Use Saved Searches for Common Queries

```bash
# In Claude, save frequently used queries
/rag-save auth-flow "how does authentication work?"

# Later, execute quickly
/rag-exec auth-flow
```

### 4. Monitor Server Status

```bash
# Periodically check server health
~/dt-cli/rag-maf status

# View logs if issues arise
~/dt-cli/rag-maf logs
```

### 5. Restart Server After Updates

```bash
# After updating dt-cli or changing configuration
~/dt-cli/rag-maf restart
```

---

## 🔄 Updating the Plugin

### Update from Git

```bash
cd ~/dt-cli
git pull origin main

# Re-run installation to update
./install.sh

# Restart MCP server
./rag-maf restart
```

### Update Dependencies

```bash
cd ~/dt-cli
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

---

## 📊 Performance Tips

### Large Codebases

For projects with >100k LOC:

1. **Incremental indexing:** Re-index only changed files
2. **Exclude unnecessary directories:** Add to `.gitignore` or configure exclusions
3. **Adjust chunk size:** Modify `.claude/rag-config.json`

### Memory Usage

```bash
# Check RAM usage
~/dt-cli/rag-maf status
ps aux | grep mcp_server

# Restart if memory grows
~/dt-cli/rag-maf restart
```

---

## 🆘 Getting Help

### Check Logs

```bash
# MCP server logs
~/dt-cli/rag-maf logs

# Installation logs
cat ~/dt-cli/logs/*.log

# Claude logs
cat ~/.claude/logs/*.log
```

### Verify Installation

```bash
# Check all components
cd ~/dt-cli

# 1. Virtual environment
ls venv/

# 2. RAG system
./rag-maf test

# 3. MCP server
./rag-maf status

# 4. Slash commands
ls .claude/commands/
```

### Common Commands

```bash
# Full restart sequence
~/dt-cli/rag-maf stop
~/dt-cli/rag-maf start
~/dt-cli/rag-maf status

# Re-index project
cd /path/to/project
~/dt-cli/rag-maf index

# Start Claude
claude
```

---

## 📚 Additional Resources

- [README](./README.md) - Overview and features
- [Ubuntu Installation Guide](./UBUNTU_DEPLOYMENT_GUIDE.md) - Server deployment
- [Quick Start](./QUICKSTART_UBUNTU.md) - Fast setup
- [Architecture](./ARCHITECTURE.md) - System design

---

## 💡 Tips & Tricks

### Auto-start MCP Server

Add to your `~/.bashrc`:

```bash
# Auto-start MCP server when shell starts
if ! pgrep -f "mcp_server/server.py" > /dev/null; then
    ~/dt-cli/rag-maf start > /dev/null 2>&1
fi
```

### Project-specific Alias

```bash
# Add to ~/.bashrc
alias rag-start='~/dt-cli/rag-maf start'
alias rag-stop='~/dt-cli/rag-maf stop'
alias rag-status='~/dt-cli/rag-maf status'
alias rag-index='~/dt-cli/rag-maf index'
```

### Quick Setup for New Projects

```bash
# Create function in ~/.bashrc
setup-rag() {
    cp -r ~/dt-cli/.claude .
    chmod +x .claude/hooks/*.sh
    ~/dt-cli/rag-maf start
    ~/dt-cli/rag-maf index
    echo "✅ RAG plugin ready! Start Claude with: claude"
}

# Usage in any project:
cd /path/to/project
setup-rag
```

---

Enjoy your RAG-powered development experience! 🚀
