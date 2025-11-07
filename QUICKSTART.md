# Quick Start Guide

Get started with the RAG-MAF plugin in minutes!

## Installation

### One-Command Install

```bash
git clone <repository-url>
cd dt-cli
./install.sh
```

That's it! The installer handles everything:
- ✅ Python environment setup
- ✅ Dependency installation
- ✅ Model downloads
- ✅ Configuration

## First Use

### 1. Start Claude Code

```bash
cd your-project
claude-code
```

The plugin will automatically:
- Start the MCP server
- Begin indexing your codebase (in background)
- Register tools with Claude Code

### 2. Try a Query

In Claude Code, use the slash command:

```
/rag-query how does authentication work?
```

You'll get relevant code snippets and documentation!

### 3. Check Status

```
/rag-status
```

See how many files are indexed and system health.

## Common Use Cases

### Finding Code Patterns

```
/rag-query error handling patterns
/rag-query database connection setup
/rag-query API endpoint implementation
```

### Documentation Search

```
/rag-query setup instructions
/rag-query deployment process
/rag-query configuration options
```

### Re-indexing After Changes

```
/rag-index
```

Run this after adding new files or making significant changes.

## Manual Control

### Start/Stop Server

```bash
# Start MCP server
./rag-maf start

# Stop MCP server
./rag-maf stop

# Check if running
./rag-maf status
```

### Manual Indexing

```bash
./rag-maf index
```

## Understanding Results

Results are ranked by relevance:

```
Found 5 results:

1. src/auth/login.py (relevance: 95%)
   Implementation of login functionality...

2. src/auth/middleware.py (relevance: 87%)
   Authentication middleware for Express...
```

Higher percentages = more relevant to your query.

## Advanced Features

### Multi-Agent Orchestration

For complex queries, the system automatically orchestrates multiple agents:

1. **Code Analyzer** - Finds relevant code patterns
2. **Documentation Retriever** - Locates documentation
3. **Context Synthesizer** - Combines information
4. **Suggestion Generator** - Provides recommendations

This happens transparently when you use `/rag-query`.

### Automatic Context

The plugin provides context to Claude automatically - no need to manually copy code snippets!

## Configuration

Edit `.claude/rag-config.json` to customize:

```json
{
  "rag": {
    "chunk_size": 1000,
    "max_results": 5
  },
  "maf": {
    "enabled": true
  }
}
```

## Troubleshooting

### Server Won't Start

```bash
# Check logs
cat /tmp/rag-maf-mcp.log

# Restart
./rag-maf stop
./rag-maf start
```

### No Results Found

1. Make sure indexing completed:
   ```
   /rag-status
   ```

2. Re-index if needed:
   ```
   /rag-index
   ```

### Slow Queries

First query after startup may be slow (model loading). Subsequent queries are fast.

## System Requirements

- Python 3.8+
- 4GB RAM (8GB recommended)
- 2GB disk space

## Performance Tips

1. **Let indexing complete** before first query
2. **Re-index periodically** as codebase changes
3. **Use specific queries** for better results
4. **Filter by file type** if needed (coming soon)

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- See [examples/](examples/) for more use cases

## Support

- 🐛 Issues: GitHub Issues
- 📖 Docs: README.md
- 💬 Discussions: GitHub Discussions

Happy coding with RAG-powered assistance! 🚀
