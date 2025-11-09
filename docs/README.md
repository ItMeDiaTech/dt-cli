# dt-cli Documentation

Welcome to the dt-cli documentation! This guide will help you navigate all available documentation.

---

## 📚 Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── guides/                # User and installation guides
├── phases/                # Implementation phase documentation
└── archive/               # Historical/deprecated documentation
```

---

## 🚀 Getting Started

**New to dt-cli?** Start here:

1. **[Main README](../README.md)** - Overview, features, quick start
2. **[Integration Guide](../INTEGRATION_GUIDE.md)** - Complete setup guide with examples
3. **[Quick Start Guide](./guides/QUICKSTART.md)** - Get up and running in 5 minutes

---

## 📖 User Guides

### Installation & Setup
- **[Installation Guide](./guides/INSTALLATION.md)** - Detailed installation instructions
- **[Quick Start](./guides/QUICKSTART.md)** - Fast track to getting started
- **[Quick Start (Open Source)](./guides/QUICKSTART_OPEN_SOURCE.md)** - Using only open source components
- **[Ubuntu Deployment](./guides/UBUNTU_DEPLOYMENT_GUIDE.md)** - Complete Ubuntu server deployment
- **[Quick Start (Ubuntu)](./guides/QUICKSTART_UBUNTU.md)** - Ubuntu-specific quick start

### Usage & Configuration
- **[User Guide](./guides/USER_GUIDE.md)** - Comprehensive user documentation
- **[Integration Guide](../INTEGRATION_GUIDE.md)** - All three usage modes explained
  - Claude Code MCP Plugin
  - Interactive Terminal UI
  - REST API

### Architecture & Design
- **[Architecture](./guides/ARCHITECTURE.md)** - System architecture and design
- **[Implementation Roadmap](./guides/IMPLEMENTATION_ROADMAP.md)** - Development roadmap

---

## 🏗️ Implementation Phases

The dt-cli project was implemented in 4 phases:

### Phase 1: RAG Foundation & Auto-Trigger
- **[Week 1: AST Chunking & BGE Embeddings](./phases/PHASE1_WEEK1_COMPLETE.md)**
  - Tree-sitter AST parsing
  - BGE embeddings with instruction prefix
  - Intelligent code chunking

- **[Week 2: Intent-Based Auto-Triggering](./phases/PHASE1_WEEK2_COMPLETE.md)**
  - Query intent classification
  - Automatic RAG vs direct LLM routing
  - Context-aware triggering

### Phase 2: Agentic Debugging
- **[Agentic Debugging Workflows](./phases/PHASE2_COMPLETE.md)**
  - Debug agent for error analysis
  - Code review agent with security checks
  - Multi-step reasoning with LangGraph

### Phase 3: Knowledge Graph
- **[Knowledge Graph Integration](./phases/PHASE3_COMPLETE.md)**
  - Dependency tracking
  - Impact analysis
  - Usage finding
  - Relationship mapping

### Phase 4: Quality & Search
- **[RAGAS Evaluation & Hybrid Search](./phases/PHASE4_COMPLETE.md)**
  - RAGAS evaluation metrics
  - Hybrid search (BM25 + semantic)
  - A/B testing framework
  - Query expansion

---

## 🎯 Use Case Guides

### For Developers
- **Codebase Navigation**: Use RAG queries to understand code
- **Bug Fixing**: Debug agent analyzes errors automatically
- **Code Review**: Catch issues before deployment
- **Refactoring**: Understand impact with knowledge graph

### For Teams
- **Knowledge Sharing**: Build shared code knowledge
- **Quality Assurance**: Automated quality checks
- **Documentation**: Generate context-aware docs
- **Onboarding**: Help new developers

---

## 🔧 Reference Documentation

### API Reference
See [Integration Guide](../INTEGRATION_GUIDE.md) for complete API documentation including:
- `/query` - RAG queries
- `/debug` - Error debugging
- `/review` - Code review
- `/graph/build` - Build knowledge graph
- `/graph/query` - Query knowledge graph
- `/evaluate` - RAGAS evaluation
- `/hybrid-search` - Hybrid search

### Configuration Reference
See [Integration Guide - Configuration](../INTEGRATION_GUIDE.md#configuration) for:
- `llm-config.yaml` - LLM and system configuration
- `.env` - Environment variables
- `.claude/mcp-config.json` - Claude Code integration

---

## 💡 Tips & Best Practices

### Performance Optimization
1. **Pre-build knowledge graph** for large codebases
2. **Tune hybrid search weights** for your specific code
3. **Adjust chunk size** based on your needs
4. **Use auto-trigger threshold** tuning for best results

### Configuration Tips
1. **Start with defaults** and adjust based on needs
2. **Use local LLMs** (Ollama) for privacy
3. **Enable caching** for better performance
4. **Monitor metrics** via statistics endpoint

### Troubleshooting
- See [Integration Guide - Troubleshooting](../INTEGRATION_GUIDE.md#troubleshooting)
- Check server logs: `/tmp/dt-cli-server.log`
- Verify health: `curl http://localhost:8765/health`
- Use interactive TUI option 7 for system stats

---

## 🔄 Version History

### v1.0 (Current)
**Complete feature set:**
- ✅ AST-based chunking
- ✅ BGE embeddings
- ✅ Auto-trigger system
- ✅ Debug agent
- ✅ Code review agent
- ✅ Knowledge graph
- ✅ RAGAS evaluation
- ✅ Hybrid search
- ✅ Interactive TUI
- ✅ Claude Code integration
- ✅ REST API

---

## 🤝 Contributing

Want to contribute? See the [Contributing Guidelines](../README.md#-development) in the main README.

### Development Documentation
- Project structure: See [Architecture](./guides/ARCHITECTURE.md)
- Running tests: See [Main README - Development](../README.md#-development)
- Code style: Python PEP 8, formatted with Black

---

## 📞 Support & Community

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/ItMeDiaTech/dt-cli/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ItMeDiaTech/dt-cli/discussions)
- **Documentation**: You're reading it!

### Reporting Bugs
When reporting bugs, please include:
1. dt-cli version
2. Python version
3. Operating system
4. Steps to reproduce
5. Error messages/logs
6. Expected vs actual behavior

### Feature Requests
We welcome feature requests! Please:
1. Check existing issues first
2. Describe the use case
3. Explain why it's valuable
4. Provide examples if possible

---

## 🗺️ Documentation Roadmap

**Upcoming documentation:**
- Video tutorials
- Example projects
- Best practices guide
- Advanced configuration guide
- Security hardening guide
- Performance tuning guide
- Docker deployment guide
- CI/CD integration guide

---

## 📦 Additional Resources

### External Resources
- [sentence-transformers Docs](https://www.sbert.net/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Rich Docs](https://rich.readthedocs.io/)

### Research Papers
- **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **RAGAS**: "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
- **BM25**: "Okapi at TREC-3"

---

## 📝 Documentation Conventions

### Code Examples
- Bash commands use `$` prompt
- Python code is syntax highlighted
- Example output is shown in comments

### Version Compatibility
- All examples tested on Python 3.8+
- API examples use curl (works on Linux/Mac/WSL)
- Windows users: use PowerShell or Git Bash

### File Paths
- Unix-style paths (`/path/to/file`)
- Windows users: replace `/` with `\`

---

## 🎓 Learning Path

**Recommended learning order:**

1. **Basics** (1 hour)
   - Read main README
   - Try Quick Start
   - Run interactive TUI

2. **Core Features** (2-3 hours)
   - Try each TUI menu option
   - Read Integration Guide
   - Experiment with API

3. **Advanced** (1-2 days)
   - Read phase documentation
   - Configure for your codebase
   - Tune performance
   - Integrate with workflow

4. **Expert** (Ongoing)
   - Read architecture docs
   - Contribute to project
   - Share experiences

---

## 🔍 Finding What You Need

### Quick Reference

| I want to... | Go to... |
|-------------|----------|
| Get started quickly | [Quick Start](./guides/QUICKSTART.md) |
| Install on Ubuntu | [Ubuntu Deployment](./guides/UBUNTU_DEPLOYMENT_GUIDE.md) |
| Use all three modes | [Integration Guide](../INTEGRATION_GUIDE.md) |
| Understand architecture | [Architecture](./guides/ARCHITECTURE.md) |
| Configure LLM | [Main README - Configuration](../README.md#%EF%B8%8F-configuration) |
| Debug issues | [Integration Guide - Troubleshooting](../INTEGRATION_GUIDE.md#troubleshooting) |
| Learn about features | [Phase Documentation](./phases/) |
| Contribute code | [Main README - Development](../README.md#-development) |

---

**Happy coding with dt-cli! 🚀**

For questions or feedback, please open an issue or discussion on GitHub.
