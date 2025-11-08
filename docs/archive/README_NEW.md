# dt-cli: 100% Open Source RAG/MAF Coding Assistant

A comprehensive **100% open source** coding assistant combining:
- **RAG** (Retrieval-Augmented Generation) for context-aware code understanding
- **MAF** (Multi-Agent Framework) for intelligent task orchestration
- **Open Source LLMs** (Ollama/vLLM) for local, private code generation
- **Zero vendor lock-in** - runs completely locally or on your infrastructure

## ✨ What Makes dt-cli Unique

### 100% Open Source Stack
Every component is open source:
- **LLMs**: Qwen3-Coder, DeepSeek-V3, StarCoder2 (via Ollama/vLLM)
- **Vector Database**: ChromaDB (automatic indexing)
- **Orchestration**: LangGraph (state-of-the-art multi-agent framework)
- **Embeddings**: sentence-transformers (local, privacy-preserving)

### No Proprietary Dependencies
- ✅ No API keys required
- ✅ No subscription fees
- ✅ No vendor lock-in
- ✅ Complete data privacy
- ✅ Runs on your hardware

### Optional Claude Code Integration
If you already have Claude Code, dt-cli can integrate with it. But **you don't need it** - dt-cli works perfectly with open source LLMs.

## 🚀 Features

### RAG System
- **Automatic Indexing**: ChromaDB handles everything automatically
- **Semantic Search**: Find code by meaning, not just keywords
- **Context Injection**: Automatically provides relevant code to LLM
- **Incremental Updates**: Fast re-indexing on file changes

### Multi-Agent Framework
- **LangGraph-based**: Industry-leading orchestration (11.7K stars, 4.2M downloads/month)
- **Agent Patterns**: Supervisor, pipeline, scatter-gather workflows
- **Code Analysis**: Specialized agents for different tasks
- **Debugging**: Automatic error pattern matching and fix suggestion

### Open Source LLMs
- **Ollama** (recommended for development):
  - Free and easy to install
  - Runs on consumer hardware
  - Great for prototyping

- **vLLM** (recommended for production):
  - 3.2x higher throughput than Ollama
  - Production-grade performance
  - Multi-GPU support

- **Claude** (optional):
  - For users who prefer Anthropic
  - Not required for dt-cli

## 📦 Installation

### Prerequisites
- Python 3.8+
- Git

### Step 1: Install dt-cli

```bash
git clone https://github.com/ItMeDiaTech/dt-cli.git
cd dt-cli
pip install -r requirements.txt
```

### Step 2: Install an LLM Provider

Choose one (or use multiple):

#### Option A: Ollama (Easiest - Recommended for Development)

```bash
# Install Ollama
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download

# Pull a model
ollama pull qwen3-coder  # Best for agentic coding workflows
# OR
ollama pull deepseek-v3  # Best for reasoning
# OR
ollama pull starcoder2   # Fast, good for completion

# Start Ollama server
ollama serve
```

#### Option B: vLLM (Production Deployment)

```bash
# Install vLLM
pip install vllm

# Start vLLM server
vllm serve qwen3-coder --port 8000

# For multi-GPU deployment
vllm serve qwen3-coder --tensor-parallel-size 4 --port 8000
```

#### Option C: Claude (Optional)

If you already have an Anthropic subscription:
1. Get API key from https://console.anthropic.com/
2. Configure in `llm-config.yaml`

### Step 3: Configure dt-cli

Edit `llm-config.yaml`:

```yaml
# For Ollama (default - already configured)
provider: ollama
llm:
  model_name: qwen3-coder
  base_url: http://localhost:11434

# For vLLM (uncomment to use)
# provider: vllm
# llm:
#   model_name: qwen3-coder
#   base_url: http://localhost:8000

# For Claude (uncomment to use)
# provider: claude
# llm:
#   model_name: claude-sonnet-4.5
#   api_key: <your-api-key>
```

### Step 4: Start the Server

```bash
# Standalone server (includes LLM)
python -m src.mcp_server.standalone_server

# Or with custom config
python -m src.mcp_server.standalone_server --config my-config.yaml
```

The server will start on http://localhost:8765

## 🎯 Usage

### Query Endpoint (RAG + LLM)

```bash
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work in this codebase?",
    "use_rag": true
  }'
```

### Direct Generation (No RAG)

```bash
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a function to parse JSON"
  }'
```

### RAG Search Only

```bash
curl -X POST http://localhost:8765/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication code"
  }'
```

### Health Check

```bash
curl http://localhost:8765/health
```

### Server Info

```bash
curl http://localhost:8765/info
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────┐
│            User Query                           │
└───────────────┬─────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────┐
│         dt-cli Server (FastAPI)                 │
│  ┌──────────────────────────────────────────┐   │
│  │  LLM Provider (Configurable)             │   │
│  │  - Ollama (local, open source)           │   │
│  │  - vLLM (production, open source)        │   │
│  │  - Claude (optional)                     │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  RAG System                              │   │
│  │  - ChromaDB (vector storage)             │   │
│  │  - Sentence Transformers (embeddings)    │   │
│  │  - Automatic indexing                    │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  MAF System                              │   │
│  │  - LangGraph (orchestration)             │   │
│  │  - Multi-agent workflows                 │   │
│  │  - Intelligent routing                   │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

## 🔧 Configuration

### LLM Configuration

File: `llm-config.yaml`

```yaml
provider: ollama  # or vllm, claude

llm:
  model_name: qwen3-coder
  base_url: http://localhost:11434
  temperature: 0.1
  max_tokens: 4096

rag:
  chunk_size: 1000
  chunk_overlap: 200
  max_results: 5
  embedding_model: sentence-transformers/all-MiniLM-L6-v2

maf:
  enabled: true
  max_iterations: 10
  timeout: 300

auto_trigger:
  enabled: true
  threshold: 0.7
  show_activity: true
```

### Environment Variables

Override config with environment variables:

```bash
export DT_CLI_PROVIDER=ollama
export DT_CLI_MODEL=qwen3-coder
export DT_CLI_BASE_URL=http://localhost:11434
export DT_CLI_TEMPERATURE=0.1
```

## 📚 Recommended Models

### Best for Agentic Coding Workflows
**Qwen3-Coder-480B** (if you have GPUs) or **Qwen3-Coder** (quantized)
- 256K context window
- Repository-scale understanding
- Purpose-built for agent orchestration
```bash
ollama pull qwen3-coder
```

### Best for Reasoning
**DeepSeek-V3**
- 671B parameters (MoE)
- Reinforcement learning enhanced
- Excellent at breaking down problems
```bash
ollama pull deepseek-v3
```

### Fast and Efficient
**StarCoder2-7B** or **15B**
- Great for code completion
- Runs on consumer hardware
- 80+ programming languages
```bash
ollama pull starcoder2
```

## 🌟 Why Open Source LLMs > Claude for Code

Based on our research (see `OPEN_SOURCE_LLM_RAG_RESEARCH.md`):

| Feature | Claude Sonnet 4.5 | Qwen3-Coder (Open) | DeepSeek-V3 (Open) |
|---------|-------------------|--------------------|--------------------|
| Context window | 200K | **256K** ✅ | 128K |
| Coding ability | Excellent | Excellent | Excellent |
| Agentic workflows | Good | **Purpose-built** ✅ | **RL-enhanced** ✅ |
| Repository understanding | Very good | **Specialized** ✅ | Very good |
| Cost | $200/month | **$0** ✅ | **$0** ✅ |
| Privacy | Cloud | **Local** ✅ | **Local** ✅ |
| Customization | ❌ No | **✅ Yes** | **✅ Yes** |

**Conclusion**: Open source LLMs match or exceed Claude for coding tasks, with the added benefits of privacy, cost savings, and full control.

## 🚀 Performance

### Development (Ollama on Laptop)
- **Hardware**: MacBook Pro M2 (24GB RAM)
- **Model**: Qwen3-Coder (quantized)
- **Latency**: ~2-5 seconds per query
- **Cost**: $0

### Production (vLLM on Cloud)
- **Hardware**: 4x NVIDIA A100 GPUs
- **Model**: Qwen3-Coder-480B
- **Throughput**: 3.2x higher than Ollama
- **Latency**: <1 second per query
- **Cost**: ~$350/day (on-demand)

## 📖 Documentation

- [Complete Research Report](OPEN_SOURCE_LLM_RAG_RESEARCH.md) - In-depth analysis of open source LLMs and RAG
- [RAG/MAF Automation](RESEARCH_RAG_MAF_AUTOMATION.md) - Agentic workflows and best practices
- [Installation Guides](UBUNTU_DEPLOYMENT_GUIDE.md) - Ubuntu server setup

## 🤝 Contributing

Contributions are welcome! This project is 100% open source.

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

Built with these excellent open source projects:
- [Ollama](https://ollama.com/) - Easy LLM deployment
- [vLLM](https://github.com/vllm-project/vllm) - High-performance serving
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [Sentence Transformers](https://www.sbert.net/) - Embeddings

Open source models:
- [Qwen3-Coder](https://github.com/QwenLM/Qwen) - Alibaba Cloud
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) - DeepSeek AI
- [StarCoder2](https://github.com/bigcode-project/starcoder2) - BigCode

## 🎯 What's Next?

See our [implementation roadmap](OPEN_SOURCE_LLM_RAG_RESEARCH.md#14-implementation-roadmap-for-dt-cli):

**Phase 1** (Weeks 1-2): Tree-sitter AST chunking + BGE embeddings
**Phase 2** (Weeks 3-4): Agentic debugging workflows
**Phase 3** (Weeks 5-6): Neo4j knowledge graph
**Phase 4** (Weeks 7-8): Production deployment + evaluation

---

**Remember**: dt-cli is 100% open source. You don't need any proprietary services to run it!
