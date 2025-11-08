# Quick Start Guide: dt-cli with Open Source LLMs

Get started with dt-cli in 5 minutes using 100% open source components!

## Step 1: Install dt-cli (1 minute)

```bash
git clone https://github.com/ItMeDiaTech/dt-cli.git
cd dt-cli
pip install -r requirements.txt
```

## Step 2: Install Ollama (2 minutes)

### macOS
```bash
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download from [https://ollama.com/download](https://ollama.com/download)

## Step 3: Pull a Model (2 minutes)

```bash
# Start Ollama server (in one terminal)
ollama serve

# Pull Qwen3-Coder (in another terminal)
ollama pull qwen3-coder
```

**Note**: The download may take a few minutes depending on your internet speed.

## Step 4: Start dt-cli Server (<1 minute)

```bash
python -m src.mcp_server.standalone_server
```

You should see:
```
INFO:     Starting standalone server on 127.0.0.1:8765
INFO:     Using LLM provider: OllamaProvider(model=qwen3-coder)
INFO:     Server is 100% open source - no proprietary dependencies!
```

## Step 5: Test It! (<1 minute)

In another terminal:

```bash
# Ask a coding question
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Write a Python function to calculate fibonacci numbers",
    "use_rag": false
  }'
```

You should get a response with generated code!

## 🎉 You're Done!

You now have a fully functional, 100% open source coding assistant running locally!

## Next Steps

### Index Your Codebase

```bash
curl -X POST http://localhost:8765/rag/index
```

### Query with RAG

```bash
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work in this codebase?",
    "use_rag": true
  }'
```

### Try Different Models

```bash
# Stop current Ollama model (Ctrl+C)

# Try DeepSeek-V3 (better at reasoning)
ollama pull deepseek-v3

# Update llm-config.yaml:
# model_name: deepseek-v3

# Restart server
python -m src.mcp_server.standalone_server
```

## Troubleshooting

### "Connection refused" error

Make sure Ollama is running:
```bash
ollama serve
```

### "Model not found" error

Pull the model first:
```bash
ollama pull qwen3-coder
```

### Server won't start

Check if port 8765 is already in use:
```bash
lsof -i :8765
# If something is using it, either kill it or start on different port:
python -m src.mcp_server.standalone_server --port 8766
```

## Configuration

Edit `llm-config.yaml` to customize:

```yaml
provider: ollama

llm:
  model_name: qwen3-coder  # Change to any Ollama model
  temperature: 0.1         # Lower = more deterministic
  max_tokens: 4096         # Maximum response length
```

## Available Models

List all available models:
```bash
curl http://localhost:8765/info
```

Pull more models:
```bash
# Best for agentic workflows
ollama pull qwen3-coder

# Best for reasoning
ollama pull deepseek-v3

# Fast and efficient
ollama pull starcoder2

# Lightweight (faster, less accurate)
ollama pull codellama:7b
```

## Resources

- [Full README](README_NEW.md) - Complete documentation
- [Research Report](OPEN_SOURCE_LLM_RAG_RESEARCH.md) - In-depth analysis
- [Configuration Guide](llm-config.yaml) - All configuration options

---

**Congratulations!** You're running a production-grade coding assistant with $0 in API costs and complete data privacy! 🎉
