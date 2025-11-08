# 🚀 Quick Start: Ubuntu Server Installation

## One-Command Installation

```bash
git clone <repository-url> dt-cli && cd dt-cli && chmod +x ubuntu-install.sh && ./ubuntu-install.sh
```

## What Gets Installed

1. ✅ **System Prerequisites** - Build tools, Python 3.8+, Git
2. ✅ **Node.js 20.x** - Required for Claude Code
3. ✅ **Claude Code CLI** - Anthropic's official CLI
4. ✅ **Python Environment** - Virtual environment with all dependencies
5. ✅ **dt-cli RAG Plugin** - Local RAG + Multi-Agent Framework
6. ✅ **MCP Server** - Model Context Protocol integration
7. ✅ **Embedding Models** - Sentence transformers (all-MiniLM-L6-v2)

## Authentication (Claude Code Max Plan)

During installation, choose one of:

### Option 1: Interactive (Easiest)
- Script provides URL
- Open in browser
- Login with your Anthropic account
- Done!

### Option 2: API Key
- Get key from: https://console.anthropic.com/settings/keys
- Paste when prompted
- Saved automatically to `~/.bashrc`

### Option 3: Skip (Configure Later)
```bash
# Configure later with:
claude auth login
# OR
export ANTHROPIC_API_KEY='your-key-here'
```

## After Installation

### 1. Reload Shell
```bash
source ~/.bashrc
```

### 2. Verify Installation
```bash
claude --version
~/dt-cli/rag-maf status
```

### 3. Start Using
```bash
# Go to your project
cd /path/to/your/project

# Start Claude Code
claude

# Use RAG commands
/rag-status
/rag-index
/rag-query how does this work?
```

## Deployment Methods

### From Local Machine to Server

**Method 1: Git Clone (Recommended)**
```bash
ssh user@server.com
git clone <repo-url> dt-cli
cd dt-cli
./ubuntu-install.sh
```

**Method 2: SCP Transfer**
```bash
# On local machine
tar -czf dt-cli.tar.gz dt-cli/
scp dt-cli.tar.gz user@server.com:~/

# On server
tar -xzf dt-cli.tar.gz
cd dt-cli
./ubuntu-install.sh
```

**Method 3: rsync**
```bash
# On local machine
rsync -avz dt-cli/ user@server.com:~/dt-cli/

# On server
cd dt-cli
./ubuntu-install.sh
```

## Troubleshooting

### Script Fails
```bash
# Run with debug output
bash -x ubuntu-install.sh
```

### Authentication Issues
```bash
# Use API key method instead
export ANTHROPIC_API_KEY='your-key'
echo 'export ANTHROPIC_API_KEY="your-key"' >> ~/.bashrc
```

### Permission Denied
```bash
chmod +x ubuntu-install.sh
sudo chown -R $USER:$USER dt-cli/
```

## Full Documentation

- 📖 [Complete Deployment Guide](./UBUNTU_DEPLOYMENT_GUIDE.md)
- 📖 [README](./README.md)
- 📖 [Architecture](./ARCHITECTURE.md)

## Support

- Installation issues: Check [UBUNTU_DEPLOYMENT_GUIDE.md](./UBUNTU_DEPLOYMENT_GUIDE.md)
- RAG plugin issues: Check logs in `~/dt-cli/logs/`
- Claude Code issues: Run `claude --help`

---

**Total Installation Time:** ~5-10 minutes (depending on internet speed)

**Disk Space Required:** ~2GB

**Prerequisites:** Ubuntu 18.04+ with sudo access

---

🎉 **You're all set! Enjoy RAG-powered Claude Code!**
