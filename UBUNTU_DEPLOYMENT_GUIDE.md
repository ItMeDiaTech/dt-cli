# Ubuntu Server Deployment Guide
## Claude Code + dt-cli RAG Plugin Installation

This guide covers multiple methods to deploy and install Claude Code with the dt-cli RAG plugin on your Ubuntu server.

---

## 🚀 Quick Start

### Method 1: Direct Clone and Install (Recommended)

**On your Ubuntu server:**

```bash
# Clone the repository
git clone <repository-url> dt-cli
cd dt-cli

# Make the installation script executable
chmod +x ubuntu-install.sh

# Run the installation
./ubuntu-install.sh
```

The script will guide you through:
- ✅ System prerequisites installation
- ✅ Node.js 20.x setup
- ✅ Claude Code installation
- ✅ Authentication setup (Max plan support)
- ✅ Python environment configuration
- ✅ RAG plugin installation
- ✅ Optional systemd service setup

---

## 📋 Pre-Installation Checklist

Before running the installation script, ensure:

- [ ] Ubuntu 18.04 or newer
- [ ] Sudo privileges
- [ ] Internet connection
- [ ] At least 2GB RAM
- [ ] At least 2GB free disk space
- [ ] Claude Code Max plan credentials ready

---

## 🔐 Authentication Methods for Claude Code Max Plan

The installation script supports three authentication methods:

### Option 1: Interactive Authentication (Easiest)

**Best for:** Servers with GUI or when you can access URLs

```bash
# During installation, choose option 1
# Follow the prompts to authenticate
```

The script will:
1. Provide an authentication URL
2. You open it in your browser
3. Log in with your Anthropic account
4. Return to terminal when complete

### Option 2: API Key Authentication

**Best for:** Headless servers, automation, CI/CD

1. Get your API key from: https://console.anthropic.com/settings/keys
2. During installation, choose option 2
3. Paste your API key when prompted

**Or set manually:**

```bash
# Set for current session
export ANTHROPIC_API_KEY='your-api-key-here'

# Make it permanent
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Option 3: SSH Tunneling for Browser-based Auth

**Best for:** Headless servers that need browser authentication

**On your local machine:**

```bash
# Create SSH tunnel
ssh -L 8080:localhost:8080 user@your-server.com

# Keep this terminal open
```

**On the server:**

```bash
# Run authentication
claude auth login --port 8080
```

**On your local machine:**
- Open browser to `http://localhost:8080`
- Complete authentication
- Return to server terminal

---

## 📦 Deployment Methods

### Method 1: Git Clone (Recommended)

**On Ubuntu server:**

```bash
# Clone from GitHub
git clone https://github.com/your-username/dt-cli.git
cd dt-cli

# Run installation
chmod +x ubuntu-install.sh
./ubuntu-install.sh
```

### Method 2: SCP (Secure Copy) from Local Machine

**On your local machine:**

```bash
# Create a tarball of the repository
tar -czf dt-cli.tar.gz dt-cli/

# Copy to server
scp dt-cli.tar.gz user@your-server.com:~/

# Connect to server
ssh user@your-server.com

# Extract and install
tar -xzf dt-cli.tar.gz
cd dt-cli
chmod +x ubuntu-install.sh
./ubuntu-install.sh
```

### Method 3: rsync (Sync Files)

**On your local machine:**

```bash
# Sync directory to server
rsync -avz --progress dt-cli/ user@your-server.com:~/dt-cli/

# Connect to server
ssh user@your-server.com

# Install
cd dt-cli
chmod +x ubuntu-install.sh
./ubuntu-install.sh
```

### Method 4: wget/curl Direct Download

**If you have a release URL:**

```bash
# Download installation script directly
wget https://raw.githubusercontent.com/your-username/dt-cli/main/ubuntu-install.sh

# Make executable
chmod +x ubuntu-install.sh

# Run (will clone repo automatically if configured)
./ubuntu-install.sh
```

### Method 5: Docker (Advanced)

**Create a Dockerfile:**

```dockerfile
FROM ubuntu:22.04

# Install prerequisites
RUN apt-get update && apt-get install -y \
    curl git build-essential python3 python3-pip python3-venv

# Copy installation script
COPY ubuntu-install.sh /tmp/
RUN chmod +x /tmp/ubuntu-install.sh

# Run installation
RUN /tmp/ubuntu-install.sh

# Set working directory
WORKDIR /workspace

CMD ["/bin/bash"]
```

**Build and run:**

```bash
docker build -t claude-rag .
docker run -it -v $(pwd):/workspace claude-rag
```

---

## 🔧 Post-Installation Configuration

### 1. Verify Installation

```bash
# Check Claude Code
claude --version

# Check Python environment
source ~/dt-cli/venv/bin/activate
python3 --version

# Check MCP server
~/dt-cli/rag-maf status
```

### 2. Test Claude Code Authentication

```bash
# Test with a simple command
claude --help

# Start an interactive session
claude
```

### 3. Configure RAG Plugin

```bash
# Index a sample codebase
cd /path/to/your/project
~/dt-cli/rag-maf index

# Start Claude Code with RAG
claude
```

Inside Claude Code:
```
/rag-status
/rag-query what does this project do?
```

### 4. Enable Systemd Service (Optional)

If you chose to create the systemd service during installation:

```bash
# Check service status
sudo systemctl status rag-maf-mcp

# Start service
sudo systemctl start rag-maf-mcp

# Enable on boot
sudo systemctl enable rag-maf-mcp

# View logs
sudo journalctl -u rag-maf-mcp -f
```

---

## 🌐 Remote Server Access Scenarios

### Scenario 1: SSH Access Only

```bash
# Connect to server
ssh user@server.com

# Clone and install
git clone <repo-url> dt-cli
cd dt-cli
./ubuntu-install.sh

# Use API key authentication (option 2)
```

### Scenario 2: Limited Internet Access

**On a machine with internet:**

```bash
# Download all dependencies
cd dt-cli
pip download -r requirements.txt -d ./packages/
npm pack @anthropic-ai/claude-code
```

**Transfer to server and install offline:**

```bash
# Copy to server
scp -r dt-cli/ user@server.com:~/

# On server
cd dt-cli
pip install --no-index --find-links=./packages/ -r requirements.txt
npm install -g ./anthropic-ai-*.tgz
```

### Scenario 3: Corporate Network / Proxy

**Set proxy before installation:**

```bash
# Set proxy environment variables
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1

# Run installation
./ubuntu-install.sh
```

### Scenario 4: Cloud Server (AWS/GCP/Azure)

**AWS EC2 Example:**

```bash
# Connect to EC2 instance
ssh -i your-key.pem ubuntu@ec2-xxx-xxx-xxx-xxx.compute.amazonaws.com

# Clone and install
git clone <repo-url> dt-cli
cd dt-cli
./ubuntu-install.sh

# Configure security group to allow necessary ports if using MCP server remotely
```

---

## 🔒 Security Best Practices

### 1. API Key Management

```bash
# Never commit API keys to git
echo "ANTHROPIC_API_KEY=your-key" > .env
echo ".env" >> .gitignore

# Use environment variables
export ANTHROPIC_API_KEY=$(cat .env | grep ANTHROPIC_API_KEY | cut -d= -f2)
```

### 2. File Permissions

```bash
# Secure your configuration files
chmod 600 ~/.bashrc
chmod 700 ~/dt-cli/.rag_data
```

### 3. Firewall Configuration

```bash
# If running MCP server on network
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 8080/tcp  # MCP server (if needed)
sudo ufw enable
```

---

## 🐛 Troubleshooting

### Installation Script Fails

```bash
# Run with verbose output
bash -x ubuntu-install.sh

# Check logs
tail -f ~/dt-cli/logs/*.log
```

### Claude Code Authentication Issues

```bash
# Clear existing auth
claude auth logout

# Re-authenticate
claude auth login

# Or use API key
export ANTHROPIC_API_KEY='your-key'
```

### Python Virtual Environment Issues

```bash
# Recreate virtual environment
cd ~/dt-cli
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Node.js Version Issues

```bash
# Remove old Node.js
sudo apt remove nodejs npm

# Reinstall Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### MCP Server Won't Start

```bash
# Check if port is in use
sudo lsof -i :8080

# Kill existing process
pkill -f mcp_server

# Restart
~/dt-cli/rag-maf start
```

### Permission Denied Errors

```bash
# Fix script permissions
chmod +x ~/dt-cli/ubuntu-install.sh
chmod +x ~/dt-cli/rag-maf

# Fix ownership
sudo chown -R $USER:$USER ~/dt-cli
```

---

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 18.04 | Ubuntu 22.04 |
| RAM | 2GB | 4GB+ |
| Disk Space | 2GB | 5GB+ |
| CPU | 2 cores | 4+ cores |
| Python | 3.8 | 3.10+ |
| Node.js | 18.x | 20.x |
| Network | HTTP/HTTPS access | Stable connection |

---

## 🔄 Updating

### Update dt-cli

```bash
cd ~/dt-cli
git pull origin main
./ubuntu-install.sh  # Re-run installation
```

### Update Claude Code

```bash
sudo npm update -g @anthropic-ai/claude-code

# Verify new version
claude --version
```

### Update Python Dependencies

```bash
cd ~/dt-cli
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

---

## 📞 Getting Help

### Check Status

```bash
# System info
uname -a
python3 --version
node --version
claude --version

# Service status
~/dt-cli/rag-maf status
sudo systemctl status rag-maf-mcp
```

### Collect Logs

```bash
# Installation logs
cat ~/dt-cli/logs/*.log

# System logs
sudo journalctl -u rag-maf-mcp -n 100

# Claude Code logs
cat ~/.claude/logs/*.log
```

### Common Commands Reference

```bash
# Start MCP server
~/dt-cli/rag-maf start

# Stop MCP server
~/dt-cli/rag-maf stop

# Check MCP server status
~/dt-cli/rag-maf status

# Index codebase
~/dt-cli/rag-maf index

# Start Claude Code session
claude

# Authenticate Claude Code
claude auth login

# Check Claude Code auth status
claude auth status
```

---

## 🎯 Quick Command Reference

```bash
# One-line installation (if script is already on server)
curl -sSL https://raw.githubusercontent.com/your-username/dt-cli/main/ubuntu-install.sh | bash

# Complete setup from scratch
git clone <repo-url> dt-cli && cd dt-cli && chmod +x ubuntu-install.sh && ./ubuntu-install.sh

# Verify everything works
source ~/.bashrc && claude --version && ~/dt-cli/rag-maf status
```

---

## ✅ Installation Verification Checklist

After installation, verify:

- [ ] `claude --version` returns version number
- [ ] `python3 --version` shows 3.8+
- [ ] `node --version` shows 18+
- [ ] `~/dt-cli/rag-maf status` runs without errors
- [ ] Claude Code authentication works
- [ ] Can start Claude Code session
- [ ] Can use `/rag-status` command in Claude Code
- [ ] MCP server responds (if running as service)

---

## 🚀 You're Ready!

Once installation is complete and verified, you can:

1. Navigate to any project directory
2. Run `claude`
3. Use RAG-powered slash commands:
   - `/rag-query <question>`
   - `/rag-index`
   - `/rag-status`
   - And more!

Enjoy your RAG-powered Claude Code experience! 🎉
