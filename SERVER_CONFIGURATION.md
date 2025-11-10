# Server Configuration Guide

## Problem
If you're running the dt-cli server on a different machine or port, the CLI needs to know where to connect.

## Solution

There are three ways to configure the server URL:

### Option 1: Environment Variable (Recommended)

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit the file and set your server URL
DT_CLI_SERVER_URL=http://192.168.1.104:58432
```

Then load the environment variable before running the CLI:

```bash
# Source the .env file (if using bash/zsh)
export $(cat .env | grep -v '^#' | xargs)

# Run the CLI
python3 -m src.cli.interactive
```

### Option 2: Command Line Argument

Pass the server URL directly when running the CLI:

```bash
python3 -m src.cli.interactive --server http://192.168.1.104:58432
```

### Option 3: Shell Export

Export the environment variable in your shell:

```bash
export DT_CLI_SERVER_URL=http://192.168.1.104:58432
python3 -m src.cli.interactive
```

## Testing the Connection

Before running the CLI, test if the server is accessible:

```bash
# Test health endpoint
curl http://192.168.1.104:58432/health

# Test query endpoint
curl -X POST http://192.168.1.104:58432/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

## Troubleshooting

### Connection Refused or Timeout

- Check if the server is running on the remote machine
- Verify the port number is correct
- Check firewall rules on the server machine
- Ensure the server is binding to `0.0.0.0` (not just `127.0.0.1`)

### Access Denied

- The server may have authentication/authorization enabled
- Check server logs for more details
- Verify you have network access to the server

### Wrong Endpoint

- Make sure the server has fully initialized before connecting
- Check server logs for initialization errors
- Try restarting the server with `--auto-port` flag

## Server Startup

To start the server on the remote machine:

```bash
# Start on default port (58432)
python3 src/mcp_server/standalone_server.py

# Start with auto-port selection if default is busy
python3 src/mcp_server/standalone_server.py --auto-port

# Start on specific port
python3 src/mcp_server/standalone_server.py --port 58433

# Start on custom host
python3 src/mcp_server/standalone_server.py --host 0.0.0.0 --port 58432
```

**Important:** To accept connections from other machines, bind to `0.0.0.0`:

```bash
python3 src/mcp_server/standalone_server.py --host 0.0.0.0 --port 58432 --auto-port
```

## Example: Remote Setup

1. On the server machine (192.168.1.104):
```bash
cd /path/to/dt-cli
python3 src/mcp_server/standalone_server.py --host 0.0.0.0 --port 58432
```

2. On the client machine:
```bash
# Set the server URL
export DT_CLI_SERVER_URL=http://192.168.1.104:58432

# Run the CLI
python3 -m src.cli.interactive
```

## Port Forwarding / SSH Tunneling

If you can't access the remote server directly, use SSH tunneling:

```bash
# Forward remote port 58432 to local port 58432
ssh -L 58432:localhost:58432 user@192.168.1.104

# In another terminal, use the CLI with localhost
export DT_CLI_SERVER_URL=http://localhost:58432
python3 -m src.cli.interactive
```
