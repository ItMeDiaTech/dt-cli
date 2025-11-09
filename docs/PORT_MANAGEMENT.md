# Port Management in dt-cli

## Overview

**Important: Only ONE application can use a network port at a time.** If you try to start the dt-cli server on a port that's already in use, it will fail.

## Automatic Port Detection

The dt-cli system includes smart port conflict resolution to handle this automatically.

## Usage

### Option 1: Let the Server Auto-Select a Port

```bash
python src/mcp_server/standalone_server.py --auto-port
```

This will:
- Try to use port 8765 (default)
- If 8765 is busy, try 8766, 8767, 8768, etc.
- Automatically use the first available port
- Log which port is being used

### Option 2: Manually Specify a Port

```bash
python src/mcp_server/standalone_server.py --port 9000
```

This will use port 9000, or fail if it's already in use.

### Option 3: Use Interactive Mode (Auto-Detects)

```bash
python dt-cli.py
# or
python dt-cli.py --intelligent
```

The interactive client will:
- Check if server is running
- Auto-start server if needed
- Automatically find an available port
- Connect to whatever port the server is using

## How It Works

### Port Availability Check

The system uses socket binding to check if a port is available:

```python
def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False
```

### Port Discovery

If the preferred port is busy, the system tries adjacent ports:

```python
def find_available_port(host: str, preferred_port: int, max_attempts: int = 10) -> int:
    """Find an available port, trying preferred port first."""
    # Try preferred port
    if is_port_available(host, preferred_port):
        return preferred_port

    # Try ports 8766, 8767, 8768, etc.
    for offset in range(1, max_attempts):
        port = preferred_port + offset
        if is_port_available(host, port):
            return port

    raise RuntimeError(f"No available port found near {preferred_port}")
```

## Common Scenarios

### Scenario 1: Port 8765 is Free
```bash
$ python src/mcp_server/standalone_server.py --auto-port
INFO - Starting standalone server on 127.0.0.1:8765
```

### Scenario 2: Port 8765 is Busy
```bash
$ python src/mcp_server/standalone_server.py --auto-port
INFO - Port 8765 is in use, using port 8766 instead
INFO - Starting standalone server on 127.0.0.1:8766
```

### Scenario 3: Without Auto-Port (Error)
```bash
$ python src/mcp_server/standalone_server.py
ERROR - Port 8765 is already in use!
ERROR - Use --auto-port to automatically find an available port
ERROR - Or specify a different port with --port
```

### Scenario 4: Interactive Client Auto-Starts
```bash
$ python dt-cli.py
Server not running. Starting server...
Port 8765 in use, using port 8766
Waiting for server to start on port 8766...
Server started successfully on http://127.0.0.1:8766!
```

## Troubleshooting

### Check What's Using a Port

**Linux/Mac:**
```bash
lsof -i :8765
# or
netstat -tuln | grep 8765
```

**Windows:**
```bash
netstat -ano | findstr :8765
```

### Kill Process on a Port

**Linux/Mac:**
```bash
lsof -ti:8765 | xargs kill -9
```

**Windows:**
```powershell
# Find PID
netstat -ano | findstr :8765
# Kill process
taskkill /PID <pid> /F
```

### Multiple Instances

If you want to run multiple dt-cli servers:

```bash
# Terminal 1
python src/mcp_server/standalone_server.py --port 8765

# Terminal 2
python src/mcp_server/standalone_server.py --port 8766

# Terminal 3
python src/mcp_server/standalone_server.py --port 8767
```

Or use auto-port for all of them:

```bash
# All will find available ports automatically
python src/mcp_server/standalone_server.py --auto-port &
python src/mcp_server/standalone_server.py --auto-port &
python src/mcp_server/standalone_server.py --auto-port &
```

## Best Practices

1. **Use `--auto-port` in development** - Prevents conflicts
2. **Use specific `--port` in production** - More predictable
3. **Let interactive client auto-start** - Most convenient
4. **Check logs for actual port** - Know where server is running

## Configuration

You can set a default port in your environment:

```bash
export DT_CLI_PORT=9000
```

Or in your shell config (~/.bashrc, ~/.zshrc):

```bash
alias dt-cli-server='python src/mcp_server/standalone_server.py --auto-port'
alias dt-cli='python dt-cli.py --intelligent'
```

## Technical Details

- **Port Range**: 8765-8774 (tries 10 ports)
- **Protocol**: TCP
- **Binding**: Reuses socket addresses (SO_REUSEADDR)
- **Timeout**: Server startup waits up to 10 seconds
- **Fallback**: Client updates URL to match actual server port

## Security Note

The server binds to `127.0.0.1` (localhost) by default, which means:
- Only accessible from your machine
- Not exposed to network
- Safe for development

To expose to network (use with caution):

```bash
python src/mcp_server/standalone_server.py --host 0.0.0.0 --port 8765
```
