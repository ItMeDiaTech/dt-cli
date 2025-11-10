# Server Restart and Initialization Fix

## Problem

Users were experiencing a "404 - /query endpoint not found" error even though the server appeared to be running. This occurred when:

1. Server started successfully
2. Client tried to use `/query` or other endpoints
3. Got 404 errors indicating endpoints weren't registered

## Root Cause

The server initialization code had a **critical ordering issue**:

1. FastAPI app was created
2. Multiple components were initialized (RAG, MAF, agents, etc.)
3. **IF ANY COMPONENT FAILED** during initialization, routes were never set up
4. Server would start and run but have **NO ENDPOINTS**

This created a confusing situation where:
- `uvicorn` reported "server running"
- Health checks and API calls returned 404
- No clear error message about what went wrong

## Solution

### 1. Route Setup Priority

**Moved route registration to happen FIRST**, before any optional components:

```python
# OLD: Routes set up last (line 253)
# Initialize RAG, MAF, agents, etc... (lines 212-249)
self._setup_routes()  # If anything above failed, this never ran!

# NEW: Routes set up FIRST (line 213)
self._setup_routes()  # Always happens, guaranteed endpoints
# Initialize components with error handling...
```

### 2. Defensive Component Initialization

Wrapped each component initialization in try/except blocks:

```python
try:
    logger.info("Initializing RAG system...")
    self.rag_engine = QueryEngine()
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}. RAG features will be unavailable.")
    self.rag_engine = None
```

This ensures:
- Server always has endpoints
- Components that fail don't crash the entire server
- Clear error messages about what failed
- Server runs with **degraded but functional** capabilities

### 3. Null-Safe Endpoint Handlers

Added defensive checks in every endpoint:

```python
@self.app.post("/query")
async def query(request: QueryRequest):
    # Check if required components are available
    if not self.llm or not self.llm_available:
        raise HTTPException(
            status_code=503,
            detail="LLM provider is not available. Please configure an LLM provider."
        )

    # Check optional components before using
    if use_rag and self.rag_engine:
        rag_results = self.rag_engine.query(...)
```

### 4. Better Logging

Enhanced startup logging to verify routes are registered:

```
INFO: Setting up API routes (priority initialization)...
INFO: Registered 23 API routes
INFO:   /health: ✓ registered
INFO:   /query: ✓ registered
INFO:   /review: ✓ registered
INFO:   /debug: ✓ registered
```

## Changes Made

### Modified Files

1. **src/mcp_server/standalone_server.py**
   - Moved `_setup_routes()` call to line 213 (was line 253)
   - Added try/except blocks around all component initializations
   - Added null checks in all endpoint handlers
   - Enhanced logging with checkmarks for route registration
   - Made server resilient to component failures

2. **src/cli/interactive.py** (from previous fix)
   - Added `DT_CLI_SERVER_URL` environment variable support
   - Updated argument parser to use env var as default

3. **Documentation**
   - Created `.env.example` with configuration templates
   - Created `SERVER_CONFIGURATION.md` with setup instructions
   - Created this `SERVER_RESTART_FIX.md` explaining the fix

## Testing

### Before Fix
```bash
# Server starts but has no endpoints
$ python3 src/mcp_server/standalone_server.py
INFO: Server running...
$ curl http://localhost:58432/health
<404 Not Found>
```

### After Fix
```bash
# Server starts with routes registered immediately
$ python3 src/mcp_server/standalone_server.py
INFO: Setting up API routes (priority initialization)...
INFO: Registered 23 API routes
INFO:   /health: ✓ registered
INFO:   /query: ✓ registered
INFO: Server running...

$ curl http://localhost:58432/health
{"status":"healthy","llm":"healthy","rag":"healthy","endpoints":{...}}
```

### Degraded Mode Testing

Even if components fail, server still works:

```bash
# RAG initialization fails
ERROR: Failed to initialize RAG system: ChromaDB not available. RAG features will be unavailable.
INFO: Server initialized successfully (degraded mode)

$ curl http://localhost:58432/health
{"status":"degraded","llm":"healthy","rag":"unhealthy",...}

$ curl -X POST http://localhost:58432/query -d '{"query":"test"}'
# Still works! Just without RAG context
{"response":"...","context_used":0}
```

## Benefits

1. **Server Always Accessible** - Endpoints registered immediately
2. **Graceful Degradation** - Failed components don't crash server
3. **Clear Error Messages** - 503 errors explain what's unavailable
4. **Better Debugging** - Logs show exactly which components failed
5. **Production Ready** - Server can run with partial functionality

## Backwards Compatibility

✅ **Fully backwards compatible**
- All existing endpoints work the same
- No API changes
- No configuration changes required
- Existing clients work without modification

## Future Improvements

Consider:
1. Health endpoint could return more detailed component status
2. Add retry logic for component initialization
3. Hot-reload capability to reinitialize failed components
4. Metrics endpoint to track component availability over time

## Related Issues

This fix addresses:
- "Server running, still getting error: The /query endpoint was not found"
- 404 errors on all endpoints despite server running
- Silent initialization failures
- Confusing server state (running but non-functional)
