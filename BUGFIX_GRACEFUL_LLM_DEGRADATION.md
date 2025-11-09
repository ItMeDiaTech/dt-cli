# Bug Fix: Graceful Degradation When LLM Provider is Unavailable

## Problem

When Ollama (or any other LLM provider) was not installed or unavailable, dt-cli would encounter a **500 Internal Server Error** when attempting code review. The error message was:

```
Error: 500
Failed to parse error response: Expecting value: line 1 column 1 (char 0)
Response: Internal Server Error
```

### Root Cause

1. **Ollama not installed**: The system was configured to use Ollama, but it was not installed on the machine
2. **No graceful degradation**: The server attempted to use the LLM provider without checking if it was actually available
3. **Poor error messages**: Error responses were not properly formatted as JSON
4. **Silent failures**: The system didn't clearly communicate when LLM features were unavailable

## Solution

Implemented comprehensive graceful degradation to allow dt-cli to function with **rule-based code analysis** even when LLM providers are unavailable.

### Changes Made

#### 1. Server Initialization (`src/mcp_server/standalone_server.py`)

**Before:**
```python
# Initialize LLM provider
self.llm = LLMProviderFactory.create_from_config(llm_config)
# Would crash if Ollama not available
```

**After:**
```python
# Initialize LLM provider with error handling
try:
    self.llm = LLMProviderFactory.create_from_config(llm_config)
    if self.llm and self.llm.check_health():
        logger.info(f"Using LLM provider: {self.llm}")
        self.llm_available = True
    else:
        logger.warning("LLM provider unhealthy. Operating with limited functionality.")
        self.llm_available = False
except Exception as e:
    logger.error(f"Failed to initialize LLM provider: {e}")
    logger.warning("Continuing without LLM support. Rule-based checks will still work.")
    self.llm = None
    self.llm_available = False
```

**Benefits:**
- Server continues to run even if LLM provider fails
- Clear logging about LLM availability
- Sets `llm_available` flag for downstream components

#### 2. Health Endpoint Improvements

**Before:**
```python
llm_healthy = self.llm.check_health()  # Would crash if self.llm is None
```

**After:**
```python
llm_healthy = self.llm.check_health() if self.llm else False

# Status determination:
# - "healthy": All systems including LLM working
# - "degraded": LLM unavailable but rule-based checks work
# - "unhealthy": Endpoints not registered properly
```

**Response format:**
```json
{
  "status": "degraded",
  "llm": "unhealthy",
  "llm_available": false,
  "rag": "healthy",
  "endpoints": {
    "query": true,
    "review": true,
    "debug": true
  }
}
```

#### 3. Review Endpoint Enhancements

**Added warning messages when LLM is unavailable:**

```python
result = review.to_dict()

if not self.llm_available:
    result['warning'] = (
        "LLM provider is not available. "
        "Review includes rule-based checks only. "
        "For advanced AI-powered analysis, please install and configure an LLM provider."
    )
    result['llm_used'] = False
else:
    result['llm_used'] = True

return result
```

**Response example (without LLM):**
```json
{
  "issues": [...],
  "summary": "Code quality: 7.5/10",
  "overall_score": 7.5,
  "metrics": {...},
  "issue_counts": {...},
  "warning": "LLM provider is not available. Review includes rule-based checks only.",
  "llm_used": false
}
```

#### 4. CLI Warning Display (`src/cli/interactive.py`)

**Added visual warning in the CLI:**

```python
if result.get('warning'):
    console.print(f"\n[yellow]⚠️  {result['warning']}[/yellow]")
```

Users now see a clear warning when LLM features are unavailable.

#### 5. Enhanced Logging (`src/debugging/review_agent.py`)

**Added detailed logging for debugging:**

```python
logger.info(f"Starting code review for {file_path or 'unnamed file'}")
logger.debug("Running security checks...")
logger.info(f"Rule-based checks found {len(issues)} issues")

if self.llm:
    logger.info("Running LLM-based advanced analysis...")
else:
    logger.info("LLM provider not available, skipping advanced analysis")
```

### What Still Works Without LLM

Even without Ollama or any LLM provider, dt-cli provides **full rule-based code analysis**:

#### Security Checks
- ✅ `eval()` and `exec()` usage detection
- ✅ Hardcoded password detection
- ✅ Shell injection vulnerabilities
- ✅ Unsafe pickle usage
- ✅ Command injection risks

#### Performance Checks
- ✅ Inefficient iteration patterns (`range(len())`)
- ✅ List concatenation in loops
- ✅ Multiple append calls optimization

#### Best Practices Checks
- ✅ Bare except clauses
- ✅ Overly broad exception handling
- ✅ Wildcard imports

#### Complexity Checks
- ✅ Function length analysis
- ✅ Code metrics calculation

### What Requires LLM

Only **advanced AI-powered analysis** requires an LLM provider:
- Deep semantic understanding
- Context-aware recommendations
- Complex pattern detection
- Natural language explanations

## Testing

Created and ran comprehensive tests to verify the fix:

```python
# Test code review without LLM
review_agent = CodeReviewAgent(llm_provider=None, rag_engine=None)
review = review_agent.review_code(test_code, file_path="test.py")

# Results:
✅ Review completed successfully!
   Overall Score: 3.0/10
   Issues Found: 5
   Summary: Code quality: 3.0/10 |  3 critical issue(s) |  1 high priority issue(s)
   Categories found: best_practices, performance, security
```

## How to Enable Full LLM Features

To enable advanced LLM-powered analysis:

### Option 1: Install Ollama (Recommended)

```bash
# 1. Install Ollama
# Visit: https://ollama.com/download

# 2. Pull a model
ollama pull qwen2.5-coder:1.5b   # For 8-16GB RAM systems
# OR
ollama pull qwen2.5-coder:32b    # For 32+ GB RAM systems

# 3. Start Ollama
ollama serve

# 4. Restart dt-cli server
```

### Option 2: Use vLLM (Production)

```bash
# 1. Install vLLM
pip install vllm

# 2. Start vLLM server
vllm serve qwen2.5-coder:32b --port 8000

# 3. Update llm-config.yaml
provider: vllm

# 4. Restart dt-cli server
```

### Option 3: Use Claude (Optional)

```bash
# 1. Get API key from https://console.anthropic.com/

# 2. Update llm-config.yaml
provider: claude
llm:
  api_key: <your-api-key>

# 3. Restart dt-cli server
```

## Impact

### Before Fix
- ❌ Server crashed when Ollama unavailable
- ❌ 500 errors with unclear messages
- ❌ No code review functionality at all
- ❌ Poor user experience

### After Fix
- ✅ Server runs successfully without LLM
- ✅ Clear warnings about LLM availability
- ✅ Full rule-based code review works
- ✅ Graceful degradation with helpful messages
- ✅ Users can still get value from the tool

## Files Modified

1. `src/mcp_server/standalone_server.py` - Server initialization and endpoints
2. `src/cli/interactive.py` - CLI warning display
3. `src/debugging/review_agent.py` - Enhanced logging

## Backward Compatibility

✅ **Fully backward compatible**
- Existing configurations work unchanged
- Systems with working LLM providers see no difference
- No breaking changes to API responses (only additions)

## Summary

This fix transforms dt-cli from a fragile system that requires LLM to function, into a **robust tool that provides value even without LLM**, while gracefully enabling advanced features when LLM is available.

**Key Principle**: Fail gracefully, communicate clearly, and provide maximum value in all scenarios.
