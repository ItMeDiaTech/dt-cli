# Code Style Guidelines for dt-cli

## No Emojis Policy

**IMPORTANT:** This codebase follows a strict no-emoji policy.

### Rules:
1. **NO emojis in code** - Python files, configuration files, or any code files
2. **NO emojis in user interface** - Terminal output, prompts, or messages
3. **NO emojis in documentation** - README files, guides, or comments
4. **NO emojis anywhere** - This is a professional codebase

### Rationale:
- Professional appearance
- Better accessibility
- Terminal compatibility
- Cleaner, more readable code
- Consistent with enterprise standards

### Examples:

**BAD:**
```python
console.print("[green]✅ Success![/green]")
console.print("[red]❌ Error occurred[/red]")
task = progress.add_task("🔍 Searching...", total=None)
```

**GOOD:**
```python
console.print("[green]Success![/green]")
console.print("[red]Error occurred[/red]")
task = progress.add_task("Searching...", total=None)
```

## Interface Guidelines

### User Prompts
- Use clear, concise text
- Avoid special characters except standard punctuation
- Use color coding (via Rich) for emphasis, not emojis

### Progress Indicators
- Use text descriptions: "Searching codebase..." not "🔍 Searching..."
- Use spinners and progress bars for visual feedback
- Use color coding: [green], [yellow], [red], [cyan]

### Error Messages
- Start with severity: "Error:", "Warning:", "Info:"
- Provide clear, actionable information
- Use color for severity level indication

## Command History

The interactive interface uses `prompt_toolkit` for command history:
- Up/Down arrows navigate previous commands
- History is saved to `~/.dt_cli_history`
- History persists across sessions

## Intelligent Mode

Users can use intelligent mode (`-i` flag) to:
- Skip the menu interface
- Go directly to RAG queries
- Natural language interaction

Usage:
```bash
python dt-cli.py --intelligent
```

## Auto-Start Server

The interactive interface automatically starts the server if not running:
- Checks server health on startup
- Starts server as background process if needed
- Waits up to 10 seconds for server to be ready
- Can be disabled with `--no-auto-start` flag

## Remember:
When contributing or modifying code:
1. Check for and remove any emojis
2. Use text-only descriptions
3. Leverage Rich library's color coding for emphasis
4. Keep interface clean and professional
