"""
Bridge to Claude Code CLI for seamless integration.
"""

from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class ClaudeCodeBridge:
    """
    Bridge between MCP server and Claude Code CLI.
    Enables zero-token overhead by providing direct tool access.
    """

    def __init__(self, rag_tools, maf_tools):
        """
        Initialize the bridge.

        Args:
            rag_tools: RAG tools instance
            maf_tools: MAF tools instance
        """
        self.rag_tools = rag_tools
        self.maf_tools = maf_tools
        logger.info("ClaudeCodeBridge initialized")

    def get_all_tools(self) -> Dict[str, Any]:
        """
        Get all available tools for Claude Code.

        Returns:
            Dictionary of all tool definitions
        """
        tools = {
            "rag": self.rag_tools.get_tools(),
            "maf": self.maf_tools.get_tools()
        }

        return tools

    def execute_tool(
        self,
        category: str,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool through the bridge.

        Args:
            category: Tool category (rag or maf)
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            Tool execution results
        """
        logger.info(f"Bridge executing {category}.{tool_name}")

        try:
            if category == "rag":
                return self.rag_tools.execute_tool(tool_name, parameters)
            elif category == "maf":
                return self.maf_tools.execute_tool(tool_name, parameters)
            else:
                return {"error": f"Unknown category: {category}"}

        except Exception as e:
            logger.error(f"Bridge execution error: {e}")
            return {"error": str(e)}

    def format_for_claude(self, results: Dict[str, Any]) -> str:
        """
        Format results for Claude Code display.

        Args:
            results: Tool execution results

        Returns:
            Formatted string for display
        """
        if results.get("error"):
            return f"Error: {results['error']}"

        # Format based on result type
        if "results" in results and isinstance(results["results"], list):
            # RAG query results
            output = [f"Found {len(results['results'])} results:\n"]

            for i, result in enumerate(results['results'][:5], 1):
                metadata = result.get('metadata', {})
                file_path = metadata.get('file_path', 'unknown')
                score = 1 - result.get('distance', 1)

                output.append(f"{i}. {file_path} (score: {score:.2f})")
                output.append(f"   {result.get('text', '')[:100]}...\n")

            return "\n".join(output)

        elif "status" in results:
            # Status results
            return json.dumps(results['status'], indent=2)

        elif "agent_results" in results:
            # MAF orchestration results
            output = ["Multi-Agent Framework Results:\n"]

            for agent_name, agent_result in results.get('agent_results', {}).items():
                output.append(f"\n{agent_name}:")
                output.append(json.dumps(agent_result, indent=2))

            return "\n".join(output)

        else:
            return json.dumps(results, indent=2)

    def get_context_for_claude(self, query: str) -> Optional[str]:
        """
        Get relevant context for Claude without token overhead.

        Args:
            query: User query

        Returns:
            Context string or None
        """
        try:
            # Use RAG to get relevant context
            rag_result = self.rag_tools.execute_tool(
                "rag_query",
                {"query": query, "n_results": 3}
            )

            if rag_result.get("success") and rag_result.get("results"):
                results = rag_result["results"]

                context_parts = ["Relevant context from codebase:\n"]

                for result in results:
                    metadata = result.get('metadata', {})
                    file_path = metadata.get('file_path', 'unknown')

                    context_parts.append(f"\nFrom {file_path}:")
                    context_parts.append(result.get('text', '')[:300])

                return "\n".join(context_parts)

            return None

        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return None
