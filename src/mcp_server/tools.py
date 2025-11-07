"""
MCP Tools for RAG and MAF functionality.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RAGTools:
    """
    MCP tools for RAG operations.
    """

    def __init__(self, query_engine):
        """
        Initialize RAG tools.

        Args:
            query_engine: RAG query engine instance
        """
        self.query_engine = query_engine

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get MCP tool definitions for RAG.

        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "rag_query",
                "description": "Query the RAG system for relevant code and documentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5
                        },
                        "file_type": {
                            "type": "string",
                            "description": "Optional file type filter (e.g., '.py', '.md')"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "rag_index",
                "description": "Index the codebase for RAG search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "root_path": {
                            "type": "string",
                            "description": "Root path to index",
                            "default": "."
                        }
                    }
                }
            },
            {
                "name": "rag_status",
                "description": "Get the status of the RAG system",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a RAG tool.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            Tool execution results
        """
        logger.info(f"Executing RAG tool: {tool_name}")

        try:
            if tool_name == "rag_query":
                return self._rag_query(parameters)
            elif tool_name == "rag_index":
                return self._rag_index(parameters)
            elif tool_name == "rag_status":
                return self._rag_status()
            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e)}

    def _rag_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG query."""
        query = params.get('query')
        n_results = params.get('n_results', 5)
        file_type = params.get('file_type')

        results = self.query_engine.query(
            query_text=query,
            n_results=n_results,
            file_type=file_type
        )

        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }

    def _rag_index(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG indexing."""
        root_path = params.get('root_path', '.')

        self.query_engine.index_codebase(root_path=root_path)

        status = self.query_engine.get_status()

        return {
            "success": True,
            "message": "Indexing complete",
            "status": status
        }

    def _rag_status(self) -> Dict[str, Any]:
        """Get RAG status."""
        status = self.query_engine.get_status()

        return {
            "success": True,
            "status": status
        }


class MAFTools:
    """
    MCP tools for Multi-Agent Framework operations.
    """

    def __init__(self, orchestrator):
        """
        Initialize MAF tools.

        Args:
            orchestrator: Agent orchestrator instance
        """
        self.orchestrator = orchestrator

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get MCP tool definitions for MAF.

        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "maf_orchestrate",
                "description": "Orchestrate multiple agents to handle a complex query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query or task for agents to handle"
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Type of task (general, code_search, doc_search)",
                            "default": "general"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "maf_status",
                "description": "Get the status of the Multi-Agent Framework",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a MAF tool.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            Tool execution results
        """
        logger.info(f"Executing MAF tool: {tool_name}")

        try:
            if tool_name == "maf_orchestrate":
                return self._maf_orchestrate(parameters)
            elif tool_name == "maf_status":
                return self._maf_status()
            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e)}

    def _maf_orchestrate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MAF orchestration."""
        query = params.get('query')
        task_type = params.get('task_type', 'general')

        results = self.orchestrator.orchestrate(
            query=query,
            task_type=task_type
        )

        return {
            "success": True,
            "query": query,
            "results": results
        }

    def _maf_status(self) -> Dict[str, Any]:
        """Get MAF status."""
        status = self.orchestrator.get_status()

        return {
            "success": True,
            "status": status
        }
