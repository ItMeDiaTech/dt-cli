"""
Enhanced Agent Orchestrator with bounded context and new agents.
"""

from typing import Dict, Any, List, Optional, Annotated, TypedDict
import logging
from langgraph.graph import StateGraph, END
from .agents import (
    CodeAnalyzerAgent,
    DocumentationRetrieverAgent,
    ContextSynthesizerAgent,
    SuggestionGeneratorAgent
)
from .advanced_agents import (
    CodeSummarizationAgent,
    DependencyMappingAgent,
    SecurityAnalysisAgent
)
from .bounded_context import BoundedContextManager
import uuid

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State passed between agents."""
    query: str
    task_type: str
    agent_results: Dict[str, Any]
    context_id: str
    current_step: str


class EnhancedAgentOrchestrator:
    """
    Enhanced orchestrator with bounded context and additional agents.
    """

    def __init__(self, rag_engine: Any = None, max_contexts: int = 1000):
        """
        Initialize enhanced orchestrator.

        Args:
            rag_engine: RAG query engine for agents to use
            max_contexts: Maximum number of contexts to keep
        """
        self.rag_engine = rag_engine
        self.context_manager = BoundedContextManager(max_contexts=max_contexts)

        # Initialize agents
        self.agents = {
            'code_analyzer': CodeAnalyzerAgent(rag_engine=rag_engine),
            'doc_retriever': DocumentationRetrieverAgent(rag_engine=rag_engine),
            'synthesizer': ContextSynthesizerAgent(rag_engine=rag_engine),
            'suggestion_gen': SuggestionGeneratorAgent(rag_engine=rag_engine),
            'code_summarizer': CodeSummarizationAgent(rag_engine=rag_engine),
            'dependency_mapper': DependencyMappingAgent(rag_engine=rag_engine),
            'security_analyzer': SecurityAnalysisAgent(rag_engine=rag_engine)
        }

        # Build the agent graph with TRUE parallel execution
        self.graph = self._build_graph()
        self.app = self.graph.compile()

        logger.info(f"EnhancedAgentOrchestrator initialized with {len(self.agents)} agents")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with proper parallel execution."""
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("analyze_code", self._run_code_analyzer)
        workflow.add_node("retrieve_docs", self._run_doc_retriever)
        workflow.add_node("synthesize", self._run_synthesizer)
        workflow.add_node("generate_suggestions", self._run_suggestion_generator)

        # PARALLEL EXECUTION: Both start simultaneously
        workflow.add_conditional_edges(
            "__start__",
            lambda x: ["analyze_code", "retrieve_docs"],
            ["analyze_code", "retrieve_docs"]
        )

        # Both must complete before synthesis
        workflow.add_edge("analyze_code", "synthesize")
        workflow.add_edge("retrieve_docs", "synthesize")

        # Final step
        workflow.add_edge("synthesize", "generate_suggestions")
        workflow.add_edge("generate_suggestions", END)

        return workflow

    def _run_code_analyzer(self, state: AgentState) -> AgentState:
        """Run the code analyzer agent."""
        logger.info("Running CodeAnalyzer")

        context = {
            'query': state['query'],
            'task_type': state['task_type']
        }

        results = self.agents['code_analyzer'].execute(context)

        state['agent_results']['CodeAnalyzer'] = results
        state['current_step'] = 'code_analysis_complete'

        return state

    def _run_doc_retriever(self, state: AgentState) -> AgentState:
        """Run the documentation retriever agent."""
        logger.info("Running DocumentationRetriever")

        context = {
            'query': state['query'],
            'task_type': state['task_type']
        }

        results = self.agents['doc_retriever'].execute(context)

        state['agent_results']['DocumentationRetriever'] = results
        state['current_step'] = 'doc_retrieval_complete'

        return state

    def _run_synthesizer(self, state: AgentState) -> AgentState:
        """Run the context synthesizer agent."""
        logger.info("Running ContextSynthesizer")

        context = {
            'query': state['query'],
            'task_type': state['task_type'],
            'agent_results': state['agent_results']
        }

        results = self.agents['synthesizer'].execute(context)

        state['agent_results']['ContextSynthesizer'] = results
        state['current_step'] = 'synthesis_complete'

        return state

    def _run_suggestion_generator(self, state: AgentState) -> AgentState:
        """Run the suggestion generator agent."""
        logger.info("Running SuggestionGenerator")

        context = {
            'query': state['query'],
            'task_type': state['task_type'],
            'agent_results': state['agent_results']
        }

        results = self.agents['suggestion_gen'].execute(context)

        state['agent_results']['SuggestionGenerator'] = results
        state['current_step'] = 'suggestions_complete'

        return state

    def orchestrate(
        self,
        query: str,
        task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Orchestrate agents to handle a query.

        Args:
            query: User query or task
            task_type: Type of task (general, code_search, doc_search)

        Returns:
            Orchestration results
        """
        logger.info(f"Orchestrating query: {query}")

        # Create context
        context_id = str(uuid.uuid4())
        self.context_manager.create_context(
            context_id=context_id,
            query=query,
            task_type=task_type
        )

        # Initialize state
        initial_state: AgentState = {
            'query': query,
            'task_type': task_type,
            'agent_results': {},
            'context_id': context_id,
            'current_step': 'start'
        }

        # Run the graph
        try:
            final_state = self.app.invoke(initial_state)

            # Update context with final results
            for agent_name, results in final_state['agent_results'].items():
                self.context_manager.update_results(
                    context_id=context_id,
                    agent_name=agent_name,
                    results=results
                )

            # Get merged results
            merged_results = self.context_manager.merge_results(context_id)

            logger.info("Orchestration complete")
            return merged_results

        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            return {
                'error': str(e),
                'context_id': context_id
            }

    def run_specialized_agent(
        self,
        agent_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a specific agent directly.

        Args:
            agent_name: Name of the agent
            context: Execution context

        Returns:
            Agent results
        """
        if agent_name not in self.agents:
            return {'error': f'Unknown agent: {agent_name}'}

        logger.info(f"Running specialized agent: {agent_name}")

        try:
            return self.agents[agent_name].execute(context)
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            return {'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            'agents': list(self.agents.keys()),
            'active_contexts': len(self.context_manager.get_active_contexts()),
            'rag_enabled': self.rag_engine is not None,
            'context_stats': self.context_manager.get_stats()
        }

    def cleanup_old_contexts(self, max_age_seconds: int = 3600):
        """Clean up contexts older than specified age."""
        self.context_manager.clear_old_contexts(max_age_seconds)
