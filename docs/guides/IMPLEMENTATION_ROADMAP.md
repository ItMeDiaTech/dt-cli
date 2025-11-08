# Implementation Roadmap: Best-in-Class Agentic RAG/MAF Coding Environment

**Goal**: Build a fully autonomous, intelligent coding assistant that automatically triggers the right capabilities at the right time.

---

## 🎯 Priority Matrix (Impact × Effort)

| Feature | Impact | Effort | Priority | Timeline |
|---------|--------|--------|----------|----------|
| **Tree-sitter AST Chunking** | 🔥🔥🔥 Very High (+25-40% quality) | 🛠️ Medium | **P0** | Week 1-2 |
| **BGE Embeddings Upgrade** | 🔥🔥 High (code-optimized) | 🛠️ Low | **P0** | Week 1 |
| **Intent-Based Auto-Triggering** | 🔥🔥🔥 Very High (UX game-changer) | 🛠️🛠️ High | **P0** | Week 2-3 |
| **Agentic Debugging** | 🔥🔥🔥 Very High (killer feature) | 🛠️🛠️ High | **P1** | Week 3-4 |
| **Knowledge Graph (Neo4j)** | 🔥🔥 High (relationships) | 🛠️🛠️🛠️ Very High | **P1** | Week 5-6 |
| **RAGAS Evaluation** | 🔥 Medium (quality assurance) | 🛠️ Medium | **P2** | Week 7 |
| **Hybrid Search** | 🔥🔥 High (better retrieval) | 🛠️🛠️ High | **P2** | Week 8 |
| **Code Review Agent** | 🔥🔥 High (automation) | 🛠️🛠️ High | **P2** | Week 9-10 |

---

## 📅 Phase 1: Foundation (Weeks 1-2) - HIGHEST IMPACT

### Week 1: Better Chunking & Embeddings

**Goal**: Improve RAG retrieval quality by 25-40%

#### Task 1.1: Tree-sitter AST-based Chunking

**Why**: Research shows +5.5 points on RepoEval. Never breaks code structure.

**Implementation Steps**:

```bash
# 1. Install tree-sitter parsers
pip install tree-sitter tree-sitter-python tree-sitter-javascript tree-sitter-typescript

# 2. Create AST chunker module
```

**Files to Create**:
- `src/rag/ast_chunker.py` - AST-based chunking
- `src/rag/parsers.py` - Multi-language parser setup
- `tests/rag/test_ast_chunking.py` - Quality tests

**Expected Code Structure**:
```python
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

class ASTChunker:
    def __init__(self):
        self.parsers = {
            '.py': Language(tspython.language()),
            '.js': Language(tsjs.language()),
            # ... more languages
        }

    def chunk_code(self, code: str, file_path: str):
        """
        Extract complete functions/classes as chunks.
        Never breaks AST nodes.
        """
        parser = self._get_parser(file_path)
        tree = parser.parse(bytes(code, "utf8"))

        return self._extract_definitions(tree, code)
```

**Expected Impact**:
- ✅ +25-40% retrieval quality
- ✅ Syntactically valid chunks always
- ✅ Better context preservation

**Time**: 3-4 days

---

#### Task 1.2: Upgrade to BGE Embeddings

**Why**: BAAI/bge-base-en-v1.5 is specifically optimized for code retrieval.

**Implementation Steps**:

```python
# Update src/rag/embeddings.py
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        self.model = SentenceTransformer(model_name)

    def embed_code(self, code: str) -> List[float]:
        # Use instruction prefix for better retrieval
        query = f"Represent this code for retrieval: {code}"
        return self.model.encode(query)
```

**Update Config**:
```yaml
# llm-config.yaml
rag:
  embedding_model: BAAI/bge-base-en-v1.5  # Changed from all-MiniLM-L6-v2
```

**Expected Impact**:
- ✅ 15-20% better code retrieval
- ✅ Same 768 dimensions (compatible)
- ✅ Drop-in replacement

**Time**: 1 day

---

### Week 2: Intent-Based Auto-Triggering

**Goal**: Automatically trigger RAG/MAF based on query intent. Eliminate manual `/rag-query` commands.

#### Task 2.1: Semantic Query Router

**Why**: 70% reduction in manual commands. Seamless UX.

**Implementation**:

**File**: `src/rag/intent_router.py`

```python
from sentence_transformers import SentenceTransformer
from typing import Literal
import numpy as np

QueryType = Literal[
    'code_search',      # Vector search
    'graph_query',      # Knowledge graph
    'direct_answer',    # Skip RAG
    'debugging',        # Debug agent
    'code_review'       # Review agent
]

class IntentRouter:
    """
    Routes queries to appropriate handlers based on semantic similarity.
    """

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Define route patterns
        self.routes = {
            'code_search': [
                "where is the authentication code?",
                "find error handling logic",
                "show me API endpoints",
                "locate database queries"
            ],
            'graph_query': [
                "what depends on this module?",
                "what imports this class?",
                "show me the call graph",
                "what functions call this?"
            ],
            'direct_answer': [
                "fix this typo",
                "add a comment here",
                "rename this variable",
                "explain what this does"
            ],
            'debugging': [
                "why is this test failing?",
                "debug this error",
                "fix this bug",
                "what's causing this exception?"
            ],
            'code_review': [
                "review this code",
                "check for issues",
                "any problems with this?",
                "is this code correct?"
            ]
        }

        # Pre-compute route embeddings
        self.route_embeddings = self._compute_route_embeddings()

    def classify(self, query: str) -> QueryType:
        """
        Classify query intent.

        Returns:
            Query type for routing
        """
        query_emb = self.model.encode(query)

        # Compute similarities
        best_route = None
        best_score = -1

        for route_name, route_emb in self.route_embeddings.items():
            similarity = np.dot(query_emb, route_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(route_emb)
            )

            if similarity > best_score:
                best_score = similarity
                best_route = route_name

        # Confidence threshold
        if best_score < 0.7:
            return 'code_search'  # Default to code search

        return best_route

    def should_use_rag(self, query: str) -> bool:
        """
        Determine if RAG should be used.
        """
        intent = self.classify(query)

        # Skip RAG for direct answers
        if intent == 'direct_answer':
            return False

        # Use RAG for everything else
        return True
```

**Integration in Standalone Server**:

```python
# Update src/mcp_server/standalone_server.py

class StandaloneMCPServer:
    def __init__(self, ...):
        # ... existing code ...

        # Add intent router
        self.intent_router = IntentRouter()

    @self.app.post("/query")
    async def query(request: QueryRequest):
        # Auto-determine if RAG should be used
        if request.use_rag is None:  # Allow override
            request.use_rag = self.intent_router.should_use_rag(request.query)

        # Classify intent for specialized handling
        intent = self.intent_router.classify(request.query)

        if intent == 'debugging':
            # Route to debug agent
            return await self.debug_agent.handle(request.query)

        elif intent == 'code_review':
            # Route to review agent
            return await self.review_agent.handle(request.query)

        elif intent == 'graph_query':
            # Use knowledge graph
            results = self.knowledge_graph.query(request.query)
            # ... generate response ...

        else:
            # Standard RAG query
            # ... existing code ...
```

**Expected Impact**:
- ✅ 70% reduction in manual `/rag-query` commands
- ✅ Intelligent routing to specialized agents
- ✅ Seamless user experience
- ✅ Skip RAG for simple edits (faster)

**Time**: 3-4 days

---

## 📅 Phase 2: Agentic Debugging (Weeks 3-4) - KILLER FEATURE

### Week 3: Error Knowledge Base

**Goal**: Learn from every error and build searchable knowledge.

#### Task 3.1: Error Pattern Indexing

**File**: `src/debugging/error_knowledge_base.py`

```python
import chromadb
from datetime import datetime
from typing import Dict, Any, List

class ErrorKnowledgeBase:
    """
    Stores and retrieves historical error patterns.
    """

    def __init__(self, chroma_client):
        self.collection = chroma_client.get_or_create_collection(
            name="error_patterns",
            metadata={"description": "Historical errors and resolutions"}
        )

    def index_error(
        self,
        error_message: str,
        stack_trace: str,
        error_type: str,
        file_path: str,
        resolution: str,
        root_cause: str
    ):
        """
        Index a resolved error for future reference.
        """
        error_id = f"error_{datetime.now().timestamp()}"

        # Combine error info for embedding
        error_text = f"""
        Error: {error_message}
        Type: {error_type}
        Stack Trace: {stack_trace}
        Root Cause: {root_cause}
        """

        self.collection.add(
            documents=[error_text],
            metadatas=[{
                'error_type': error_type,
                'file': file_path,
                'resolution': resolution,
                'root_cause': root_cause,
                'timestamp': datetime.now().isoformat()
            }],
            ids=[error_id]
        )

    def find_similar_errors(
        self,
        error_message: str,
        error_type: str = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical errors.

        Returns:
            List of similar errors with resolutions
        """
        # Build filter
        where = {}
        if error_type:
            where['error_type'] = error_type

        results = self.collection.query(
            query_texts=[error_message],
            n_results=k,
            where=where if where else None
        )

        similar_errors = []
        for i in range(len(results['documents'][0])):
            similar_errors.append({
                'error': results['documents'][0][i],
                'resolution': results['metadatas'][0][i]['resolution'],
                'root_cause': results['metadatas'][0][i]['root_cause'],
                'similarity': 1 - results['distances'][0][i]
            })

        return similar_errors
```

**Integration with Test Runners**:

```python
# src/debugging/test_watcher.py

import subprocess
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TestWatcher(FileSystemEventHandler):
    """
    Watches for test failures and triggers debugging.
    """

    def __init__(self, error_kb, debug_agent):
        self.error_kb = error_kb
        self.debug_agent = debug_agent
        self.test_pattern = re.compile(r'test_.*\.py$')

    def on_modified(self, event):
        if self.test_pattern.search(event.src_path):
            # Run tests
            result = subprocess.run(
                ['pytest', event.src_path, '-v'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                # Test failed - trigger debugging
                self._handle_test_failure(
                    event.src_path,
                    result.stdout + result.stderr
                )

    def _handle_test_failure(self, test_file: str, output: str):
        """
        Automatically debug test failure.
        """
        # Parse error from output
        error_info = self._parse_test_error(output)

        # Search for similar errors
        similar = self.error_kb.find_similar_errors(
            error_info['message'],
            error_info['type']
        )

        # Trigger debug agent
        self.debug_agent.analyze_error(
            error_info,
            similar_errors=similar,
            auto_fix=True  # Automatically suggest fixes
        )
```

**Time**: 4-5 days

---

### Week 4: LangGraph Debug Workflow

**Goal**: Fully autonomous debugging agent.

#### Task 4.1: Debug Agent Implementation

**File**: `src/debugging/debug_agent.py`

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

class DebugState(TypedDict):
    """State for debugging workflow."""
    error_message: str
    stack_trace: str
    error_type: str
    context: Dict[str, Any]
    similar_errors: List[Dict]
    root_cause: str
    suggested_fix: str
    fix_confidence: float

class DebugAgent:
    """
    Autonomous debugging agent using LangGraph.
    """

    def __init__(self, rag_engine, error_kb, llm_provider):
        self.rag = rag_engine
        self.error_kb = error_kb
        self.llm = llm_provider

        # Build workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph debugging workflow.
        """
        workflow = StateGraph(DebugState)

        # Add nodes
        workflow.add_node("gather_context", self._gather_context)
        workflow.add_node("analyze_error", self._analyze_error)
        workflow.add_node("find_similar", self._find_similar_errors)
        workflow.add_node("identify_root_cause", self._identify_root_cause)
        workflow.add_node("generate_fix", self._generate_fix)
        workflow.add_node("verify_fix", self._verify_fix)

        # Define flow
        workflow.set_entry_point("gather_context")
        workflow.add_edge("gather_context", "find_similar")
        workflow.add_edge("find_similar", "analyze_error")
        workflow.add_edge("analyze_error", "identify_root_cause")
        workflow.add_edge("identify_root_cause", "generate_fix")
        workflow.add_edge("generate_fix", "verify_fix")
        workflow.add_edge("verify_fix", END)

        return workflow.compile()

    def _gather_context(self, state: DebugState) -> DebugState:
        """
        Gather all relevant context for debugging.
        """
        # Extract files from stack trace
        files = self._extract_files_from_stack(state['stack_trace'])

        # Retrieve code from RAG
        context = {
            'stack_trace_files': [],
            'related_tests': [],
            'recent_changes': []
        }

        for file_info in files:
            # Get file content
            code = self.rag.retrieve_file(file_info['file'])
            context['stack_trace_files'].append({
                'file': file_info['file'],
                'line': file_info['line'],
                'code': code
            })

            # Find related tests
            tests = self.rag.find_tests_for_file(file_info['file'])
            context['related_tests'].extend(tests)

        state['context'] = context
        return state

    def _find_similar_errors(self, state: DebugState) -> DebugState:
        """
        Find similar historical errors.
        """
        similar = self.error_kb.find_similar_errors(
            state['error_message'],
            state['error_type'],
            k=5
        )

        state['similar_errors'] = similar
        return state

    def _analyze_error(self, state: DebugState) -> DebugState:
        """
        Analyze error with LLM.
        """
        prompt = f"""
        Analyze this error:

        Error: {state['error_message']}
        Type: {state['error_type']}
        Stack Trace: {state['stack_trace']}

        Context:
        {self._format_context(state['context'])}

        Similar Past Errors:
        {self._format_similar_errors(state['similar_errors'])}

        Provide a detailed analysis of what's happening.
        """

        analysis = self.llm.generate(prompt)
        state['analysis'] = analysis
        return state

    def _identify_root_cause(self, state: DebugState) -> DebugState:
        """
        Identify root cause.
        """
        prompt = f"""
        Based on this analysis:
        {state['analysis']}

        Identify the root cause of the error.
        Be specific about:
        1. What went wrong
        2. Why it happened
        3. Where in the code it originated
        """

        root_cause = self.llm.generate(prompt)
        state['root_cause'] = root_cause
        return state

    def _generate_fix(self, state: DebugState) -> DebugState:
        """
        Generate suggested fix.
        """
        prompt = f"""
        Root Cause: {state['root_cause']}

        Generate a fix for this issue.
        Provide:
        1. Exact code changes needed
        2. Explanation of why this fixes it
        3. How to verify the fix works
        4. Confidence level (0-1)

        If similar errors were resolved before, use those solutions as reference:
        {self._format_similar_errors(state['similar_errors'])}
        """

        fix_response = self.llm.generate(prompt)

        # Extract confidence (simple regex for now)
        import re
        confidence_match = re.search(r'confidence:?\s*(0?\.\d+|1\.0)', fix_response.lower())
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5

        state['suggested_fix'] = fix_response
        state['fix_confidence'] = confidence
        return state

    def _verify_fix(self, state: DebugState) -> DebugState:
        """
        Verify the suggested fix.
        """
        # In production: Actually apply fix and run tests
        # For now: Just add verification step
        state['verified'] = state['fix_confidence'] > 0.7
        return state

    async def analyze_error(
        self,
        error_info: Dict[str, Any],
        similar_errors: List[Dict] = None,
        auto_fix: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze an error and suggest fixes.

        Args:
            error_info: Error information
            similar_errors: Optional similar errors
            auto_fix: Whether to automatically apply fix

        Returns:
            Debug results including suggested fix
        """
        initial_state = {
            'error_message': error_info['message'],
            'stack_trace': error_info['stack_trace'],
            'error_type': error_info['type'],
            'context': {},
            'similar_errors': similar_errors or []
        }

        # Run workflow
        result = await self.workflow.ainvoke(initial_state)

        # Index this error for future reference
        if result.get('verified'):
            self.error_kb.index_error(
                error_message=result['error_message'],
                stack_trace=result['stack_trace'],
                error_type=result['error_type'],
                file_path=result['context']['stack_trace_files'][0]['file'],
                resolution=result['suggested_fix'],
                root_cause=result['root_cause']
            )

        return result
```

**Expected Impact**:
- ✅ Automatic error analysis
- ✅ Learning from past errors
- ✅ Suggested fixes with confidence
- ✅ Self-improving over time

**Time**: 5-6 days

---

## 📅 Phase 3: Knowledge Graph (Weeks 5-6) - CODE RELATIONSHIPS

### Week 5-6: Neo4j Integration

**Goal**: Understand code relationships, dependencies, and call graphs.

#### Implementation Overview

**Files to Create**:
- `src/knowledge_graph/neo4j_client.py` - Neo4j connection
- `src/knowledge_graph/code_graph_builder.py` - Build graph from codebase
- `src/knowledge_graph/graph_queries.py` - Pre-built queries

**Example Queries This Enables**:
- "What functions call `authenticate()`?"
- "What would break if I change this class?"
- "Show me all database queries in the auth flow"
- "What tests cover this function?"

**Expected Impact**:
- ✅ Deep code understanding
- ✅ Impact analysis before changes
- ✅ Dependency tracking
- ✅ Better context for LLM

**Time**: 8-10 days

---

## 📅 Phase 4: Evaluation & Hybrid Search (Weeks 7-8)

### Week 7: RAGAS Evaluation

**Goal**: Measure and improve RAG quality continuously.

**Implementation**:
```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)

class RAGEvaluator:
    def evaluate_system(self):
        """
        Continuous evaluation of RAG quality.
        """
        results = evaluate(
            test_dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ]
        )

        # Track over time
        self.metrics_history.append(results)
```

### Week 8: Hybrid Search

**Goal**: Combine vector + BM25 + graph search.

**Expected Impact**:
- ✅ 20-40% better retrieval than vector alone
- ✅ Handles exact matches + semantic matches
- ✅ Leverages graph relationships

---

## 🎯 Auto-Triggering Best Practices

Based on research, here's when to automatically trigger different capabilities:

### Trigger Matrix

| User Query Pattern | Auto-Trigger | Why |
|-------------------|--------------|-----|
| "where is...", "find...", "show me..." | ✅ Vector Search | Exploratory query |
| "what calls...", "what depends..." | ✅ Knowledge Graph | Relationship query |
| "why is this failing...", "debug..." | ✅ Debug Agent | Error/failure query |
| "review this...", "any issues..." | ✅ Review Agent | Code review request |
| "fix typo", "rename X" | ❌ Direct LLM | Simple edit, skip RAG |
| "explain this code" | 🟡 Maybe RAG | If code not in context |

### Implementation

```python
class AutoTriggerSystem:
    """
    Intelligent auto-triggering based on query analysis.
    """

    def analyze_and_route(self, query: str, context: Dict) -> str:
        """
        Analyze query and automatically trigger appropriate system.
        """
        # 1. Check if context is already sufficient
        if self._has_sufficient_context(context):
            return 'direct_llm'  # Skip RAG

        # 2. Classify intent
        intent = self.intent_router.classify(query)

        # 3. Route to appropriate system
        routing = {
            'code_search': 'vector_rag',
            'graph_query': 'knowledge_graph',
            'debugging': 'debug_agent',
            'code_review': 'review_agent',
            'direct_answer': 'direct_llm'
        }

        return routing.get(intent, 'vector_rag')  # Default to RAG

    def _has_sufficient_context(self, context: Dict) -> bool:
        """
        Determine if current context is sufficient.
        """
        # Skip RAG if:
        # - Specific file already open
        # - 10+ files in context
        # - Follow-up question in conversation

        if context.get('files_open', 0) >= 10:
            return True

        if context.get('is_followup', False):
            return True

        return False
```

---

## 📊 Success Metrics

Track these KPIs to measure progress:

| Metric | Baseline | Target | How to Measure |
|--------|----------|--------|----------------|
| **Retrieval Quality** | Current | +30% | RAGAS context_precision |
| **Auto-trigger Accuracy** | N/A | >85% | Manual review |
| **Debug Success Rate** | N/A | >70% | Fix acceptance rate |
| **Query Latency** | Current | <500ms P95 | Prometheus metrics |
| **User Satisfaction** | N/A | >8/10 | User survey |

---

## 🚀 Quick Start: Week 1 Implementation

Want to start NOW? Here's the immediate action plan:

```bash
# 1. Install tree-sitter
pip install tree-sitter tree-sitter-python tree-sitter-javascript

# 2. Create AST chunker
# See implementation above

# 3. Update embeddings to BGE
pip install sentence-transformers

# 4. Test improvement
# Run benchmarks before/after
```

---

## 📝 Implementation Checklist

### Phase 1 (Weeks 1-2)
- [ ] Install tree-sitter parsers
- [ ] Implement ASTChunker class
- [ ] Add multi-language support
- [ ] Create tests for AST chunking
- [ ] Upgrade to BGE embeddings
- [ ] Benchmark quality improvement
- [ ] Implement IntentRouter class
- [ ] Add route definitions
- [ ] Integrate with standalone server
- [ ] Test auto-triggering

### Phase 2 (Weeks 3-4)
- [ ] Create ErrorKnowledgeBase class
- [ ] Implement error indexing
- [ ] Create TestWatcher for auto-debugging
- [ ] Implement DebugAgent with LangGraph
- [ ] Add context gathering
- [ ] Add fix generation
- [ ] Add fix verification
- [ ] Test end-to-end debugging flow

### Phase 3 (Weeks 5-6)
- [ ] Install Neo4j Community Edition
- [ ] Create Neo4j client
- [ ] Implement code graph builder
- [ ] Index codebase into graph
- [ ] Create pre-built queries
- [ ] Integrate with RAG system
- [ ] Test hybrid retrieval

### Phase 4 (Weeks 7-8)
- [ ] Install RAGAS
- [ ] Create evaluation dataset
- [ ] Implement continuous evaluation
- [ ] Add BM25 indexing
- [ ] Implement hybrid search
- [ ] Benchmark improvements
- [ ] Set up monitoring dashboard

---

## 🎯 Priority Recommendation

**Start with Phase 1** - it has the highest ROI:
- Tree-sitter: +25-40% quality for medium effort
- BGE embeddings: +15-20% quality for low effort
- Intent router: Massive UX improvement

**Total time**: 1-2 weeks
**Expected impact**: +40-60% overall improvement

**Should we start implementing Phase 1 now?**
