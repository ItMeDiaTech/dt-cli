"""
Tests for evaluation and hybrid search functionality.
"""

import pytest
from src.evaluation import (
    RAGASEvaluator,
    RAGEvaluation,
    ABTester,
    HybridSearch,
    BM25,
    QueryRewriter,
    SearchResult
)


class TestRAGASEvaluator:
    """Test RAGAS evaluation metrics."""

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = RAGASEvaluator()
        assert evaluator is not None

    def test_context_relevance(self):
        """Test context relevance metric."""
        evaluator = RAGASEvaluator()

        query = "how to authenticate users"
        contexts = [
            "User authentication can be done using OAuth or JWT tokens.",
            "The login function validates credentials and returns a token."
        ]

        score = evaluator._context_relevance(query, contexts)
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have some relevance

    def test_context_relevance_empty(self):
        """Test context relevance with empty contexts."""
        evaluator = RAGASEvaluator()

        score = evaluator._context_relevance("test query", [])
        assert score == 0.0

    def test_answer_faithfulness(self):
        """Test answer faithfulness metric."""
        evaluator = RAGASEvaluator()

        contexts = [
            "The function returns True if authentication succeeds."
        ]
        answer = "The function returns True when authentication is successful."

        score = evaluator._answer_faithfulness(contexts, answer)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be faithful

    def test_answer_faithfulness_unfaithful(self):
        """Test answer faithfulness with unfaithful answer."""
        evaluator = RAGASEvaluator()

        contexts = [
            "The function returns True if authentication succeeds."
        ]
        answer = "The system uses blockchain for payments."  # Completely different

        score = evaluator._answer_faithfulness(contexts, answer)
        assert score < 0.3  # Should be low

    def test_answer_relevance(self):
        """Test answer relevance metric."""
        evaluator = RAGASEvaluator()

        query = "how does authentication work?"
        answer = "Authentication validates user credentials and returns a token."

        score = evaluator._answer_relevance(query, answer)
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should be relevant

    def test_evaluate_complete(self):
        """Test complete evaluation."""
        evaluator = RAGASEvaluator()

        query = "how to authenticate users"
        contexts = [
            "User authentication uses JWT tokens.",
            "The auth function validates credentials."
        ]
        answer = "Authentication validates credentials and uses JWT tokens."

        evaluation = evaluator.evaluate(query, contexts, answer)

        assert isinstance(evaluation, RAGEvaluation)
        assert evaluation.context_relevance >= 0
        assert evaluation.answer_faithfulness >= 0
        assert evaluation.answer_relevance >= 0
        assert evaluation.overall_score >= 0

    def test_evaluate_with_ground_truth(self):
        """Test evaluation with ground truth."""
        evaluator = RAGASEvaluator()

        query = "how does login work?"
        contexts = [
            "Login validates username and password.",
            "Returns JWT token on success."
        ]
        answer = "Login checks credentials and returns JWT."
        ground_truth = "Login validates credentials and returns a JWT token."

        evaluation = evaluator.evaluate(query, contexts, answer, ground_truth)

        assert evaluation.context_precision > 0
        assert evaluation.context_recall > 0

    def test_batch_evaluate(self):
        """Test batch evaluation."""
        evaluator = RAGASEvaluator()

        queries = [
            "how to authenticate",
            "how to authorize"
        ]
        contexts_list = [
            ["Authentication uses tokens"],
            ["Authorization checks permissions"]
        ]
        answers = [
            "Use tokens for auth",
            "Check permissions for authorization"
        ]

        evaluations = evaluator.batch_evaluate(queries, contexts_list, answers)

        assert len(evaluations) == 2
        assert all(isinstance(e, RAGEvaluation) for e in evaluations)

    def test_aggregate_metrics(self):
        """Test metric aggregation."""
        evaluator = RAGASEvaluator()

        # Create some evaluations
        evaluations = [
            RAGEvaluation(
                query="test1",
                retrieved_contexts=["ctx"],
                generated_answer="ans",
                context_relevance=0.8,
                answer_faithfulness=0.7,
                answer_relevance=0.9,
                overall_score=0.8
            ),
            RAGEvaluation(
                query="test2",
                retrieved_contexts=["ctx"],
                generated_answer="ans",
                context_relevance=0.6,
                answer_faithfulness=0.8,
                answer_relevance=0.7,
                overall_score=0.7
            )
        ]

        aggregated = evaluator.aggregate_metrics(evaluations)

        assert 'context_relevance_mean' in aggregated
        assert 'answer_faithfulness_mean' in aggregated
        assert 'overall_score_mean' in aggregated
        assert aggregated['total_evaluations'] == 2

    def test_evaluation_to_dict(self):
        """Test converting evaluation to dict."""
        evaluation = RAGEvaluation(
            query="test query",
            retrieved_contexts=["context 1", "context 2"],
            generated_answer="test answer",
            context_relevance=0.8,
            overall_score=0.75
        )

        result = evaluation.to_dict()

        assert result['query'] == "test query"
        assert 'metrics' in result
        assert result['metrics']['context_relevance'] == 0.8
        assert result['retrieved_contexts_count'] == 2


class TestABTester:
    """Test A/B testing functionality."""

    def test_init(self):
        """Test A/B tester initialization."""
        evaluator = RAGASEvaluator()
        tester = ABTester(evaluator)

        assert tester is not None
        assert tester.evaluator == evaluator

    def test_run_experiment(self):
        """Test running an experiment."""
        evaluator = RAGASEvaluator()
        tester = ABTester(evaluator)

        results = tester.run_experiment(
            "baseline",
            queries=["test query"],
            contexts_list=[["test context"]],
            answers=["test answer"]
        )

        assert results['experiment'] == "baseline"
        assert 'metrics' in results
        assert results['sample_size'] == 1

    def test_compare_experiments(self):
        """Test comparing two experiments."""
        evaluator = RAGASEvaluator()
        tester = ABTester(evaluator)

        # Run two experiments
        tester.run_experiment(
            "baseline",
            queries=["test"],
            contexts_list=[["context"]],
            answers=["answer"]
        )

        tester.run_experiment(
            "improved",
            queries=["test"],
            contexts_list=[["better context"]],
            answers=["better answer"]
        )

        comparison = tester.compare_experiments("baseline", "improved")

        assert comparison['experiment_a'] == "baseline"
        assert comparison['experiment_b'] == "improved"
        assert 'improvements' in comparison

    def test_get_best_experiment(self):
        """Test finding best experiment."""
        evaluator = RAGASEvaluator()
        tester = ABTester(evaluator)

        # Run experiments with different scores
        tester.run_experiment(
            "poor",
            queries=["test"],
            contexts_list=[[]],  # No context = low score
            answers=["test"]
        )

        tester.run_experiment(
            "good",
            queries=["authentication test"],
            contexts_list=[["authentication uses tokens"]],
            answers=["authentication uses tokens"]
        )

        best = tester.get_best_experiment()

        # "good" should score higher due to better alignment
        assert best in ["poor", "good"]


class TestBM25:
    """Test BM25 keyword scoring."""

    def test_init(self):
        """Test BM25 initialization."""
        bm25 = BM25()
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75

    def test_fit(self):
        """Test BM25 fitting."""
        bm25 = BM25()

        documents = [
            "User authentication system",
            "Password validation logic",
            "Token generation for users"
        ]

        bm25.fit(documents)

        assert bm25.num_docs == 3
        assert len(bm25.doc_lengths) == 3
        assert bm25.avg_doc_length > 0
        assert len(bm25.idf) > 0

    def test_score(self):
        """Test BM25 scoring."""
        bm25 = BM25()

        documents = [
            "User authentication with password",
            "Database connection settings",
            "Password hashing algorithm"
        ]

        bm25.fit(documents)

        # Query about authentication
        score_0 = bm25.score("user authentication password", documents[0], 0)
        score_1 = bm25.score("user authentication password", documents[1], 1)
        score_2 = bm25.score("user authentication password", documents[2], 2)

        # First document should score highest (contains all query terms)
        assert score_0 > score_1
        # Second document should score lowest (irrelevant)
        assert score_1 < score_2


class TestQueryRewriter:
    """Test query rewriting."""

    def test_init(self):
        """Test query rewriter initialization."""
        rewriter = QueryRewriter()
        assert rewriter is not None

    def test_expand_query(self):
        """Test query expansion."""
        rewriter = QueryRewriter()

        expanded = rewriter.expand_query("fix the function error")

        assert len(expanded) > 1
        assert "fix the function error" in expanded
        # Should contain variations
        assert any("method" in q for q in expanded) or any("exception" in q for q in expanded)

    def test_rewrite_for_code(self):
        """Test code-specific rewriting."""
        rewriter = QueryRewriter()

        # Question about implementation
        rewritten = rewriter.rewrite_for_code("how does authentication work?")
        assert "implementation" in rewritten.lower()

        # Question about error
        rewritten = rewriter.rewrite_for_code("why is this error happening?")
        assert "exception" in rewritten.lower() or "stack" in rewritten.lower()


class TestHybridSearch:
    """Test hybrid search functionality."""

    def test_init(self):
        """Test hybrid search initialization."""
        search = HybridSearch(semantic_weight=0.7, keyword_weight=0.3)

        assert search.semantic_weight == 0.7
        assert search.keyword_weight == 0.3

    def test_index_documents(self):
        """Test document indexing."""
        search = HybridSearch()

        documents = [
            "User authentication module",
            "Database connection handler",
            "Password hashing utility"
        ]

        search.index_documents(documents)

        assert len(search.documents) == 3
        assert search.bm25.num_docs == 3

    def test_search_keyword_only(self):
        """Test search with keyword scores only."""
        search = HybridSearch(semantic_weight=0.0, keyword_weight=1.0)

        documents = [
            "User authentication with password validation",
            "Database connection pooling",
            "Password encryption algorithm"
        ]

        search.index_documents(documents)

        # Search for authentication
        results = search.search("user authentication password", top_k=3)

        assert len(results) <= 3
        assert isinstance(results[0], SearchResult)
        # First result should be about authentication
        assert "authentication" in results[0].text.lower()

    def test_search_hybrid(self):
        """Test hybrid search."""
        search = HybridSearch(semantic_weight=0.5, keyword_weight=0.5)

        documents = [
            "User authentication system",
            "Login validation logic",
            "Database queries"
        ]

        metadata = [
            {"file": "auth.py"},
            {"file": "login.py"},
            {"file": "db.py"}
        ]

        search.index_documents(documents, metadata)

        # Provide semantic scores (simulated)
        semantic_scores = [0.9, 0.7, 0.1]  # First doc most similar

        results = search.search(
            "authentication",
            semantic_scores=semantic_scores,
            top_k=2
        )

        assert len(results) == 2
        assert results[0].metadata["file"] in ["auth.py", "login.py"]
        assert results[0].combined_score > 0

    def test_search_with_expansion(self):
        """Test search with query expansion."""
        search = HybridSearch()

        documents = [
            "Function implementation details",
            "Method definition and usage",
            "Class structure"
        ]

        search.index_documents(documents)

        # Search for "function" (should also match "method")
        results = search.search("function", use_query_expansion=True, top_k=3)

        assert len(results) > 0

    def test_search_result_to_dict(self):
        """Test search result conversion."""
        result = SearchResult(
            text="test document",
            metadata={"file": "test.py"},
            semantic_score=0.8,
            keyword_score=0.6,
            combined_score=0.7,
            rank=1
        )

        result_dict = result.to_dict()

        assert result_dict['text'] == "test document"
        assert result_dict['scores']['semantic'] == 0.8
        assert result_dict['scores']['keyword'] == 0.6
        assert result_dict['rank'] == 1

    def test_tune_weights(self):
        """Test weight tuning."""
        search = HybridSearch()

        documents = [
            "authentication code",
            "login function",
            "database query",
            "password hash"
        ]

        search.index_documents(documents)

        # Simulated validation data
        queries = ["authentication", "database"]
        ground_truth_indices = [[0, 1], [2]]  # Relevant docs for each query
        semantic_scores_list = [
            [0.9, 0.7, 0.1, 0.2],  # Scores for query 1
            [0.1, 0.2, 0.9, 0.1]   # Scores for query 2
        ]

        best_weights = search.tune_weights(
            queries,
            ground_truth_indices,
            semantic_scores_list
        )

        assert len(best_weights) == 2
        assert 0.0 <= best_weights[0] <= 1.0
        assert 0.0 <= best_weights[1] <= 1.0


class TestIntegration:
    """Integration tests."""

    def test_ragas_with_hybrid_search(self):
        """Test RAGAS evaluation on hybrid search results."""
        evaluator = RAGASEvaluator()
        search = HybridSearch()

        # Index documents
        documents = [
            "User authentication validates credentials and returns JWT token",
            "Database connection uses connection pooling",
            "Password hashing uses bcrypt algorithm"
        ]

        search.index_documents(documents)

        # Perform search
        query = "how does authentication work?"
        semantic_scores = [0.9, 0.2, 0.3]
        results = search.search(query, semantic_scores=semantic_scores, top_k=3)

        # Extract contexts and generate answer (simulated)
        contexts = [r.text for r in results]
        answer = "Authentication validates credentials and returns a JWT token"

        # Evaluate
        evaluation = evaluator.evaluate(query, contexts, answer)

        assert evaluation.overall_score > 0
        assert evaluation.context_relevance > 0

    def test_ab_testing_different_weights(self):
        """Test A/B testing different hybrid search weights."""
        evaluator = RAGASEvaluator()
        tester = ABTester(evaluator)

        documents = ["doc1 about auth", "doc2 about db", "doc3 about auth"]

        # Test semantic-heavy configuration
        search_a = HybridSearch(semantic_weight=0.9, keyword_weight=0.1)
        search_a.index_documents(documents)

        results_a = search_a.search("auth", semantic_scores=[0.9, 0.1, 0.8])
        contexts_a = [r.text for r in results_a]

        tester.run_experiment(
            "semantic_heavy",
            queries=["auth"],
            contexts_list=[contexts_a],
            answers=["auth response"]
        )

        # Test keyword-heavy configuration
        search_b = HybridSearch(semantic_weight=0.1, keyword_weight=0.9)
        search_b.index_documents(documents)

        results_b = search_b.search("auth", semantic_scores=[0.9, 0.1, 0.8])
        contexts_b = [r.text for r in results_b]

        tester.run_experiment(
            "keyword_heavy",
            queries=["auth"],
            contexts_list=[contexts_b],
            answers=["auth response"]
        )

        # Compare
        comparison = tester.compare_experiments("semantic_heavy", "keyword_heavy")

        assert 'improvements' in comparison


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
