"""
Result explainability - explain why results were returned.

Provides:
- Relevance score breakdown
- Matched terms highlighting
- Context snippets
- Similarity explanations
"""

from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class ResultExplainer:
    """
    Explains why search results were returned.
    """

    def __init__(self):
        """Initialize result explainer."""
        pass

    def explain_result(
        self,
        query: str,
        result: Dict[str, Any],
        score_breakdown: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single result.

        Args:
            query: Search query
            result: Search result
            score_breakdown: Optional score breakdown

        Returns:
            Explanation dictionary
        """
        explanation = {
            'result_id': result.get('id'),
            'relevance_score': result.get('score', 0),
            'score_breakdown': score_breakdown or {},
            'matched_terms': self._find_matched_terms(query, result),
            'context_snippets': self._extract_context_snippets(query, result),
            'similarity_explanation': self._explain_similarity(query, result),
            'ranking_factors': self._explain_ranking_factors(result, score_breakdown)
        }

        return explanation

    def explain_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate explanations for all results.

        Args:
            query: Search query
            results: List of search results
            pipeline_info: Optional pipeline information

        Returns:
            Comprehensive explanations
        """
        explanations = {
            'query': query,
            'query_analysis': self._analyze_query(query),
            'total_results': len(results),
            'result_explanations': [],
            'pipeline_info': pipeline_info or {}
        }

        for idx, result in enumerate(results):
            result_explanation = self.explain_result(query, result)
            result_explanation['rank'] = idx + 1
            explanations['result_explanations'].append(result_explanation)

        # Add comparative insights
        explanations['comparative_insights'] = self._generate_comparative_insights(
            results
        )

        return explanations

    def _find_matched_terms(
        self,
        query: str,
        result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Find matched terms between query and result.

        Args:
            query: Search query
            result: Search result

        Returns:
            List of matched terms with positions
        """
        content = result.get('content', '')
        metadata = result.get('metadata', {})

        # Tokenize query
        query_terms = set(self._tokenize(query.lower()))

        matched_terms = []

        # Find exact matches
        for term in query_terms:
            if term in content.lower():
                # Find positions
                positions = [
                    m.start() for m in re.finditer(
                        re.escape(term),
                        content.lower()
                    )
                ]

                if positions:
                    matched_terms.append({
                        'term': term,
                        'type': 'exact',
                        'count': len(positions),
                        'positions': positions[:5]  # Limit to first 5
                    })

        # Check metadata matches
        file_path = metadata.get('file_path', '')
        if any(term in file_path.lower() for term in query_terms):
            matched_terms.append({
                'term': 'filename',
                'type': 'metadata',
                'matched_in': 'file_path'
            })

        return matched_terms

    def _extract_context_snippets(
        self,
        query: str,
        result: Dict[str, Any],
        snippet_length: int = 150
    ) -> List[Dict[str, str]]:
        """
        Extract context snippets around matched terms.

        Args:
            query: Search query
            result: Search result
            snippet_length: Length of snippet

        Returns:
            List of context snippets
        """
        content = result.get('content', '')
        query_terms = set(self._tokenize(query.lower()))

        snippets = []

        for term in query_terms:
            # Find term in content
            pattern = re.compile(re.escape(term), re.IGNORECASE)

            for match in pattern.finditer(content):
                start = max(0, match.start() - snippet_length // 2)
                end = min(len(content), match.end() + snippet_length // 2)

                snippet = content[start:end]

                # Highlight matched term
                highlighted = pattern.sub(f'**{match.group()}**', snippet)

                snippets.append({
                    'term': term,
                    'snippet': highlighted,
                    'position': match.start()
                })

                # Limit snippets per term
                if len(snippets) >= 3:
                    break

        return snippets[:5]  # Return max 5 snippets

    def _explain_similarity(
        self,
        query: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain semantic similarity.

        Args:
            query: Search query
            result: Search result

        Returns:
            Similarity explanation
        """
        distance = result.get('distance', 0)
        similarity = 1 - distance if distance else result.get('score', 0)

        # Categorize similarity
        if similarity >= 0.9:
            category = 'very_high'
            description = 'Very high semantic similarity'
        elif similarity >= 0.7:
            category = 'high'
            description = 'High semantic similarity'
        elif similarity >= 0.5:
            category = 'moderate'
            description = 'Moderate semantic similarity'
        elif similarity >= 0.3:
            category = 'low'
            description = 'Low semantic similarity'
        else:
            category = 'very_low'
            description = 'Very low semantic similarity'

        return {
            'similarity_score': similarity,
            'distance': distance,
            'category': category,
            'description': description
        }

    def _explain_ranking_factors(
        self,
        result: Dict[str, Any],
        score_breakdown: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Explain factors that contributed to ranking.

        Args:
            result: Search result
            score_breakdown: Optional score breakdown

        Returns:
            List of ranking factors
        """
        factors = []

        if score_breakdown:
            # Add factors from breakdown
            for factor, score in sorted(
                score_breakdown.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                factors.append({
                    'factor': factor,
                    'score': score,
                    'contribution': self._describe_contribution(score)
                })

        # Add metadata factors
        metadata = result.get('metadata', {})

        if 'file_path' in metadata:
            factors.append({
                'factor': 'file_location',
                'value': metadata['file_path'],
                'contribution': 'Provides context about code location'
            })

        if 'function_name' in metadata:
            factors.append({
                'factor': 'function_name',
                'value': metadata['function_name'],
                'contribution': 'Specific function reference'
            })

        return factors

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query characteristics.

        Args:
            query: Search query

        Returns:
            Query analysis
        """
        terms = self._tokenize(query.lower())

        return {
            'length': len(query),
            'term_count': len(terms),
            'unique_terms': len(set(terms)),
            'avg_term_length': sum(len(t) for t in terms) / max(len(terms), 1),
            'contains_code': self._contains_code_pattern(query),
            'query_type': self._classify_query_type(query)
        }

    def _generate_comparative_insights(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate insights by comparing results.

        Args:
            results: List of results

        Returns:
            Comparative insights
        """
        if not results:
            return {}

        scores = [r.get('score', 0) for r in results]

        insights = {
            'score_range': {
                'min': min(scores) if scores else 0,
                'max': max(scores) if scores else 0,
                'avg': sum(scores) / len(scores) if scores else 0
            },
            'score_distribution': self._categorize_scores(scores),
            'top_result_advantage': self._calculate_top_advantage(scores)
        }

        # File distribution
        files = [r.get('metadata', {}).get('file_path', '') for r in results]
        unique_files = len(set(files))

        insights['file_distribution'] = {
            'total_files': unique_files,
            'results_per_file': len(results) / max(unique_files, 1)
        }

        return insights

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            List of terms
        """
        # Simple tokenization
        return re.findall(r'\w+', text.lower())

    def _contains_code_pattern(self, query: str) -> bool:
        """
        Check if query contains code patterns.

        Args:
            query: Search query

        Returns:
            True if contains code patterns
        """
        code_patterns = [
            r'def\s+\w+',  # Python function
            r'class\s+\w+',  # Class definition
            r'\w+\(',  # Function call
            r'import\s+\w+',  # Import statement
            r'\w+\.\w+',  # Dot notation
        ]

        return any(re.search(pattern, query) for pattern in code_patterns)

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type.

        Args:
            query: Search query

        Returns:
            Query type
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ['how', 'what', 'why', 'when', 'where']):
            return 'question'
        elif self._contains_code_pattern(query):
            return 'code_search'
        elif any(word in query_lower for word in ['error', 'bug', 'fix', 'issue']):
            return 'troubleshooting'
        elif any(word in query_lower for word in ['example', 'usage', 'implement']):
            return 'how_to'
        else:
            return 'general'

    def _describe_contribution(self, score: float) -> str:
        """
        Describe score contribution.

        Args:
            score: Contribution score

        Returns:
            Description
        """
        if score >= 0.8:
            return 'Very strong contribution'
        elif score >= 0.6:
            return 'Strong contribution'
        elif score >= 0.4:
            return 'Moderate contribution'
        elif score >= 0.2:
            return 'Weak contribution'
        else:
            return 'Minimal contribution'

    def _categorize_scores(self, scores: List[float]) -> Dict[str, int]:
        """
        Categorize score distribution.

        Args:
            scores: List of scores

        Returns:
            Distribution counts
        """
        distribution = {
            'very_high': 0,
            'high': 0,
            'moderate': 0,
            'low': 0
        }

        for score in scores:
            if score >= 0.8:
                distribution['very_high'] += 1
            elif score >= 0.6:
                distribution['high'] += 1
            elif score >= 0.4:
                distribution['moderate'] += 1
            else:
                distribution['low'] += 1

        return distribution

    def _calculate_top_advantage(self, scores: List[float]) -> Optional[float]:
        """
        Calculate advantage of top result over second.

        Args:
            scores: List of scores

        Returns:
            Advantage percentage or None
        """
        if len(scores) < 2:
            return None

        top_score = scores[0]
        second_score = scores[1]

        if second_score == 0:
            return None

        advantage = ((top_score - second_score) / second_score) * 100
        return round(advantage, 2)


# Global instance
result_explainer = ResultExplainer()
