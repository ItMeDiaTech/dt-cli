"""
Advanced NLP-based query understanding.

Features:
- Intent classification (search, explain, find examples, debug)
- Entity extraction (class names, function names, concepts)
- Query expansion with synonyms
- Question reformulation
- Context-aware query enhancement
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class QueryIntent:
    """Query intent types."""

    SEARCH = "search"
    EXPLAIN = "explain"
    FIND_EXAMPLES = "find_examples"
    DEBUG = "debug"
    FIND_USES = "find_uses"
    FIND_TESTS = "find_tests"
    FIND_DOCUMENTATION = "find_documentation"
    COMPARE = "compare"
    UNKNOWN = "unknown"


class AdvancedQueryParser:
    """
    Advanced query parser with NLP capabilities.

    MEDIUM PRIORITY FIX: Added input validation for all operations.
    """

    # MEDIUM PRIORITY FIX: Maximum query length to prevent abuse
    MAX_QUERY_LENGTH = 10000

    def __init__(self):
        """Initialize query parser."""
        # Intent patterns
        self.intent_patterns = {
            QueryIntent.EXPLAIN: [
                r'\bhow (does|do|can|to)\b',
                r'\bwhat (is|are|does)\b',
                r'\bwhy\b',
                r'\bexplain\b',
                r'\bdescribe\b'
            ],
            QueryIntent.FIND_EXAMPLES: [
                r'\bexamples?\b',
                r'\bshow me\b',
                r'\bdemo\b',
                r'\bsample\b',
                r'\bhow to use\b'
            ],
            QueryIntent.DEBUG: [
                r'\berror\b',
                r'\bbug\b',
                r'\bissue\b',
                r'\bproblem\b',
                r'\bfail(s|ed|ing)?\b',
                r'\bfix\b',
                r'\btroubleshoot\b'
            ],
            QueryIntent.FIND_USES: [
                r'\bwhere (is|are).*(used|called)\b',
                r'\bfind (all )?uses\b',
                r'\breferences?\b',
                r'\bcallers?\b'
            ],
            QueryIntent.FIND_TESTS: [
                r'\btests?\b',
                r'\btest cases?\b',
                r'\bunittest\b',
                r'\bspec\b'
            ],
            QueryIntent.FIND_DOCUMENTATION: [
                r'\bdocs?\b',
                r'\bdocumentation\b',
                r'\bapi reference\b',
                r'\bguide\b',
                r'\breadme\b'
            ],
            QueryIntent.COMPARE: [
                r'\b(vs|versus|compared? to?)\b',
                r'\bdifference\b',
                r'\balternative\b'
            ]
        }

        # Entity patterns
        self.entity_patterns = {
            'class': r'\b[A-Z][a-zA-Z0-9]*(?:Controller|Service|Manager|Handler|Model|View)?\b',
            'function': r'\b[a-z_][a-z0-9_]*\(\)',
            'variable': r'\b[a-z_][a-z0-9_]*\b',
            'file': r'\b[\w-]+\.(py|js|ts|java|cpp|go|rs)\b'
        }

        # Synonym mappings for query expansion
        self.synonyms = {
            'function': ['method', 'procedure', 'routine', 'def'],
            'class': ['type', 'object', 'interface'],
            'error': ['exception', 'failure', 'issue', 'bug'],
            'test': ['unittest', 'spec', 'test case'],
            'api': ['endpoint', 'route', 'service'],
            'config': ['configuration', 'settings', 'options'],
            'database': ['db', 'data store', 'persistence'],
            'authentication': ['auth', 'login', 'access control'],
            'cache': ['caching', 'cached', 'memoization'],
        }

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse query with advanced NLP understanding.

        MEDIUM PRIORITY FIX: Validate input before processing.

        Args:
            query: User query

        Returns:
            Parsed query information
        """
        # MEDIUM PRIORITY FIX: Validate input
        if not isinstance(query, str):
            logger.error(f"Invalid query type: expected str, got {type(query)}")
            return {
                'error': 'Invalid query type',
                'original_query': '',
                'intent': QueryIntent.UNKNOWN,
                'entities': {},
                'keywords': [],
                'expanded_query': '',
                'reformulated_queries': [],
                'is_question': False,
                'complexity': 'simple'
            }

        # MEDIUM PRIORITY FIX: Check length limit
        if len(query) > self.MAX_QUERY_LENGTH:
            logger.warning(f"Query too long ({len(query)} chars), truncating to {self.MAX_QUERY_LENGTH}")
            query = query[:self.MAX_QUERY_LENGTH]

        # MEDIUM PRIORITY FIX: Handle empty query
        if not query.strip():
            logger.warning("Empty query provided")
            return {
                'original_query': query,
                'intent': QueryIntent.UNKNOWN,
                'entities': {},
                'keywords': [],
                'expanded_query': query,
                'reformulated_queries': [],
                'is_question': False,
                'complexity': 'simple'
            }

        try:
            result = {
                'original_query': query,
                'intent': self._classify_intent(query),
                'entities': self._extract_entities(query),
                'keywords': self._extract_keywords(query),
                'expanded_query': self._expand_query(query),
                'reformulated_queries': self._reformulate_query(query),
                'is_question': self._is_question(query),
                'complexity': self._assess_complexity(query)
            }

            return result

        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return {
                'error': f'Parsing failed: {str(e)}',
                'original_query': query,
                'intent': QueryIntent.UNKNOWN,
                'entities': {},
                'keywords': [],
                'expanded_query': query,
                'reformulated_queries': [],
                'is_question': False,
                'complexity': 'simple'
            }

    def _classify_intent(self, query: str) -> str:
        """
        Classify query intent.

        Args:
            query: User query

        Returns:
            Intent type
        """
        query_lower = query.lower()

        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        return QueryIntent.SEARCH

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract code entities from query.

        Args:
            query: User query

        Returns:
            Extracted entities by type
        """
        entities = {}

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query)
            if matches:
                entities[entity_type] = list(set(matches))

        return entities

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords.

        Args:
            query: User query

        Returns:
            List of keywords
        """
        # Remove stop words
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            'how', 'what', 'where', 'when', 'why', 'which', 'who', 'does', 'do'
        }

        # Tokenize
        words = re.findall(r'\b\w+\b', query.lower())

        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _expand_query(self, query: str) -> str:
        """
        Expand query with synonyms.

        Args:
            query: User query

        Returns:
            Expanded query
        """
        expanded_terms = []

        query_lower = query.lower()

        # Check for synonyms
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                # Add original term and some synonyms
                expanded_terms.extend([term] + synonyms[:2])

        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"

        return query

    def _reformulate_query(self, query: str) -> List[str]:
        """
        Generate alternative query formulations.

        Args:
            query: User query

        Returns:
            List of reformulated queries
        """
        reformulated = []

        query_lower = query.lower()

        # If it's a question, create statement version
        if self._is_question(query):
            # Remove question words
            statement = re.sub(r'^(how|what|why|where|when)\s+(does|do|is|are)\s+', '', query_lower)
            reformulated.append(statement)

        # Add "code" or "implementation" suffix for clarity
        if not any(word in query_lower for word in ['code', 'implementation', 'example']):
            reformulated.append(f"{query} code")
            reformulated.append(f"{query} implementation")

        return reformulated[:3]  # Limit to 3 alternatives

    def _is_question(self, query: str) -> bool:
        """
        Check if query is a question.

        Args:
            query: User query

        Returns:
            True if question
        """
        question_indicators = [
            r'^\s*(how|what|why|where|when|who|which)',
            r'\?$'
        ]

        query_lower = query.lower()

        return any(re.search(pattern, query_lower) for pattern in question_indicators)

    def _assess_complexity(self, query: str) -> str:
        """
        Assess query complexity.

        Args:
            query: User query

        Returns:
            Complexity level ('simple', 'medium', 'complex')
        """
        # Simple metrics for complexity
        word_count = len(query.split())
        has_boolean = bool(re.search(r'\b(and|or|not)\b', query.lower()))
        has_quotes = '"' in query or "'" in query

        if word_count <= 3 and not has_boolean:
            return 'simple'
        elif word_count <= 8 and not has_boolean:
            return 'medium'
        else:
            return 'complex'

    def enhance_query_for_search(self, query: str) -> Dict[str, Any]:
        """
        Enhance query specifically for search.

        Args:
            query: User query

        Returns:
            Enhanced query information
        """
        parsed = self.parse_query(query)

        # Build optimized search queries
        search_queries = [parsed['original_query']]

        # Add expanded query if different
        if parsed['expanded_query'] != parsed['original_query']:
            search_queries.append(parsed['expanded_query'])

        # Add reformulated queries
        search_queries.extend(parsed['reformulated_queries'])

        # Add entity-focused queries
        if parsed['entities']:
            for entity_type, entities in parsed['entities'].items():
                for entity in entities:
                    search_queries.append(f"{entity} {query}")

        return {
            'primary_query': parsed['original_query'],
            'search_queries': list(set(search_queries)),
            'intent': parsed['intent'],
            'entities': parsed['entities'],
            'filters': self._suggest_filters(parsed),
            'boost_terms': self._identify_boost_terms(parsed)
        }

    def _suggest_filters(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest search filters based on parsed query.

        Args:
            parsed_query: Parsed query information

        Returns:
            Suggested filters
        """
        filters = {}

        # Intent-based filters
        if parsed_query['intent'] == QueryIntent.FIND_TESTS:
            filters['file_pattern'] = '*test*.py|*spec*'

        elif parsed_query['intent'] == QueryIntent.FIND_DOCUMENTATION:
            filters['file_pattern'] = '*.md|*.rst|README*'

        elif parsed_query['intent'] == QueryIntent.DEBUG:
            filters['boost_terms'] = ['error', 'exception', 'try', 'catch']

        # Entity-based filters
        if 'class' in parsed_query['entities']:
            filters['prefer_definitions'] = True

        return filters

    def _identify_boost_terms(self, parsed_query: Dict[str, Any]) -> List[str]:
        """
        Identify terms to boost in search.

        Args:
            parsed_query: Parsed query information

        Returns:
            Terms to boost
        """
        boost_terms = []

        # Boost entity names
        for entity_list in parsed_query['entities'].values():
            boost_terms.extend(entity_list)

        # Boost keywords
        boost_terms.extend(parsed_query['keywords'][:5])

        return list(set(boost_terms))


class QueryRecommender:
    """
    Recommends queries based on context and history.
    """

    def __init__(self, query_learning_system=None):
        """
        Initialize query recommender.

        Args:
            query_learning_system: Query learning system instance
        """
        self.query_learning_system = query_learning_system

    def recommend_queries(
        self,
        current_query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recommend queries.

        Args:
            current_query: Current query (optional)
            context: Context information (optional)
            top_k: Number of recommendations

        Returns:
            List of recommended queries
        """
        recommendations = []

        # Based on current query
        if current_query and self.query_learning_system:
            similar = self.query_learning_system.get_similar_queries(current_query, max_results=3)
            for sim_query in similar:
                recommendations.append({
                    'query': sim_query['query'],
                    'reason': f"Similar to current query ({sim_query['similarity']:.0%} match)",
                    'score': sim_query['similarity']
                })

        # Based on popular queries
        if self.query_learning_system:
            popular = self.query_learning_system.get_popular_queries(days=7, top_k=3)
            for pop_query in popular:
                recommendations.append({
                    'query': pop_query['query'],
                    'reason': f"Popular query ({pop_query['count']} uses)",
                    'score': pop_query['frequency']
                })

        # Based on context
        if context:
            context_queries = self._generate_context_queries(context)
            for ctx_query in context_queries:
                recommendations.append(ctx_query)

        # Remove duplicates and sort by score
        seen = set()
        unique_recommendations = []

        for rec in recommendations:
            if rec['query'] not in seen:
                seen.add(rec['query'])
                unique_recommendations.append(rec)

        unique_recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)

        return unique_recommendations[:top_k]

    def _generate_context_queries(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate context-based query recommendations.

        Args:
            context: Context information

        Returns:
            List of context queries
        """
        queries = []

        # File-based context
        if 'current_file' in context:
            file_name = context['current_file']
            queries.append({
                'query': f"related to {file_name}",
                'reason': "Based on current file",
                'score': 0.7
            })

        # Function-based context
        if 'current_function' in context:
            func_name = context['current_function']
            queries.append({
                'query': f"how does {func_name} work",
                'reason': "Based on current function",
                'score': 0.8
            })

        return queries


# Global instances
query_parser = AdvancedQueryParser()
query_recommender = QueryRecommender()
