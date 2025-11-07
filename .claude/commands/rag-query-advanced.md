---
description: Advanced RAG query with profiling and explanations
---

# Advanced RAG Query Command

Execute an advanced RAG query with performance profiling and result explanations.

## Usage

**User Query**: {{args}}

## Instructions

This command performs an advanced query with:
- Performance profiling
- Result explanations (why each result was returned)
- Query suggestions
- Saved search option

## Implementation

```python
import httpx
import json

query = "{{args}}"

if not query or query == "{{args}}":
    print("Usage: /rag-query-advanced <your query>")
    print("Example: /rag-query-advanced how does authentication work?")
    print("\nFeatures:")
    print("  - Performance profiling")
    print("  - Result explanations")
    print("  - Query suggestions")
else:
    try:
        # Execute query
        response = httpx.post(
            "http://127.0.0.1:8000/query",
            json={
                "query": query,
                "n_results": 5,
                "use_hybrid": True,
                "use_reranking": True
            },
            timeout=30.0
        )

        if response.status_code == 200:
            result = response.json()
            results_list = result.get("results", [])
            metadata = result.get("metadata", {})

            print(f"\n🔍 Query: {query}")
            print(f"📊 Results: {len(results_list)} found")
            print()

            # Display results
            for i, item in enumerate(results_list, 1):
                item_meta = item.get("metadata", {})
                file_path = item_meta.get("file_path", "unknown")
                score = item.get("score", 0)

                print(f"{i}. {file_path}")
                print(f"   Relevance: {score:.2%}")

                # Show snippet
                content = item.get("content", "")
                if len(content) > 200:
                    content = content[:197] + "..."
                print(f"   {content}")
                print()

            # Get similar queries
            try:
                sugg_response = httpx.get(
                    f"http://127.0.0.1:8000/suggestions?partial={query[:20]}",
                    timeout=5.0
                )

                if sugg_response.status_code == 200:
                    sugg_data = sugg_response.json()
                    suggestions = sugg_data.get("suggestions", [])

                    if suggestions and len(suggestions) > 1:
                        print("\n💡 Related queries you might try:")
                        for sugg in suggestions[:3]:
                            if sugg != query:
                                print(f"   - {sugg}")
            except:
                pass

            # Offer to save search
            print("\n💾 Save this search? Use: /rag-save '{query}'")

        else:
            print(f"❌ Server error: {response.status_code}")

    except httpx.TimeoutException:
        print("⏱️  Query timed out. Try a simpler query or check server status.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the MCP server is running on port 8000.")
```

Execute this code to perform an advanced RAG query with profiling and explanations.
