---
description: Display system metrics and performance dashboard
---

# Metrics Dashboard Command

Display comprehensive system metrics and performance statistics.

## Usage

No arguments required.

## Instructions

Shows system health, query performance, cache stats, and recent activity.

## Implementation

```python
import httpx
import json

try:
    # Get all metrics
    response = httpx.get(
        "http://127.0.0.1:8765/metrics",
        timeout=10.0
    )

    if response.status_code == 200:
        metrics = response.json()

        print("=" * 70)
        print("[=] RAG SYSTEM METRICS DASHBOARD".center(70))
        print("=" * 70)
        print()

        # Health Status
        health = metrics.get("health", {})
        if health:
            status = health.get("status", "unknown").upper()
            status_emoji = "[OK]" if status == "HEALTHY" else "[!]"

            print(f"{status_emoji} SYSTEM HEALTH: {status}")

            error_rate = health.get("error_rate", 0)
            memory_mb = health.get("memory_usage_mb", 0)
            latency = health.get("avg_query_latency_ms", 0)

            print(f"   Error Rate: {error_rate:.2%}")
            print(f"   Memory: {memory_mb:.1f} MB")
            print(f"   Avg Latency: {latency:.1f} ms")
            print()

        # Query Performance
        perf = metrics.get("performance", {})
        if perf:
            print("[!] QUERY PERFORMANCE (7 days)")

            total = perf.get("total_queries", 0)
            qpd = perf.get("queries_per_day", 0)
            avg_time = perf.get("avg_execution_time_ms", 0)
            p95_time = perf.get("p95_execution_time_ms", 0)

            print(f"   Total Queries: {total}")
            print(f"   Queries/Day: {qpd:.1f}")
            print(f"   Avg Time: {avg_time:.1f} ms")
            print(f"   P95 Time: {p95_time:.1f} ms")

            feedback = perf.get("avg_feedback_score")
            if feedback is not None:
                print(f"   Avg Feedback: {feedback:.2f}/1.0")

            print()

        # Cache Statistics
        cache = metrics.get("cache", {})
        if cache:
            print("[@] CACHE STATISTICS")

            entries = cache.get("total_entries", 0)
            size_mb = cache.get("total_size_mb", 0)
            avg_access = cache.get("average_access_count", 0)

            print(f"   Total Entries: {entries}")
            print(f"   Total Size: {size_mb:.2f} MB")
            print(f"   Avg Access Count: {avg_access:.1f}")
            print()

        # Get popular queries
        try:
            history_response = httpx.get(
                "http://127.0.0.1:8765/query-history?days=1",
                timeout=5.0
            )

            if history_response.status_code == 200:
                history = history_response.json()
                popular = history.get("popular_queries", [])

                if popular:
                    print("[FIRE] TOP QUERIES (24h)")
                    for i, query_info in enumerate(popular[:5], 1):
                        query = query_info.get("query", "")
                        count = query_info.get("count", 0)

                        if len(query) > 50:
                            query = query[:47] + "..."

                        print(f"   {i}. {query} ({count}x)")

                    print()

        except:
            pass

        print("=" * 70)

        print("\n[i] Refresh with: /rag-metrics")

    else:
        print(f"[X] Server error: {response.status_code}")

except Exception as e:
    print(f"[X] Error: {e}")
    print("Make sure the MCP server is running on port 8000.")
```

Execute this code to display the metrics dashboard.
