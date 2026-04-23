"""
Latency Evaluation — measures P50/P95/P99 for pipeline and recommendation endpoints.
"""

import asyncio
import time
from typing import List, Dict, Any
import httpx
import numpy as np
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TARGET_INGEST_MS = 10_000   # 10s SLA
TARGET_RECOMMEND_MS = 3_000  # 3s SLA


async def measure_latency(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs,
) -> float:
    """Returns latency in milliseconds."""
    start = time.monotonic()
    resp = await getattr(client, method)(url, **kwargs)
    return (time.monotonic() - start) * 1000


def percentiles(latencies: List[float]) -> Dict[str, float]:
    a = np.array(latencies)
    return {
        "p50": round(float(np.percentile(a, 50)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
        "p99": round(float(np.percentile(a, 99)), 1),
        "mean": round(float(np.mean(a)), 1),
        "min": round(float(np.min(a)), 1),
        "max": round(float(np.max(a)), 1),
    }


async def run_latency_eval(client_ids: List[str]) -> Dict[str, Any]:
    ingest_latencies: List[float] = []
    recommend_latencies: List[float] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        print("=== LATENCY EVALUATION ===")
        for cid in client_ids:
            # Ingest
            try:
                lat = await measure_latency(client, "post", f"{API_BASE}/ingest/{cid}")
                ingest_latencies.append(lat)
                sla_ok = "✓" if lat < TARGET_INGEST_MS else "✗ OVER SLA"
                print(f"  ingest/{cid}: {lat:.0f}ms {sla_ok}")
            except Exception as e:
                print(f"  ingest/{cid}: ERROR {e}")

            # Recommend
            try:
                lat = await measure_latency(client, "post", f"{API_BASE}/recommend/{cid}")
                recommend_latencies.append(lat)
                sla_ok = "✓" if lat < TARGET_RECOMMEND_MS else "✗ OVER SLA"
                print(f"  recommend/{cid}: {lat:.0f}ms {sla_ok}")
            except Exception as e:
                print(f"  recommend/{cid}: ERROR {e}")

    results = {
        "ingest": {
            **percentiles(ingest_latencies),
            "sla_ms": TARGET_INGEST_MS,
            "sla_violations": sum(1 for l in ingest_latencies if l > TARGET_INGEST_MS),
        } if ingest_latencies else {},
        "recommend": {
            **percentiles(recommend_latencies),
            "sla_ms": TARGET_RECOMMEND_MS,
            "sla_violations": sum(1 for l in recommend_latencies if l > TARGET_RECOMMEND_MS),
        } if recommend_latencies else {},
    }

    print("\n=== INGEST LATENCY ===")
    for k, v in results["ingest"].items():
        print(f"  {k}: {v}")
    print("\n=== RECOMMEND LATENCY ===")
    for k, v in results["recommend"].items():
        print(f"  {k}: {v}")

    return results


async def main():
    client_ids = [f"C{100+i}" for i in range(5)]
    await run_latency_eval(client_ids)


if __name__ == "__main__":
    asyncio.run(main())
