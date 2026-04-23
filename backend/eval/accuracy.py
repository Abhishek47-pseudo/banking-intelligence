"""
Accuracy Evaluation — Precision@3, Recall@3, MRR.
Usage: python -m backend.eval.eval_runner --test_set data/eval/ground_truth.json
"""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Tuple
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


async def evaluate(test_set_path: str) -> Dict[str, float]:
    with open(test_set_path) as f:
        ground_truth = json.load(f)

    precision_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    n = len(ground_truth)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for item in ground_truth:
            cid = item["client_id"]
            expected: List[str] = item["expected_products"]

            # Get recommendations
            try:
                resp = await client.post(f"{API_BASE}/recommend/{cid}")
                resp.raise_for_status()
                data = resp.json()
                predicted = [r["product"] for r in data.get("recommendations", [])][:3]
            except Exception as e:
                print(f"  ERROR for {cid}: {e}")
                n -= 1
                continue

            # Precision@3
            hits = [p for p in predicted if p in expected]
            precision = len(hits) / 3 if predicted else 0.0
            # Recall@3
            recall = len(hits) / len(expected) if expected else 0.0

            # MRR: reciprocal rank of first correct recommendation
            rr = 0.0
            for rank, pred in enumerate(predicted, 1):
                if pred in expected:
                    rr = 1.0 / rank
                    break

            precision_sum += precision
            recall_sum += recall
            mrr_sum += rr

            print(f"  {cid}: P@3={precision:.2f} R@3={recall:.2f} RR={rr:.2f} | Predicted: {predicted}")

    if n == 0:
        return {"precision_at_3": 0, "recall_at_3": 0, "mrr": 0, "n_evaluated": 0}

    results = {
        "precision_at_3": round(precision_sum / n, 3),
        "recall_at_3": round(recall_sum / n, 3),
        "mrr": round(mrr_sum / n, 3),
        "n_evaluated": n,
    }
    print("\n=== ACCURACY RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", default="data/eval/ground_truth.json")
    args = parser.parse_args()
    asyncio.run(evaluate(args.test_set))


if __name__ == "__main__":
    main()
