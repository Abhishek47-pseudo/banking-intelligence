"""
Eval Runner — combines accuracy, hallucination, and latency evaluations.
Usage: python -m backend.eval.eval_runner [--test_set data/eval/ground_truth.json]
"""

import argparse
import asyncio
import json
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
CLIENT_IDS = [f"C{100+i}" for i in range(10)]


async def run_all(test_set_path: str):
    print("=" * 60)
    print("AI BANKING INTELLIGENCE — EVALUATION SUITE")
    print("=" * 60)

    # 1. Accuracy
    print("\n[1/3] ACCURACY EVALUATION")
    if os.path.exists(test_set_path):
        from backend.eval.accuracy import evaluate
        acc_results = await evaluate(test_set_path)
    else:
        print(f"  Ground truth not found at {test_set_path}. Skipping accuracy eval.")
        acc_results = {}

    # 2. Hallucination
    print("\n[2/3] HALLUCINATION DETECTION")
    from backend.eval.hallucination import run_hallucination_eval
    hall_results = await run_hallucination_eval(CLIENT_IDS[:5])

    # 3. Latency
    print("\n[3/3] LATENCY BENCHMARKS")
    from backend.eval.latency import run_latency_eval
    lat_results = await run_latency_eval(CLIENT_IDS[:3])

    # Summary
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  Accuracy P@3:      {acc_results.get('precision_at_3', 'N/A')}")
    print(f"  Accuracy MRR:      {acc_results.get('mrr', 'N/A')}")
    print(f"  Hallucination Rate:{hall_results.get('overall_hallucination_rate', 'N/A')}")
    print(f"  Ingest P95:        {lat_results.get('ingest', {}).get('p95', 'N/A')}ms")
    print(f"  Recommend P95:     {lat_results.get('recommend', {}).get('p95', 'N/A')}ms")

    # Save results
    combined = {
        "accuracy": acc_results,
        "hallucination": hall_results,
        "latency": lat_results,
    }
    os.makedirs("data/eval", exist_ok=True)
    with open("data/eval/eval_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print("\n  Results saved to data/eval/eval_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", default="data/eval/ground_truth.json")
    args = parser.parse_args()
    asyncio.run(run_all(args.test_set))


if __name__ == "__main__":
    main()
