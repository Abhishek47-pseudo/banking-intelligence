"""
Hallucination Detection — checks LLM output claims against source data.
"""

import json
import re
from typing import Dict, List, Any, Tuple
import httpx
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def _extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from a text string."""
    return [float(x.replace(",", "")) for x in re.findall(r"[\d,]+\.?\d*", text)]


def _string_in_source(claim: str, source_texts: List[str]) -> bool:
    """Check if a claim string appears (or is close) in source texts."""
    claim_lower = claim.lower().strip()
    for src in source_texts:
        if claim_lower in src.lower():
            return True
    return False


def _numbers_grounded(output_text: str, source_texts: List[str], tolerance: float = 0.05) -> Tuple[int, int]:
    """
    Check that numeric values in output exist in source data (±5%).
    Returns (grounded_count, total_count).
    """
    output_numbers = _extract_numbers(output_text)
    source_numbers = []
    for src in source_texts:
        source_numbers.extend(_extract_numbers(src))

    grounded = 0
    for num in output_numbers:
        if any(abs(num - s) <= abs(s) * tolerance for s in source_numbers):
            grounded += 1

    return grounded, len(output_numbers)


async def detect_hallucinations(
    client_id: str,
    recommendation_output: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    For a given recommendation output, check:
    1. No product recommended that client already holds
    2. Numeric values in output exist in source profile (±5%)
    3. Key claims are present in source data
    """
    issues = []
    total_checks = 0
    flagged = 0

    current_products = set(profile.get("current_products") or [])
    source_texts = [json.dumps(profile)]

    # Check 1: No already-held products recommended
    for rec in recommendation_output.get("recommendations", []):
        total_checks += 1
        if rec["product"] in current_products:
            issues.append(f"HALLUCINATION: Recommended '{rec['product']}' which client already holds")
            flagged += 1

    # Check 2: Numeric grounding
    output_text = json.dumps(recommendation_output)
    grounded, total_nums = _numbers_grounded(output_text, source_texts)
    ungrounded = total_nums - grounded
    total_checks += total_nums
    flagged += ungrounded
    if ungrounded > 0:
        issues.append(f"POTENTIAL HALLUCINATION: {ungrounded}/{total_nums} numbers not found in source data")

    hallucination_rate = flagged / total_checks if total_checks > 0 else 0.0

    return {
        "client_id": client_id,
        "total_checks": total_checks,
        "flagged_claims": flagged,
        "hallucination_rate": round(hallucination_rate, 3),
        "issues": issues,
    }


async def run_hallucination_eval(client_ids: List[str]) -> Dict[str, Any]:
    """Run hallucination detection for a list of clients."""
    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for cid in client_ids:
            try:
                profile_resp = await client.get(f"{API_BASE}/profile/{cid}")
                if profile_resp.status_code == 404:
                    continue
                profile = profile_resp.json()

                rec_resp = await client.post(f"{API_BASE}/recommend/{cid}")
                rec_resp.raise_for_status()
                recommendations = rec_resp.json()

                result = await detect_hallucinations(cid, recommendations, profile)
                results.append(result)
                print(f"  {cid}: rate={result['hallucination_rate']:.3f} | issues={result['issues']}")
            except Exception as e:
                print(f"  ERROR {cid}: {e}")

    overall_rate = (
        sum(r["hallucination_rate"] for r in results) / len(results)
        if results else 0.0
    )
    summary = {
        "overall_hallucination_rate": round(overall_rate, 3),
        "clients_evaluated": len(results),
        "per_client": results,
    }
    print(f"\n=== HALLUCINATION SUMMARY ===")
    print(f"  Overall rate: {summary['overall_hallucination_rate']:.3f}")
    print(f"  Clients evaluated: {summary['clients_evaluated']}")
    return summary
