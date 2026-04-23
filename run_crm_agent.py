import argparse
import asyncio
import json
import re
from datetime import timedelta

from backend.agents.crm_agent import run_crm_agent
from backend.observability.usage_logger import append_usage_history_jsonl


def _parse_retry_after_seconds(msg: str) -> float | None:
    """
    Parse Groq-style: "Please try again in 1m6.528s."
    Returns seconds (float) or None.
    """
    m = re.search(r"try again in\s+(\d+)m(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = float(m.group(2))
    delta = timedelta(minutes=minutes, seconds=seconds)
    return max(0.0, delta.total_seconds())


async def _amain() -> int:
    parser = argparse.ArgumentParser(description="Run CRM agent for a client_id.")
    parser.add_argument("--client-id", required=True, help="Client ID to load from CRM mock data")
    parser.add_argument(
        "--save-usage",
        action="store_true",
        help="Append token/tool usage snapshot to data/usage/usage_history.jsonl",
    )
    parser.add_argument("--run-name", default="manual_crm_agent", help="Run label stored in usage history")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries on rate-limit errors")
    args = parser.parse_args()

    attempt = 0
    while True:
        try:
            result = await run_crm_agent(args.client_id)
            break
        except Exception as e:
            attempt += 1
            wait_s = _parse_retry_after_seconds(str(e))
            if wait_s is None or attempt > args.max_retries:
                raise
            # Add small cushion so we don't re-hit the limit.
            wait_s = min(wait_s + 2.0, 300.0)
            print(f"\nRate-limited. Waiting {wait_s:.1f}s then retrying ({attempt}/{args.max_retries})...")
            await asyncio.sleep(wait_s)

    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

    if args.save_usage:
        path = append_usage_history_jsonl(run_name=args.run_name, agent="crm_agent")
        print(f"\nSaved usage snapshot to: {path}")

    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())

