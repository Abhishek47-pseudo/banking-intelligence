from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

log = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_candidate_blocks(text: str) -> list[str]:
    """
    Return candidate JSON-like substrings ordered by likelihood.
    Keeps logic lightweight and robust to typical LLM formatting noise.
    """
    candidates: list[str] = []
    if not text:
        return candidates

    # 1) Prefer fenced code blocks first (```json ... ```).
    for m in _FENCE_RE.finditer(text):
        block = (m.group(1) or "").strip()
        if block:
            candidates.append(block)

    # 2) Then attempt to find the largest {...} block (greedy).
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0).strip())

    # 3) And the largest [...] block (sometimes outputs are arrays).
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        candidates.append(m.group(0).strip())

    # 4) Finally, the whole string.
    candidates.append(text.strip())

    # Deduplicate while preserving order.
    seen = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def safe_json_loads(
    raw: Any,
    *,
    default: Optional[Any] = None,
    expect: type | tuple[type, ...] = (dict, list),
    logger: logging.Logger | None = None,
    context: str = "",
) -> Any:
    """
    Best-effort JSON parsing for LLM outputs.

    - Handles extra prose around JSON.
    - Handles ```json fenced blocks.
    - Tries multiple candidate substrings.
    - Returns `default` on failure.
    """
    lg = logger or log
    if raw is None:
        return default

    if isinstance(raw, (dict, list)) and isinstance(raw, expect):
        return raw

    text = raw if isinstance(raw, str) else str(raw)
    for candidate in _extract_candidate_blocks(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, expect):
                return parsed
        except Exception:
            continue

    if context:
        lg.warning("safe_json_parse_failed", context=context, raw_preview=text[:200])
    else:
        lg.warning("safe_json_parse_failed", raw_preview=text[:200])
    return default

