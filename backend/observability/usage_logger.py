"""
LangChain callback handler to log:
- LLM calls + token usage (when available)
- tool invocations (local tools are 0-token, but we still count calls)
- rate-limit refresh times (parses Groq "Please try again in ...")

This is intentionally lightweight: it works for OpenAI + Groq via LangChain
and degrades gracefully when providers don't return token metadata.
"""

from __future__ import annotations

import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

log = structlog.get_logger(__name__)


current_agent: ContextVar[Optional[str]] = ContextVar("current_agent", default=None)


def _parse_retry_after(text: str) -> Optional[Tuple[timedelta, str]]:
    """
    Parse strings like:
      "Please try again in 12m48.96s."
      "Please try again in 7m23.232s."
    Returns (delta, human_string) or None.
    """
    m = re.search(r"try again in\s+(\d+)m(\d+(?:\.\d+)?)s", text, re.IGNORECASE)
    if not m:
        return None
    minutes = int(m.group(1))
    seconds = float(m.group(2))
    delta = timedelta(minutes=minutes, seconds=seconds)
    return delta, f"{minutes}m{seconds:.3f}s"


def _extract_usage(llm_output: Any) -> Dict[str, int]:
    """
    Best-effort extraction across providers.
    Returns dict with any of: prompt_tokens, completion_tokens, total_tokens.
    """
    usage: Dict[str, int] = {}

    # LLMResult.llm_output often holds provider metadata.
    if isinstance(llm_output, dict):
        # OpenAI-like
        u = llm_output.get("token_usage") or llm_output.get("usage") or llm_output.get("usage_metadata")
        if isinstance(u, dict):
            for k_src, k_dst in (
                ("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
                ("input_tokens", "prompt_tokens"),
                ("output_tokens", "completion_tokens"),
            ):
                v = u.get(k_src)
                if isinstance(v, int):
                    usage[k_dst] = v

    # If we got prompt+completion but not total, derive.
    if "total_tokens" not in usage and ("prompt_tokens" in usage or "completion_tokens" in usage):
        usage["total_tokens"] = int(usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))

    return usage


@dataclass
class _Counters:
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    attempted_tokens: int = 0  # e.g., provider rejected but reported "Requested X"
    last_rate_limit_refresh_at: Optional[str] = None
    last_rate_limit_wait: Optional[str] = None


@dataclass
class UsageSnapshot:
    by_agent: Dict[str, _Counters] = field(default_factory=dict)
    by_tool: Dict[str, _Counters] = field(default_factory=dict)

    def _agent(self, agent: str) -> _Counters:
        if agent not in self.by_agent:
            self.by_agent[agent] = _Counters()
        return self.by_agent[agent]

    def _tool(self, tool: str) -> _Counters:
        if tool not in self.by_tool:
            self.by_tool[tool] = _Counters()
        return self.by_tool[tool]


_snapshot: UsageSnapshot = UsageSnapshot()
_callback_singleton: Optional["UsageLoggerCallback"] = None


def reset_usage() -> None:
    global _snapshot
    _snapshot = UsageSnapshot()


def get_usage_snapshot() -> UsageSnapshot:
    return _snapshot


def record_tool_steps(agent: str, intermediate_steps: Any) -> None:
    """
    Record tool usage from AgentExecutor results.

    `intermediate_steps` is typically a list of (AgentAction, observation).
    We count tool names. Tokens are 0 here because tools are local Python calls.
    """
    try:
        if not intermediate_steps:
            return
        c_agent = _snapshot._agent(agent)
        for step in intermediate_steps:
            # step can be tuple(Action, Observation)
            action = step[0] if isinstance(step, (tuple, list)) and step else None
            tool_name = getattr(action, "tool", None) or getattr(action, "tool_name", None)
            if not tool_name:
                continue
            c_agent.tool_calls += 1
            _snapshot._tool(str(tool_name)).tool_calls += 1
    except Exception:
        # Never break pipeline just because of logging.
        return


def get_usage_callbacks() -> list["UsageLoggerCallback"]:
    """
    Return a singleton callback instance wrapped in a list, suitable for passing
    to both LLM constructors and AgentExecutors.
    """
    global _callback_singleton
    if _callback_singleton is None:
        _callback_singleton = UsageLoggerCallback()
    return [_callback_singleton]


class UsageLoggerCallback(BaseCallbackHandler):
    """
    Logs token usage when providers return it, plus tool usage counts.
    Tagging is based on `current_agent` contextvar set by the orchestrator.
    """

    def on_llm_start(self, serialized: Dict[str, Any], prompts: Any, **kwargs: Any) -> Any:
        agent = current_agent.get() or "unknown_agent"
        _snapshot._agent(agent).llm_calls += 1
        return None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        agent = current_agent.get() or "unknown_agent"
        usage = _extract_usage(getattr(response, "llm_output", None))

        c = _snapshot._agent(agent)
        c.prompt_tokens += int(usage.get("prompt_tokens", 0))
        c.completion_tokens += int(usage.get("completion_tokens", 0))
        c.total_tokens += int(usage.get("total_tokens", 0))

        # Optional: emit per-call debug line if token data exists.
        if usage:
            log.info("llm_usage", agent=agent, **usage)
        return None

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        agent = current_agent.get() or "unknown_agent"
        msg = str(error)

        # Groq rate-limit errors include "Requested N" tokens. Capture this as attempted.
        m_req = re.search(r"Requested\s+(\d+)", msg)
        if m_req:
            _snapshot._agent(agent).attempted_tokens += int(m_req.group(1))

        parsed = _parse_retry_after(msg)
        if parsed:
            delta, wait_str = parsed
            refresh_at = (datetime.now(timezone.utc) + delta).isoformat()
            c = _snapshot._agent(agent)
            c.last_rate_limit_refresh_at = refresh_at
            c.last_rate_limit_wait = wait_str
            log.warning(
                "llm_rate_limited",
                agent=agent,
                wait=wait_str,
                refresh_at_utc=refresh_at,
            )
        return None

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        agent = current_agent.get() or "unknown_agent"
        tool_name = (serialized or {}).get("name") or (serialized or {}).get("id") or "unknown_tool"

        _snapshot._agent(agent).tool_calls += 1
        _snapshot._tool(tool_name).tool_calls += 1
        return None

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        return None

