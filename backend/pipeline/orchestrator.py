"""
Pipeline Orchestrator — runs all 4 agents concurrently with retry logic.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog
from langchain_core.language_models import BaseChatModel

from backend.agents.transaction_agent import run_transaction_agent, TransactionOutput
from backend.agents.crm_agent import run_crm_agent, CRMOutput
from backend.agents.interaction_agent import run_interaction_agent, InteractionOutput
from backend.agents.product_agent import run_product_agent, ProductOutput
from backend.llm.llm_factory import get_llm
from backend.observability.usage_logger import current_agent, get_usage_snapshot, reset_usage

log = structlog.get_logger(__name__)

MAX_RETRIES = 3
AGENT_TIMEOUT = 30.0  # seconds
BACKOFF_BASE = 2.0


@dataclass
class AgentResult:
    agent_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    attempts: int = 0


@dataclass
class PipelineResult:
    client_id: str
    transaction: Optional[TransactionOutput] = None
    crm: Optional[CRMOutput] = None
    interaction: Optional[InteractionOutput] = None
    product: Optional[ProductOutput] = None
    agent_results: List[AgentResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    pipeline_timestamp: Optional[str] = None
    partial_failure: bool = False
    failed_agents: List[str] = field(default_factory=list)


async def _run_with_retry(
    name: str,
    coro_fn,
    *args,
    **kwargs,
) -> AgentResult:
    """Run an agent coroutine with exponential backoff retry and timeout."""
    token = current_agent.set(name)
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.monotonic()
        try:
            log.info("agent_start", agent=name, attempt=attempt)
            output = await asyncio.wait_for(
                coro_fn(*args, **kwargs),
                timeout=AGENT_TIMEOUT,
            )
            latency = (time.monotonic() - start) * 1000
            log.info("agent_success", agent=name, latency_ms=round(latency, 1))
            current_agent.reset(token)
            return AgentResult(
                agent_name=name, success=True, output=output,
                latency_ms=latency, attempts=attempt,
            )
        except asyncio.TimeoutError as e:
            log.warning("agent_timeout", agent=name, attempt=attempt)
            err = f"TimeoutError after {AGENT_TIMEOUT}s"
        except Exception as e:
            log.warning("agent_error", agent=name, attempt=attempt, error=str(e))
            err = str(e)

        if attempt < MAX_RETRIES:
            wait = BACKOFF_BASE ** attempt
            log.info("agent_retry", agent=name, wait_s=wait)
            await asyncio.sleep(wait)

    latency = (time.monotonic() - start) * 1000
    current_agent.reset(token)
    return AgentResult(
        agent_name=name, success=False, error=err,
        latency_ms=latency, attempts=MAX_RETRIES,
    )


async def run_pipeline(client_id: str, llm: Optional[BaseChatModel] = None) -> PipelineResult:
    """
    Async orchestrator: runs all 4 agents concurrently.
    Partial failures are tolerated — pipeline continues with reduced confidence.
    llm defaults to the factory singleton (OpenAI → Groq fallback).
    """
    if llm is None:
        llm = get_llm()
    pipeline_start = time.monotonic()
    reset_usage()
    log.info("pipeline_start", client_id=client_id)

    # Phase 1: Run Transaction + CRM + Interaction concurrently (no inter-dependencies)
    tx_task = _run_with_retry("transaction_agent", run_transaction_agent, client_id, llm)
    crm_task = _run_with_retry("crm_agent", run_crm_agent, client_id, llm)
    int_task = _run_with_retry("interaction_agent", run_interaction_agent, client_id, llm)

    tx_result, crm_result, int_result = await asyncio.gather(
        tx_task, crm_task, int_task
    )

    # Build partial current profile for product agent
    current_profile = {}
    if tx_result.success and tx_result.output:
        tx: TransactionOutput = tx_result.output
        current_profile.update({
            "client_id": client_id,
            "monthly_spend": tx.monthly_spend,
            "international_usage": tx.international_usage,
            "top_categories": tx.top_categories,
        })
    if crm_result.success and crm_result.output:
        crm: CRMOutput = crm_result.output
        current_profile.update({
            "income_band": crm.income_band,
            "current_products": crm.products_held,
        })

    # Phase 2: Product agent (depends on other agents' outputs for similarity)
    prod_result = await _run_with_retry(
        "product_agent", run_product_agent, client_id, current_profile, llm
    )

    all_results = [tx_result, crm_result, int_result, prod_result]
    failed = [r.agent_name for r in all_results if not r.success]
    total_latency = (time.monotonic() - pipeline_start) * 1000

    from datetime import datetime, timezone
    result = PipelineResult(
        client_id=client_id,
        transaction=tx_result.output if tx_result.success else None,
        crm=crm_result.output if crm_result.success else None,
        interaction=int_result.output if int_result.success else None,
        product=prod_result.output if prod_result.success else None,
        agent_results=all_results,
        total_latency_ms=total_latency,
        pipeline_timestamp=datetime.now(timezone.utc).isoformat(),
        partial_failure=len(failed) > 0,
        failed_agents=failed,
    )
    log.info(
        "pipeline_complete",
        client_id=client_id,
        total_ms=round(total_latency, 1),
        failed_agents=failed,
        partial_failure=result.partial_failure,
    )

    # Usage summary (best-effort; token metadata depends on provider responses)
    snap = get_usage_snapshot()
    for agent_name, c in snap.by_agent.items():
        log.info(
            "usage_summary_agent",
            agent=agent_name,
            llm_calls=c.llm_calls,
            tool_calls=c.tool_calls,
            prompt_tokens=c.prompt_tokens,
            completion_tokens=c.completion_tokens,
            total_tokens=c.total_tokens,
            attempted_tokens=c.attempted_tokens,
            rate_limit_refresh_at_utc=c.last_rate_limit_refresh_at,
            rate_limit_wait=c.last_rate_limit_wait,
        )

    for tool_name, c in snap.by_tool.items():
        log.info("usage_summary_tool", tool=tool_name, calls=c.tool_calls)

    return result
