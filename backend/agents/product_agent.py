"""
Product Agent — LangChain Tool-Calling Agent.
Collaborative filtering: finds similar clients and detects product gaps.
"""

import os, json
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.language_models import BaseChatModel
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.llm.llm_factory import get_llm
from backend.observability.usage_logger import get_usage_callbacks, record_tool_steps
from backend.utils.safe_json import safe_json_loads


# ── Output Schemas ────────────────────────────────────────────────────────────

class ProductGap(BaseModel):
    product: str
    adoption_rate_among_similar: float
    relevance_score: float
    reason: str


class ProductOutput(BaseModel):
    product_gaps: List[Dict[str, Any]]
    similar_client_count: int
    confidence_score: float


ADOPTION_THRESHOLD = 0.40  # flag if > 40% of similar clients hold a product


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def load_all_profiles(limit: int = 100) -> str:
    """Load all enriched client profiles from SQL store for similarity comparison."""
    # Lazy import to avoid circular dependency
    import asyncio
    from backend.storage import sql_store

    async def _fetch():
        profiles = await sql_store.list_all_profiles()
        return [
            {
                "client_id": p.client_id,
                "income_band": p.income_band,
                "age_band": p.age_band,
                "monthly_spend": p.monthly_spend or 0,
                "international_usage": p.international_usage or False,
                "churn_risk": p.churn_risk or False,
                "current_products": p.current_products or [],
                "spend_trend": p.spend_trend,
                "top_categories": p.top_categories or [],
            }
            for p in profiles[:limit]
        ]

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _fetch())
                data = future.result(timeout=10)
        else:
            data = loop.run_until_complete(_fetch())
    except Exception:
        data = []

    return json.dumps(data)


@tool
def find_similar_clients(current_profile_json: str, all_profiles_json: str) -> str:
    """
    Find top-10 most similar clients via cosine similarity on feature vectors.
    Features: income_band (encoded), monthly_spend, international_usage (bool→float).
    """
    current = json.loads(current_profile_json)
    all_profiles = json.loads(all_profiles_json)

    if not all_profiles:
        return json.dumps({"similar_clients": [], "count": 0})

    # Encode income band
    band_enc = {"low": 0, "mid": 1, "high": 2, "ultra-high": 3}

    def encode(p):
        return np.array([
            band_enc.get(p.get("income_band", "mid"), 1),
            min(float(p.get("monthly_spend", 0)) / 100000, 10.0),
            1.0 if p.get("international_usage") else 0.0,
        ], dtype=float)

    current_id = current.get("client_id", "")
    current_vec = encode(current)
    current_norm = np.linalg.norm(current_vec)

    similarities = []
    for p in all_profiles:
        if p.get("client_id") == current_id:
            continue
        vec = encode(p)
        norm = np.linalg.norm(vec)
        if current_norm > 0 and norm > 0:
            sim = float(np.dot(current_vec, vec) / (current_norm * norm))
        else:
            sim = 0.0
        similarities.append({"profile": p, "similarity": sim})

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top10 = similarities[:10]

    return json.dumps({
        "similar_clients": [x["profile"] for x in top10],
        "similarities": [x["similarity"] for x in top10],
        "count": len(top10),
    })


@tool
def detect_product_gaps(current_profile_json: str, similar_result_json: str) -> str:
    """
    Compare current client's products against similar clients.
    Flag products owned by > 40% of similar clients that this client lacks.
    """
    current = json.loads(current_profile_json)
    similar_data = json.loads(similar_result_json)

    current_products = set(current.get("current_products", []) or [])
    similar_clients = similar_data.get("similar_clients", [])
    n = len(similar_clients)

    if n == 0:
        return json.dumps({"gaps": [], "similar_count": 0})

    # Count product adoption across similar clients
    product_counts: Dict[str, int] = {}
    for sc in similar_clients:
        for prod in sc.get("current_products", []) or []:
            product_counts[prod] = product_counts.get(prod, 0) + 1

    gaps = []
    for prod, count in product_counts.items():
        adoption_rate = count / n
        if adoption_rate > ADOPTION_THRESHOLD and prod not in current_products:
            gaps.append({
                "product": prod,
                "adoption_rate_among_similar": round(adoption_rate, 3),
            })

    return json.dumps({"gaps": gaps, "similar_count": n})


@tool
def score_gaps(gaps_json: str, current_profile_json: str) -> str:
    """Rank detected gaps by spend relevance and similarity score."""
    gaps_data = json.loads(gaps_json)
    current = json.loads(current_profile_json)
    gaps = gaps_data.get("gaps", [])

    top_cats = set(current.get("top_categories", []) or [])
    intl = current.get("international_usage", False)
    income = current.get("income_band", "mid")

    # Simple relevance heuristics
    relevance_hints = {
        "forex_card": 0.3 if intl else 0.0,
        "credit_card": 0.2 if income in ("high", "ultra-high") else 0.1,
        "mutual_fund": 0.2 if income in ("high", "ultra-high") else 0.05,
        "insurance": 0.15,
        "fd": 0.1,
        "home_loan": 0.05,
        "personal_loan": 0.05,
    }

    for gap in gaps:
        base = relevance_hints.get(gap["product"], 0.1)
        gap["relevance_score"] = round(min(1.0, gap["adoption_rate_among_similar"] + base), 3)
        gap["reason"] = (
            f"{gap['adoption_rate_among_similar']*100:.0f}% of similar clients hold "
            f"{gap['product']}; relevant given client profile"
        )

    gaps.sort(key=lambda x: x["relevance_score"], reverse=True)
    return json.dumps({"scored_gaps": gaps, "similar_count": gaps_data.get("similar_count", 0)})


# ── Agent Factory ─────────────────────────────────────────────────────────────

def build_product_agent(llm: BaseChatModel) -> AgentExecutor:
    tools = [load_all_profiles, find_similar_clients, detect_product_gaps, score_gaps]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a product gap analysis agent for a bank using collaborative filtering. "
         "Steps: load_all_profiles → find_similar_clients → detect_product_gaps → score_gaps. "
         "Return JSON matching ProductOutput schema with product_gaps list, "
         "similar_client_count, and confidence_score."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        callbacks=get_usage_callbacks(),
    )


async def run_product_agent(
    client_id: str,
    current_profile: Dict[str, Any],
    llm: Optional[BaseChatModel] = None,
) -> ProductOutput:
    if llm is None:
        llm = get_llm()
    executor = build_product_agent(llm)
    result = await executor.ainvoke(
        {
            "input": (
                f"Find product gaps for client {client_id}. "
                f"Current profile summary: {json.dumps(current_profile)}"
            )
        },
        config={"callbacks": get_usage_callbacks()},
    )
    record_tool_steps("product_agent", result.get("intermediate_steps"))
    raw = result.get("output", "{}")
    data = safe_json_loads(raw, default={}, context="product_agent.output")

    gaps = data.get("product_gaps", data.get("scored_gaps", []))
    n_similar = int(data.get("similar_client_count", data.get("similar_count", 0)))
    conf = min(1.0, 0.5 + n_similar * 0.05) if n_similar > 0 else 0.3

    return ProductOutput(
        product_gaps=gaps,
        similar_client_count=n_similar,
        confidence_score=round(conf, 2),
    )
