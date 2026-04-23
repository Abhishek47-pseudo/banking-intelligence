"""
Transaction Agent — LangChain Tool-Calling Agent.
Loads, categorises, aggregates, and analyses client transaction data.
"""

import os, json
from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.language_models import BaseChatModel
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.llm.llm_factory import get_llm
from backend.observability.usage_logger import get_usage_callbacks, record_tool_steps


# ── Output Schema ─────────────────────────────────────────────────────────────

class TransactionOutput(BaseModel):
    monthly_spend: int
    top_categories: List[str]
    international_usage: bool
    avg_txn_size: float
    spend_trend: str  # increasing | stable | decreasing
    anomalies_flagged: List[str]
    confidence_score: float
    months_of_data: int


# ── MCC Category Map (simplified) ────────────────────────────────────────────

MCC_MAP = {
    "5411": "groceries", "5812": "dining", "4511": "travel",
    "5541": "fuel", "5912": "healthcare", "5732": "electronics",
    "7011": "travel", "4722": "travel", "5311": "shopping",
}


# ── Tools ─────────────────────────────────────────────────────────────────────

_TX_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mock", "transactions.csv"
)


@tool
def load_transactions(client_id: str) -> str:
    """Load raw transaction records for a client from the CSV store.
    Returns a JSON string of transaction records."""
    try:
        df = pd.read_csv(_TX_DATA_PATH)
        client_df = df[df["client_id"] == client_id]
        if client_df.empty:
            return json.dumps({"error": f"No transactions found for {client_id}"})
        return client_df.to_json(orient="records")
    except FileNotFoundError:
        # Return synthetic data if CSV doesn't exist yet
        import random
        random.seed(hash(client_id))
        records = []
        categories = ["groceries", "travel", "dining", "utilities", "fuel", "shopping"]
        for i in range(50):
            records.append({
                "transaction_id": f"TXN{i:06d}",
                "client_id": client_id,
                "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "amount": round(random.uniform(200, 25000), 2),
                "category": random.choice(categories),
                "mcc": str(random.randint(1000, 9999)),
                "merchant": f"Merchant_{i}",
                "is_international": random.random() < 0.1,
            })
        return json.dumps(records)


@tool
def categorize_transactions(transactions_json: str) -> str:
    """Categorize transactions using MCC lookup. Returns JSON with category assigned."""
    records = json.loads(transactions_json)
    if isinstance(records, dict) and "error" in records:
        return transactions_json
    for r in records:
        mcc = str(r.get("mcc", ""))
        if mcc in MCC_MAP:
            r["category"] = MCC_MAP[mcc]
    return json.dumps(records)


@tool
def aggregate_monthly(transactions_json: str) -> str:
    """Aggregate transactions monthly per category. Returns spend summary JSON."""
    records = json.loads(transactions_json)
    if isinstance(records, dict) and "error" in records:
        return transactions_json
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    grouped = df.groupby(["year_month", "category"])["amount"].sum().reset_index()
    monthly_total = df.groupby("year_month")["amount"].sum().reset_index()
    return json.dumps({
        "by_category": grouped.to_dict(orient="records"),
        "monthly_total": monthly_total.to_dict(orient="records"),
        "total_transactions": len(df),
        "unique_months": df["year_month"].nunique(),
    })


@tool
def detect_patterns(transactions_json: str) -> str:
    """Detect international usage, spend trend, and Z-score anomalies."""
    records = json.loads(transactions_json)
    if isinstance(records, dict) and "error" in records:
        return transactions_json
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # International usage
    intl_flag = bool(df.get("is_international", pd.Series(dtype=bool)).any()) if "is_international" in df.columns else False

    # Monthly totals for trend + anomaly
    monthly = df.groupby("year_month")["amount"].sum().reset_index()
    amounts = monthly["amount"].values

    # Spend trend (linear regression slope)
    if len(amounts) >= 2:
        x = np.arange(len(amounts))
        slope = np.polyfit(x, amounts, 1)[0]
        mean_spend = np.mean(amounts)
        if slope > mean_spend * 0.05:
            trend = "increasing"
        elif slope < -mean_spend * 0.05:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # Anomaly detection via Z-score
    anomalies = []
    if len(amounts) >= 3:
        mean, std = np.mean(amounts), np.std(amounts)
        if std > 0:
            zscores = (amounts - mean) / std
            for i, z in enumerate(zscores):
                if abs(z) > 2:
                    anomalies.append(monthly.iloc[i]["year_month"])

    avg_txn = float(df["amount"].mean())
    top_cats = df.groupby("category")["amount"].sum().nlargest(3).index.tolist()

    return json.dumps({
        "international_usage": intl_flag,
        "spend_trend": trend,
        "anomalies_flagged": anomalies,
        "avg_txn_size": round(avg_txn, 2),
        "top_categories": top_cats,
        "unique_months": len(monthly),
        "monthly_avg_spend": round(float(np.mean(amounts)), 2) if len(amounts) else 0,
    })


# ── Confidence Scoring ────────────────────────────────────────────────────────

def _compute_confidence(months: int) -> float:
    if months >= 12:
        return 1.0
    if months >= 6:
        return 0.75
    if months >= 3:
        return 0.5
    return 0.25


# ── Agent Factory ─────────────────────────────────────────────────────────────

def build_transaction_agent(llm: BaseChatModel) -> AgentExecutor:
    tools = [load_transactions, categorize_transactions, aggregate_monthly, detect_patterns]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a transaction analysis agent for a bank. "
         "Use the provided tools in sequence: load_transactions → categorize_transactions → "
         "aggregate_monthly → detect_patterns. "
         "After all tools, return a JSON object matching TransactionOutput schema exactly. "
         "Do not add extra commentary."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
        callbacks=get_usage_callbacks(),
    )


async def run_transaction_agent(client_id: str, llm: Optional[BaseChatModel] = None) -> TransactionOutput:
    """Entry point called by the orchestrator."""
    if llm is None:
        llm = get_llm()
    executor = build_transaction_agent(llm)
    result = await executor.ainvoke(
        {"input": f"Analyse transactions for client {client_id}."},
        config={"callbacks": get_usage_callbacks()},
    )
    record_tool_steps("transaction_agent", result.get("intermediate_steps"))
    raw = result.get("output", "{}")
    # Parse JSON from LLM output
    import re
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        data = json.loads(match.group())
    else:
        data = {}

    months = data.get("unique_months", data.get("months_of_data", 0))
    return TransactionOutput(
        monthly_spend=int(data.get("monthly_avg_spend", data.get("monthly_spend", 0))),
        top_categories=data.get("top_categories", []),
        international_usage=bool(data.get("international_usage", False)),
        avg_txn_size=float(data.get("avg_txn_size", 0.0)),
        spend_trend=data.get("spend_trend", "stable"),
        anomalies_flagged=data.get("anomalies_flagged", []),
        confidence_score=_compute_confidence(months),
        months_of_data=months,
    )
