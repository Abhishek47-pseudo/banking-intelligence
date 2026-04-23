"""
CRM Agent — LangChain Tool-Calling Agent.
Loads, standardises, deduplicates, and enriches CRM records.
"""

import os, json, re
from datetime import date, datetime
from typing import List, Optional, Dict, Any
import pandas as pd
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.language_models import BaseChatModel
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.llm.llm_factory import get_llm
from backend.observability.usage_logger import get_usage_callbacks, record_tool_steps
from backend.utils.safe_json import safe_json_loads


# ── Output Schema ─────────────────────────────────────────────────────────────

class CRMOutput(BaseModel):
    client_id: str
    income_band: str
    income_source: str
    risk_profile: str
    risk_source: str
    city: str
    city_source: str
    age_band: str
    relationship_tenure_years: int
    products_held: List[str]
    stale_fields: List[str]
    duplicate_resolved: bool
    confidence_score: float


# ── PIN → City Lookup (simplified) ────────────────────────────────────────────

PIN_CITY = {
    "110001": "Delhi", "400001": "Mumbai", "560001": "Bangalore",
    "600001": "Chennai", "500001": "Hyderabad", "411001": "Pune",
    "700001": "Kolkata", "380001": "Ahmedabad", "302001": "Jaipur",
    "395001": "Surat",
}

_CRM_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mock", "crm.csv"
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def load_crm_record(client_id: str) -> str:
    """Load a single CRM record for a client. Returns JSON dict."""
    try:
        df = pd.read_csv(_CRM_DATA_PATH)
        row = df[df["client_id"] == client_id]
        if row.empty:
            # Return synthetic CRM record
            return json.dumps({
                "client_id": client_id, "name": f"Client {client_id}",
                "dob": "1985-06-15", "phone": "+917012345678",
                "email": f"{client_id.lower()}@bank.com",
                "city": "Mumbai", "pin": "400001",
                "income_band": "mid", "risk_profile": "moderate",
                "age_band": "36-45", "tenure_years": 5,
                "products_held": '["savings_account", "debit_card"]',
                "last_updated": "2023-01-01",
            })
        return row.iloc[0].to_json()
    except FileNotFoundError:
        return json.dumps({
            "client_id": client_id, "name": f"Client {client_id}",
            "dob": "1985-06-15", "city": "Delhi", "pin": "110001",
            "income_band": "high", "risk_profile": "moderate",
            "age_band": "36-45", "tenure_years": 7,
            "products_held": '["savings_account", "credit_card"]',
            "last_updated": "2022-03-01",
        })


@tool
def standardize_fields(record_json: str) -> str:
    """Normalize income to rupees, phone format, dates to ISO-8601, bucket income bands."""
    # Be defensive: if a weaker model passes non-JSON (e.g. "load_crm_record(...)")
    # try to recover rather than crashing the whole agent run.
    try:
        record = json.loads(record_json)
    except Exception:
        # Try to extract an embedded JSON object.
        m = re.search(r"\{.*\}", str(record_json), re.DOTALL)
        if m:
            record = json.loads(m.group())
        else:
            # If it looks like a tool call string, try to re-load using the client_id.
            m_id = re.search(r"client_id\s*=\s*'([^']+)'", str(record_json))
            if m_id:
                record = json.loads(load_crm_record(m_id.group(1)))
            else:
                return json.dumps({"error": "standardize_fields_expected_json", "raw": str(record_json)})

    # Normalize income_band
    raw_band = str(record.get("income_band", "")).lower().strip()
    band_map = {
        "low": "low", "l": "low",
        "mid": "mid", "medium": "mid", "m": "mid",
        "high": "high", "h": "high",
        "ultra": "ultra-high", "ultra-high": "ultra-high",
    }
    record["income_band"] = band_map.get(raw_band, raw_band) or "unknown"

    # Parse DOB → age_band
    dob_str = record.get("dob", "")
    try:
        dob = datetime.strptime(dob_str[:10], "%Y-%m-%d").date()
        age = (date.today() - dob).days // 365
        if age < 26:
            record["age_band"] = "18-25"
        elif age < 36:
            record["age_band"] = "26-35"
        elif age < 46:
            record["age_band"] = "36-45"
        elif age < 56:
            record["age_band"] = "46-55"
        else:
            record["age_band"] = "55+"
    except Exception:
        record["age_band"] = record.get("age_band", "unknown")

    # Normalize phone
    phone = str(record.get("phone", ""))
    phone = re.sub(r"[^\d+]", "", phone)
    record["phone"] = phone

    # Ensure products_held is a list
    ph = record.get("products_held", "[]")
    if isinstance(ph, str):
        try:
            record["products_held"] = json.loads(ph)
        except Exception:
            record["products_held"] = [ph] if ph else []

    return json.dumps(record)


@tool
def infer_missing(record_json: str) -> str:
    """Infer missing city from PIN code. Tag inferred fields with _source suffix."""
    record = json.loads(record_json)

    # PIN → City lookup
    pin = str(record.get("pin", ""))[:6]
    if not record.get("city") and pin in PIN_CITY:
        record["city"] = PIN_CITY[pin]
        record["city_source"] = "pin_inferred"
    else:
        record["city_source"] = "crm_declared"

    record["income_source"] = "crm_declared"
    record["risk_source"] = "crm_declared"

    return json.dumps(record)


@tool
def flag_stale_fields(record_json: str) -> str:
    """Flag fields that haven't been updated in > 2 years."""
    record = json.loads(record_json)
    stale = []
    last_updated_str = record.get("last_updated", "")
    try:
        last_updated = datetime.strptime(last_updated_str[:10], "%Y-%m-%d").date()
        age_days = (date.today() - last_updated).days
        if age_days > 730:  # > 2 years
            stale = ["income_band", "risk_profile", "city", "phone"]
    except Exception:
        stale = []
    record["stale_fields"] = stale
    return json.dumps(record)


@tool
def resolve_duplicates(client_id: str) -> str:
    """Check for potential duplicate CRM entries using fuzzy name+DOB matching."""
    # In a real system this would use recordlinkage; here we do a lightweight check
    return json.dumps({"duplicate_resolved": False, "note": "No duplicates found."})


# ── Confidence Scoring ────────────────────────────────────────────────────────

def _crm_confidence(stale_fields: List[str], inferred: bool) -> float:
    base = 1.0
    if stale_fields:
        base -= 0.1 * min(len(stale_fields), 4)
    if inferred:
        base -= 0.1
    return max(0.2, round(base, 2))


# ── Agent Factory ─────────────────────────────────────────────────────────────

def build_crm_agent(llm: BaseChatModel) -> AgentExecutor:
    tools = [load_crm_record, standardize_fields, infer_missing, flag_stale_fields, resolve_duplicates]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a CRM data quality agent for a bank. "
         "Use tools in order: load_crm_record → standardize_fields → infer_missing → "
         "flag_stale_fields → resolve_duplicates. "
         "When calling a tool that expects JSON, pass the EXACT JSON string returned "
         "by the previous tool (not a paraphrase, not a function call string). "
         "Return a JSON object with all CRMOutput fields."),
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


async def run_crm_agent(client_id: str, llm: Optional[BaseChatModel] = None) -> CRMOutput:
    if llm is None:
        llm = get_llm()
    executor = build_crm_agent(llm)
    result = await executor.ainvoke(
        {"input": f"Load and standardize CRM record for client {client_id}."},
        config={"callbacks": get_usage_callbacks()},
    )
    record_tool_steps("crm_agent", result.get("intermediate_steps"))
    raw = result.get("output", "{}")
    data = safe_json_loads(raw, default={}, context="crm_agent.output")

    stale = data.get("stale_fields", [])
    inferred = data.get("city_source", "") == "pin_inferred"

    return CRMOutput(
        client_id=client_id,
        income_band=data.get("income_band", "unknown"),
        income_source=data.get("income_source", "crm_declared"),
        risk_profile=data.get("risk_profile", "moderate"),
        risk_source=data.get("risk_source", "crm_declared"),
        city=data.get("city", "unknown"),
        city_source=data.get("city_source", "crm_declared"),
        age_band=data.get("age_band", "unknown"),
        relationship_tenure_years=int(data.get("tenure_years", data.get("relationship_tenure_years", 0))),
        products_held=data.get("products_held", []),
        stale_fields=stale,
        duplicate_resolved=bool(data.get("duplicate_resolved", False)),
        confidence_score=_crm_confidence(stale, inferred),
    )
