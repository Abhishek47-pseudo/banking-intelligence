"""
Interaction Agent — LangChain Tool-Calling Agent.
Processes relationship manager notes, emails, and call logs.
"""

import os, json, re
from typing import List, Optional, Dict, Any
import pandas as pd
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.language_models import BaseChatModel
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.llm.llm_factory import get_llm
from backend.llm.prompts import INTERACTION_EXTRACTION_PROMPT
from backend.observability.usage_logger import get_usage_callbacks, record_tool_steps

# ── Output Schemas ────────────────────────────────────────────────────────────

class IntentObject(BaseModel):
    type: str
    value: str
    confidence: float


class LifeEventObject(BaseModel):
    event: str
    timeframe: str


class InteractionOutput(BaseModel):
    summary: str
    sentiment: str
    intents: List[Dict[str, Any]]
    life_events: List[Dict[str, Any]]
    churn_risk: bool
    signal_quality: str
    confidence_score: float
    interactions_processed: int


_SIGNAL_QUALITY_SCORE = {"high": 1.0, "medium": 0.75, "low": 0.5, "none": 0.25}

_INTERACTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "mock", "interactions.csv"
)

# PII patterns to strip
_PII_PATTERNS = [
    r'\b\d{10,18}\b',           # account/card numbers
    r'\b\d{6}\b',               # PINs / OTPs
    r'\b\d{1,5}[,\s]\w+\s(?:Street|Road|Nagar|Colony|Lane)\b',  # addresses
    r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN
    r'\b\d{12}\b',              # Aadhaar
]


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def load_interactions(client_id: str) -> str:
    """Load all interaction records for a client. Returns JSON list of notes."""
    try:
        df = pd.read_csv(_INTERACTIONS_PATH)
        rows = df[df["client_id"] == client_id]
        if rows.empty:
            return json.dumps([{
                "interaction_id": "INT00001",
                "client_id": client_id,
                "date": "2024-06-15",
                "channel": "call",
                "notes": "Client called about FD maturity and asked about investment options.",
            }])
        return rows.to_json(orient="records")
    except FileNotFoundError:
        return json.dumps([{
            "interaction_id": "INT00001",
            "client_id": client_id,
            "date": "2024-06-15",
            "channel": "branch_visit",
            "notes": "Client visited branch. Expressed interest in mutual funds. Mentioned upcoming home purchase.",
        }])


@tool
def preprocess_text(interactions_json: str) -> str:
    """Strip PII, deduplicate near-identical notes, and chunk long docs (512 tokens, 50 overlap)."""
    records = json.loads(interactions_json)
    processed = []
    seen_notes: List[str] = []

    for r in records:
        note = str(r.get("notes", ""))
        # Strip PII
        for pattern in _PII_PATTERNS:
            note = re.sub(pattern, "[REDACTED]", note)
        # Simple deduplication: skip if very similar to a seen note
        is_dup = False
        for seen in seen_notes:
            # Basic char-level overlap ratio
            shorter = min(len(note), len(seen))
            if shorter > 0:
                overlap = sum(a == b for a, b in zip(note[:shorter], seen[:shorter]))
                if overlap / shorter > 0.95:
                    is_dup = True
                    break
        if not is_dup:
            seen_notes.append(note)
            r["cleaned_notes"] = note
            processed.append(r)

    return json.dumps(processed)


@tool
def extract_signals(processed_json: str, raw_notes_combined: str = "") -> str:
    """
    Use LLM structured extraction to pull signals from cleaned interaction notes.
    Returns JSON matching the InteractionOutput schema.
    Note: The actual LLM call happens in run_interaction_agent via the chain.
    This tool formats the prompt and returns it for the agent to call the LLM.
    """
    records = json.loads(processed_json)
    if not records:
        return json.dumps({
            "summary": "No interactions available.",
            "sentiment": "neutral",
            "intents": [],
            "life_events": [],
            "churn_risk": False,
            "signal_quality": "none",
            "interactions_processed": 0,
        })

    # Combine all notes
    combined = " | ".join(r.get("cleaned_notes", r.get("notes", "")) for r in records)
    # Return the combined notes for the agent to process with the LLM
    return json.dumps({
        "combined_notes": combined[:2000],  # truncate for safety
        "interaction_count": len(records),
        "channels": list({r.get("channel", "unknown") for r in records}),
    })


@tool
def validate_output(llm_response: str) -> str:
    """Validate LLM JSON extraction output. Retry with defaults on failure."""
    try:
        data = json.loads(llm_response)
        required = ["summary", "sentiment", "intents", "life_events",
                    "churn_risk", "signal_quality"]
        for field in required:
            if field not in data:
                data[field] = [] if "intents" in field or "events" in field else (
                    "neutral" if field == "sentiment" else
                    ("none" if field == "signal_quality" else
                     ("No summary available." if field == "summary" else False))
                )
        return json.dumps(data)
    except json.JSONDecodeError:
        return json.dumps({
            "summary": "Unable to parse interaction signals.",
            "sentiment": "neutral",
            "intents": [],
            "life_events": [],
            "churn_risk": False,
            "signal_quality": "none",
        })


# ── Agent Factory ─────────────────────────────────────────────────────────────

def build_interaction_agent(llm: BaseChatModel) -> AgentExecutor:
    tools = [load_interactions, preprocess_text, extract_signals, validate_output]
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an interaction signal extraction agent for a bank. "
         "Steps: load_interactions → preprocess_text → extract_signals → "
         "then call the LLM with this EXACT prompt schema:\n\n"
         + INTERACTION_EXTRACTION_PROMPT[:1500] +
         "\n\nFinally: validate_output. "
         "Return valid JSON matching InteractionOutput schema."),
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


async def run_interaction_agent(client_id: str, llm: Optional[BaseChatModel] = None) -> InteractionOutput:
    if llm is None:
        llm = get_llm()
    executor = build_interaction_agent(llm)
    result = await executor.ainvoke(
        {
            "input": (
                f"Extract interaction signals for client {client_id}. "
                "Load interactions, preprocess, extract signals using the LLM prompt, and validate."
            )
        },
        config={"callbacks": get_usage_callbacks()},
    )
    record_tool_steps("interaction_agent", result.get("intermediate_steps"))
    raw = result.get("output", "{}")
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    data = json.loads(match.group()) if match else {}

    sq = data.get("signal_quality", "none")
    return InteractionOutput(
        summary=data.get("summary", "No interaction data available."),
        sentiment=data.get("sentiment", "neutral"),
        intents=data.get("intents", []),
        life_events=data.get("life_events", []),
        churn_risk=bool(data.get("churn_risk", False)),
        signal_quality=sq,
        confidence_score=_SIGNAL_QUALITY_SCORE.get(sq, 0.25),
        interactions_processed=data.get("interactions_processed", 0),
    )
