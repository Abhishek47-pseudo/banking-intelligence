"""
LLM Generation Layer — LCEL chain for recommendation generation.
Provider-agnostic: uses the LLM factory (OpenAI → Groq fallback).
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from backend.llm.prompts import RECOMMENDATION_PROMPT
from backend.llm.llm_factory import get_llm          # ← factory import
from backend.utils.safe_json import safe_json_loads
from backend.pipeline.confidence import (
    LOW_CONFIDENCE_WARN,
    should_suppress_recommendations,
)

log = logging.getLogger(__name__)


# ── Output Schemas ────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    product: str
    reason: str
    data_source: str
    confidence: float


class RecommendationOutput(BaseModel):
    briefing: str
    recommendations: List[RecommendationItem]
    talking_points: List[str]
    confidence_score: float
    recommendation_id: str


# ── Generation Chain ──────────────────────────────────────────────────────────

def _build_chain(llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_template(RECOMMENDATION_PROMPT)
    return prompt | llm | StrOutputParser()


async def generate_recommendations(
    client_profile_text: str,
    retrieved_context: str,
    confidence_score: float,
    llm: Optional[BaseChatModel] = None,
) -> RecommendationOutput:
    """
    Generate recommendations using the LCEL chain.

    Args:
        client_profile_text: Enriched natural language profile from enricher.py
        retrieved_context:   Assembled context from hybrid retriever
        confidence_score:    Merged pipeline confidence (0–1)
        llm:                 Optional model (uses factory singleton if None)
    """
    if llm is None:
        llm = get_llm()

    chain = _build_chain(llm)
    low_confidence_note = (
        "Data confidence is below 60%. Treat recommendations as indicative."
        if confidence_score < LOW_CONFIDENCE_WARN else ""
    )
    if should_suppress_recommendations(confidence_score):
        return RecommendationOutput(
            briefing=(
                "Insufficient reliable signals to generate product recommendations. "
                "Please verify client data and re-run ingestion. "
                + ((" " + low_confidence_note) if low_confidence_note else "")
            ),
            recommendations=[],
            talking_points=[
                "Confirm income band and recent transaction activity.",
                "Collect updated interaction notes (goals, timelines, risk appetite).",
            ],
            confidence_score=confidence_score,
            recommendation_id=str(uuid.uuid4()),
        )

    try:
        raw_output = await chain.ainvoke({
            "client_profile": client_profile_text,
            "retrieved_context": retrieved_context,
            "confidence_score": confidence_score,
            "low_confidence_note": low_confidence_note,
        })

        data = safe_json_loads(raw_output, default=None, context="generator.recommendations")
        if not isinstance(data, dict):
            raise ValueError("No valid JSON object found in LLM response")
        recs = [
            RecommendationItem(
                product=r.get("product", ""),
                reason=r.get("reason", ""),
                data_source=r.get("data_source", "profile"),
                confidence=float(r.get("confidence", confidence_score)),
            )
            for r in data.get("recommendations", [])[:3]
        ]

        log.info(
            "recommendations_generated",
            confidence_score=confidence_score,
            count=len(recs),
            low_confidence=confidence_score < LOW_CONFIDENCE_WARN,
        )
        return RecommendationOutput(
            briefing=data.get("briefing", ""),
            recommendations=recs,
            talking_points=data.get("talking_points", [])[:3],
            confidence_score=confidence_score,
            recommendation_id=str(uuid.uuid4()),
        )

    except Exception as e:
        log.error(f"Recommendation generation failed: {e}")
        return RecommendationOutput(
            briefing=f"Unable to generate recommendations. Error: {str(e)[:120]}",
            recommendations=[],
            talking_points=[],
            confidence_score=confidence_score,
            recommendation_id=str(uuid.uuid4()),
        )

