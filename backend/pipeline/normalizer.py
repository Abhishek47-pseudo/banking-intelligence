"""
Normalizer — merges all 4 agent outputs into a unified Client 360 Profile.
Applies source priority rules and computes weighted confidence score.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from backend.agents.transaction_agent import TransactionOutput
from backend.agents.crm_agent import CRMOutput
from backend.agents.interaction_agent import InteractionOutput
from backend.agents.product_agent import ProductOutput
from backend.pipeline.orchestrator import PipelineResult

log = logging.getLogger(__name__)

# Confidence weights per agent
WEIGHTS = {
    "transaction": 0.35,
    "crm": 0.30,
    "interaction": 0.25,
    "product": 0.10,
}


class Client360Profile(BaseModel):
    client_id: str
    # Financial
    monthly_spend: Optional[int]
    top_categories: Optional[List[str]]
    international_usage: Optional[bool]
    avg_txn_size: Optional[float]
    spend_trend: Optional[str]
    anomalies_flagged: Optional[List[str]]
    # CRM
    income_band: Optional[str]
    income_source: Optional[str]
    risk_profile: Optional[str]
    city: Optional[str]
    city_source: Optional[str]
    age_band: Optional[str]
    relationship_tenure_years: Optional[int]
    # Products
    current_products: Optional[List[str]]
    product_gaps: Optional[List[Dict[str, Any]]]
    similar_client_count: Optional[int]
    # Interactions
    interaction_summary: Optional[str]
    sentiment: Optional[str]
    intents: Optional[List[Dict[str, Any]]]
    life_events: Optional[List[Dict[str, Any]]]
    churn_risk: Optional[bool]
    signal_quality: Optional[str]
    # Meta
    stale_fields: Optional[List[str]]
    merged_confidence_score: float
    pipeline_timestamp: Optional[str]
    partial_failure: bool
    failed_agents: Optional[List[str]]


def _resolve_income_band(
    crm: Optional[CRMOutput],
    tx: Optional[TransactionOutput],
) -> tuple[Optional[str], Optional[str]]:
    """Priority: CRM declared > transaction-inferred > default."""
    if crm and crm.income_band and crm.income_band != "unknown":
        return crm.income_band, crm.income_source
    if tx and tx.monthly_spend:
        spend = tx.monthly_spend
        if spend > 150000:
            band = "ultra-high"
        elif spend > 75000:
            band = "high"
        elif spend > 30000:
            band = "mid"
        else:
            band = "low"
        log.info(f"income_band inferred from transaction: {band}")
        return band, "transaction_inferred"
    return "unknown", "default"


def _resolve_city(
    crm: Optional[CRMOutput],
) -> tuple[Optional[str], Optional[str]]:
    """Priority: CRM declared > PIN-inferred (already done in CRM agent)."""
    if crm:
        return crm.city, crm.city_source
    return None, None


def _weighted_confidence(
    tx: Optional[TransactionOutput],
    crm: Optional[CRMOutput],
    interaction: Optional[InteractionOutput],
    product: Optional[ProductOutput],
) -> float:
    score = 0.0
    score += WEIGHTS["transaction"] * (tx.confidence_score if tx else 0.0)
    score += WEIGHTS["crm"] * (crm.confidence_score if crm else 0.0)
    score += WEIGHTS["interaction"] * (interaction.confidence_score if interaction else 0.0)
    score += WEIGHTS["product"] * (product.confidence_score if product else 0.0)
    return round(score, 3)


def normalize(pipeline_result: PipelineResult) -> Client360Profile:
    """
    Merge all 4 agent outputs into a Client 360 Profile applying source priority rules.
    """
    tx = pipeline_result.transaction
    crm = pipeline_result.crm
    interaction = pipeline_result.interaction
    product = pipeline_result.product

    income_band, income_source = _resolve_income_band(crm, tx)
    city, city_source = _resolve_city(crm)

    # Risk profile: CRM only, never infer
    risk_profile = crm.risk_profile if crm else None

    # Stale fields
    stale = crm.stale_fields if crm else []

    confidence = _weighted_confidence(tx, crm, interaction, product)

    return Client360Profile(
        client_id=pipeline_result.client_id,
        # Financial
        monthly_spend=tx.monthly_spend if tx else None,
        top_categories=tx.top_categories if tx else None,
        international_usage=tx.international_usage if tx else None,
        avg_txn_size=tx.avg_txn_size if tx else None,
        spend_trend=tx.spend_trend if tx else None,
        anomalies_flagged=tx.anomalies_flagged if tx else None,
        # CRM
        income_band=income_band,
        income_source=income_source,
        risk_profile=risk_profile,
        city=city,
        city_source=city_source,
        age_band=crm.age_band if crm else None,
        relationship_tenure_years=crm.relationship_tenure_years if crm else None,
        # Products
        current_products=crm.products_held if crm else [],
        product_gaps=product.product_gaps if product else [],
        similar_client_count=product.similar_client_count if product else 0,
        # Interactions
        interaction_summary=interaction.summary if interaction else None,
        sentiment=interaction.sentiment if interaction else "neutral",
        intents=interaction.intents if interaction else [],
        life_events=interaction.life_events if interaction else [],
        churn_risk=interaction.churn_risk if interaction else False,
        signal_quality=interaction.signal_quality if interaction else "none",
        # Meta
        stale_fields=stale,
        merged_confidence_score=confidence,
        pipeline_timestamp=pipeline_result.pipeline_timestamp,
        partial_failure=pipeline_result.partial_failure,
        failed_agents=pipeline_result.failed_agents,
    )
