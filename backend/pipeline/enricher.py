"""
Enricher — converts structured Client 360 Profile → natural language paragraphs
used as context for the RAG + LLM generation layer.
"""

from backend.pipeline.normalizer import Client360Profile
from backend.llm.prompts import ENRICHMENT_TEMPLATE, LOW_CONFIDENCE_DISCLAIMER


def enrich(profile: Client360Profile) -> str:
    """
    Convert a Client360Profile into a natural language paragraph block.
    Returns the enriched text string ready to be embedded and stored in FAISS.
    """
    # Anomaly note
    anomalies = profile.anomalies_flagged or []
    if anomalies:
        anomaly_note = f"Anomalous spend flagged in: {', '.join(anomalies)}."
    else:
        anomaly_note = ""

    # Products
    current_prods = ", ".join(profile.current_products or []) or "none on record"

    # Product gaps
    gaps = profile.product_gaps or []
    if gaps:
        gap_strs = [
            f"{g['product']} ({g.get('adoption_rate_among_similar', 0)*100:.0f}% adoption among similar clients)"
            for g in gaps[:3]
        ]
        product_gaps_nl = "Gaps identified: " + "; ".join(gap_strs)
    else:
        product_gaps_nl = "No significant product gaps identified."

    # Top categories
    cats = ", ".join(profile.top_categories or ["general spending"])

    # Confidence note
    conf = profile.merged_confidence_score
    if conf < 0.6:
        low_conf_note = LOW_CONFIDENCE_DISCLAIMER
    else:
        low_conf_note = ""

    try:
        text = ENRICHMENT_TEMPLATE.format(
            client_id=profile.client_id,
            age_band=profile.age_band or "unknown",
            city=profile.city or "unknown",
            tenure=profile.relationship_tenure_years or 0,
            monthly_spend=profile.monthly_spend or 0,
            top_categories=cats,
            spend_trend=profile.spend_trend or "stable",
            international_usage="Yes" if profile.international_usage else "No",
            anomaly_note=anomaly_note,
            current_products=current_prods,
            product_gaps_natural_language=product_gaps_nl,
            interaction_summary=profile.interaction_summary or "No interaction data.",
            sentiment=profile.sentiment or "neutral",
            churn_risk="Yes" if profile.churn_risk else "No",
            merged_confidence_score=conf,
            low_confidence_disclaimer=low_conf_note,
        )
    except (KeyError, ValueError) as e:
        # Fallback plain text
        text = (
            f"Client {profile.client_id} profile. "
            f"Income: {profile.income_band}. City: {profile.city}. "
            f"Monthly spend: {profile.monthly_spend}. "
            f"Data confidence: {conf:.0%}."
        )

    return text.strip()
