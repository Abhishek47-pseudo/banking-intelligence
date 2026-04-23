"""
Confidence scoring utilities.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Centralized thresholds so behavior is consistent across modules.
LOW_CONFIDENCE_WARN = 0.60
LOW_CONFIDENCE_SUPPRESS = 0.45


def compute_confidence(months_of_data: int) -> float:
    """Transaction confidence based on months of data available."""
    if months_of_data >= 12:
        return 1.0
    if months_of_data >= 6:
        return 0.75
    if months_of_data >= 3:
        return 0.5
    return 0.25


def signal_quality_to_confidence(signal_quality: str) -> float:
    """Map interaction signal quality to a 0–1 confidence score."""
    mapping = {"high": 1.0, "medium": 0.75, "low": 0.5, "none": 0.25}
    return mapping.get(signal_quality.lower(), 0.25)


def weighted_confidence(
    transaction_score: float,
    crm_score: float,
    interaction_score: float,
    product_score: float,
) -> float:
    """
    Weighted average confidence across all four agents.
    Weights: transaction 0.35, CRM 0.30, interaction 0.25, product 0.10.
    """
    return round(
        0.35 * transaction_score
        + 0.30 * crm_score
        + 0.25 * interaction_score
        + 0.10 * product_score,
        3,
    )


def should_suppress_recommendations(confidence_score: float) -> bool:
    """
    Confidence-aware behavior gate.
    Keep minimal: a single rule used by generation layer.
    """
    if confidence_score < LOW_CONFIDENCE_SUPPRESS:
        log.warning(
            "low_confidence_suppress",
            confidence_score=confidence_score,
            threshold=LOW_CONFIDENCE_SUPPRESS,
        )
        return True
    return False
