"""
Confidence scoring utilities.
"""


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
