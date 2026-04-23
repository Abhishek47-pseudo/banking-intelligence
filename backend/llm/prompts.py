"""
All LLM prompt templates for the Banking Intelligence Platform.
"""

# ── Interaction Extraction ────────────────────────────────────────────────────

INTERACTION_EXTRACTION_PROMPT = """
You are extracting structured signals from bank relationship manager notes.
Return ONLY valid JSON. Do not add commentary or markdown.

Schema:
{{
  "summary": string,           // 2-3 sentences, factual, third-person
  "sentiment": "positive" | "neutral" | "negative" | "anxious",
  "intents": [
    {{ "type": "product_interest" | "complaint" | "inquiry",
      "value": string,
      "confidence": float }}    // 0.0–1.0
  ],
  "life_events": [
    {{ "event": string, "timeframe": string }}
  ],
  "churn_risk": boolean,
  "signal_quality": "high" | "medium" | "low" | "none"
}}

Rules:
- Do NOT infer intent not explicitly stated or strongly implied
- If no useful signal exists, return signal_quality: "none"
- confidence reflects how explicitly the intent was stated

Few-shot examples:
Input: "Client called re FD maturity next month, asked about forex options for Europe trip in March"
Output: {{
  "summary": "Client inquired about FD maturity and forex options for an upcoming Europe trip.",
  "sentiment": "neutral",
  "intents": [
    {{"type": "product_interest", "value": "forex_card", "confidence": 0.92}},
    {{"type": "inquiry", "value": "fd_maturity_options", "confidence": 0.95}}
  ],
  "life_events": [{{"event": "international_travel", "timeframe": "March"}}],
  "churn_risk": false,
  "signal_quality": "high"
}}

Input: {raw_notes}
Output:"""

# ── Client Profile Enrichment Template ───────────────────────────────────────

ENRICHMENT_TEMPLATE = """
Client {client_id} ({age_band}, {city}) has been a customer for {tenure} years.

Financial behaviour: Monthly spend of ₹{monthly_spend:,}, primarily on {top_categories}.
Spend trend is {spend_trend}. International transactions: {international_usage}.
{anomaly_note}

Product holdings: {current_products}.
Identified gaps: {product_gaps_natural_language}.

Preferences & intent: {interaction_summary}
Sentiment: {sentiment}. Churn risk: {churn_risk}.

Data confidence: {merged_confidence_score:.0%}.
{low_confidence_disclaimer}
""".strip()

LOW_CONFIDENCE_DISCLAIMER = (
    "Note: Profile is based on limited data. "
    "Recommendations should be treated as indicative."
)

# ── Recommendation Generation ─────────────────────────────────────────────────

RECOMMENDATION_PROMPT = """
You are a senior banking relationship manager assistant.
Given the client profile and similar client context below, generate:
1. A 3-sentence client briefing
2. Up to 3 product recommendations, each with a specific reason tied to the client's data
3. 3 talking points for the relationship manager

Confidence level: {confidence_score:.0%}
{low_confidence_note}

Client profile:
{client_profile}

Similar client context:
{retrieved_context}

Rules:
- Ground every recommendation in the client data — no generic advice
- If confidence < 60%, prefix each recommendation with "Based on available data, ..."
- Never recommend a product the client already holds
- Format output as valid JSON matching this schema:
{{
  "briefing": "string",
  "recommendations": [
    {{
      "product": "string",
      "reason": "string",
      "data_source": "string",
      "confidence": float
    }}
  ],
  "talking_points": ["string", "string", "string"]
}}
"""
