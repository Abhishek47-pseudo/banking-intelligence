"""
SQLAlchemy models for the Banking Intelligence Platform.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, Text, JSON, ForeignKey, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class ClientProfile(Base):
    """Enriched Client 360 profile stored after pipeline run."""
    __tablename__ = "client_profiles"

    client_id = Column(String(64), primary_key=True, index=True)
    # Financial summary fields
    monthly_spend = Column(Integer, nullable=True)
    top_categories = Column(JSON, nullable=True)        # List[str]
    international_usage = Column(Boolean, default=False)
    avg_txn_size = Column(Float, nullable=True)
    spend_trend = Column(String(20), nullable=True)     # increasing|stable|decreasing
    anomalies_flagged = Column(JSON, nullable=True)     # List[str]
    # CRM fields
    income_band = Column(String(32), nullable=True)
    income_source = Column(String(32), nullable=True)
    risk_profile = Column(String(32), nullable=True)
    risk_source = Column(String(32), nullable=True)
    city = Column(String(64), nullable=True)
    city_source = Column(String(32), nullable=True)
    age_band = Column(String(32), nullable=True)
    relationship_tenure_years = Column(Integer, nullable=True)
    stale_fields = Column(JSON, nullable=True)          # List[str]
    duplicate_resolved = Column(Boolean, default=False)
    # Interaction fields
    interaction_summary = Column(Text, nullable=True)
    sentiment = Column(String(32), nullable=True)
    intents = Column(JSON, nullable=True)              # List[IntentObject]
    life_events = Column(JSON, nullable=True)          # List[LifeEventObject]
    churn_risk = Column(Boolean, default=False)
    signal_quality = Column(String(16), nullable=True)
    # Product fields
    current_products = Column(JSON, nullable=True)     # List[str]
    product_gaps = Column(JSON, nullable=True)          # List[ProductGap]
    similar_client_count = Column(Integer, nullable=True)
    # Meta
    merged_confidence_score = Column(Float, nullable=True)
    pipeline_timestamp = Column(DateTime, default=datetime.utcnow)
    partial_failure = Column(Boolean, default=False)
    failed_agents = Column(JSON, nullable=True)
    # Embedding flag
    embedding_id = Column(String(128), nullable=True)  # for FAISS lookup

    feedback_events = relationship("FeedbackEvent", back_populates="client")

    __table_args__ = (
        Index("ix_client_income_band", "income_band"),
        Index("ix_client_churn_risk", "churn_risk"),
    )


class FeedbackEvent(Base):
    """Tracks recommendation feedback for the learning loop."""
    __tablename__ = "feedback_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(64), ForeignKey("client_profiles.client_id"), index=True)
    recommendation_id = Column(String(128), nullable=False)
    product = Column(String(128), nullable=True)
    outcome = Column(String(16), nullable=False)        # accepted|rejected|pending
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    client = relationship("ClientProfile", back_populates="feedback_events")


class ProductRelevanceScore(Base):
    """Per-client per-product relevance score updated by feedback loop."""
    __tablename__ = "product_relevance_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(64), index=True)
    product = Column(String(128))
    score = Column(Float, default=0.0)
    rejection_count = Column(Integer, default=0)
    acceptance_count = Column(Integer, default=0)
    flagged_for_review = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
