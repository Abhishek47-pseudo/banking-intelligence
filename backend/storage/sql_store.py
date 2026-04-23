"""
Async SQL store — all database interactions for the platform.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.storage.models import Base, ClientProfile, FeedbackEvent, ProductRelevanceScore

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/banking.db")

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── Client Profile ────────────────────────────────────────────────────────────

async def upsert_client_profile(profile_dict: Dict[str, Any]) -> ClientProfile:
    """Insert or update a client profile."""
    async with AsyncSessionLocal() as session:
        stmt = sqlite_insert(ClientProfile).values(**profile_dict)
        stmt = stmt.on_conflict_do_update(
            index_elements=["client_id"],
            set_={k: v for k, v in profile_dict.items() if k != "client_id"}
        )
        await session.execute(stmt)
        await session.commit()
        return await get_client_profile(profile_dict["client_id"])


async def get_client_profile(client_id: str) -> Optional[ClientProfile]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ClientProfile).where(ClientProfile.client_id == client_id)
        )
        return result.scalar_one_or_none()


async def list_all_profiles() -> List[ClientProfile]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ClientProfile))
        return result.scalars().all()


async def filter_profiles(
    income_band: Optional[str] = None,
    has_churn_risk: Optional[bool] = None,
    city: Optional[str] = None,
    exclude_ids: Optional[List[str]] = None,
) -> List[ClientProfile]:
    """Structured pre-filter for hybrid RAG retrieval."""
    async with AsyncSessionLocal() as session:
        q = select(ClientProfile)
        if income_band:
            q = q.where(ClientProfile.income_band == income_band)
        if has_churn_risk is not None:
            q = q.where(ClientProfile.churn_risk == has_churn_risk)
        if city:
            q = q.where(ClientProfile.city == city)
        if exclude_ids:
            q = q.where(ClientProfile.client_id.notin_(exclude_ids))
        result = await session.execute(q)
        return result.scalars().all()


# ── Feedback ──────────────────────────────────────────────────────────────────

async def record_feedback(
    client_id: str,
    recommendation_id: str,
    product: str,
    outcome: str,
    rejection_reason: Optional[str] = None,
) -> FeedbackEvent:
    async with AsyncSessionLocal() as session:
        event = FeedbackEvent(
            client_id=client_id,
            recommendation_id=recommendation_id,
            product=product,
            outcome=outcome,
            rejection_reason=rejection_reason,
            created_at=datetime.utcnow(),
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        return event


# ── Product Relevance Scores ──────────────────────────────────────────────────

async def get_relevance_score(client_id: str, product: str) -> Optional[ProductRelevanceScore]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ProductRelevanceScore)
            .where(ProductRelevanceScore.client_id == client_id)
            .where(ProductRelevanceScore.product == product)
        )
        return result.scalar_one_or_none()


async def increment_relevance(client_id: str, product: str, delta: float = 1.0) -> None:
    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            select(ProductRelevanceScore)
            .where(ProductRelevanceScore.client_id == client_id)
            .where(ProductRelevanceScore.product == product)
        )
        row = existing.scalar_one_or_none()
        if row:
            row.score = row.score + delta
            row.acceptance_count = row.acceptance_count + 1
            row.updated_at = datetime.utcnow()
        else:
            session.add(ProductRelevanceScore(
                client_id=client_id, product=product,
                score=delta, acceptance_count=1
            ))
        await session.commit()


async def decrement_relevance(client_id: str, product: str, delta: float = 1.0) -> None:
    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            select(ProductRelevanceScore)
            .where(ProductRelevanceScore.client_id == client_id)
            .where(ProductRelevanceScore.product == product)
        )
        row = existing.scalar_one_or_none()
        if row:
            row.score = row.score - delta
            row.rejection_count = row.rejection_count + 1
            if row.rejection_count >= 3:
                row.flagged_for_review = True
            row.updated_at = datetime.utcnow()
        else:
            session.add(ProductRelevanceScore(
                client_id=client_id, product=product,
                score=-delta, rejection_count=1
            ))
        await session.commit()
