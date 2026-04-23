"""
Feedback Loop — captures recommendation outcomes and updates profiles + FAISS index.
"""

import logging
from typing import Optional

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from backend.storage import sql_store
from backend.rag.vector_store import get_vector_store

log = structlog.get_logger(__name__)

scheduler = AsyncIOScheduler()


async def process_feedback(
    client_id: str,
    recommendation_id: str,
    product: str,
    outcome: str,
    rejection_reason: Optional[str] = None,
) -> None:
    """
    Process a recommendation outcome:
    - accepted: increment relevance score, re-embed profile, log event
    - rejected: decrement score, flag if 3+ rejections, log event
    - pending:  log event only
    """
    # Record event in SQL
    await sql_store.record_feedback(
        client_id=client_id,
        recommendation_id=recommendation_id,
        product=product,
        outcome=outcome,
        rejection_reason=rejection_reason,
    )
    log.info("feedback_recorded",
             client_id=client_id, product=product, outcome=outcome)

    if outcome == "accepted":
        await sql_store.increment_relevance(client_id, product)
        await _refresh_client_embedding(client_id)
        log.info("relevance_incremented", client_id=client_id, product=product)

    elif outcome == "rejected":
        await sql_store.decrement_relevance(client_id, product)
        # Check if flagged
        score_row = await sql_store.get_relevance_score(client_id, product)
        if score_row and score_row.flagged_for_review:
            log.warning("product_flagged_for_review",
                        client_id=client_id, product=product,
                        rejection_count=score_row.rejection_count)


async def _refresh_client_embedding(client_id: str) -> None:
    """Re-embed and upsert client profile into FAISS after a positive outcome."""
    try:
        profile = await sql_store.get_client_profile(client_id)
        if not profile:
            return

        from backend.pipeline.normalizer import Client360Profile
        from backend.pipeline.enricher import enrich

        # Reconstruct a minimal profile object for enrichment
        p360 = Client360Profile(
            client_id=client_id,
            monthly_spend=profile.monthly_spend,
            top_categories=profile.top_categories,
            international_usage=profile.international_usage,
            avg_txn_size=profile.avg_txn_size,
            spend_trend=profile.spend_trend,
            anomalies_flagged=profile.anomalies_flagged,
            income_band=profile.income_band,
            income_source=profile.income_source,
            risk_profile=profile.risk_profile,
            city=profile.city,
            city_source=profile.city_source,
            age_band=profile.age_band,
            relationship_tenure_years=profile.relationship_tenure_years,
            current_products=profile.current_products,
            product_gaps=profile.product_gaps,
            similar_client_count=profile.similar_client_count,
            interaction_summary=profile.interaction_summary,
            sentiment=profile.sentiment,
            intents=profile.intents,
            life_events=profile.life_events,
            churn_risk=profile.churn_risk,
            signal_quality=profile.signal_quality,
            stale_fields=profile.stale_fields,
            merged_confidence_score=profile.merged_confidence_score or 0.5,
            pipeline_timestamp=str(profile.pipeline_timestamp),
            partial_failure=False,
            failed_agents=[],
        )

        enriched = enrich(p360)
        store = get_vector_store()
        store.upsert_profile(
            client_id=client_id,
            enriched_text=enriched,
            metadata={
                "income_band": profile.income_band,
                "churn_risk": profile.churn_risk or False,
                "merged_confidence_score": profile.merged_confidence_score or 0.0,
            },
        )
        store.save_all()
        log.info("embedding_refreshed", client_id=client_id)

    except Exception as e:
        log.error("embedding_refresh_failed", client_id=client_id, error=str(e))


async def weekly_rebuild() -> None:
    """
    Weekly batch job: rebuild FAISS index from all profiles.
    Triggered by APScheduler every Sunday at 02:00.
    """
    log.info("weekly_rebuild_start")
    try:
        profiles = await sql_store.list_all_profiles()
        from backend.pipeline.normalizer import Client360Profile
        from backend.pipeline.enricher import enrich

        store = get_vector_store()
        store._indices = {k: None for k in store._indices}  # reset indices

        for profile in profiles:
            try:
                p360 = Client360Profile(
                    client_id=profile.client_id,
                    monthly_spend=profile.monthly_spend,
                    top_categories=profile.top_categories,
                    international_usage=profile.international_usage,
                    avg_txn_size=profile.avg_txn_size,
                    spend_trend=profile.spend_trend,
                    anomalies_flagged=profile.anomalies_flagged,
                    income_band=profile.income_band,
                    income_source=profile.income_source,
                    risk_profile=profile.risk_profile,
                    city=profile.city,
                    city_source=profile.city_source,
                    age_band=profile.age_band,
                    relationship_tenure_years=profile.relationship_tenure_years,
                    current_products=profile.current_products,
                    product_gaps=profile.product_gaps,
                    similar_client_count=profile.similar_client_count,
                    interaction_summary=profile.interaction_summary,
                    sentiment=profile.sentiment,
                    intents=profile.intents,
                    life_events=profile.life_events,
                    churn_risk=profile.churn_risk,
                    signal_quality=profile.signal_quality,
                    stale_fields=profile.stale_fields,
                    merged_confidence_score=profile.merged_confidence_score or 0.5,
                    pipeline_timestamp=str(profile.pipeline_timestamp),
                    partial_failure=False,
                    failed_agents=[],
                )
                enriched = enrich(p360)
                store.upsert_profile(
                    client_id=profile.client_id,
                    enriched_text=enriched,
                    metadata={
                        "income_band": profile.income_band,
                        "churn_risk": profile.churn_risk or False,
                        "merged_confidence_score": profile.merged_confidence_score or 0.0,
                    },
                )
            except Exception as e:
                log.warning("rebuild_profile_failed", client_id=profile.client_id, error=str(e))

        store.save_all()
        log.info("weekly_rebuild_complete", profiles_rebuilt=len(profiles))

    except Exception as e:
        log.error("weekly_rebuild_failed", error=str(e))


def start_scheduler() -> None:
    """Register and start the APScheduler weekly rebuild job."""
    scheduler.add_job(weekly_rebuild, "cron", day_of_week="sun", hour=2, minute=0)
    scheduler.start()
    log.info("feedback_scheduler_started")
