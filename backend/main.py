"""
FastAPI Application — AI Banking Client Intelligence Platform.
All routes are async. FAISS index, SQL engine, and LLM initialised on startup.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from backend.storage import sql_store
from backend.rag.vector_store import get_vector_store
from backend.pipeline.orchestrator import run_pipeline, PipelineResult
from backend.pipeline.normalizer import normalize, Client360Profile
from backend.pipeline.enricher import enrich
from backend.rag.retriever import hybrid_retrieve, assemble_context
from backend.rag.reranker import mmr_rerank
from backend.llm.generator import generate_recommendations, RecommendationOutput
from backend.llm.llm_factory import get_llm          # ← single source of truth
from backend.feedback.feedback_loop import process_feedback, start_scheduler

log = structlog.get_logger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup_begin")
    # 1. Init DB
    await sql_store.init_db()
    log.info("db_initialised")
    # 2. Load FAISS indices
    store = get_vector_store()
    store.load_all()
    log.info("faiss_loaded")
    # 3. Warm up LLM (singleton)
    get_llm()
    log.info("llm_ready")
    # 4. Start APScheduler
    start_scheduler()
    log.info("scheduler_started")
    log.info("startup_complete")
    yield
    log.info("shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Banking Client Intelligence Platform",
    description="Agent-based RAG pipeline for personalised banking recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────────────────────

class IngestResponse(BaseModel):
    client_id: str
    status: str
    partial_failure: bool
    failed_agents: List[str]
    merged_confidence_score: float
    total_latency_ms: float
    pipeline_timestamp: Optional[str]


class RecommendRequest(BaseModel):
    income_band: Optional[str] = None
    city: Optional[str] = None
    has_churn_risk: Optional[bool] = None


class FeedbackRequest(BaseModel):
    client_id: str
    recommendation_id: str
    product: str
    outcome: str  # accepted | rejected | pending
    rejection_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    faiss_profiles_loaded: bool
    db_url: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ingest/{client_id}", response_model=IngestResponse, tags=["Pipeline"])
async def ingest_client(client_id: str) -> IngestResponse:
    """
    Run the full 4-agent pipeline for a client.
    Agents run concurrently. Results are normalised and stored in SQL + FAISS.
    """
    log.info("ingest_request", client_id=client_id)
    llm = get_llm()

    try:
        pipeline_result: PipelineResult = await run_pipeline(client_id, llm)
    except Exception as e:
        log.error("pipeline_failed", client_id=client_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    # Normalise
    profile: Client360Profile = normalize(pipeline_result)

    # Enrich → natural language
    enriched_text = enrich(profile)

    # Persist to SQL
    profile_dict = profile.model_dump()
    profile_dict["pipeline_timestamp"] = (
        pipeline_result.pipeline_timestamp
    )
    await sql_store.upsert_client_profile(profile_dict)

    # Upsert into FAISS
    store = get_vector_store()
    store.upsert_profile(
        client_id=client_id,
        enriched_text=enriched_text,
        metadata={
            "income_band": profile.income_band,
            "churn_risk": profile.churn_risk or False,
            "merged_confidence_score": profile.merged_confidence_score,
        },
    )
    if profile.intents:
        store.upsert_intents(client_id, profile.intents)
    if profile.product_gaps:
        store.upsert_product_gaps(client_id, profile.product_gaps)
    store.save_all()

    return IngestResponse(
        client_id=client_id,
        status="success" if not pipeline_result.partial_failure else "partial",
        partial_failure=pipeline_result.partial_failure,
        failed_agents=pipeline_result.failed_agents,
        merged_confidence_score=profile.merged_confidence_score,
        total_latency_ms=pipeline_result.total_latency_ms,
        pipeline_timestamp=pipeline_result.pipeline_timestamp,
    )


@app.get("/profile/{client_id}", response_model=Dict[str, Any], tags=["Profile"])
async def get_profile(client_id: str) -> Dict[str, Any]:
    """Return enriched Client 360 profile from SQL store."""
    profile = await sql_store.get_client_profile(client_id)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"No profile found for client {client_id}. Run /ingest first."
        )
    # Return as dict (SQLAlchemy model → dict)
    cols = profile.__table__.columns.keys()
    return {c: getattr(profile, c) for c in cols}


@app.post("/recommend/{client_id}", response_model=RecommendationOutput, tags=["Recommendations"])
async def recommend(client_id: str, req: RecommendRequest = RecommendRequest()) -> RecommendationOutput:
    """
    Run hybrid RAG retrieval + LLM generation to produce personalised recommendations.
    """
    # Fetch stored profile
    profile_row = await sql_store.get_client_profile(client_id)
    if not profile_row:
        raise HTTPException(
            status_code=404,
            detail=f"No profile for {client_id}. Run /ingest/{client_id} first."
        )

    # Rebuild Client360Profile for enrichment
    from backend.pipeline.normalizer import Client360Profile
    profile = Client360Profile(
        client_id=client_id,
        monthly_spend=profile_row.monthly_spend,
        top_categories=profile_row.top_categories,
        international_usage=profile_row.international_usage,
        avg_txn_size=profile_row.avg_txn_size,
        spend_trend=profile_row.spend_trend,
        anomalies_flagged=profile_row.anomalies_flagged,
        income_band=profile_row.income_band,
        income_source=profile_row.income_source,
        risk_profile=profile_row.risk_profile,
        city=profile_row.city,
        city_source=profile_row.city_source,
        age_band=profile_row.age_band,
        relationship_tenure_years=profile_row.relationship_tenure_years,
        current_products=profile_row.current_products,
        product_gaps=profile_row.product_gaps,
        similar_client_count=profile_row.similar_client_count,
        interaction_summary=profile_row.interaction_summary,
        sentiment=profile_row.sentiment,
        intents=profile_row.intents,
        life_events=profile_row.life_events,
        churn_risk=profile_row.churn_risk,
        signal_quality=profile_row.signal_quality,
        stale_fields=profile_row.stale_fields,
        merged_confidence_score=profile_row.merged_confidence_score or 0.5,
        pipeline_timestamp=str(profile_row.pipeline_timestamp),
        partial_failure=False,
        failed_agents=[],
    )

    enriched_profile = enrich(profile)

    # Hybrid retrieval
    docs = await hybrid_retrieve(
        query=enriched_profile[:500],  # truncated query
        client_id=client_id,
        income_band=req.income_band or profile_row.income_band,
        has_churn_risk=req.has_churn_risk,
        city=req.city,
        top_k=5,
        fetch_k=20,
    )

    # MMR re-ranking
    docs = await mmr_rerank(enriched_profile[:300], docs, top_k=5)

    # Assemble context
    context = assemble_context(docs)

    # Generate recommendations
    llm = get_llm()
    result = await generate_recommendations(
        client_profile_text=enriched_profile,
        retrieved_context=context,
        confidence_score=profile_row.merged_confidence_score or 0.5,
        llm=llm,
    )
    return result


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    """Capture recommendation outcome and trigger feedback loop."""
    if req.outcome not in ("accepted", "rejected", "pending"):
        raise HTTPException(status_code=422, detail="outcome must be accepted|rejected|pending")

    background_tasks.add_task(
        process_feedback,
        client_id=req.client_id,
        recommendation_id=req.recommendation_id,
        product=req.product,
        outcome=req.outcome,
        rejection_reason=req.rejection_reason,
    )
    return {"status": "feedback_queued", "client_id": req.client_id, "outcome": req.outcome}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    store = get_vector_store()
    return HealthResponse(
        status="ok",
        faiss_profiles_loaded=store._indices.get("client_profiles_index") is not None,
        db_url=os.getenv("DATABASE_URL", "not_set"),
    )
