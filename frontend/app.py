"""
Streamlit Frontend — AI Banking Client Intelligence Platform
Pages:
  1. Client Lookup — ingest + enriched profile
  2. Recommendations — top-3 with reasons and talking points
  3. Feedback — thumbs up/down on each recommendation
  4. Pipeline Monitor — per-agent status and latency
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Banking Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg: #0f0f1a;
    --surface: #1a1a2e;
    --surface2: #252540;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --border: rgba(99,102,241,0.2);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid var(--border);
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 0.5rem 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: var(--primary); }
.metric-label { font-size: 0.78rem; color: var(--text-muted); margin-top: 4px; }

/* Confidence bar */
.conf-bar-wrap { margin: 0.75rem 0; }
.conf-bar {
    height: 8px;
    border-radius: 4px;
    background: var(--surface2);
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-green { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b981; }
.badge-amber { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b; }
.badge-red   { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef4444; }
.badge-blue  { background: rgba(99,102,241,0.15); color: #6366f1; border: 1px solid #6366f1; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--primary);
    border-left: 3px solid var(--primary);
    padding-left: 10px;
    margin: 1.5rem 0 0.75rem 0;
}

/* Recommendation card */
.rec-card {
    background: linear-gradient(135deg, var(--surface) 0%, rgba(99,102,241,0.05) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin: 0.75rem 0;
    position: relative;
}
.rec-number {
    position: absolute;
    top: -12px;
    left: 16px;
    background: var(--primary);
    color: white;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
}

/* Agent status pill */
.agent-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 4px;
}
.agent-ok   { background: rgba(16,185,129,0.1); border: 1px solid #10b981; color: #10b981; }
.agent-fail { background: rgba(239,68,68,0.1);  border: 1px solid #ef4444; color: #ef4444; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Button overrides */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(99,102,241,0.4);
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs) -> Optional[Dict]:
    try:
        resp = httpx.request(method, f"{API_BASE}{path}", timeout=120.0, **kwargs)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.text[:200]}")
        return None
    except httpx.ConnectError:
        st.error("⚠️ Cannot connect to backend. Make sure FastAPI is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def _confidence_bar(score: float, label: str = "Data Confidence"):
    pct = int(score * 100)
    if score >= 0.8:
        colour = "#10b981"
    elif score >= 0.6:
        colour = "#f59e0b"
    else:
        colour = "#ef4444"
    st.markdown(f"""
    <div class='conf-bar-wrap'>
        <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
            <span style='font-size:0.85rem;color:#94a3b8;'>{label}</span>
            <span style='font-size:0.85rem;font-weight:600;color:{colour};'>{pct}%</span>
        </div>
        <div class='conf-bar'>
            <div class='conf-fill' style='width:{pct}%;background:{colour};'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _badge(text: str, colour: str = "blue"):
    st.markdown(f"<span class='badge badge-{colour}'>{text}</span>", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────

if "current_client" not in st.session_state:
    st.session_state.current_client = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "pipeline_meta" not in st.session_state:
    st.session_state.pipeline_meta = None
if "feedback_sent" not in st.session_state:
    st.session_state.feedback_sent = {}


# ── Sidebar Navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:2.5rem;'>🏦</div>
        <div style='font-size:1rem;font-weight:700;color:#6366f1;margin-top:4px;'>Banking Intelligence</div>
        <div style='font-size:0.72rem;color:#94a3b8;'>AI-Powered Client Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    page = st.radio(
        "Navigate",
        ["🔍 Client Lookup", "💡 Recommendations", "👍 Feedback", "📊 Pipeline Monitor"],
        label_visibility="collapsed",
    )
    st.divider()

    # Quick client selector
    st.markdown("<div style='font-size:0.8rem;color:#94a3b8;margin-bottom:6px;'>QUICK CLIENT</div>", unsafe_allow_html=True)
    quick_clients = [f"C{100+i}" for i in range(20)]
    selected_quick = st.selectbox("Select client", quick_clients, label_visibility="collapsed")
    if st.button("Load Client", use_container_width=True):
        st.session_state.current_client = selected_quick
        st.rerun()

    st.divider()
    # Health check
    health = _api("GET", "/health")
    if health and health.get("status") == "ok":
        st.markdown("<div class='agent-pill agent-ok'>● Backend Online</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='agent-pill agent-fail'>● Backend Offline</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: CLIENT LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🔍 Client Lookup":
    st.markdown("# 🔍 Client Intelligence Lookup")
    st.markdown("<div style='color:#94a3b8;margin-bottom:1.5rem;'>Run the full AI pipeline to build a Client 360 profile</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        client_id = st.text_input(
            "Client ID",
            value=st.session_state.current_client or "C100",
            placeholder="e.g. C100, C101 ...",
        )
    with col2:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        run_pipeline = st.button("🚀 Run Pipeline", use_container_width=True)

    if run_pipeline or (st.session_state.current_client and client_id != st.session_state.current_client):
        st.session_state.current_client = client_id
        with st.spinner(f"Running 4-agent pipeline for {client_id}..."):
            t0 = time.time()
            result = _api("POST", f"/ingest/{client_id}")
            elapsed = time.time() - t0
        if result:
            st.session_state.pipeline_meta = result
            profile = _api("GET", f"/profile/{client_id}")
            if profile:
                st.session_state.profile = profile
                st.session_state.recommendations = None
                st.success(f"✅ Pipeline completed in {elapsed:.1f}s")

    # Display profile
    profile = st.session_state.profile
    if profile:
        conf = profile.get("merged_confidence_score") or 0.0
        _confidence_bar(conf)

        if profile.get("partial_failure"):
            failed = ", ".join(profile.get("failed_agents", []))
            st.warning(f"⚠️ Partial pipeline failure — agents failed: {failed}")

        # Key metrics row
        st.markdown("<div class='section-header'>Financial Overview</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        with cols[0]:
            st.metric("Monthly Spend", f"₹{(profile.get('monthly_spend') or 0):,}")
        with cols[1]:
            st.metric("Spend Trend", profile.get("spend_trend") or "—")
        with cols[2]:
            st.metric("Avg Transaction", f"₹{(profile.get('avg_txn_size') or 0):,.0f}")
        with cols[3]:
            st.metric("International", "Yes" if profile.get("international_usage") else "No")

        # Top categories
        cats = profile.get("top_categories") or []
        if cats:
            st.markdown("**Top Spending Categories:**")
            for cat in cats:
                _badge(cat.title(), "blue")

        # CRM info
        st.markdown("<div class='section-header'>Client Profile</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='card'>
                <div style='font-size:0.8rem;color:#94a3b8;'>INCOME BAND</div>
                <div style='font-size:1.1rem;font-weight:600;margin-top:4px;'>
                    {profile.get('income_band', '—').upper()}
                </div>
                <div style='font-size:0.72rem;color:#6366f1;'>
                    Source: {profile.get('income_source', '—')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='card'>
                <div style='font-size:0.8rem;color:#94a3b8;'>RISK PROFILE</div>
                <div style='font-size:1.1rem;font-weight:600;margin-top:4px;'>
                    {profile.get('risk_profile', '—').title() if profile.get('risk_profile') else '—'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            churn = profile.get("churn_risk", False)
            churn_color = "#ef4444" if churn else "#10b981"
            st.markdown(f"""
            <div class='card'>
                <div style='font-size:0.8rem;color:#94a3b8;'>CHURN RISK</div>
                <div style='font-size:1.1rem;font-weight:600;margin-top:4px;color:{churn_color};'>
                    {"⚠️ HIGH" if churn else "✓ LOW"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Products & Interactions
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'>Current Products</div>", unsafe_allow_html=True)
            products = profile.get("current_products") or []
            if products:
                for p in products:
                    st.markdown(f"• {p.replace('_', ' ').title()}")
            else:
                st.markdown("_No products on record_")

        with col2:
            st.markdown("<div class='section-header'>Interaction Summary</div>", unsafe_allow_html=True)
            summary = profile.get("interaction_summary") or "_No interactions recorded_"
            sentiment = profile.get("sentiment") or "neutral"
            s_colour = {"positive": "green", "negative": "red", "anxious": "amber"}.get(sentiment, "blue")
            _badge(sentiment.title(), s_colour)
            st.markdown(f"_{summary}_")

        # Anomalies
        anomalies = profile.get("anomalies_flagged") or []
        if anomalies:
            st.markdown("<div class='section-header'>⚠️ Spend Anomalies</div>", unsafe_allow_html=True)
            for a in anomalies:
                st.warning(f"Unusual spend detected: **{a}**")

        # Stale fields warning
        stale = profile.get("stale_fields") or []
        if stale:
            st.info(f"📋 **Stale data fields** (>2 years old): {', '.join(stale)}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💡 Recommendations":
    st.markdown("# 💡 Product Recommendations")
    st.markdown("<div style='color:#94a3b8;margin-bottom:1.5rem;'>AI-generated recommendations powered by hybrid RAG</div>", unsafe_allow_html=True)

    if not st.session_state.current_client:
        st.info("👈 Load a client first from the Client Lookup page.")
        st.stop()

    cid = st.session_state.current_client
    st.markdown(f"**Client:** `{cid}`")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔮 Generate Recommendations", use_container_width=True):
            with st.spinner("Running hybrid RAG + GPT-4o..."):
                recs = _api("POST", f"/recommend/{cid}", json={})
            if recs:
                st.session_state.recommendations = recs
                st.success("Recommendations generated!")

    if st.session_state.recommendations:
        recs = st.session_state.recommendations
        conf = recs.get("confidence_score", 0.0)
        _confidence_bar(conf, "Recommendation Confidence")

        # Client briefing
        st.markdown("<div class='section-header'>Client Briefing</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='card'>
            <p style='margin:0;line-height:1.6;color:#e2e8f0;'>{recs.get('briefing', 'No briefing generated.')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        st.markdown("<div class='section-header'>Product Recommendations</div>", unsafe_allow_html=True)
        rec_items = recs.get("recommendations", [])
        for i, rec in enumerate(rec_items, 1):
            product = rec.get("product", "").replace("_", " ").title()
            reason = rec.get("reason", "")
            source = rec.get("data_source", "profile")
            rec_conf = rec.get("confidence", conf)
            rec_id = f"{recs.get('recommendation_id', 'unknown')}_{i}"

            st.markdown(f"""
            <div class='rec-card'>
                <div class='rec-number'>{i}</div>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-top:4px;'>
                    <div style='font-size:1.05rem;font-weight:700;color:#e2e8f0;'>{product}</div>
                    <div style='font-size:0.8rem;color:#6366f1;font-weight:600;'>{rec_conf*100:.0f}% confidence</div>
                </div>
                <div style='font-size:0.88rem;color:#94a3b8;margin-top:6px;line-height:1.5;'>{reason}</div>
                <div style='margin-top:8px;'>
                    <span class='badge badge-blue'>📊 {source.replace('_',' ').title()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Talking points
        talking_points = recs.get("talking_points", [])
        if talking_points:
            st.markdown("<div class='section-header'>🗣️ RM Talking Points</div>", unsafe_allow_html=True)
            for j, tp in enumerate(talking_points, 1):
                st.markdown(f"""
                <div class='card' style='padding:0.75rem 1.25rem;'>
                    <span style='color:#6366f1;font-weight:600;'>{j}.</span>
                    <span style='color:#e2e8f0;margin-left:8px;'>{tp}</span>
                </div>
                """, unsafe_allow_html=True)

        # Low confidence warning
        if conf < 0.6:
            st.warning(
                "⚠️ **Low data confidence (<60%).** Recommendations are indicative. "
                "Gather more client data before acting on these suggestions."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "👍 Feedback":
    st.markdown("# 👍 Recommendation Feedback")
    st.markdown("<div style='color:#94a3b8;margin-bottom:1.5rem;'>Rate recommendations to improve future personalisation</div>", unsafe_allow_html=True)

    if not st.session_state.recommendations:
        st.info("👈 Generate recommendations first from the Recommendations page.")
        st.stop()

    cid = st.session_state.current_client
    recs = st.session_state.recommendations
    rec_id_base = recs.get("recommendation_id", "unknown")

    st.markdown(f"**Client:** `{cid}` | **Recommendation Session:** `{rec_id_base[:8]}...`")

    for i, rec in enumerate(recs.get("recommendations", []), 1):
        product = rec.get("product", "")
        product_display = product.replace("_", " ").title()
        rec_uid = f"{rec_id_base}_{i}"
        fb_key = f"{cid}_{rec_uid}"

        already_sent = st.session_state.feedback_sent.get(fb_key)

        st.markdown(f"""
        <div class='rec-card'>
            <div style='font-size:1rem;font-weight:600;color:#e2e8f0;'>{i}. {product_display}</div>
            <div style='font-size:0.85rem;color:#94a3b8;margin-top:4px;'>{rec.get('reason','')}</div>
        </div>
        """, unsafe_allow_html=True)

        if already_sent:
            outcome = already_sent
            c = "green" if outcome == "accepted" else "red"
            st.markdown(f"<span class='badge badge-{c}'>✓ Feedback submitted: {outcome}</span>", unsafe_allow_html=True)
        else:
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("👍 Accept", key=f"accept_{rec_uid}", use_container_width=True):
                    result = _api("POST", "/feedback", json={
                        "client_id": cid,
                        "recommendation_id": rec_uid,
                        "product": product,
                        "outcome": "accepted",
                    })
                    if result:
                        st.session_state.feedback_sent[fb_key] = "accepted"
                        st.rerun()
            with col2:
                if st.button("👎 Reject", key=f"reject_{rec_uid}", use_container_width=True):
                    reason = st.text_input("Rejection reason (optional)", key=f"reason_{rec_uid}")
                    result = _api("POST", "/feedback", json={
                        "client_id": cid,
                        "recommendation_id": rec_uid,
                        "product": product,
                        "outcome": "rejected",
                        "rejection_reason": reason,
                    })
                    if result:
                        st.session_state.feedback_sent[fb_key] = "rejected"
                        st.rerun()

        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PIPELINE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Pipeline Monitor":
    st.markdown("# 📊 Pipeline Monitor")
    st.markdown("<div style='color:#94a3b8;margin-bottom:1.5rem;'>Real-time agent status and performance metrics</div>", unsafe_allow_html=True)

    meta = st.session_state.pipeline_meta

    if not meta:
        st.info("Run the pipeline for a client first to see monitor data.")
    else:
        cid = meta.get("client_id", "—")
        conf = meta.get("merged_confidence_score", 0)
        lat = meta.get("total_latency_ms", 0)
        ts = meta.get("pipeline_timestamp", "—")
        failed = meta.get("failed_agents", [])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Client", cid)
        with col2:
            st.metric("Total Latency", f"{lat:.0f}ms")
        with col3:
            status_txt = "✅ Success" if not meta.get("partial_failure") else f"⚠️ Partial ({len(failed)} failed)"
            st.metric("Status", status_txt)

        _confidence_bar(conf, "Merged Pipeline Confidence")

        st.markdown("<div class='section-header'>Agent Status</div>", unsafe_allow_html=True)
        agent_names = ["transaction_agent", "crm_agent", "interaction_agent", "product_agent"]
        agent_icons = {"transaction_agent": "💳", "crm_agent": "👤", "interaction_agent": "💬", "product_agent": "📦"}

        cols = st.columns(4)
        for col, name in zip(cols, agent_names):
            with col:
                icon = agent_icons.get(name, "🤖")
                ok = name not in failed
                cls = "agent-ok" if ok else "agent-fail"
                status_dot = "●"
                label = name.replace("_", " ").title()
                st.markdown(
                    f"<div class='card' style='text-align:center;'>"
                    f"<div style='font-size:1.8rem;'>{icon}</div>"
                    f"<div style='font-size:0.8rem;font-weight:600;margin-top:6px;'>{label}</div>"
                    f"<div class='agent-pill {cls}' style='margin-top:8px;display:inline-flex;'>"
                    f"{status_dot} {'OK' if ok else 'FAILED'}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(f"<div class='section-header'>Last Run</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='card'>
            <div><span style='color:#94a3b8;'>Timestamp:</span> {ts}</div>
            <div><span style='color:#94a3b8;'>Pipeline Mode:</span> {'Partial Success' if meta.get('partial_failure') else 'Full Success'}</div>
            <div><span style='color:#94a3b8;'>Confidence Score:</span> {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Health ping
    st.markdown("<div class='section-header'>System Health</div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Health"):
        health = _api("GET", "/health")
        if health:
            st.json(health)
