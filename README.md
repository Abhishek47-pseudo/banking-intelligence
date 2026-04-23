# AI Banking Client Intelligence Platform (Agentic RAG)

This repository contains a working prototype of an **agent-based Banking Client Intelligence platform**:

- **Backend**: FastAPI service that runs a **4-agent pipeline** to build a *Client 360* profile, stores it in **SQLite**, embeds it into **FAISS**, and serves **hybrid-RAG** recommendations.
- **Frontend**: Streamlit app that lets you ingest a client, view the enriched profile, generate recommendations, and submit feedback.

The code lives in `banking-intelligence/`.

**Architecture deep dive**: see `ARCHITECTURE.md` (agentic pipeline + RAG/LLM specifics).

---

## What this system does

### Build a Client 360 profile (pipeline)
For a given `client_id`, the backend runs **four LangChain tool-calling agents**:

- **Transaction Agent** (`backend/agents/transaction_agent.py`): loads transactions, categorizes, aggregates monthly spend, detects spend trend + anomalies.
- **CRM Agent** (`backend/agents/crm_agent.py`): loads CRM record, standardizes fields, infers missing city from PIN, flags stale fields.
- **Interaction Agent** (`backend/agents/interaction_agent.py`): loads interaction notes, strips PII-like patterns, and uses an LLM prompt to extract structured intents/life-events/sentiment/churn risk.
- **Product Agent** (`backend/agents/product_agent.py`): collaborative filtering over existing stored profiles to detect product gaps (products held by similar clients but not by this client).

Outputs are merged into a single **Client 360** object by the normalizer (`backend/pipeline/normalizer.py`) with a **weighted confidence score**.

### Generate recommendations (Hybrid RAG)
After a profile is ingested, the backend can produce recommendations via:

- **SQL pre-filter** (structured): optional filtering by `income_band`, `city`, `has_churn_risk`
- **FAISS similarity search** over embedded *enriched profiles*
- **MMR re-ranking** for diversity (`backend/rag/reranker.py`)
- **LLM JSON generation** of briefing + up to 3 recommendations + talking points (`backend/llm/generator.py`)

### Feedback loop (lightweight learning)
The Streamlit UI can send feedback (`accepted` / `rejected` / `pending`).

- Feedback events are stored in SQLite (`backend/storage/models.py`)
- Accepted/rejected updates a per-client per-product **relevance score**
- Accepted feedback triggers **re-embedding** and upserting the profile into FAISS
- A weekly scheduled job (APScheduler) can rebuild FAISS from all stored profiles (`backend/feedback/feedback_loop.py`)

---

## Architecture at a glance

- **API**: FastAPI (`backend/main.py`)
  - startup: init DB → load FAISS → warm up LLM → start scheduler
- **Storage**:
  - SQLite database at `./data/banking.db` (default)
  - FAISS indices persisted under `./data/faiss_index/`
- **UI**: Streamlit (`frontend/app.py`)
  - connects to backend through `API_BASE`

Ports:

- Backend: `8000`
- Frontend: `8501`

---

## Repository layout

```text
RAGProject/
  README.md
  banking-intelligence/
    requirements.txt
    .env.example
    backend/
      main.py                 # FastAPI app + routes
      agents/                 # 4 pipeline agents
      pipeline/               # orchestrator + normalizer + enricher
      rag/                    # embeddings + FAISS store + retriever + reranker
      llm/                    # LLM factory + prompts + generator
      storage/                # SQLAlchemy models + async store
      feedback/               # feedback processing + scheduler
      eval/                   # evaluation utilities (accuracy/latency/etc.)
    frontend/
      app.py                  # Streamlit UI
    data/
      mock/
        generate_mock_data.py # generates CSV mock data
        transactions.csv      # generated
        crm.csv               # generated
        interactions.csv      # generated
```

---

## Local development (classic)

### Prereqs
- Python **3.11** recommended

### 1) Install dependencies
From `banking-intelligence/`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configure environment
Create `banking-intelligence/.env` from `.env.example`.

### 3) Generate mock data (optional but useful)
From `banking-intelligence/`:

```bash
python data/mock/generate_mock_data.py
```

### 4) Run the backend
From `banking-intelligence/`:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 5) Run the frontend
From `banking-intelligence/`:

```bash
set API_BASE=http://localhost:8000   # Windows cmd
$env:API_BASE="http://localhost:8000" # PowerShell
streamlit run frontend/app.py --server.port=8501
```

---

## Configuration

The backend reads configuration from environment variables (see `.env.example`):

- **`OPENAI_API_KEY`**: enables OpenAI chat model + OpenAI embeddings
- **`GROQ_API_KEY`**: enables Groq chat model (used as fallback if OpenAI fails)
- **`DATABASE_URL`**: default `sqlite+aiosqlite:///./data/banking.db`
- **`FAISS_INDEX_PATH`**: default `./data/faiss_index`
- **`LOG_LEVEL`**: default `INFO`

### LLM provider fallback (chat)
The backend uses `backend/llm/llm_factory.py`:

- Primary: OpenAI (`gpt-4o`) if `OPENAI_API_KEY` is set
- Fallback: Groq (`llama-3.3-70b-versatile`) if `GROQ_API_KEY` is set
- Runtime fallback: if the primary errors during a call, it retries on the fallback

### Embeddings provider fallback
Embeddings are selected in `backend/rag/embeddings.py`:

- Primary: OpenAI `text-embedding-3-small` (1536 dims)
- Fallback: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384 dims)

**Important**: FAISS indices are *not compatible* across embedding dimension changes.
If you switch embedding providers, delete your FAISS directory (default `./data/faiss_index/`) and rebuild by re-ingesting profiles.

---

## API endpoints

Backend base: `http://localhost:8000`

- **`POST /ingest/{client_id}`**: run the 4-agent pipeline, store in SQLite, upsert into FAISS
- **`GET /profile/{client_id}`**: fetch stored Client 360 profile from SQLite
- **`POST /recommend/{client_id}`**: run hybrid retrieval + LLM JSON generation for recommendations
- **`POST /feedback`**: submit recommendation feedback (queued in background)
- **`GET /health`**: basic system health response

### Example cURL

```bash
curl -X POST http://localhost:8000/ingest/C100
curl http://localhost:8000/profile/C100
curl -X POST http://localhost:8000/recommend/C100 -H "Content-Type: application/json" -d "{}"
curl -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d "{\"client_id\":\"C100\",\"recommendation_id\":\"demo_1\",\"product\":\"mutual_fund\",\"outcome\":\"accepted\"}"
curl http://localhost:8000/health
```

---

## Frontend (Streamlit)

The UI in `frontend/app.py` includes:

- **Client Lookup**: run pipeline + show profile and confidence
- **Recommendations**: generate up to 3 recommendations + talking points
- **Feedback**: accept/reject recommendations
- **Pipeline Monitor**: per-agent status and latency based on latest run metadata

The frontend connects to the backend via:

- `API_BASE` (default `http://localhost:8000`)

---

## Testing

From `banking-intelligence/`:

```bash
pytest -q
```

There is also a simple async pipeline runner:

```bash
python test_pipeline.py
```

---

## Notes / limitations (current version)

- Data sources are currently **mock CSVs** generated under `data/mock/`.
- FAISS metadata filtering is implemented as a **post-filter** (FAISS itself doesn’t natively filter by metadata).
- The product agent loads profiles from SQL using a simplified event-loop fallback approach; it’s designed for prototype-scale data.

