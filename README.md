# Marketing SMS Platform (Maroc Telecom Demo)

Persona-driven, catalog-aware SMS campaign generation. This platform combines:

- **Segmentation CSV data** (usage family + persona)  
- **Offer & handset catalogs** (CSV + optional Qdrant vector retrieval)  
- **RAG compose API (FastAPI)** that builds a structured JSON prompt  
- **Mock or Live LLM generation** (Streamlit UI toggles)  
- **Telegram delivery** of the generated SMS  

> Open the Streamlit UI, select audience + CTA/brand, the app retrieves structured context via the compose FastAPI service, then an LLM (mock templates or remote vLLM) produces a single compliant marketing SMS.

---
## Architecture Overview
```
   +---------------+          +-------------------+         +--------------------+
   |    Browser    |    --->  |    Streamlit UI   |  --->   | FastAPI Compose API|
   +---------------+          +---------+---------+         +---------+----------+
                                          |                             |
                                          | (Qdrant client + CSV)       |
                                          v                             v
                                   +-------------+               +-------------+
                                   |  Qdrant     |               |  LLM (mock  |
                                   |  (cloud)    |               |  or vLLM)   |
                                   +-------------+               +-------------+
```
- **Streamlit UI** (`ui/app.py`): User workflow, persona selection, generation states, preview & Telegram send.
- **FastAPI** (`app/main.py` + `service.py` + `catalog.py`): RAG style composition: loads CSV catalogs & (optionally) queries Qdrant collections (`offres`, `smartphones`). Returns `llm_input_json`.
- **Qdrant**: Optional vector recall for offers/handsets (provide `QDRANT_URL` & key).
- **LLM modes**: 
  - `Mock`: deterministic template rotation (fast preview).
  - `Live inference`: user supplies a public (ngrok) base to a Colab-hosted vLLM server. UI validates `/health` & `/v1/models`.

---
## Key Features
| Feature | Description |
|---------|-------------|
| Persona + Family selection | Dependent dropdowns drive context & insights text. |
| RAG Compose | FastAPI assembles structured JSON from segmentation + catalog. |
| Mock SMS Generation | Rotating curated templates (speed & predictability). |
| Live vLLM Inference | OpenAI-compatible chat payload to finetuned Mistral (ngrok base). |
| Mobile Preview | Styled device mockup with offer metrics. |
| Telegram Delivery | Send generated SMS to an operator via bot. |
| JSON Transparency | Expand to see raw `llm_input_json`. |

---
## Repository Layout
```
app/                # FastAPI compose service
ui/                 # Streamlit front-end
segmentationRAG/    # Segmentation + catalog CSVs
requirements.txt    # Python dependencies
.env.example        # Environment variable template
```

---
## Environment Variables
Copy `.env.example` to `.env` (DO NOT COMMIT `.env`).

| Variable | Component | Purpose |
|----------|-----------|---------|
| `CATALOG_PATH` | FastAPI | Directory holding `offres.csv`, `smartphones.csv` |
| `QDRANT_URL` | FastAPI | Qdrant Cloud base URL (no trailing slash) |
| `QDRANT_API_KEY` | FastAPI | Qdrant API key (header `api-key`) |
| `QDRANT_OFFRES_COLLECTION` | FastAPI | Offer collection name |
| `QDRANT_SMARTPHONES_COLLECTION` | FastAPI | Handset collection name |
| `LLM_BASE_URL` | FastAPI | (Optional) internal LLM service (if not using ngrok UI path) |
| `LLM_API_KEY` | FastAPI/UI | Bearer token for LLM if required |
| `API_BASE_URL` | Streamlit | Public FastAPI compose base URL |
| `LLM_MODE` | Streamlit | Default mode (`Mock` or `Live inference`) |
| `LIVE_LLM_URL` | Streamlit | Pre-seeded live vLLM base (optional) |
| `LIVE_LLM_API_KEY` | Streamlit | API key for live LLM if required |
| `TELEGRAM_BOT_TOKEN` | Streamlit | Bot token for sending SMS |
| `SEGMENTATION_PATH` | Both | Path to segmentation CSV |

---
## Local Development
1. Create virtual env & install deps:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   cp .env.example .env  # fill values
   ```
2. Run FastAPI compose API:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
3. Run Streamlit UI:
   ```bash
   streamlit run ui/app.py
   ```
4. Open: http://localhost:8501 (default Streamlit) — ensure `API_BASE_URL=http://localhost:8000`.

---
## Live Inference (vLLM via Colab + ngrok)
1. Launch provided Colab notebook (link in UI) & run all cells.
2. Copy the printed public ngrok base (e.g. `https://xxxx.ngrok-free.app`).
3. In UI set LLM Mode = Live inference.
4. Paste base URL, validate: the UI hits `/health` then `/v1/models`.
5. On success you see a green chip “Connected to vLLM”.
6. Generate SMS → intermediate states: RAG fetch → LLM generation → success.

Payload (chat/completions):
```json
{
  "model": "/content/drive/MyDrive/mt_iam/mistral7b_iam_sms_lora/merged_4bit",
  "messages": [
    {"role": "system", "content": "French marketing copy constraints ..."},
    {"role": "user", "content": "<llm_input_json>"}
  ],
  "temperature": 0.8,
  "top_p": 0.9,
  "max_tokens": 140
}
```

---
## Qdrant Integration
- Provide `QDRANT_URL` and `QDRANT_API_KEY`.
- Ensure collections named in env exist (`offres`, `smartphones`).
- The compose API optionally performs similarity retrieval to enrich `offer_context`.
- In n8n or other orchestrators, call the FastAPI endpoints **not** Qdrant directly for composed JSON.

---
## Deployment on Render
### FastAPI Service
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Set env vars (omit secrets from repo; fill in dashboard).

### Streamlit UI
- Build: same as above.
- Start: `streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0`
- Set `API_BASE_URL` to public FastAPI URL.

### Optional
- Add custom domain for Streamlit (e.g. `campaigns.example.com`).
- Turn on auto deploys on main branch.

---
## Security & Operational Notes
| Concern | Recommendation |
|---------|----------------|
| Secrets in repo | Keep `.env` out of Git. Rotate any accidentally committed keys immediately. |
| Qdrant key scope | Use least privilege / rotate periodically. |
| LLM endpoint exposure | Ngrok URLs are ephemeral; require API key if you expose longer term. |
| Input validation | Consider adding stricter schema validation in FastAPI for production. |
| Rate limiting | Add a reverse proxy (Cloudflare / API Gateway) if public traffic grows. |

---
## Testing
Add basic unit tests (pytest included) for:
- Template selection logic
- Compose service JSON structure
- Live LLM wrapper (mock the HTTP call)

Run:
```bash
pytest -q
```

---
## Roadmap Ideas
- Persist historical generations (PostgreSQL)
- AB test multiple template variants
- Multi-language expansion
- Analytics dashboard (usage metrics)
- OAuth / SSO for recruiter logins

---
## License
Add a license if distributing (MIT / Apache-2.0 recommended). Currently none specified.

---
## Quick Start TL;DR
```bash
cp .env.example .env  # fill values
pip install -r requirements.txt
uvicorn app.main:app --port 8000 &
streamlit run ui/app.py
```
Open the UI, choose persona, generate SMS. Switch to Live inference if you have a running vLLM endpoint.

---
**Note:** If any secret (e.g. Telegram bot token, Qdrant API key) was ever committed, rotate it now.
