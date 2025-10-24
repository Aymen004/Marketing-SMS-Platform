# Marketing SMS Platform - RAG + Finetuned LLM - n8n automation

Persona-driven, catalog-aware SMS campaign generation. This platform combines real customer usage segmentation, vector-based offer retrieval (Qdrant RAG), and a finetuned telecom-specialized LLM to produce compliant marketing SMS automatically. 
[Streamlit Demo](https://streamlit-ui-mgwb.onrender.com) (You may 30s for Render service warm-up)

https://github.com/user-attachments/assets/e798c2ae-42df-456e-a061-6835ea36304c

## Overview
This project showcases an end-to-end marketing pipeline prototype:
- Transforms raw segmentation + product catalogs into high-conversion contextual SMS.
- Uses Qdrant-powered semantic retrieval to surface the most relevant offer or device for a customer usage profile.
- Leverages a 4-bit quantized, finetuned Mistral-7B-Instruct-v0.2 model trained on prior Maroc Telecom promotional messages (style, tone, constraints).
- Supports both mock deterministic templates (fast iteration) and live inference (Remotely hosted GPU).
- Includes a production-minded automation design (PySpark segmentation → Airflow scheduling → RAG enrichment → LLM generation → Telegram API delivery → scalable orchestration via n8n).
- Deployed Streamlit UI (public demo): https://streamlit-ui-mgwb.onrender.com

---
## Architecture Overview
```
1. User Interaction:
   +-------------------+
   |   Browser/User    |
   | - Selects persona |
   +---------+---------+
             |
             v
2. Streamlit UI:
   +-------------------+
   |  Streamlit        |
   | (ui/app.py)       |
   | - Dropdowns       |
   | - Calls FastAPI   |
   +---------+---------+
             |
             v
3. FastAPI Backend:
   +-------------------+
   | FastAPI           |
   | (app/main.py)     |
   | - Receives request|
   | - Queries Qdrant  |
   +---------+---------+
             |
             v
4. Qdrant RAG:
   +-------------------+
   |     Qdrant        |
   |   (Vector DB)     |
   | - Semantic search |
   | - Returns JSON    |
   +---------+---------+
             |
             v
5. LLM Generation:
   +-------------------+     +-------------------+
   | Finetuned LLM     |     |   Mock Mode       |
   | (Mistral-7B)      |  OR |                   |
   |  Generates SMS    |     |  (Templates)      |
   +-------------------+     +-------------------+
             |
             v
6. Result Display:
   +-------------------+
   |  Streamlit UI     |
   | - SMS Preview     |
   | - Telegram Send   |
   +-------------------+
```
- **Streamlit UI** (`ui/app.py`): User workflow, persona selection, generation states, preview & Telegram send.
- **FastAPI** (`app/main.py` + `service.py` + `catalog.py`): RAG style composition: loads CSV catalogs and queries Qdrant collections (`offres`, `smartphones`). Returns `llm_input_json`.
- **Qdrant**: Vector recall for offers/handsets (provide `QDRANT_URL` & key).
- **LLM modes**: 
  - `Mock`: deterministic template rotation (fast preview).
  - `Live inference`: Live inference using finetuned Mistral model with HuggingFace inference endpoint (Cloud GPU).

---
## Key Features
| Feature | Description |
|---------|-------------|
| Persona + Family selection | Dependent dropdowns drive context & insights text. |
| Qdrant-backed RAG | Retrieves semantically aligned offers/devices for the usage persona. |
| JSON Transparency | Full structured payload shown for debugging & trust. |
| Finetuned LLM | Telecom‑style constrained generation (pricing + CTA discipline). |
| Mock Mode | Fast deterministic templates for UX validation. |
| Mobile Preview | Realistic phone frame + offer metrics. |
| Telegram Delivery | Operator testing channel (extensible). |

---
## Repository Layout
```
app/                # FastAPI compose service
ui/                 # Streamlit front-end
segmentationRAG/    # Segmentation + catalog CSVs + build_index.py (Qdrant)
finetuning/         # LoRA training scripts, dataset samples, merge + 4bit quantization notes
requirements.txt    # Python dependencies
.env.example        # Environment variable template
```

---
## Environment Variables
Copy `.env.example` to `.env` .

| Variable | Component | Purpose |
|----------|-----------|---------|
| `CATALOG_PATH` | FastAPI | Directory holding `offres.csv`, `smartphones.csv` |
| `QDRANT_URL` | FastAPI | Qdrant Cloud base URL (no trailing slash) |
| `QDRANT_API_KEY` | FastAPI | Qdrant API key (header `api-key`) |
| `QDRANT_OFFRES_COLLECTION` | FastAPI | Offer collection name |
| `QDRANT_SMARTPHONES_COLLECTION` | FastAPI | Handset collection name |
| `LLM_BASE_URL` | FastAPI | (Optional) internal LLM service (if not using ngrok UI path) |
| `API_BASE_URL` | Streamlit | Public FastAPI compose base URL |
| `LLM_MODE` | Streamlit | Default mode (`Mock` or `Live inference`) |
| `LIVE_LLM_URL` | Streamlit | Pre-seeded live vLLM base (optional) |
| `LIVE_LLM_API_KEY` | Streamlit | API key for live LLM if required |
| `TELEGRAM_BOT_TOKEN` | Streamlit | Bot token for sending SMS |
| `SEGMENTATION_PATH` | Both | Path to segmentation CSV |
---
## Customer Segmentation (PySpark + Airflow)
A PySpark data processing job (prototype DAG in Airflow) ingests raw telecom usage KPIs (data volume tiers, voice minute distributions, SMS counts, roaming events, handset metadata) and:
- Normalizes & buckets usage signals
- Assigns usage family (famille) + refined persona labels
- Outputs a clean, denormalized segmentation_sms.csv consumed by both:
  - Streamlit UI (dropdown population)
  - FastAPI compose service (to enrich llm_input_json)

Rationale: fast iterative offline segmentation without overfitting; Airflow scheduling makes the audience refresh reproducible.

## Qdrant Vector Index (build_index.py)
The script segmentationRAG/build_index.py:
- Loads offres.csv & smartphones.csv
- Builds semantic text representations (SentenceTransformer intfloat/multilingual-e5-base)
- Creates (or recreates) two Qdrant collections: offres and smartphones
- Upserts vector embeddings + rich payload (price, validity, volumes, brand, model)
- Normalized fields (ints/floats) enable post-retrieval filtering / scoring
Outcome: The compose FastAPI service can perform similarity search to pull the most contextually aligned offer or handset for a given persona/famille before prompting the LLM.

> This vector layer upgrades simple CSV lookups into intent/semantic-aware retrieval, improving personalization quality for the generated SMS.

## RAG & LLM Stack
Core intelligence layer:
- Retrieval: Qdrant (semantic vectors) augments CSV catalog matches to refine offer/handset context for each audience (usage family + persona).
- Composition: FastAPI service merges segmentation record + retrieved offer context → structured `llm_input_json`.
- Generation: Two modes  
  - Mock rotation (deterministic, zero cost)  
  - Live inference: Finetuned Mistral-7B-Instruct-v0.2 via HuggingFace endpoint (4-bit quantized for cost efficiency).

## Finetuning & Optimization
- Base model: Mistral-7B-Instruct-v0.2
- Domain corpus: Historical Maroc Telecom SMS & promotional copy (cleaned, normalized, deduped)
- Objective: Style adaptation (tone, brevity, CTA discipline), compliance (price, volume, duration fidelity)
- Techniques:
  - LoRA fine-tune → merged & 4-bit quantized for deployment efficiency
  - Prompt schema: System role enforces constraints; user payload = JSON context
  - Deterministic rotation fallback ensures UX continuity when LLM unavailable

## End-to-End Flow
```
Segmentation (PySpark) → Orchestrated (Airflow) → FastAPI Compose (RAG + catalogs) 
    → LLM Inference (finetuned Mistral) → SMS Preview (Streamlit) → Delivery (Telegram Bot)
```
## Live Demo
- Streamlit UI (deployed):   
  Explore: persona selection → contextual insights → RAG JSON → generated SMS → send via Telegram.


## Automated Pipeline n8n Prototype 
An automated orchestration prototype (n8n + Airflow) was used in production :
1. Batch customer clustering / segmentation (PySpark jobs)
2. Scheduled via Airflow (daily audience refresh)
3. n8n workflow calls Compose API (RAG enrichment for offers/handsets)
4. 4-bit finetuned LLM generates targeted SMS
5. Delivery channel  (Telegram API prototype → extensible to SMS gateways)

---
## Quick Start TL;DR
```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --port 8000 &
streamlit run ui/app.py
```
