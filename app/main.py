"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .catalog import load_catalog
from .config import settings
from .models import (
    ComposeOffrePayload,
    ComposeResponse,
    ComposeSmartphonePayload,
)
from .service import ComposeService, to_llm_response

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IAM Compose API",
    description="Segmentation -> RAG -> LLM payload composer",
    version="0.2.0",
)

# Add CORS middleware to allow requests from Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (can be restricted to specific Streamlit Cloud URLs)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_service() -> ComposeService:
    catalog = load_catalog(settings.catalog_path)
    return ComposeService(
        catalog=catalog,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        offres_collection=settings.qdrant_offres_collection,
        smartphones_collection=settings.qdrant_smartphones_collection,
    )


ServiceDep = Annotated[ComposeService, Depends(get_service)]


@app.get("/")
async def root(request: Request):
    logger.info(f"Wake-up request from {request.client.host}")
    logger.info(f"Headers: {dict(request.headers)}")
    return {"message": "Backend is awake"}


@app.post("/compose/offre", response_model=ComposeResponse)
def compose_offre(payload: ComposeOffrePayload, service: ServiceDep) -> ComposeResponse:
    llm_payload = service.compose_offre(
        persona=payload.persona,
        famille=payload.famille,
        cta=payload.tag_offre,
    )
    return ComposeResponse(**to_llm_response(llm_payload))


@app.post("/compose/smartphone", response_model=ComposeResponse)
def compose_smartphone(payload: ComposeSmartphonePayload, service: ServiceDep) -> ComposeResponse:
    llm_payload = service.compose_smartphone(
        persona=payload.persona,
        famille=payload.famille,
        hset_brand=payload.hset_brand,
    )
    return ComposeResponse(**to_llm_response(llm_payload))