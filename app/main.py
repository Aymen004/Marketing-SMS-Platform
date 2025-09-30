"""FastAPI application entrypoint."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI

from .catalog import load_catalog
from .config import settings
from .models import (
    ComposeOffrePayload,
    ComposeResponse,
    ComposeSmartphonePayload,
    HealthResponse,
)
from .service import ComposeService, to_llm_response

app = FastAPI(
    title="IAM Compose API",
    description="Segmentation -> RAG -> LLM payload composer",
    version="0.2.0",
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


@app.get("/health", response_model=HealthResponse)
def health(service: ServiceDep) -> HealthResponse:
    status = "ok"
    qdrant_status = "offline"
    if service.qdrant:
        qdrant_status = "connected"
    return HealthResponse(status=status, qdrant=qdrant_status, catalog_version=service.catalog.version)


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