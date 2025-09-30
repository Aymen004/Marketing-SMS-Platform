"""Pydantic data models used by the API."""
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class ComposeOffrePayload(BaseModel):
    persona: str = Field(..., description="Marketing persona, e.g. CHURN_Competiteur")
    famille: str = Field(..., description="Marketing family such as RISQUE_Churn")
    tag_offre: str = Field(..., description="CTA code like *3 or *78")


class ComposeSmartphonePayload(BaseModel):
    persona: str = Field(..., description="Equipment persona, e.g. OPPORTUNITE_AchatSmartphone")
    famille: str = Field(..., description="Family, usually OPPORTUNITE_Achat_Equipement")
    hset_brand: str = Field("", description="Existing handset brand to bias selection")


class ComposeResponse(BaseModel):
    llm_input_json: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    qdrant: Optional[str] = None
    catalog_version: Optional[str] = None