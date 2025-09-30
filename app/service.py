"""Business logic for compose endpoints."""
from __future__ import annotations

import json
import logging
import random
from datetime import date
from typing import Any, Dict, List, Optional

from dateutil.relativedelta import relativedelta

try:  # optional dependency for local tests
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:  # pragma: no cover - executed when qdrant-client missing
    QdrantClient = None  # type: ignore
    rest = None  # type: ignore

from .catalog import CatalogData

LOGGER = logging.getLogger(__name__)

MONTHS_FR = [
    "janvier",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
]


class ComposeService:
    ROTATING_CTAS = {"*6", "*5", "*4", "*3", "*2", "*1", "*22", "*88", "Forfait_Business"}

    def __init__(
        self,
        catalog: CatalogData,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        offres_collection: str = "offres",
        smartphones_collection: str = "smartphones",
    ) -> None:
        self.catalog = catalog
        self.offres_collection = offres_collection
        self.smartphones_collection = smartphones_collection
        self.qdrant_url = qdrant_url
        self.qdrant: Optional[QdrantClient] = None
        if qdrant_url and QdrantClient is not None:
            try:
                self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=2.0)
                self.qdrant.get_collections()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Unable to reach Qdrant at %s: %s", qdrant_url, exc)
                self.qdrant = None

    # ---------- Helpers ----------
    @staticmethod
    def _deadline_eom_fr() -> str:
        today = date.today()
        end = (today.replace(day=1) + relativedelta(months=1)) - relativedelta(days=1)
        return f"{end.day} {MONTHS_FR[end.month - 1]}"

    @staticmethod
    def _fmt_volume(value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        try:
            mb = float(value)
        except ValueError:
            return None
        if mb >= 1024:
            gb = mb / 1024
            if abs(gb - round(gb)) < 1e-6:
                return f"{int(round(gb))} Go"
            return f"{gb:.1f} Go".replace(".0", "")
        return f"{int(mb)} Mo"

    @staticmethod
    def _fmt_minutes(value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        try:
            mins = int(float(value))
        except ValueError:
            return None
        if mins < 0:
            return "illimite"
        return f"{mins} min"

    @staticmethod
    def _pick_discount(price: Optional[float]) -> Optional[int]:
        if price is None:
            return None
        promo = int(round(price * 0.75))
        return max(promo, 1)

    @staticmethod
    def _norm_brand(value: str) -> str:
        lookup = {
            "IPHONE": "APPLE",
            "APPLE": "APPLE",
            "SAMSUNG": "SAMSUNG",
            "XIAOMI": "XIAOMI",
            "OPPO": "OPPO",
            "HONOR": "HONOR",
            "TECNO": "TECNO",
        }
        upper = (value or "").upper()
        for key, mapped in lookup.items():
            if key in upper:
                return mapped
        return ""

    # ---------- Offer composition ----------
    def compose_offre(self, persona: str, famille: str, cta: str) -> Dict:
        candidates = self._query_offre(famille, cta)
        offer = {"offre": f"Pass {cta}"}
        price = None
        if candidates:
            ranked = self._rank_offres(candidates, persona, famille, cta)
            if ranked:
                if persona == "PROFIL_Econome":
                    chosen = min(
                        ranked,
                        key=lambda item: self._safe_cast(item.get("prix_dh"))
                        if self._safe_cast(item.get("prix_dh")) is not None
                        else float("inf"),
                    )
                elif famille != "OPPORTUNITE_Achat_Equipement" and len(ranked) > 1:
                    chosen = random.choice(ranked)
                else:
                    chosen = ranked[0]
                offer["offre"] = chosen.get("libelle") or offer["offre"]
                price = self._safe_cast(chosen.get("prix_dh"))
                volume = self._fmt_volume(chosen.get("volume"))
                minutes = self._fmt_minutes(chosen.get("minutes"))
                sms = chosen.get("sms")
                validite = chosen.get("validite_jours")
                if volume:
                    offer["volume"] = volume
                if minutes:
                    offer["minutes"] = minutes
                if sms:
                    offer["sms"] = int(float(sms))
                if validite:
                    offer["validite"] = f"{int(float(validite))} jours"
                if price is not None:
                    offer["prix_dh"] = price
                if chosen.get("zone"):
                    offer["destinations"] = chosen.get("zone")
                if chosen.get("link"):
                    offer["details"] = chosen.get("link")
        llm_input = self._base_llm_payload(
            persona=persona,
            famille=famille,
            cta=cta,
            offer_context=offer,
            link=offer.get("details") or "https://bit.ly/Recharge_IAM",
            price=price,
        )
        return llm_input

    def compose_smartphone(self, persona: str, famille: str, hset_brand: str) -> Dict:
        brand = self._norm_brand(hset_brand)
        candidates = self._query_smartphone(brand)
        chosen = self._select_smartphone_candidate(candidates, persona, brand)

        offer = {"offre": "Smartphone"}
        price = None
        cta = ""
        link = "https://offres.iam.ma/smartphones"
        if chosen:
            offer["modele"] = chosen.get("modele")
            offer["capacite"] = chosen.get("capacite")
            price = self._safe_cast(chosen.get("prix_dh"))
            if price is not None:
                offer["prix_dh"] = price
            if chosen.get("link"):
                link = chosen["link"]
            if chosen.get("cta"):
                cta = chosen.get("cta", "")

        llm_input = self._base_llm_payload(
            persona=persona,
            famille=famille,
            cta=cta,
            offer_context=offer,
            link=link,
            price=price,
        )
        return llm_input

    # ---------- Internals ----------
    def _base_llm_payload(
        self,
        persona: str,
        famille: str,
        cta: str,
        offer_context: Dict,
        link: str,
        price: Optional[float],
    ) -> Dict:
        payload = {
            "persona": persona,
            "famille": famille,
            "cta": cta,
            "deadline": self._deadline_eom_fr(),
            "offer_context": offer_context,
            "promo_context": None,
            "links": {"details": link},
        }
        discount = self._pick_discount(price)
        if discount:
            payload["promo_context"] = {"prix_promo_dh": discount}
        return payload

    def _rank_offres(self, candidates: List[Dict], persona: str, famille: str, cta: str) -> List[Dict]:
        if not candidates:
            return []

        def volume_mb(item: Dict) -> float:
            try:
                return float(item.get("volume") or 0)
            except (TypeError, ValueError):
                return 0.0

        if famille == "USAGE_Internet" and persona in {"PROFIL_Internet", "PROFIL_ReseauxSociaux"}:
            return sorted(candidates, key=volume_mb, reverse=True)

        if famille.startswith("RISQUE_") or persona.startswith("CHURN_") or persona == "PROFIL_Econome":
            return sorted(candidates, key=lambda item: self._safe_cast(item.get("prix_dh")) or 0)

        if cta in self.ROTATING_CTAS:
            return list(candidates)

        return candidates

    def _query_offre(self, famille: str, cta: str) -> List[Dict]:
        candidates: List[Dict] = []
        if self.qdrant and rest is not None:
            try:
                candidates = self._query_qdrant(
                    collection=self.offres_collection,
                    filters=[
                        rest.FieldCondition(key="type", match=rest.MatchValue(value="offre")),
                        rest.FieldCondition(key="cta", match=rest.MatchValue(value=cta)),
                        rest.FieldCondition(key="famille", match=rest.MatchValue(value=famille)),
                    ],
                )
                if not candidates:
                    candidates = self._query_qdrant(
                        collection=self.offres_collection,
                        filters=[
                            rest.FieldCondition(key="type", match=rest.MatchValue(value="offre")),
                            rest.FieldCondition(key="cta", match=rest.MatchValue(value=cta)),
                        ],
                    )
            except Exception as exc:  # pragma: no cover - fallback path
                LOGGER.warning("Qdrant query failed, reverting to local catalogue: %s", exc)
        if not candidates:
            candidates = self._local_offres(famille=famille, cta=cta)
        return candidates

    def _query_smartphone(self, brand: str) -> List[Dict]:
        candidates: List[Dict] = []
        if self.qdrant and rest is not None:
            try:
                filters = [rest.FieldCondition(key="type", match=rest.MatchValue(value="smartphone"))]
                if brand:
                    filters.append(rest.FieldCondition(key="marque", match=rest.MatchValue(value=brand)))
                candidates = self._query_qdrant(
                    collection=self.smartphones_collection,
                    filters=filters,
                )
            except Exception as exc:  # pragma: no cover - fallback path
                LOGGER.warning("Qdrant smartphone query failed, fallback to local list: %s", exc)
        if not candidates:
            candidates = self._local_smartphones(brand=brand)
        return candidates

    def _query_qdrant(self, collection: str, filters: List[Any]) -> List[Dict]:
        if not self.qdrant or rest is None:
            return []
        query_filter = rest.Filter(must=filters)
        try:
            points, _ = self.qdrant.scroll(
                collection_name=collection,
                scroll_filter=query_filter,
                limit=50,
            )
        except Exception as exc:  # pragma: no cover - network failure
            LOGGER.debug("Qdrant scroll error: %s", exc)
            return []
        payloads = [getattr(point, "payload", {}) or {} for point in points]
        payloads.sort(key=lambda item: self._safe_cast(item.get("prix_dh")) or 0)
        return payloads

    def _local_offres(self, famille: str, cta: str) -> List[Dict]:
        strict = [o for o in self.catalog.offres if o.get("cta") == cta and o.get("famille") == famille]
        fallback = [o for o in self.catalog.offres if o.get("cta") == cta]
        candidates = strict or fallback
        candidates.sort(key=lambda item: self._safe_cast(item.get("prix_dh")) or 0)
        return candidates

    def _local_smartphones(self, brand: str) -> List[Dict]:
        records = self.catalog.smartphones[:]
        if brand:
            filtered = [s for s in records if s.get("marque", "").upper() == brand]
            if filtered:
                records = filtered
        records.sort(key=lambda item: self._safe_cast(item.get("prix_dh")) or 0)
        return records

    @staticmethod
    def _safe_cast(value: Optional[str]) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _select_smartphone_candidate(self, candidates: List[Dict], persona: str, brand: str) -> Optional[Dict]:
        if not candidates:
            return None

        priced_pairs = [(item, self._safe_cast(item.get("prix_dh"))) for item in candidates]
        priced_values = [pair for pair in priced_pairs if pair[1] is not None]

        if priced_values:
            sorted_asc = [item for item, _ in sorted(priced_values, key=lambda pair: pair[1])]  # cheapest first
        else:
            sorted_asc = [item for item, _ in priced_pairs]

        sorted_desc = list(reversed(sorted_asc))
        pool: List[Dict]

        if persona == "OPPORTUNITE_AchatSmartphone":
            pool = sorted_asc[:15]
        elif persona == "OPPORTUNITE_PerformanceSmartphone":
            pool = sorted_desc[:15]
        elif persona == "OPPORTUNITE_AchatNouveaute" and brand:
            brand_items = [item for item in candidates if (item.get("marque", "").upper() == brand)]
            if brand_items:
                brand_priced = [(item, self._safe_cast(item.get("prix_dh"))) for item in brand_items]
                brand_with_price = [pair for pair in brand_priced if pair[1] is not None]
                if brand_with_price:
                    pool = [item for item, _ in sorted(brand_with_price, key=lambda pair: pair[1], reverse=True)[:3]]
                else:
                    pool = brand_items[:3]
            else:
                pool = []
        else:
            pool = sorted_asc

        if not pool:
            pool = sorted_asc
        if not pool:
            return None
        return random.choice(pool)

def to_llm_response(payload: Dict) -> Dict:
    return {
        "llm_input_json": json.dumps(payload, ensure_ascii=False),
        "metadata": {
            "deadline": payload.get("deadline", ""),
            "cta": payload.get("cta", ""),
        },
    }
