"""Catalog loading helpers."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CatalogData:
    offres: List[Dict]
    smartphones: List[Dict]
    version: Optional[str]


def _detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        if ";" in sample and "," not in sample:
            return ";"
        return ","


def _read_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        sample = handle.read(1024)
        handle.seek(0)
        delimiter = _detect_delimiter(sample)
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader)


def load_catalog(base_path: Path) -> CatalogData:
    offres_path = base_path / "offres.csv"
    smartphones_path = base_path / "smartphones.csv"

    offres = _read_csv(offres_path)
    smartphones = _read_csv(smartphones_path)

    version = None
    if offres:
        version = offres[0].get("version_catalogue") or version
    if smartphones and not version:
        version = smartphones[0].get("version_catalogue")

    return CatalogData(offres=offres, smartphones=smartphones, version=version)