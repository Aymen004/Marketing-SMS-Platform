# build_index.py
import csv, os, sys
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL","http://localhost:6333")
COL_OFFRES = "offres"
COL_SMARTS = "smartphones"
EMB_MODEL  = "intfloat/multilingual-e5-base"

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = []
        for r in rdr:
            rows.append({k:(v.strip() if isinstance(v,str) else v) for k,v in r.items()})
        return rows

def to_int(x):
    try:
        if x is None or x=="":
            return None
        return int(float(str(x).replace(",",".")))
    except:
        return None

def offre_text(r):
    # Texte sémantique stable
    parts = [
        f"CTA {r.get('cta','')}",
        r.get("famille",""),
        r.get("libelle",""),
        f"volume {r.get('volume','')}",
        f"minutes {r.get('minutes','')}",
        f"sms {r.get('sms','')}",
        f"validite {r.get('validite_jours','')} jours",
        f"prix {r.get('prix_dh','')} DH",
        r.get("zone","") or ""
    ]
    return " | ".join([p for p in parts if p])

def smart_text(r):
    parts = [
        f"SMARTPHONE {r.get('marque','')} {r.get('modele','')}",
        f"capacite {r.get('capacite','')}",
        f"prix {r.get('prix_dh','')} DH",
        f"gamme {r.get('gamme','')}",
    ]
    return " | ".join([p for p in parts if p])

def ensure_collection(client, name, size):
    names = [c.name for c in client.get_collections().collections]
    if name not in names:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE)
        )

def upsert_points(client, name, vectors, payloads, start_id=1):
    points = []
    for i,(v,p) in enumerate(zip(vectors, payloads), start=start_id):
        points.append({"id": i, "vector": v, "payload": p})
    client.upsert(collection_name=name, points=points)

if __name__ == "__main__":
    offres_csv  = sys.argv[1]  # ex: data/offres.csv
    smarts_csv  = sys.argv[2]  # ex: data/smartphones.csv

    m  = SentenceTransformer(EMB_MODEL)
    qc = QdrantClient(url=QDRANT_URL)

    # -------- OFFRES ----------
    offres = load_csv(offres_csv)
    ensure_collection(qc, COL_OFFRES, size=m.get_sentence_embedding_dimension())
    texts_o = [ "query: " + offre_text(r) for r in offres ]
    vecs_o  = m.encode(texts_o, normalize_embeddings=True).tolist()
    payloads_o = []
    for r in offres:
        payloads_o.append({
            "type":"offre",
            "id": r.get("id"),
            "cta": r.get("cta"),
            "famille": r.get("famille"),
            "libelle": r.get("libelle"),
            "volume": to_int(r.get("volume")),
            "minutes": to_int(r.get("minutes")),
            "sms": to_int(r.get("sms")),
            "validite_jours": to_int(r.get("validite_jours")),
            "prix_dh": float(str(r.get("prix_dh","")).replace(",", ".")) if r.get("prix_dh") else None,
            "zone": r.get("zone"),
            "link": r.get("link"),
            "version_catalogue": r.get("version_catalogue"),
            "raw": r
        })
    upsert_points(qc, COL_OFFRES, vecs_o, payloads_o, start_id=1)

    # -------- SMARTPHONES ----------
    smarts = load_csv(smarts_csv)
    ensure_collection(qc, COL_SMARTS, size=m.get_sentence_embedding_dimension())
    texts_s = [ "query: " + smart_text(r) for r in smarts ]
    vecs_s  = m.encode(texts_s, normalize_embeddings=True).tolist()
    payloads_s = []
    for r in smarts:
        payloads_s.append({
            "type":"smartphone",
            "id": r.get("id"),
            "marque": (r.get("marque") or "").upper(),
            "modele": r.get("modele"),
            "capacite": r.get("capacite"),
            "prix_dh": float(str(r.get("prix_dh","")).replace(",", ".")) if r.get("prix_dh") else None,
            "gamme": r.get("gamme"),
            "version_catalogue": r.get("version_catalogue"),
            "link": r.get("link"),
            "raw": r
        })
    upsert_points(qc, COL_SMARTS, vecs_s, payloads_s, start_id=100001)

    print("✅ Qdrant index chargé (offres + smartphones).")