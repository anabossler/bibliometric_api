"""
OpenAlex → VOSviewer exporter OPTIMIZADO
- CSV "Scopus-like" compatible con VOSviewer (pestaña Scopus)
- MMR avanzado con SBERT o TF-IDF (fallback)
- OPTIMIZADO para evitar Out of Memory en Neo4j

Requisitos pip (instalar dentro del venv):
pip install pandas numpy python-dotenv neo4j scikit-learn sentence-transformers

Si no quieres SBERT: omite sentence-transformers y el script usa TF-IDF.
"""

import os
import sys
import re
import csv
import time
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SBERT opcional
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ================== CONFIG ==================
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD  = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DB   = os.getenv("NEO4J_DATABASE", "circulareconomy")

# Labels (por si tu grafo usa otros)
PAPER_LABEL   = os.getenv("PAPER_LABEL", "Paper")
AUTHOR_LABEL  = os.getenv("AUTHOR_LABEL", "Author")
JOURNAL_LABEL = os.getenv("JOURNAL_LABEL", "Journal")
CONCEPT_LABEL = os.getenv("CONCEPT_LABEL", "Concept")

# Relación alternativas (si tu esquema usa otros nombres)
AUTH_RELS   = ["AUTHORED", "AUTHORED_BY"]
JOUR_RELS   = ["PUBLISHED_IN", "APPEARS_IN"]
CONC_RELS   = ["HAS_CONCEPT", "HAS_FIELD_OF_STUDY"]

# Tamaños OPTIMIZADOS para evitar OOM
PAGE_SIZE   = int(os.getenv("EXPORT_BATCH_SIZE", "100"))   # REDUCIDO de 300 a 100
MMR_LIMIT   = int(os.getenv("MMR_MAX_ROWS", "3000"))

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "vosviewer_exports")
os.makedirs(OUT_DIR, exist_ok=True)

# ================== LIMPIEZA / UTIL ==================

_ALIAS = {
    "circular economy": "Circular Economy",
    "sustainability": "Sustainability",
    "zero waste": "Zero Waste",
    "green chemistry": "Green Chemistry",
    "waste management": "Waste Management",
    "recycling": "Recycling",
    "life cycle assessment": "Life Cycle Assessment",
}

_STOPLIKE = {
    "article","review","paper","study","research","introduction","conclusion",
    "methods","results","dataset","analysis","human","humans","medicine",
    "medical","clinical"
}

def clean_concepts(concepts_list) -> str:
    if not concepts_list or not isinstance(concepts_list, list):
        return ""
    seen, out = set(), []
    for c in concepts_list:
        if not isinstance(c, str):
            continue
        w = c.strip().lower()
        if not w or w in _STOPLIKE:
            continue
        if w in _ALIAS:
            w2 = _ALIAS[w]
        else:
            w2 = " ".join(s.capitalize() for s in w.split())
        if len(w2.split()) > 6:
            continue
        k = w2.lower()
        if k not in seen:
            seen.add(k)
            out.append(w2)
    return "; ".join(out)

def clean_abstract(text: str) -> str:
    if not text:
        return ""
    patt = [
        r"©\s*\d{4}.*?rights reserved\.?",
        r"all rights reserved\.?",
        r"this is an open access article.*?license\.",
        r"creative commons.*?license",
        r"supplementary material.*",
        r"https?://\S+|doi:\s*\S+|10\.\d{4,9}/\S+",
        r"conflict of interest.*",
    ]
    s = re.sub(r"\s+", " ", text).strip()
    for p in patt:
        s = re.sub(p, " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def join_semi(items: List[str]) -> str:
    items = [x for x in (items or []) if isinstance(x, str) and x.strip()]
    return "; ".join(dict.fromkeys([x.strip() for x in items]))

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ================== QUERIES OPTIMIZADAS ==================

COUNT_QUERY = f"""
MATCH (p:{PAPER_LABEL})
WHERE p.title IS NOT NULL
RETURN count(p) AS total
"""

# QUERY OPTIMIZADA: Sin COLLECT masivo, usando relaciones directas
PAGED_QUERY_OPTIMIZED = f"""
MATCH (p:{PAPER_LABEL})
WHERE p.title IS NOT NULL
WITH p
ORDER BY p.cited_by_count DESC
SKIP $offset
LIMIT $limit
RETURN
  p.openalex_id           AS openalex_id,
  p.doi                   AS doi,
  p.title                 AS title,
  p.publication_year      AS year,
  p.cited_by_count        AS cited_by,
  p.abstract              AS abstract,
  p.type                  AS document_type,
  p.language              AS language
"""

# Queries separadas para relaciones (más eficiente)
GET_AUTHORS_QUERY = f"""
MATCH (p:{PAPER_LABEL} {{openalex_id: $pid}})
OPTIONAL MATCH (p)<-[:AUTHORED]-(a:{AUTHOR_LABEL})
RETURN collect(DISTINCT a.display_name) AS authors,
       collect(DISTINCT a.openalex_id) AS author_ids
LIMIT 1
"""

GET_JOURNAL_QUERY = f"""
MATCH (p:{PAPER_LABEL} {{openalex_id: $pid}})
OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:{JOURNAL_LABEL})
RETURN j.display_name AS journal
LIMIT 1
"""

GET_CONCEPTS_QUERY = f"""
MATCH (p:{PAPER_LABEL} {{openalex_id: $pid}})
OPTIONAL MATCH (p)-[:HAS_CONCEPT]->(c:{CONCEPT_LABEL})
RETURN collect(DISTINCT c.display_name) AS concepts
LIMIT 1
"""

MMR_QUERY = f"""
MATCH (p:{PAPER_LABEL})
WHERE p.title IS NOT NULL
  AND p.abstract IS NOT NULL
  AND p.abstract <> ''
WITH p
ORDER BY p.cited_by_count DESC
LIMIT $limit
OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:{JOURNAL_LABEL})
OPTIONAL MATCH (p)<-[:AUTHORED]-(a:{AUTHOR_LABEL})
WITH p, j, collect(DISTINCT a.display_name) AS authors
RETURN
  p.openalex_id      AS openalex_id,
  p.title            AS title,
  p.publication_year AS year,
  p.cited_by_count   AS cited_by,
  p.abstract         AS abstract,
  j.display_name     AS journal,
  authors
"""

# ================== NEO4J ==================

def connect_driver():
    drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))
    with drv.session(database=NEO4J_DB) as s:
        s.run("RETURN 1").consume()
    return drv

def db_stats(drv) -> Dict[str, int]:
    stats = {}
    with drv.session(database=NEO4J_DB) as s:
        for L in [PAPER_LABEL, AUTHOR_LABEL, JOURNAL_LABEL, "Institution", CONCEPT_LABEL, "Country"]:
            try:
                c = s.run(f"MATCH (n:{L}) RETURN count(n) AS c").single()["c"]
            except Exception:
                c = 0
            stats[L] = c
        try:
            cabs = s.run(f"MATCH (p:{PAPER_LABEL}) WHERE p.abstract IS NOT NULL AND p.abstract<>'' RETURN count(p) AS c").single()["c"]
        except Exception:
            cabs = 0
        try:
            cites = s.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()["c"]
        except Exception:
            cites = 0
    stats["Papers_with_abstract"] = cabs
    stats["Citations"] = cites
    return stats

# ================== EXPORT SCOPUS-LIKE OPTIMIZADO ==================

def to_scopus_row(rec: dict) -> dict:
    """Convierte un registro a formato Scopus"""
    authors = join_semi(rec.get("authors_raw") or [])
    author_ids = join_semi([str(x) for x in (rec.get("author_ids_raw") or []) if x])
    concepts = clean_concepts(rec.get("concepts_raw") or [])
    
    return {
        "Authors": authors,
        "Author(s) ID": author_ids,
        "Title": rec.get("title") or "",
        "Year": rec.get("year") or "",
        "Source title": rec.get("journal") or "",
        "Abstract": clean_abstract(rec.get("abstract") or ""),
        "Cited by": rec.get("cited_by") or 0,
        "DOI": rec.get("doi") or "",
        "Author Keywords": concepts,
        "Affiliations": "",
        "Volume": "",
        "Issue": "",
        "Page start": "",
        "Page end": "",
        "EID": rec.get("openalex_id") or "",
    }

def fetch_paper_relations(session, paper_id: str) -> dict:
    """Obtiene autores, journal y conceptos de un paper (queries separadas)"""
    result = {"authors_raw": [], "author_ids_raw": [], "journal": "", "concepts_raw": []}
    
    try:
        # Autores
        auth_data = session.run(GET_AUTHORS_QUERY, pid=paper_id).single()
        if auth_data:
            result["authors_raw"] = auth_data["authors"] or []
            result["author_ids_raw"] = auth_data["author_ids"] or []
        
        # Journal
        jour_data = session.run(GET_JOURNAL_QUERY, pid=paper_id).single()
        if jour_data and jour_data["journal"]:
            result["journal"] = jour_data["journal"]
        
        # Conceptos
        conc_data = session.run(GET_CONCEPTS_QUERY, pid=paper_id).single()
        if conc_data:
            result["concepts_raw"] = conc_data["concepts"] or []
    except Exception as e:
        print(f"  ⚠️  Error obteniendo relaciones para {paper_id}: {e}")
    
    return result

def export_scopus_csv(drv, hard_limit: Optional[int] = None) -> str:
    """Exporta papers a CSV formato Scopus con queries optimizadas"""
    
    with drv.session(database=NEO4J_DB) as ses:
        total = ses.run(COUNT_QUERY).single()["total"]
    
    if hard_limit:
        total = min(total, hard_limit)

    ts = now_stamp()
    out_path = os.path.join(OUT_DIR, f"scopus_vosviewer_{ts}.csv")

    exported = 0
    offset = 0

    fieldnames = [
        "Authors","Author(s) ID","Title","Year","Source title","Abstract",
        "Cited by","DOI","Author Keywords","Affiliations","Volume","Issue",
        "Page start","Page end","EID"
    ]
    
    print(f"Total registros a exportar: {total:,}")
    print(f"Tamaño de lote: {PAGE_SIZE}")
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        
        with drv.session(database=NEO4J_DB) as ses:
            while exported < total:
                lim = min(PAGE_SIZE, total - exported)
                
                try:
                    # Query principal (solo papers, sin JOINs pesados)
                    res = ses.run(PAGED_QUERY_OPTIMIZED, offset=offset, limit=lim)
                    batch = res.data()
                    
                    if not batch:
                        break
                    
                    # Para cada paper, obtener relaciones en queries separadas
                    for paper in batch:
                        paper_id = paper.get("openalex_id")
                        
                        if paper_id:
                            # Obtener relaciones
                            relations = fetch_paper_relations(ses, paper_id)
                            paper.update(relations)
                        
                        # Escribir fila
                        w.writerow(to_scopus_row(paper))
                    
                    exported += len(batch)
                    offset += len(batch)
                    
                    # Progreso cada 500 registros
                    if exported % 500 == 0 or exported == total:
                        print(f"  → Exportados: {exported:,}/{total:,} ({exported/total*100:.1f}%)")
                    
                    # Pequeña pausa para no saturar
                    if exported % 1000 == 0:
                        time.sleep(0.1)
                
                except Exception as e:
                    print(f"\n❌ Error en lote {offset}-{offset+lim}: {e}")
                    print(f"   Continuando con siguiente lote...")
                    offset += lim
                    continue
    
    print(f"\n✅ CSV Scopus listo: {out_path}")
    print(f"   Registros exportados: {exported:,}")
    return out_path

# ================== MMR ==================

def create_embeddings(texts: List[str], method: str = "sbert") -> np.ndarray:
    if method == "sbert" and SentenceTransformer is not None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"  Cargando modelo SBERT: {model_name}")
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        return emb
    
    # TF-IDF fallback
    print("  Usando TF-IDF (SBERT no disponible)")
    vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.8, stop_words="english"
    )
    X = vec.fit_transform(texts).astype(np.float32)
    X = X / (np.linalg.norm(X.toarray(), axis=1, keepdims=True) + 1e-12)
    return X

def relevance_meta(papers: pd.DataFrame, query_terms: List[str] = None) -> np.ndarray:
    if not query_terms:
        query_terms = ["circular","economy","sustainability","waste","recycling","green"]
    
    scores = []
    for _, r in papers.iterrows():
        cites = int(r.get("cited_by", 0) or 0)
        sc_cit = min(np.log1p(cites) / np.log1p(1000), 1.0)
        
        year = int(r.get("year", 2000) or 2000)
        sc_year = (year - 2000) / (2024 - 2000) if year >= 2000 else 0.0
        
        txt = f"{str(r.get('title','')).lower()} {str(r.get('abstract','')).lower()}"
        sc_terms = min(sum(1 for t in query_terms if t in txt) / len(query_terms), 1.0)
        
        abs_len = len(str(r.get("abstract","")).split())
        sc_abs = min(abs_len / 200, 1.0)
        
        scores.append(0.4*sc_cit + 0.2*sc_year + 0.3*sc_terms + 0.1*sc_abs)
    
    return np.array(scores, dtype=np.float32)

def mmr_select(df: pd.DataFrame, Z: np.ndarray, k: int = 50, lam: float = 0.6,
               qterms: List[str] = None) -> pd.DataFrame:
    n = len(df)
    if n == 0 or k == 0:
        return df.head(0)
    k = min(k, n)
    
    meta = relevance_meta(df, qterms)
    q = Z.mean(axis=0, keepdims=True)
    
    def norm(X): 
        nv = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (nv + 1e-12)
    
    Z = norm(Z)
    Q = norm(q)
    rel_q = cosine_similarity(Z, Q).ravel()
    rel = 0.7*meta + 0.3*rel_q

    chosen = [int(np.argmax(rel))]
    pool = set(range(n)) - set(chosen)

    while len(chosen) < k and pool:
        best_idx, best_score = None, -1e9
        Ze = Z[chosen]
        for i in list(pool):
            div = np.max(cosine_similarity(Z[[i]], Ze))
            score = lam*rel[i] - (1-lam)*div
            if score > best_score:
                best_score, best_idx = score, i
        chosen.append(best_idx)
        pool.remove(best_idx)

    out = df.iloc[chosen].copy().reset_index(drop=True)
    out["mmr_rank"] = np.arange(1, len(out)+1)
    out["relevance_score"] = rel[chosen]
    out["query_similarity"] = rel_q[chosen]
    out["metadata_relevance"] = meta[chosen]
    return out

def diversity_metrics(Z: np.ndarray) -> Dict[str, float]:
    if Z.shape[0] < 2:
        return {"n": int(Z.shape[0]), "diameter_cos": 0.0, "mean_distance": 0.0,
                "std_distance": 0.0, "p90_distance": 0.0, "p95_distance": 0.0,
                "spectral_entropy": 0.0, "participation_ratio": 1.0}
    
    S = cosine_similarity(Z)
    D = 1.0 - S
    iu = np.triu_indices(D.shape[0], 1)
    d = D[iu]
    
    try:
        C = Z - Z.mean(axis=0, keepdims=True)
        cov = (C.T @ C) / max(1, Z.shape[0]-1)
        ev = np.linalg.eigvalsh(cov)
        ev = np.clip(ev, 1e-12, None)
        p = ev/ev.sum()
        H = float(-np.sum(p*np.log(p)))
        PR = float((ev.sum()**2)/np.sum(ev**2))
    except Exception:
        H, PR = 0.0, 1.0
    
    return {
        "n": int(Z.shape[0]),
        "diameter_cos": float(d.max()),
        "mean_distance": float(d.mean()),
        "std_distance": float(d.std()),
        "p90_distance": float(np.percentile(d, 90)),
        "p95_distance": float(np.percentile(d, 95)),
        "spectral_entropy": H,
        "participation_ratio": PR,
    }

def run_mmr(drv, k: int = 50, method: str = "sbert", lam: float = 0.6) -> str:
    print(f"\n🔬 MMR → método={method}, k={k}, λ={lam}")
    
    with drv.session(database=NEO4J_DB) as s:
        rows = s.run(MMR_QUERY, limit=MMR_LIMIT).data()
    
    if not rows:
        print("❌ No hay papers elegibles para MMR.")
        return ""
    
    df = pd.DataFrame([{
        "openalex_id": r["openalex_id"], 
        "title": r["title"], 
        "year": r["year"],
        "cited_by": r["cited_by"] or 0, 
        "abstract": clean_abstract(r["abstract"] or ""),
        "journal": r["journal"] or "", 
        "authors": join_semi(r["authors"] or [])
    } for r in rows])

    # Filtrar abstracts muy cortos
    df["abs_words"] = df["abstract"].str.split().str.len()
    df = df[df["abs_words"] >= 20].reset_index(drop=True)
    
    print(f"  Papers con abstract válido: {len(df):,}")
    
    texts = [f"{t}. {t}. {a}" for t,a in zip(df["title"], df["abstract"])]
    Z = create_embeddings(texts, method=method)
    
    reps = mmr_select(df, Z, k=k, lam=lam, 
                      qterms=["circular","economy","sustainability","waste","recycling"])
    
    Zsel = Z[reps.index.values]
    metrics = diversity_metrics(Zsel)

    ts = now_stamp()
    out_reps = os.path.join(OUT_DIR, f"mmr_{method}_{ts}.csv")
    reps.to_csv(out_reps, index=False, encoding="utf-8")
    
    out_div = os.path.join(OUT_DIR, f"mmr_diversity_{ts}.csv")
    pd.DataFrame([metrics]).to_csv(out_div, index=False)

    print(f"\n✅ MMR guardado:")
    print(f"  • Representantes: {out_reps}")
    print(f"  • Métricas:       {out_div}")
    print(f"  • n={len(reps)}, años {reps['year'].min()}–{reps['year'].max()}")
    print(f"  • Citas medias: {reps['cited_by'].mean():.1f}")
    
    return out_reps

# ================== MAIN ==================

def main():
    print("\n" + "="*60)
    print("  EXPORTADOR OPENALEX → VOSviewer (OPTIMIZADO)")
    print("="*60)
    
    try:
        drv = connect_driver()
        print(f"✓ Conectado a Neo4j DB: {NEO4J_DB}")
    except Exception as e:
        print(f"❌ Error conectando a Neo4j: {e}")
        sys.exit(1)

    stats = db_stats(drv)
    print("\n📊 ESTADÍSTICAS:")
    for k, v in stats.items():
        print(f"  • {k}: {v:,}")

    print(f"\n📁 Salida: {OUT_DIR}\n")
    print("Opciones:")
    print("1) Exportar CSV formato Scopus (VOSviewer)")
    print("2) MMR avanzado (SBERT/TF-IDF) - representantes")
    print("3) Ambos\n")

    try:
        opt = input("👉 Opción (1-3): ").strip()
        
        if opt == "1":
            lim = input("Límite opcional de registros (ENTER = todo): ").strip()
            lim = int(lim) if lim else None
            export_scopus_csv(drv, lim)
            
        elif opt == "2":
            k = input("Número de representantes (default 50): ").strip()
            k = int(k) if k else 50
            method = input("Embeddings [sbert|tfidf] (default sbert): ").strip().lower() or "sbert"
            lam = input("λ (relevancia vs. diversidad, 0..1, default 0.6): ").strip()
            lam = float(lam) if lam else 0.6
            run_mmr(drv, k=k, method=method, lam=lam)
            
        elif opt == "3":
            print("\n🔄 Exportando CSV primero...")
            export_scopus_csv(drv, None)
            print("\n🔄 Ejecutando MMR...")
            run_mmr(drv, k=50, method="sbert", lam=0.6)
            
        else:
            print("❌ Opción no válida.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido por usuario.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        drv.close()
    
    print(f"\n✅ Proceso completado. Archivos en: {OUT_DIR}")

if __name__ == "__main__":
    main()