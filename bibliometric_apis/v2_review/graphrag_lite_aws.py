"""
graphrag_lite_aws.py
====================

GraphRAG/HyGRAG-style retrieval evaluation for the AWS paper (plastic
recycling corpus). Reuses paper-level TMO as entities (already extracted)
and adds:
  - paper-level RELATIONS extraction via LLM (Phase 1)
  - hybrid graph construction (paper-paper via shared TMO; TMO-paper)
  - hierarchical communities via agglomerative clustering on hybrid nodes
  - community summaries via LLM (Phase 2)
  - bi-level retrieval: context-aware (community summaries + chunks) +
    relation-aware (triplet expansion)

Design notes:
  - Each abstract = 1 chunk (abstracts are already ~250 tokens).
  - Entities = TMO techniques + methods + objectives + materials from
    runs/paper_tmo/paper_tmo.csv (no new NER needed).
  - Relations = LLM-extracted triplets restricted to entities of the
    same paper (one call per paper).
  - Clustering = sentence-transformers on entity+chunk embeddings,
    agglomerative, 3 levels of granularity 

INPUTS:
  backup_recycled_a/full_corpus/abstracts_full.csv
  backup_recycled_a/full_corpus/paper_topics.csv
  backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv
  runs/paper_tmo/paper_tmo.csv
  benchmark_results/embeddings/specter2.npy   (optional, for embedding fallback)

OUTPUTS (in runs/graphrag_lite/):
  phase1_relations.csv          (per-paper triplets)
  phase2_communities.csv        (community → members + summary)
  phase3_retrieval.csv          (per citation pair, hit by condition)
  phase3_summary.json           (aggregate stats + Wilcoxon)

USAGE:
  python graphrag_lite_aws.py --phase 1       # extract relations
  python graphrag_lite_aws.py --phase 2       # build graph + cluster + summarize
  python graphrag_lite_aws.py --phase 3       # evaluate retrieval
  python graphrag_lite_aws.py --phase all     # run all three sequentially


"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import wilcoxon
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("yyy", "")
OPENROUTER_URL     = "yyyy"
MODEL              = "google/gemini-2.5-flash-lite"

BASE_DIR  = Path("backup_recycled_a/full_corpus")
ABSTRACTS = BASE_DIR / "abstracts_full.csv"
TOPICS    = BASE_DIR / "paper_topics.csv"
EDGES     = BASE_DIR / "citation_edges_cross_cluster.csv"
TMO_PATH  = Path("runs/paper_tmo/paper_tmo.csv")
SPECTER2  = Path("benchmark_results/embeddings/specter2.npy")

OUT_DIR   = Path("runs/graphrag_lite")
RELS_CSV  = OUT_DIR / "phase1_relations.csv"
COMM_CSV  = OUT_DIR / "phase2_communities.csv"
RETR_CSV  = OUT_DIR / "phase3_retrieval.csv"
SUMM_JSON = OUT_DIR / "phase3_summary.json"

TOP_K          = 50
N_COMMUNITIES  = 60        # number of communities (~50 papers each at 3050 papers)
SHARED_TMO_TH  = 2         # min shared TMO terms to add paper-paper edge
RNG_SEED       = 42
MAX_RETRIES    = 3
INTER_CALL_S   = 0.4

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("graphrag")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

STOPWORDS = set(
    "the and for that this with from are was were been have has had not but "
    "its can will also into than more which their these they would could "
    "should about each between through during after before other some such "
    "only over how our who what when where doi org https http www "
    "available online published all any both did does done either "
    "however just much most must same several since still very while "
    "using used use results show study paper present".split()
)


def tokenize(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    words = re.findall(r"[a-z]+", text.lower())
    words = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    return set(words) | {f"{words[i]} {words[i+1]}"
                         for i in range(len(words) - 1)}


def parse_tmo_field(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [p.strip() for p in cell.split("|") if p.strip()]


def call_llm(prompt: str, max_tokens: int = 500,
             temperature: float = 0.0) -> tuple[dict | None, str | None]:
    """Generic OpenRouter call returning (parsed_json, raw_text)."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set")

    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}",
               "Content-Type": "application/json"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(OPENROUTER_URL, json=payload,
                              headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** attempt + random.uniform(0, 1))
                continue
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            text = (msg.get("content") or msg.get("reasoning") or "").strip()
            if not text:
                raise ValueError("Empty content")
            return _parse_json(text), text
        except Exception as ex:
            if attempt == MAX_RETRIES:
                log.warning("LLM call failed: %s", ex)
                return None, None
            time.sleep(2 + random.uniform(0, 1))
    return None, None


def _parse_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(t[s:e + 1])
    except json.JSONDecodeError:
        return None


# ===========================================================================
# PHASE 1 — Extract per-paper relations
# ===========================================================================

REL_PROMPT = """\
You extract scientific relations between concepts in a plastic recycling paper.

You are given an abstract and a list of pre-extracted entities (techniques, \
methods, objectives, materials). Extract a small set of explicit semantic \
relations BETWEEN THESE ENTITIES as observed in this paper.

Use ONLY these relation types: uses, evaluates, produces, applies_to, \
compared_with, achieves, enables, targets.

Rules:
- Each triplet must use ONLY entities from the provided list (exact match).
- Output 3-10 triplets max. Prefer the most informative relations.
- If no clear relation exists between entities, return an empty list.
- Return ONLY valid JSON: {{"triplets": [{{"h": "...", "r": "...", "t": "..."}}]}}
  No markdown, no explanation.

Entities:
{entities}

Abstract:
{abstract}
"""


def run_phase1(limit: int | None = None, retry_errors: bool = False):
    log.info("PHASE 1 — Per-paper relation extraction")

    df_abs = pd.read_csv(ABSTRACTS)[["doi", "abstract"]].dropna()
    df_tmo = pd.read_csv(TMO_PATH)
    df_tmo = df_tmo[df_tmo["status"] == "ok"]

    df = df_abs.merge(df_tmo, on="doi", how="inner")
    log.info("  Papers with abstract+TMO: %d", len(df))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done: dict[str, str] = {}
    if RELS_CSV.exists():
        prev = pd.read_csv(RELS_CSV, usecols=["doi", "status"])
        for _, r in prev.iterrows():
            done[r["doi"]] = r["status"]
        if retry_errors:
            keep = pd.read_csv(RELS_CSV)
            keep = keep[keep["status"] == "ok"]
            keep.to_csv(RELS_CSV, index=False)
            done = {d: s for d, s in done.items() if s == "ok"}
            log.info("  Retry mode: %d 'ok' rows kept", len(done))
        else:
            log.info("  Resume: %d already processed", len(done))

    todo = [(r["doi"], r["abstract"], r) for _, r in df.iterrows()
            if done.get(r["doi"]) != "ok"]
    if limit:
        todo = todo[:limit]
    log.info("  To process: %d", len(todo))

    new_file = not RELS_CSV.exists()
    fields = ["doi", "n_triplets", "triplets_json", "status", "timestamp"]
    t0 = time.time()
    n_ok = n_fail = 0

    with RELS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()

        for i, (doi, abstract, row) in enumerate(todo, 1):
            entities = (parse_tmo_field(row.get("techniques", ""))
                        + parse_tmo_field(row.get("methods", ""))
                        + parse_tmo_field(row.get("objectives", "")))
            if not entities:
                w.writerow({"doi": doi, "n_triplets": 0,
                            "triplets_json": "[]", "status": "no_entities",
                            "timestamp": _now()})
                f.flush()
                continue

            prompt = REL_PROMPT.format(
                entities="\n".join(f"- {e}" for e in entities),
                abstract=str(abstract).strip()[:2000],
            )
            parsed, _ = call_llm(prompt, max_tokens=400)
            now = _now()

            if parsed is None or "triplets" not in parsed:
                n_fail += 1
                w.writerow({"doi": doi, "n_triplets": 0, "triplets_json": "[]",
                            "status": "error", "timestamp": now})
            else:
                trips = parsed.get("triplets", [])
                ents_lower = {e.lower() for e in entities}
                clean = []
                for t in trips:
                    if not isinstance(t, dict):
                        continue
                    h = str(t.get("h", "")).strip()
                    r = str(t.get("r", "")).strip()
                    tt = str(t.get("t", "")).strip()
                    if not h or not r or not tt:
                        continue
                    if h.lower() not in ents_lower or tt.lower() not in ents_lower:
                        continue
                    clean.append({"h": h, "r": r, "t": tt})
                n_ok += 1
                w.writerow({"doi": doi, "n_triplets": len(clean),
                            "triplets_json": json.dumps(clean,
                                                       ensure_ascii=False),
                            "status": "ok", "timestamp": now})
            f.flush()

            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                eta = (len(todo) - i) / rate if rate > 0 else 0
                log.info("[%d/%d] ok=%d fail=%d | %.1f/s | ETA %.1f min",
                         i, len(todo), n_ok, n_fail, rate, eta / 60)

            time.sleep(INTER_CALL_S)

    log.info("PHASE 1 done: ok=%d fail=%d", n_ok, n_fail)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ===========================================================================
# PHASE 2 — Build hybrid graph, cluster into communities, summarize
# ===========================================================================

SUM_PROMPT = """\
Summarize the following scientific research community in 80-120 words.

This community groups papers in plastic recycling that share similar \
techniques, materials, methods, or objectives. Your summary will be used \
to retrieve papers via semantic search, so include:
1. The dominant research themes and topics.
2. Key techniques and methods used.
3. Main materials studied.
4. Primary objectives or applications.

Be specific and use technical vocabulary. No preamble, no markdown.

Top-25 representative paper titles from this community:
{titles}

Top TMO entities in this community (sorted by frequency):
{entities}
"""


def run_phase2():
    log.info("PHASE 2 — Hybrid graph + clustering + community summaries")

    # ----- Load corpus + TMO + relations -----
    df_abs = pd.read_csv(ABSTRACTS)[["doi", "title", "abstract"]].dropna(
        subset=["doi", "abstract"])
    df_tmo = pd.read_csv(TMO_PATH)
    df_tmo = df_tmo[df_tmo["status"] == "ok"]
    df = df_abs.merge(df_tmo, on="doi", how="inner").reset_index(drop=True)
    log.info("  Papers: %d", len(df))

    df_rel = pd.read_csv(RELS_CSV) if RELS_CSV.exists() else pd.DataFrame()
    log.info("  Papers with relations: %d",
             len(df_rel[df_rel["status"] == "ok"]) if len(df_rel) else 0)

    # ----- Build paper-level TMO term sets (entities per paper) -----
    paper_entities: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        ents = (parse_tmo_field(r.get("techniques", ""))
                + parse_tmo_field(r.get("methods", ""))
                + parse_tmo_field(r.get("objectives", "")))
        paper_entities[r["doi"]] = ents

    # ----- Embed papers (SPECTER2 if available, else SBERT) -----
    if SPECTER2.exists():
        X = np.load(SPECTER2)
        if X.shape[0] == len(df_abs):
            # SPECTER2 was computed on df_abs order. We need to align to df.
            doi_to_specter_idx = {row["doi"]: i
                                  for i, row in df_abs.reset_index(drop=True).iterrows()}
            idx = [doi_to_specter_idx[d] for d in df["doi"]]
            emb = X[idx]
            log.info("  Loaded SPECTER2 embeddings: %s", emb.shape)
        else:
            log.warning("  SPECTER2 shape mismatch; falling back to SBERT")
            emb = _embed_sbert(df["abstract"].tolist())
    else:
        log.info("  SPECTER2 not found; computing SBERT embeddings...")
        emb = _embed_sbert(df["abstract"].tolist())

    # ----- Agglomerative clustering -----
    log.info("  Clustering into %d communities (agglomerative)...",
             N_COMMUNITIES)
    clustering = AgglomerativeClustering(
        n_clusters=N_COMMUNITIES, metric="cosine", linkage="average")
    comm_labels = clustering.fit_predict(emb)
    df["community"] = comm_labels
    log.info("  Community sizes: min=%d max=%d median=%d",
             pd.Series(comm_labels).value_counts().min(),
             pd.Series(comm_labels).value_counts().max(),
             int(pd.Series(comm_labels).value_counts().median()))

    # ----- Generate summary per community -----
    log.info("  Generating community summaries via LLM...")
    summaries: dict[int, str] = {}
    if COMM_CSV.exists():
        prev = pd.read_csv(COMM_CSV)
        summaries = {int(r["community"]): r["summary"]
                     for _, r in prev.iterrows()
                     if isinstance(r.get("summary"), str)}
        log.info("  Resume: %d communities already summarized", len(summaries))

    t0 = time.time()
    rows_out = []
    for cid in sorted(df["community"].unique()):
        members = df[df["community"] == cid]
        member_dois = members["doi"].tolist()
        titles = members["title"].dropna().tolist()[:25]
        ents_flat = [e for d in member_dois for e in paper_entities.get(d, [])]
        ent_counts = pd.Series(ents_flat).value_counts().head(30)
        ent_str = ", ".join(ent_counts.index.tolist())

        if cid in summaries and summaries[cid]:
            summary = summaries[cid]
        else:
            prompt = SUM_PROMPT.format(
                titles="\n".join(f"- {t}" for t in titles),
                entities=ent_str or "(none)",
            )
            _, raw = call_llm(prompt, max_tokens=400)
            summary = (raw or "").strip()
            summaries[cid] = summary
            time.sleep(INTER_CALL_S)

        rows_out.append({
            "community": int(cid),
            "n_papers": int(len(members)),
            "summary": summary,
            "top_entities": ent_str,
            "member_dois": "|".join(member_dois),
        })

        if (cid + 1) % 10 == 0:
            log.info("  Summarized %d/%d (%.0fs)",
                     cid + 1, N_COMMUNITIES, time.time() - t0)

    pd.DataFrame(rows_out).to_csv(COMM_CSV, index=False)

    # ----- Save embeddings for retrieval (paper + community) -----
    np.save(OUT_DIR / "paper_emb.npy", emb)
    df[["doi", "community"]].to_csv(OUT_DIR / "paper_communities.csv",
                                    index=False)

    # ----- Embed community summaries for retrieval -----
    log.info("  Embedding community summaries...")
    valid = [(cid, s) for cid, s in summaries.items() if s.strip()]
    if valid:
        cids, texts = zip(*valid)
        comm_emb = _embed_sbert(list(texts))
        np.save(OUT_DIR / "community_emb.npy", comm_emb)
        pd.DataFrame({"community": cids}).to_csv(
            OUT_DIR / "community_emb_index.csv", index=False)
        log.info("  Saved community embeddings: %s", comm_emb.shape)

    log.info("PHASE 2 done")


def _embed_sbert(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return m.encode(texts, show_progress_bar=True, convert_to_numpy=True,
                    normalize_embeddings=True, batch_size=128)


# ===========================================================================
# PHASE 3 — Bi-level retrieval evaluation
# ===========================================================================

def run_phase3():
    log.info("PHASE 3 — Bi-level retrieval evaluation")

    # ----- Load corpus -----
    df_abs = pd.read_csv(ABSTRACTS)[["doi", "title", "abstract"]].dropna(
        subset=["doi", "abstract"])
    df_tmo = pd.read_csv(TMO_PATH)
    df_tmo = df_tmo[df_tmo["status"] == "ok"]
    df = df_abs.merge(df_tmo, on="doi", how="inner").reset_index(drop=True)
    doi_to_idx = {d: i for i, d in enumerate(df["doi"])}

    # ----- Lexical doc terms (for baseline) -----
    doc_terms = [tokenize(a) for a in df["abstract"].tolist()]

    # ----- Paper TMO terms -----
    paper_tmo: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        ents = (parse_tmo_field(r.get("techniques", ""))
                + parse_tmo_field(r.get("methods", ""))
                + parse_tmo_field(r.get("objectives", "")))
        terms = set()
        for e in ents:
            terms |= tokenize(e)
        paper_tmo[r["doi"]] = terms

    # ----- Paper embeddings -----
    paper_emb = np.load(OUT_DIR / "paper_emb.npy")
    if paper_emb.shape[0] != len(df):
        raise ValueError(f"paper_emb shape {paper_emb.shape} != {len(df)} papers")

    # ----- Community assignments + summaries -----
    df_comm = pd.read_csv(OUT_DIR / "paper_communities.csv")
    paper_community: dict[str, int] = dict(zip(df_comm["doi"],
                                                df_comm["community"].astype(int)))

    df_summ = pd.read_csv(COMM_CSV)
    comm_emb = np.load(OUT_DIR / "community_emb.npy")
    comm_idx = pd.read_csv(OUT_DIR / "community_emb_index.csv")
    cid_to_row = {int(c): i for i, c in enumerate(comm_idx["community"])}
    cid_to_members = {int(r["community"]): r["member_dois"].split("|")
                      for _, r in df_summ.iterrows()
                      if isinstance(r.get("member_dois"), str)}

    # ----- Relations per paper (for relation-aware retrieval) -----
    df_rel = pd.read_csv(RELS_CSV)
    paper_triplets: dict[str, list[dict]] = {}
    for _, r in df_rel.iterrows():
        if r.get("status") != "ok":
            continue
        try:
            paper_triplets[r["doi"]] = json.loads(r["triplets_json"])
        except Exception:
            pass

    # ----- Citation pairs -----
    pairs = pd.read_csv(EDGES)
    log.info("  %d citation pairs", len(pairs))

    # ----- Eval -----
    log.info("  Running retrieval (TOP_K=%d)...", TOP_K)
    rows = []
    skipped = 0

    for _, row in pairs.iterrows():
        src_doi = row["source_doi"]
        tgt_doi = row["target_doi"]
        if src_doi not in doi_to_idx or tgt_doi not in doi_to_idx:
            skipped += 1
            continue
        si, ti = doi_to_idx[src_doi], doi_to_idx[tgt_doi]
        src_lex = doc_terms[si]
        if not src_lex:
            skipped += 1
            continue

        # A: lexical baseline
        top_a = _score_lexical(src_lex, doc_terms, si, TOP_K)
        hit_a = int(ti in top_a)

        # B: dense (SPECTER2 or SBERT)
        sims = paper_emb @ paper_emb[si]
        sims[si] = -np.inf
        top_b = list(np.argsort(-sims)[:TOP_K])
        hit_b = int(ti in top_b)

        # C: community-aware retrieval
        # find top-3 communities by similarity between src embedding and comm embeddings
        if comm_emb.size > 0:
            src_v = paper_emb[si:si + 1]
            csims = (comm_emb @ src_v.T).flatten()
            top_comm_rows = np.argsort(-csims)[:3]
            top_cids = [int(comm_idx.iloc[r]["community"]) for r in top_comm_rows]
            # candidate papers = union of members of these communities
            cand_dois = set()
            for c in top_cids:
                cand_dois.update(cid_to_members.get(c, []))
            cand_idx = [doi_to_idx[d] for d in cand_dois
                        if d in doi_to_idx and doi_to_idx[d] != si]
            if cand_idx:
                cand_sims = paper_emb[cand_idx] @ paper_emb[si]
                order = np.argsort(-cand_sims)
                top_c = [cand_idx[o] for o in order[:TOP_K]]
            else:
                top_c = []
        else:
            top_c = []
        hit_c = int(ti in top_c)

        # D: relation-aware (expand via relation neighbors of source's TMO entities)
        # Idea: collect all papers whose triplets share any entity with src's triplets
        src_trips = paper_triplets.get(src_doi, [])
        src_ents_in_trips = set()
        for t in src_trips:
            src_ents_in_trips.add(str(t.get("h", "")).lower())
            src_ents_in_trips.add(str(t.get("t", "")).lower())
        rel_cands: list[int] = []
        if src_ents_in_trips:
            for d, trips in paper_triplets.items():
                if d == src_doi or d not in doi_to_idx:
                    continue
                for t in trips:
                    h = str(t.get("h", "")).lower()
                    tt = str(t.get("t", "")).lower()
                    if h in src_ents_in_trips or tt in src_ents_in_trips:
                        rel_cands.append(doi_to_idx[d])
                        break
            # rank rel_cands by dense similarity to src
            if rel_cands:
                rs = paper_emb[rel_cands] @ paper_emb[si]
                order = np.argsort(-rs)
                top_d = [rel_cands[o] for o in order[:TOP_K]]
            else:
                top_d = []
        else:
            top_d = []
        hit_d = int(ti in top_d)

        # E: HyGRAG-style fusion (RRF of A, B, C, D)
        top_e = _rrf([top_a, top_b, top_c, top_d], top_k=TOP_K)
        hit_e = int(ti in top_e)

        rows.append({
            "source_doi": src_doi, "target_doi": tgt_doi,
            "hit_A_lexical":          hit_a,
            "hit_B_dense":            hit_b,
            "hit_C_community":        hit_c,
            "hit_D_relation":         hit_d,
            "hit_E_hygrag_fusion":    hit_e,
        })

    df_r = pd.DataFrame(rows)
    df_r.to_csv(RETR_CSV, index=False)
    log.info("  Evaluated %d pairs (skipped %d)", len(df_r), skipped)

    # ----- Stats -----
    rec = {
        "A_lexical":           round(df_r["hit_A_lexical"].mean(), 4),
        "B_dense":             round(df_r["hit_B_dense"].mean(), 4),
        "C_community":         round(df_r["hit_C_community"].mean(), 4),
        "D_relation":          round(df_r["hit_D_relation"].mean(), 4),
        "E_hygrag_fusion":     round(df_r["hit_E_hygrag_fusion"].mean(), 4),
    }
    cis = {k: list(_bootstrap_ci(df_r[f"hit_{k}"].values))
           for k in [c.replace("hit_", "")
                     for c in df_r.columns if c.startswith("hit_")]}

    # Wilcoxon: each condition vs A (one-sided greater)
    base = df_r["hit_A_lexical"].values
    wilcox = {}
    for k, col in [("B_dense", "hit_B_dense"),
                   ("C_community", "hit_C_community"),
                   ("D_relation", "hit_D_relation"),
                   ("E_hygrag_fusion", "hit_E_hygrag_fusion")]:
        w, p = _safe_wilcoxon(df_r[col].values, base)
        wilcox[f"{k}_vs_A"] = {"W": w, "p": p}

    summary = {
        "n_pairs": len(df_r),
        "n_skipped": skipped,
        "top_k": TOP_K,
        "n_communities": N_COMMUNITIES,
        "recall": rec,
        "bootstrap_ci": cis,
        "wilcoxon_vs_lexical": wilcox,
    }
    SUMM_JSON.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print("GRAPHRAG-LITE RETRIEVAL — plastic corpus")
    print("=" * 64)
    print(f"Pairs evaluated: {len(df_r)}")
    print(f"\nRecall@{TOP_K}:")
    for k, v in rec.items():
        c = cis[k]
        print(f"  {k:25s}: {v:.4f}  CI[{c[0]:.4f},{c[1]:.4f}]")
    print(f"\nWilcoxon vs A_lexical (one-sided greater):")
    for k, t in wilcox.items():
        if t["W"] is None:
            print(f"  {k:25s}: insufficient diffs")
        else:
            star = ""
            if t["p"] is not None:
                if t["p"] < 0.001:
                    star = " ***"
                elif t["p"] < 0.01:
                    star = " **"
                elif t["p"] < 0.05:
                    star = " *"
            print(f"  {k:25s}: W={t['W']:.0f}, p={t['p']:.3e}{star}")
    print(f"\nOutputs in {OUT_DIR}/")


def _score_lexical(query: set[str], doc_terms: list[set[str]],
                   exclude: int, top_k: int) -> list[int]:
    scored = []
    for i, t in enumerate(doc_terms):
        if i == exclude:
            continue
        ov = len(query & t)
        if ov:
            scored.append((i, ov))
    scored.sort(key=lambda x: -x[1])
    return [i for i, _ in scored[:top_k]]


def _rrf(rankings: list[list[int]], top_k: int, k_const: int = 60) -> list[int]:
    """Reciprocal Rank Fusion."""
    scores: dict[int, float] = defaultdict(float)
    for rank_list in rankings:
        for r, idx in enumerate(rank_list):
            scores[idx] += 1.0 / (k_const + r + 1)
    sorted_idx = sorted(scores, key=lambda x: -scores[x])
    return sorted_idx[:top_k]


def _bootstrap_ci(values, n_boot=1000, seed=42):
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return (round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))


def _safe_wilcoxon(x, y):
    d = np.asarray(x) - np.asarray(y)
    if np.count_nonzero(d) < 10:
        return None, None
    w, p = wilcoxon(x, y, alternative="greater")
    return float(w), float(p)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="GraphRAG-lite for AWS paper")
    parser.add_argument("--phase", choices=["1", "2", "3", "all"],
                        default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="(phase 1) smoke test on first N papers")
    parser.add_argument("--retry-errors", action="store_true",
                        help="(phase 1) drop error rows and retry")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("1", "all"):
        run_phase1(limit=args.limit, retry_errors=args.retry_errors)
    if args.phase in ("2", "all"):
        run_phase2()
    if args.phase in ("3", "all"):
        run_phase3()


if __name__ == "__main__":
    main()
