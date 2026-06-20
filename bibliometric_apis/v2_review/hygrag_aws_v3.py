"""
hygrag_aws_v3.py
================

 HyGRAG-style pipeline for the AWS revision paper.

Five phases, independent and resumable:

  Phase 0  recompute_embeddings    : SBERT + ChemBERTa + SPECTER2 (local; free)
                                     Optional: Gemini / Qwen3 / NV-Embed (API)
  Phase 1  relations               : reuses runs/paper_tmo/paper_tmo.csv plus
                                     runs/graphrag_lite/phase1_relations.csv
                                     (already extracted; SKIP if present)
  Phase 2  hybrid_graph            : chunks + entities + relations →
                                     hybrid NetworkX graph + Node2Vec via PyG
  Phase 3  hierarchical_clustering : 3-level recursive agglomerative clustering
                                     on Node2Vec embeddings; cohesive LLM
                                     summaries at each level (~$2 with Flash-Lite)
  Phase 4  retrieval_evaluation    : 9 conditions vs 670 cross-cluster citation
                                     pairs; bootstrap CI; Wilcoxon vs lexical

Inputs :
  backup_recycled_a/full_corpus/abstracts_full.csv
  backup_recycled_a/full_corpus/paper_topics.csv
  backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv
  runs/paper_tmo/paper_tmo.csv
  runs/graphrag_lite/phase1_relations.csv

Outputs (runs/hygrag_v3/):
  phase0_embeddings/<model>.npy           # aligned with paper_tmo.csv order
  phase0_doi_order.csv                    # the canonical row→doi mapping
  phase2_hybrid_graph.gml                 # full hybrid graph
  phase2_node2vec.npy                     # node2vec embeddings
  phase2_node_index.csv                   # row → (node_id, node_type)
  phase3_clusters_L{1,2,3}.csv            # community assignments per level
  phase3_summaries_L{1,2,3}.csv           # cohesive LLM summaries
  phase3_summaries_emb_L{1,2,3}.npy       # summary embeddings (SBERT)
  phase4_results.csv                      # hit-by-condition per pair
  phase4_summary.json                     # aggregated table for the paper

Usage:
  python hygrag_aws_v3.py --phase 0
  python hygrag_aws_v3.py --phase 0 --include-api    # add Gemini/Qwen3/NV-Embed
  python hygrag_aws_v3.py --phase 2
  python hygrag_aws_v3.py --phase 3
  python hygrag_aws_v3.py --phase 4
  python hygrag_aws_v3.py --phase all
"""

from __future__ import annotations

import argparse
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

# Heavy imports lazy-loaded per phase

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hygrag")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR  = Path("backup_recycled_a/full_corpus")
ABSTRACTS = BASE_DIR / "abstracts_full.csv"
TOPICS    = BASE_DIR / "paper_topics.csv"
EDGES     = BASE_DIR / "citation_edges_cross_cluster.csv"
TMO_PATH  = Path("runs/paper_tmo/paper_tmo.csv")
RELS_PATH = Path("runs/graphrag_lite/phase1_relations.csv")

OUT_DIR    = Path("runs/hygrag_v3")
EMB_DIR    = OUT_DIR / "phase0_embeddings"
DOI_ORDER  = OUT_DIR / "phase0_doi_order.csv"

GRAPH_PATH    = OUT_DIR / "phase2_hybrid_graph.gml"
NODE2VEC_PATH = OUT_DIR / "phase2_node2vec.npy"
NODE_INDEX    = OUT_DIR / "phase2_node_index.csv"
GRAPH_STATS   = OUT_DIR / "phase2_graph_stats.json"

RESULTS_CSV   = OUT_DIR / "phase4_results.csv"
SUMMARY_JSON  = OUT_DIR / "phase4_summary.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Node2Vec hyperparams (Grover & Leskovec 2016 defaults)
N2V_DIM            = 128
N2V_WALK_LENGTH    = 20
N2V_NUM_WALKS      = 10
N2V_P              = 1.0   # return parameter
N2V_Q              = 1.0   # in-out parameter (1.0 = DeepWalk-equivalent)
N2V_EPOCHS         = 5
N2V_BATCH_SIZE     = 128
N2V_LR             = 0.01
N2V_CONTEXT_SIZE   = 10
N2V_NEG_SAMPLES    = 1

# Hybrid graph thresholds
SHARED_ENTITY_TH   = 3   # min shared entities to add chunk-chunk edge (HyGRAG l=3)

# Clustering
N_CLUSTERS_L1      = 60
N_CLUSTERS_L2      = 12
N_CLUSTERS_L3      = 4
SUMMARY_MAX_TOKENS = 400

# Retrieval
TOP_K              = 50
RNG_SEED           = 42

# LLM config (for summaries)
MODEL              = "google/gemini-2.5-flash-lite"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
INTER_CALL_S       = 0.4
MAX_RETRIES        = 3

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


def normalize_entity(e: str) -> str:
    """Normalize entity surface form to its canonical key."""
    return re.sub(r"\s+", " ", str(e).strip().lower())


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_corpus() -> pd.DataFrame:
    """Load and align: abstracts + topics + TMO; produce canonical order."""
    df_abs = pd.read_csv(ABSTRACTS)
    if "title" not in df_abs.columns:
        df_abs["title"] = ""
    df_abs = df_abs[["doi", "title", "abstract"]].dropna(subset=["doi", "abstract"])
    df_top = pd.read_csv(TOPICS)[["doi", "cluster"]].dropna()
    df_top["cluster"] = df_top["cluster"].astype(int)

    df_tmo = pd.read_csv(TMO_PATH)
    df_tmo = df_tmo[df_tmo["status"] == "ok"].reset_index(drop=True)
    # canonical row order = paper_tmo.csv order
    df = (df_tmo[["doi", "techniques", "methods", "objectives"]]
          .merge(df_abs, on="doi", how="inner")
          .merge(df_top, on="doi", how="inner")
          .reset_index(drop=True))
    return df


def bootstrap_ci(values, n_boot=1000, seed=42):
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return (round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))


def safe_wilcoxon(x, y):
    from scipy.stats import wilcoxon
    d = np.asarray(x) - np.asarray(y)
    if np.count_nonzero(d) < 10:
        return None, None
    w, p = wilcoxon(x, y, alternative="greater")
    return float(w), float(p)


# ===========================================================================
# PHASE 0  --  Recompute embeddings on the aligned 3043-paper TMO corpus
# ===========================================================================

def run_phase0(include_api: bool = False):
    log.info("PHASE 0 — Recomputing embeddings on aligned TMO corpus")
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    df = load_corpus()
    log.info("  Aligned corpus: %d papers", len(df))

    df[["doi"]].to_csv(DOI_ORDER, index=False)
    log.info("  Canonical order saved → %s", DOI_ORDER)

    abstracts = df["abstract"].astype(str).tolist()
    titles    = df["title"].astype(str).tolist()
    # SPECTER-style input: "title [SEP] abstract"
    sep_input = [f"{t} [SEP] {a}" for t, a in zip(titles, abstracts)]

    # ---- SBERT (MiniLM) ----
    p = EMB_DIR / "sbert.npy"
    if not p.exists():
        log.info("  → SBERT (sentence-transformers/all-MiniLM-L6-v2)")
        import gc
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        X = m.encode(abstracts, batch_size=32, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)
        np.save(p, X)
        log.info("    saved %s  shape=%s", p, X.shape)
        del m, X; gc.collect()
    else:
        log.info("  ✓ SBERT cached")

    # ---- ChemBERTa ----
    p = EMB_DIR / "chemberta.npy"
    if not p.exists():
        log.info("  → ChemBERTa (DeepChem/ChemBERTa-77M-MTR)")
        import gc
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("DeepChem/ChemBERTa-77M-MTR")
        X = m.encode(abstracts, batch_size=16, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)
        np.save(p, X)
        log.info("    saved %s  shape=%s", p, X.shape)
        del m, X; gc.collect()
    else:
        log.info("  ✓ ChemBERTa cached")

    # ---- SPECTER2 ----
    p = EMB_DIR / "specter2.npy"
    if not p.exists():
        log.info("  → SPECTER2 (allenai/specter2_base + proximity adapter)")
        X = _embed_specter2(sep_input)
        np.save(p, X)
        log.info("    saved %s  shape=%s", p, X.shape)
    else:
        log.info("  ✓ SPECTER2 cached")

    if include_api:
        log.info("  --include-api set; API embeddings TBD (Gemini/Qwen3/NV-Embed)")
        log.info("  Skipping in this run; locals are sufficient for HyGRAG pipeline")

    log.info("PHASE 0 done")


def _embed_specter2(texts):
    """SPECTER2 with proximity adapter, CLS-token embeddings."""
    import torch
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    import gc
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf",
                       load_as="proximity", set_active=True)
    model.eval()

    embs = []
    batch_size = 4
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :].numpy()
        embs.append(cls)
        del inputs, out
        if (i // batch_size) % 10 == 0:
            gc.collect()
        if (i // batch_size) % 20 == 0:
            log.info("    SPECTER2 batch %d/%d",
                     i // batch_size + 1, n_batches)
    X = np.vstack(embs)
    # normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


# ===========================================================================
# PHASE 2  --  Hybrid graph + Node2Vec
# ===========================================================================

def run_phase2():
    log.info("PHASE 2 — Hybrid graph + Node2Vec")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import networkx as nx

    df = load_corpus()
    log.info("  Papers: %d", len(df))

    # ----- Per-paper entities (TMO) -----
    entities_by_doi: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        ents = (parse_tmo_field(r["techniques"])
                + parse_tmo_field(r["methods"])
                + parse_tmo_field(r["objectives"]))
        entities_by_doi[r["doi"]] = {normalize_entity(e) for e in ents if e}

    # ----- Per-paper relations (already extracted) -----
    rel_by_doi: dict[str, list[dict]] = {}
    if RELS_PATH.exists():
        df_rel = pd.read_csv(RELS_PATH)
        df_rel = df_rel[df_rel["status"] == "ok"]
        for _, r in df_rel.iterrows():
            try:
                trips = json.loads(r["triplets_json"])
                if trips:
                    rel_by_doi[r["doi"]] = trips
            except Exception:
                continue
        log.info("  Loaded %d papers with relation triplets", len(rel_by_doi))
    else:
        log.warning("  Phase 1 relations missing; entity-entity edges will be sparse")

    # ----- Build hybrid graph -----
    log.info("  Building hybrid graph...")
    G = nx.Graph()

    # Add chunk (paper) nodes
    for _, r in df.iterrows():
        G.add_node(f"chunk::{r['doi']}", node_type="chunk", doi=r["doi"])

    # Add entity nodes (unique normalized TMO terms)
    all_entities: set[str] = set()
    for ents in entities_by_doi.values():
        all_entities |= ents
    for e in all_entities:
        G.add_node(f"entity::{e}", node_type="entity", label=e)
    log.info("  Nodes: %d chunks + %d entities = %d",
             len(df), len(all_entities), G.number_of_nodes())

    # Chunk-Entity edges (membership)
    n_ce = 0
    for doi, ents in entities_by_doi.items():
        cn = f"chunk::{doi}"
        for e in ents:
            G.add_edge(cn, f"entity::{e}", edge_type="membership")
            n_ce += 1
    log.info("  Chunk-Entity edges: %d", n_ce)

    # Entity-Entity edges (from LLM-extracted triplets)
    n_ee = 0
    for doi, trips in rel_by_doi.items():
        for t in trips:
            h = normalize_entity(t.get("h", ""))
            tt = normalize_entity(t.get("t", ""))
            if h and tt and h != tt:
                hn, tn = f"entity::{h}", f"entity::{tt}"
                if G.has_node(hn) and G.has_node(tn):
                    if G.has_edge(hn, tn):
                        G[hn][tn]["weight"] = G[hn][tn].get("weight", 1) + 1
                    else:
                        G.add_edge(hn, tn, edge_type="relation",
                                   relation=t.get("r", ""), weight=1)
                        n_ee += 1
    log.info("  Entity-Entity edges: %d", n_ee)

    # Chunk-Chunk edges (shared entities >= threshold)
    log.info("  Computing chunk-chunk edges (shared entities ≥ %d)...",
             SHARED_ENTITY_TH)
    dois = df["doi"].tolist()
    n_cc = 0
    # inverted index for speed
    entity_to_papers: dict[str, set[str]] = defaultdict(set)
    for doi, ents in entities_by_doi.items():
        for e in ents:
            entity_to_papers[e].add(doi)
    # count shared entities per pair
    pair_count: dict[tuple, int] = defaultdict(int)
    for e, papers in entity_to_papers.items():
        plist = sorted(papers)
        if len(plist) < 2:
            continue
        # only count pairs through entities with reasonable specificity
        if len(plist) > 200:  # skip uber-generic entities like "plastic"
            continue
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                pair_count[(plist[i], plist[j])] += 1
    for (a, b), n in pair_count.items():
        if n >= SHARED_ENTITY_TH:
            G.add_edge(f"chunk::{a}", f"chunk::{b}",
                       edge_type="shared_entity", weight=n)
            n_cc += 1
    log.info("  Chunk-Chunk edges: %d", n_cc)
    log.info("  Total edges: %d", G.number_of_edges())

    # Save graph
    nx.write_gml(G, GRAPH_PATH)
    log.info("  Graph saved → %s", GRAPH_PATH)

    stats = {
        "n_chunks": len(df),
        "n_entities": len(all_entities),
        "n_chunk_entity": n_ce,
        "n_entity_entity": n_ee,
        "n_chunk_chunk": n_cc,
        "total_edges": G.number_of_edges(),
        "n_components": int(sum(1 for _ in __import__("networkx").connected_components(G))),
    }
    GRAPH_STATS.write_text(json.dumps(stats, indent=2))

    # ----- Node2Vec via pure-Python node2vec library (Mac-friendly) -----
    log.info("  Training Node2Vec (pure-Python): dim=%d walks=%d len=%d p=%.1f q=%.1f...",
             N2V_DIM, N2V_NUM_WALKS, N2V_WALK_LENGTH, N2V_P, N2V_Q)

    from node2vec import Node2Vec as N2VLib

    n2v_model = N2VLib(
        G,
        dimensions=N2V_DIM,
        walk_length=N2V_WALK_LENGTH,
        num_walks=N2V_NUM_WALKS,
        p=N2V_P,
        q=N2V_Q,
        workers=2,
        seed=RNG_SEED,
        quiet=False,
    )
    model_w2v = n2v_model.fit(window=N2V_CONTEXT_SIZE,
                              min_count=1,
                              batch_words=4,
                              epochs=N2V_EPOCHS,
                              seed=RNG_SEED)

    node_list = list(G.nodes())
    emb = np.zeros((len(node_list), N2V_DIM), dtype=np.float32)
    for i, n in enumerate(node_list):
        if str(n) in model_w2v.wv:
            emb[i] = model_w2v.wv[str(n)]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms
    np.save(NODE2VEC_PATH, emb)
    log.info("  Node2Vec embeddings saved → %s  shape=%s", NODE2VEC_PATH,
             emb.shape)

    df_idx = pd.DataFrame({
        "row": range(len(node_list)),
        "node_id": node_list,
        "node_type": [G.nodes[n]["node_type"] for n in node_list],
    })
    df_idx.to_csv(NODE_INDEX, index=False)
    log.info("  Node index saved → %s", NODE_INDEX)

    # ---- Entity embeddings (independent SBERT representation) ----
    log.info("  Embedding %d entities via SBERT (for relation-aware retrieval)...",
             len(all_entities))
    from sentence_transformers import SentenceTransformer
    sbert_e = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    entity_list = sorted(all_entities)
    entity_vecs = sbert_e.encode(entity_list, batch_size=64,
                                  show_progress_bar=True,
                                  convert_to_numpy=True,
                                  normalize_embeddings=True)
    np.save(OUT_DIR / "phase2_entity_emb.npy", entity_vecs)
    pd.DataFrame({"entity": entity_list}).to_csv(
        OUT_DIR / "phase2_entity_index.csv", index=False)
    log.info("  Entity embeddings saved → phase2_entity_emb.npy  shape=%s",
             entity_vecs.shape)

    log.info("PHASE 2 done")


# ===========================================================================
# PHASE 3  --  Hierarchical clustering + cohesive LLM summaries
# ===========================================================================

def run_phase3():
    log.info("PHASE 3 — Hierarchical clustering + cohesive summaries")

    import requests
    from dotenv import load_dotenv
    from sklearn.cluster import AgglomerativeClustering
    from sentence_transformers import SentenceTransformer

    load_dotenv(Path.home() / "Desktop/openalex/.env")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")

    df_idx = pd.read_csv(NODE_INDEX)
    emb = np.load(NODE2VEC_PATH)
    log.info("  Loaded %d nodes, emb shape=%s", len(df_idx), emb.shape)

    df_corpus = load_corpus()
    titles_by_doi = dict(zip(df_corpus["doi"], df_corpus["title"]))

    # Load relation triplets per DOI (for community-level relation summaries)
    rel_by_doi: dict[str, list[dict]] = {}
    if RELS_PATH.exists():
        df_rel = pd.read_csv(RELS_PATH)
        df_rel = df_rel[df_rel["status"] == "ok"]
        for _, r in df_rel.iterrows():
            try:
                trips = json.loads(r["triplets_json"])
                if trips:
                    rel_by_doi[r["doi"]] = trips
            except Exception:
                continue
        log.info("  Loaded %d papers with relation triplets", len(rel_by_doi))

    # ---- Per-level clustering on node embeddings ----
    cluster_sizes = [N_CLUSTERS_L1, N_CLUSTERS_L2, N_CLUSTERS_L3]

    # L1: cluster all nodes (chunks + entities)
    log.info("  L1: agglomerative clustering into %d communities (all nodes)...",
             N_CLUSTERS_L1)
    cl_L1 = AgglomerativeClustering(
        n_clusters=N_CLUSTERS_L1, metric="cosine", linkage="average")
    labels_L1 = cl_L1.fit_predict(emb)
    df_idx["L1"] = labels_L1

    # For L2, L3 we cluster the community centroids of the previous level
    centroids_L1 = np.zeros((N_CLUSTERS_L1, emb.shape[1]))
    for c in range(N_CLUSTERS_L1):
        centroids_L1[c] = emb[labels_L1 == c].mean(axis=0)
    log.info("  L2: clustering %d L1 centroids → %d communities",
             N_CLUSTERS_L1, N_CLUSTERS_L2)
    cl_L2 = AgglomerativeClustering(
        n_clusters=N_CLUSTERS_L2, metric="cosine", linkage="average")
    L1_to_L2 = cl_L2.fit_predict(centroids_L1)
    df_idx["L2"] = [L1_to_L2[l] for l in labels_L1]

    centroids_L2 = np.zeros((N_CLUSTERS_L2, emb.shape[1]))
    for c in range(N_CLUSTERS_L2):
        centroids_L2[c] = centroids_L1[L1_to_L2 == c].mean(axis=0)
    log.info("  L3: clustering %d L2 centroids → %d communities",
             N_CLUSTERS_L2, N_CLUSTERS_L3)
    cl_L3 = AgglomerativeClustering(
        n_clusters=N_CLUSTERS_L3, metric="cosine", linkage="average")
    L2_to_L3 = cl_L3.fit_predict(centroids_L2)
    df_idx["L3"] = [L2_to_L3[l2] for l2 in df_idx["L2"]]

    df_idx.to_csv(OUT_DIR / "phase3_node_clusters.csv", index=False)
    log.info("  Cluster assignments saved")

    # ---- Cohesive LLM summaries per cluster per level ----
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for level, n_clusters in [(1, N_CLUSTERS_L1), (2, N_CLUSTERS_L2),
                              (3, N_CLUSTERS_L3)]:
        out_csv = OUT_DIR / f"phase3_summaries_L{level}.csv"
        emb_out = OUT_DIR / f"phase3_summaries_emb_L{level}.npy"

        # resume if already done
        done_summaries: dict[int, str] = {}
        if out_csv.exists():
            prev = pd.read_csv(out_csv)
            done_summaries = {int(r["cluster"]): r["summary"]
                              for _, r in prev.iterrows()
                              if isinstance(r.get("summary"), str)}
            log.info("  L%d: resume — %d already summarized", level,
                     len(done_summaries))

        log.info("  L%d: generating %d cohesive summaries via LLM...",
                 level, n_clusters)
        rows = []
        t0 = time.time()
        for cid in range(n_clusters):
            mask = df_idx[f"L{level}"] == cid
            members = df_idx[mask]
            chunks = members[members["node_type"] == "chunk"]
            entities = members[members["node_type"] == "entity"]

            paper_dois = [n.replace("chunk::", "") for n in chunks["node_id"]]
            sample_titles = [titles_by_doi.get(d, d)
                             for d in paper_dois[:25] if d in titles_by_doi]
            entity_labels = [n.replace("entity::", "")
                             for n in entities["node_id"]][:40]

            # Collect top relation triplets from this community
            community_entity_set = set(entity_labels)
            community_trips = []
            for d in paper_dois:
                for t in rel_by_doi.get(d, []):
                    h = str(t.get("h", "")).strip().lower()
                    rel = str(t.get("r", "")).strip()
                    tt = str(t.get("t", "")).strip().lower()
                    # prioritize triplets whose entities are in this community
                    if h in community_entity_set or tt in community_entity_set:
                        community_trips.append(f"({h}, {rel}, {tt})")
            # dedup & cap
            seen = set()
            triplet_strs = []
            for t in community_trips:
                if t not in seen:
                    seen.add(t)
                    triplet_strs.append(t)
                    if len(triplet_strs) >= 30:
                        break

            if cid in done_summaries and done_summaries[cid]:
                summary = done_summaries[cid]
            else:
                summary = _llm_summarize(
                    api_key, sample_titles, entity_labels,
                    triplet_strs, level)
                time.sleep(INTER_CALL_S)

            rows.append({
                "cluster": cid,
                "level": level,
                "n_chunks": int(len(chunks)),
                "n_entities": int(len(entities)),
                "summary": summary,
                "top_entities": " | ".join(entity_labels[:20]),
            })

            if (cid + 1) % 10 == 0 or cid + 1 == n_clusters:
                log.info("    L%d: %d/%d   elapsed=%.0fs",
                         level, cid + 1, n_clusters, time.time() - t0)

        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(out_csv, index=False)
        log.info("  L%d summaries saved → %s", level, out_csv)

        # embed summaries
        texts = df_sum["summary"].fillna("").astype(str).tolist()
        X = sbert.encode(texts, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
        np.save(emb_out, X)
        log.info("  L%d summary embeddings saved → %s  shape=%s",
                 level, emb_out, X.shape)

    log.info("PHASE 3 done")


SUMMARY_PROMPT = """You are a research synthesizer for a plastic recycling literature corpus.

Generate a 100-150 word cohesive synthesis (NOT a list) of the research \
community below. The synthesis must integrate THREE information sources:
- representative paper titles (context)
- key entities (what is studied)
- relation triplets (h, relation, t) extracted from these papers (HOW \
  entities interact)

Your synthesis should:

1. Identify the unifying research thread connecting these papers (a shared \
   material class, technique family, or application domain).
2. Explain how the dominant techniques relate to the stated objectives — \
   USE THE TRIPLETS to ground these connections (e.g., "the community \
   applies pyrolysis to mixed plastic waste, which produces bio-oil and \
   aromatic compounds").
3. Surface emergent insight: what does this community collectively reveal \
   about HOW the entities interact, that is not obvious from any single \
   paper.

Be technical, specific, and use domain vocabulary. Output a single \
paragraph of flowing prose. No bullet points, no preamble, no markdown.

Granularity level of this community: L{level} (level 1 = finest, level 3 = broadest).

Up to 25 representative paper titles:
{titles}

Top TMO entities for this community:
{entities}

Representative relation triplets (h, relation, t) from the community:
{relations}
"""


def _llm_summarize(api_key: str, titles: list[str], entities: list[str],
                   relations: list[str], level: int) -> str:
    import requests
    prompt = SUMMARY_PROMPT.format(
        level=level,
        titles="\n".join(f"- {t}" for t in titles) or "(none)",
        entities=", ".join(entities) or "(none)",
        relations="\n".join(f"- {r}" for r in relations) or "(none)",
    )
    payload = {
        "model": MODEL,
        "max_tokens": SUMMARY_MAX_TOKENS,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(OPENROUTER_URL, json=payload,
                              headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            text = (msg.get("content") or msg.get("reasoning") or "").strip()
            if text:
                return text
        except Exception as ex:
            if attempt == MAX_RETRIES - 1:
                log.warning("    LLM summary failed: %s", ex)
            time.sleep(2)
    return ""


# ===========================================================================
# PHASE 4  --  Retrieval evaluation (9 conditions)
# ===========================================================================

def run_phase4():
    log.info("PHASE 4 — Retrieval evaluation")

    df = load_corpus()
    doi_to_idx = {d: i for i, d in enumerate(df["doi"])}

    # Lexical doc terms
    doc_terms = [tokenize(a) for a in df["abstract"].tolist()]

    # Paper TMO terms (for sanity / lexical-TMO baseline reuse)
    paper_tmo_terms = []
    for _, r in df.iterrows():
        ents = (parse_tmo_field(r["techniques"])
                + parse_tmo_field(r["methods"])
                + parse_tmo_field(r["objectives"]))
        s = set()
        for e in ents:
            s |= tokenize(e)
        paper_tmo_terms.append(s)

    # ---- Load all paper-level embeddings (aligned with df by row) ----
    log.info("  Loading paper-level embeddings...")
    embs = {}
    for name in ["sbert", "chemberta", "specter2", "qwen3", "semcse"]:
        p = EMB_DIR / f"{name}.npy"
        if p.exists():
            X = np.load(p)
            if X.shape[0] != len(df):
                log.warning("  %s shape mismatch (%d vs %d); skipping",
                            name, X.shape[0], len(df))
                continue
            embs[name] = X
            log.info("    %s: %s", name, X.shape)

    # ---- Load entity embeddings (HyGRAG-fidelity) ----
    entity_embs = None
    entity_list = []
    entity_to_emb_idx = {}
    p_ent = OUT_DIR / "phase2_entity_emb.npy"
    p_ent_idx = OUT_DIR / "phase2_entity_index.csv"
    if p_ent.exists() and p_ent_idx.exists():
        entity_embs = np.load(p_ent)
        df_ent = pd.read_csv(p_ent_idx)
        entity_list = df_ent["entity"].tolist()
        entity_to_emb_idx = {e: i for i, e in enumerate(entity_list)}
        log.info("    entity embeddings: %s", entity_embs.shape)

    # ---- Load Node2Vec (subset to chunk rows in df order) ----
    df_idx = pd.read_csv(NODE_INDEX)
    n2v_full = np.load(NODE2VEC_PATH)
    chunks = df_idx[df_idx["node_type"] == "chunk"].copy()
    chunks["doi"] = chunks["node_id"].str.replace("chunk::", "", regex=False)
    chunk_doi_to_row = dict(zip(chunks["doi"], chunks["row"]))
    n2v = np.zeros((len(df), n2v_full.shape[1]))
    for i, d in enumerate(df["doi"]):
        if d in chunk_doi_to_row:
            n2v[i] = n2v_full[chunk_doi_to_row[d]]
    # re-normalize
    nn = np.linalg.norm(n2v, axis=1, keepdims=True)
    nn[nn == 0] = 1
    n2v = n2v / nn
    log.info("    node2vec: %s", n2v.shape)

    # ---- Load community assignments + summaries (for bi-level retrieval) ----
    # Load relation triplets per DOI (for relation-aware retrieval, condition J)
    rel_by_doi: dict[str, list[dict]] = {}
    paper_entity_set: dict[str, set[str]] = defaultdict(set)
    if RELS_PATH.exists():
        df_rel = pd.read_csv(RELS_PATH)
        df_rel = df_rel[df_rel["status"] == "ok"]
        for _, r in df_rel.iterrows():
            try:
                trips = json.loads(r["triplets_json"])
                if trips:
                    rel_by_doi[r["doi"]] = trips
                    for t in trips:
                        h = str(t.get("h", "")).strip().lower()
                        tt = str(t.get("t", "")).strip().lower()
                        if h: paper_entity_set[r["doi"]].add(h)
                        if tt: paper_entity_set[r["doi"]].add(tt)
            except Exception:
                continue
        log.info("  Loaded relations for %d papers", len(rel_by_doi))

    # Inverted index: entity → papers that mention it in triplets
    entity_to_papers: dict[str, set[str]] = defaultdict(set)
    for d, ents in paper_entity_set.items():
        for e in ents:
            entity_to_papers[e].add(d)

    df_clusters = pd.read_csv(OUT_DIR / "phase3_node_clusters.csv")
    chunk_clusters = df_clusters[df_clusters["node_type"] == "chunk"].copy()
    chunk_clusters["doi"] = chunk_clusters["node_id"].str.replace(
        "chunk::", "", regex=False)
    doi_to_L1 = dict(zip(chunk_clusters["doi"], chunk_clusters["L1"]))
    L1_to_dois: dict[int, list[str]] = defaultdict(list)
    for d, l in doi_to_L1.items():
        L1_to_dois[int(l)].append(d)

    summary_embs = {}
    for level in [1, 2, 3]:
        p = OUT_DIR / f"phase3_summaries_emb_L{level}.npy"
        if p.exists():
            summary_embs[level] = np.load(p)
            log.info("    L%d summaries: %s", level, summary_embs[level].shape)

    # ---- Citation pairs ----
    pairs = pd.read_csv(EDGES)
    log.info("  %d citation pairs to evaluate", len(pairs))

    log.info("  Running 9 retrieval conditions @ K=%d...", TOP_K)
    rows = []
    skipped = 0

    for _, row in pairs.iterrows():
        src_doi, tgt_doi = row["source_doi"], row["target_doi"]
        if src_doi not in doi_to_idx or tgt_doi not in doi_to_idx:
            skipped += 1
            continue
        si, ti = doi_to_idx[src_doi], doi_to_idx[tgt_doi]
        src_lex = doc_terms[si]
        if not src_lex:
            skipped += 1
            continue

        # A. Lexical baseline
        top_a = _score_lexical(src_lex, doc_terms, si, TOP_K)
        hit_a = int(ti in top_a)

        # B. SBERT dense
        top_b = _score_dense(embs["sbert"], si, TOP_K) if "sbert" in embs else []
        hit_b = int(ti in top_b)

        # C. ChemBERTa dense
        top_c = _score_dense(embs["chemberta"], si, TOP_K) if "chemberta" in embs else []
        hit_c = int(ti in top_c)

        # D. SPECTER2 dense
        top_d = _score_dense(embs["specter2"], si, TOP_K) if "specter2" in embs else []
        hit_d = int(ti in top_d)

        # E. Node2Vec (structural over hybrid graph)
        top_e = _score_dense(n2v, si, TOP_K)
        hit_e = int(ti in top_e)

        # F. Community-aware (HyGRAG-lite: top-k communities at L1 → expand members)
        top_f = _score_community(src_doi, doi_to_L1, L1_to_dois,
                                  summary_embs.get(1), embs.get("sbert"),
                                  doi_to_idx, si, TOP_K)
        hit_f = int(ti in top_f)

        # G. HyGRAG bi-level fusion: RRF over (Lexical, SPECTER2, Node2Vec, Community)
        top_g = _rrf([top_a, top_d, top_e, top_f], TOP_K)
        hit_g = int(ti in top_g)

        # H. SPECTER2 + TMO lexical RRF (citation-aware + structured terms)
        top_h_tmo = _score_lexical(paper_tmo_terms[si], doc_terms, si, TOP_K)
        top_h = _rrf([top_d, top_h_tmo], TOP_K)
        hit_h = int(ti in top_h)

        # I. Full RRF: all dense + node2vec + lexical
        all_lists = [top_a, top_b, top_c, top_d, top_e]
        top_i = _rrf(all_lists, TOP_K)
        hit_i = int(ti in top_i)

        # L. Qwen3 dense (SOTA multilingual MTEB #1)
        top_l = _score_dense(embs["qwen3"], si, TOP_K) if "qwen3" in embs else []
        hit_l = int(ti in top_l)

        # M. RRF best-of-class: lexical + SBERT + SPECTER2 + Qwen3
        top_m = _rrf([top_a, top_b, top_d, top_l], TOP_K) if top_l else top_i
        hit_m = int(ti in top_m)

        # S. SemCSE (Brinner & Zarriess, EMNLP 2025) — semantic contrastive
        #    explicitly designed against citation-based training paradigm
        top_s = _score_dense(embs["semcse"], si, TOP_K) if "semcse" in embs else []
        hit_s = int(ti in top_s)

        # T. RRF best-with-SemCSE: lex + SBERT + SPECTER2 + Qwen3 + SemCSE
        top_t = _rrf([top_a, top_b, top_d, top_l, top_s], TOP_K) if top_s else top_m
        hit_t = int(ti in top_t)

        # J. Relation-aware retrieval (HyGRAG-fidelity):
        #    1. retrieve top-k entities by similarity to query (SBERT)
        #    2. expand to papers mentioning those entities in triplets
        #    3. rank by SPECTER2 similarity
        src_ents = paper_entity_set.get(src_doi, set())
        if src_ents and entity_embs is not None and "sbert" in embs:
            # Build query vector as mean of src entity embeddings
            src_ent_idxs = [entity_to_emb_idx[e] for e in src_ents
                            if e in entity_to_emb_idx]
            if src_ent_idxs:
                q_vec = entity_embs[src_ent_idxs].mean(axis=0)
                q_vec /= max(np.linalg.norm(q_vec), 1e-9)
                # Find top-30 entities semantically close to query
                e_sims = entity_embs @ q_vec
                top_ent_idx = np.argsort(-e_sims)[:30]
                top_entities = {entity_list[i] for i in top_ent_idx}
                # Expand to papers
                rel_cand_dois = set()
                for e in top_entities:
                    rel_cand_dois.update(entity_to_papers.get(e, set()))
                rel_cand_dois.discard(src_doi)
                rel_cand_idx = [doi_to_idx[d] for d in rel_cand_dois
                                if d in doi_to_idx and doi_to_idx[d] != si]
                if rel_cand_idx and "specter2" in embs:
                    sims = embs["specter2"][rel_cand_idx] @ embs["specter2"][si]
                    order = np.argsort(-sims)
                    top_j = [rel_cand_idx[o] for o in order[:TOP_K]]
                else:
                    top_j = []
            else:
                top_j = []
        else:
            top_j = []
        hit_j = int(ti in top_j)

        # K. HyGRAG-FULL: bi-level (community) + relation + dense, fused
        top_k_full = _rrf([top_d, top_e, top_f, top_j], TOP_K)
        hit_k = int(ti in top_k_full)

        rows.append({
            "source_doi": src_doi, "target_doi": tgt_doi,
            "hit_A_lexical":         hit_a,
            "hit_B_sbert":           hit_b,
            "hit_C_chemberta":       hit_c,
            "hit_D_specter2":        hit_d,
            "hit_E_node2vec":        hit_e,
            "hit_F_community":       hit_f,
            "hit_G_hygrag_bilevel":  hit_g,
            "hit_H_specter2_tmo":    hit_h,
            "hit_I_full_rrf":        hit_i,
            "hit_J_relation":        hit_j,
            "hit_K_hygrag_full":     hit_k,
            "hit_L_qwen3":           hit_l,
            "hit_M_rrf_best":        hit_m,
            "hit_S_semcse":          hit_s,
            "hit_T_rrf_with_semcse": hit_t,
        })

    df_r = pd.DataFrame(rows)
    df_r.to_csv(RESULTS_CSV, index=False)
    log.info("  Evaluated %d pairs (skipped %d)", len(df_r), skipped)

    # ---- Stats ----
    cols = [c for c in df_r.columns if c.startswith("hit_")]
    rec = {c.replace("hit_", ""): round(df_r[c].mean(), 4) for c in cols}
    cis = {c.replace("hit_", ""): list(bootstrap_ci(df_r[c].values)) for c in cols}

    base = df_r["hit_A_lexical"].values
    wilcox = {}
    for c in cols:
        if c == "hit_A_lexical":
            continue
        w, p = safe_wilcoxon(df_r[c].values, base)
        wilcox[f"{c}_vs_A"] = {"W": w, "p": p}

    summary = {
        "n_pairs": len(df_r),
        "n_skipped": skipped,
        "top_k": TOP_K,
        "recall": rec,
        "bootstrap_ci": cis,
        "wilcoxon_vs_lexical": wilcox,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("HYGRAG-AWS v3 RETRIEVAL — plastic corpus")
    print("=" * 70)
    print(f"Pairs evaluated: {len(df_r)}\n")
    print(f"Recall@{TOP_K}:")
    for k, v in rec.items():
        c = cis[k]
        print(f"  {k:25s}: {v:.4f}  CI[{c[0]:.4f},{c[1]:.4f}]")
    print(f"\nWilcoxon vs A_lexical (one-sided greater):")
    for name, t in wilcox.items():
        if t["W"] is None:
            print(f"  {name:35s}: insufficient diffs")
        else:
            star = ""
            if t["p"] is not None:
                if t["p"] < 0.001:   star = " ***"
                elif t["p"] < 0.01:  star = " **"
                elif t["p"] < 0.05:  star = " *"
            print(f"  {name:35s}: W={t['W']:.0f}, p={t['p']:.3e}{star}")
    print(f"\nOutputs in {OUT_DIR}/")


def _score_lexical(query, doc_terms, exclude, top_k):
    if not query:
        return []
    scored = []
    for i, t in enumerate(doc_terms):
        if i == exclude:
            continue
        ov = len(query & t)
        if ov:
            scored.append((i, ov))
    scored.sort(key=lambda x: -x[1])
    return [i for i, _ in scored[:top_k]]


def _score_dense(X, si, top_k):
    sims = X @ X[si]
    sims[si] = -np.inf
    return list(np.argsort(-sims)[:top_k])


def _score_community(src_doi, doi_to_L1, L1_to_dois, summary_embs_L1,
                     paper_embs, doi_to_idx, si, top_k, n_communities=5):
    if summary_embs_L1 is None or paper_embs is None or src_doi not in doi_to_L1:
        return []
    # find top communities by similarity between src embedding and L1 summaries
    src_v = paper_embs[si]
    sims = summary_embs_L1 @ src_v
    top_comm = np.argsort(-sims)[:n_communities]
    cand = set()
    for cid in top_comm:
        cand.update(L1_to_dois.get(int(cid), []))
    cand_idx = [doi_to_idx[d] for d in cand
                if d in doi_to_idx and doi_to_idx[d] != si]
    if not cand_idx:
        return []
    cs = paper_embs[cand_idx] @ src_v
    order = np.argsort(-cs)
    return [cand_idx[o] for o in order[:top_k]]


def _rrf(rankings, top_k, k_const=60):
    scores: dict[int, float] = defaultdict(float)
    for rl in rankings:
        for r, idx in enumerate(rl):
            scores[idx] += 1.0 / (k_const + r + 1)
    return sorted(scores, key=lambda x: -scores[x])[:top_k]


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["0", "2", "3", "4", "all"], default="all")
    ap.add_argument("--include-api", action="store_true",
                    help="(phase 0) also compute Gemini/Qwen3/NV-Embed via API")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("0", "all"):
        run_phase0(include_api=args.include_api)
    if args.phase in ("2", "all"):
        run_phase2()
    if args.phase in ("3", "all"):
        run_phase3()
    if args.phase in ("4", "all"):
        run_phase4()


if __name__ == "__main__":
    main()
