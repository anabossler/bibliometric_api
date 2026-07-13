"""
k_sensitivity.py
=======================

Sensitivity of the CSC-score to the number of clusters k.

The CSC-score is defined as

    CSC = 1 - (S_cross * D_w)

where

    S_cross : fraction of within-corpus citations that cross cluster boundaries
    D_w     : citation-flow-weighted mean vocabulary divergence (1 - RBO)
              between cluster pairs

Both components depend on the partition, hence on k. This script re-clusters
the corpus for k in a user-specified range, recomputes both components from
scratch for each k, and reports CSC(k).

Requires the FULL within-corpus citation edge list (not only the edges that
happen to be cross-cluster at k=6).

Outputs:
  runs/k_sensitivity/k_sensitivity.csv
  runs/k_sensitivity/k_sensitivity.json
  runs/k_sensitivity/k_sensitivity.png

Usage:
  python k_sensitivity.py --edges data_checkpoints/edges.csv
  python k_sensitivity.py --edges data_checkpoints/edges.csv --k-min 2 --k-max 12
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ksens")

# ---------------------------------------------------------------------------
# Paths (aligned with step14/step15)
# ---------------------------------------------------------------------------

DOI_ORDER = Path("runs/hygrag_v3/phase0_doi_order.csv")
EMB_PATH = Path("runs/hygrag_v3/phase0_embeddings/sbert.npy")
ABSTRACTS = Path("backup_recycled_a/full_corpus/abstracts_full.csv")

OUT_DIR = Path("runs/k_sensitivity")

SEED = 42
TOP_N = 50      # vocabulary list length for RBO
P_RBO = 0.9     # RBO persistence parameter (as in the paper)

STOPWORDS = set(
    "the and for that this with from are was were been have has had not but "
    "its can will also into than more which their these they would could "
    "should about each between through during after before other some such "
    "only over how our who what when where doi org https http www "
    "available online published all any both did does done either however "
    "just much most must same several since still very while using used use "
    "results show study paper present".split()
)


# ---------------------------------------------------------------------------
# Metric components
# ---------------------------------------------------------------------------

def rbo(list1: list, list2: list, p: float = P_RBO) -> float:
    """Rank-biased overlap between two ranked lists."""
    if not list1 or not list2:
        return 0.0
    shorter, longer = (list1, list2) if len(list1) <= len(list2) else (list2, list1)
    seen_short, agreement = set(), 0.0
    for depth, item in enumerate(shorter, start=1):
        seen_short.add(item)
        overlap = len(seen_short & set(longer[:depth]))
        agreement += (overlap / depth) * (p ** (depth - 1))
    return (1.0 - p) * agreement


def cluster_vocabularies(labels: np.ndarray, docs: list[str], top_n: int = TOP_N
                         ) -> dict[int, list[str]]:
    """Rank each cluster's vocabulary by class-based TF-IDF (c-TF-IDF)."""
    joined = []
    cluster_ids = sorted(set(labels.tolist()))
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        joined.append(" ".join(docs[i] for i in idx))

    vec = TfidfVectorizer(
        stop_words=list(STOPWORDS),
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(joined)
    terms = np.array(vec.get_feature_names_out())

    vocab = {}
    for row, cid in enumerate(cluster_ids):
        scores = matrix[row].toarray().ravel()
        order = np.argsort(-scores)[:top_n]
        vocab[cid] = [t for t in terms[order] if scores[terms.tolist().index(t)] > 0] \
            if False else list(terms[order])
    return vocab


def compute_components(labels: np.ndarray,
                       docs: list[str],
                       edges_idx: list[tuple[int, int]]) -> dict:
    """Compute S_cross, D_w and CSC for one partition."""
    n_edges = len(edges_idx)
    if n_edges == 0:
        raise ValueError("No usable citation edges.")

    # ---- S_cross: fraction of within-corpus citations crossing clusters ----
    flow = Counter()          # (ci, cj) unordered -> citation count
    n_cross = 0
    for src, tgt in edges_idx:
        ci, cj = int(labels[src]), int(labels[tgt])
        if ci != cj:
            n_cross += 1
            flow[tuple(sorted((ci, cj)))] += 1
    s_cross = n_cross / n_edges

    # ---- D_w: flow-weighted mean vocabulary divergence ----
    vocab = cluster_vocabularies(labels, docs)

    divergences, weights = [], []
    for (ci, cj), w in flow.items():
        d_ij = 1.0 - rbo(vocab[ci], vocab[cj], p=P_RBO)
        divergences.append(d_ij)
        weights.append(w)

    if not divergences:
        # k=1 degenerate case: no cross-cluster pairs exist
        d_w, d_unweighted = float("nan"), float("nan")
    else:
        divergences = np.asarray(divergences)
        weights = np.asarray(weights, dtype=float)
        d_w = float(np.average(divergences, weights=weights))
        d_unweighted = float(divergences.mean())

    csc = 1.0 - (s_cross * d_w) if divergences.size else 1.0

    return {
        "n_clusters": int(len(set(labels.tolist()))),
        "n_edges": n_edges,
        "n_cross_edges": n_cross,
        "s_cross": round(s_cross, 4),
        "d_w": round(d_w, 4) if divergences.size else None,
        "d_unweighted": round(d_unweighted, 4) if divergences.size else None,
        "csc": round(csc, 4),
        "cluster_sizes": {int(c): int(n) for c, n in
                          sorted(Counter(labels.tolist()).items())},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True,
                    help="Citation edge list. May be the raw OpenAlex reference "
                         "dump (citing,cited as W-IDs); it is filtered down to "
                         "within-corpus edges automatically.")
    ap.add_argument("--source-col", default="citing")
    ap.add_argument("--target-col", default="cited")
    ap.add_argument("--corpus", default="corpus_plastic_recycling.csv",
                    help="Corpus CSV providing the OpenAlex-ID to DOI mapping")
    ap.add_argument("--corpus-id-col", default="id",
                    help="Column in --corpus holding the OpenAlex work ID")
    ap.add_argument("--corpus-doi-col", default="doi")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--linkage", default="average",
                    choices=["average", "ward", "complete"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- corpus in canonical order ----
    order = pd.read_csv(DOI_ORDER)
    dois = order["doi"].astype(str).str.lower().str.strip().tolist()
    doi_to_idx = {d: i for i, d in enumerate(dois)}
    log.info("Corpus: %d papers in canonical order", len(dois))

    emb = np.load(EMB_PATH)
    if emb.shape[0] != len(dois):
        raise ValueError(f"Embedding rows ({emb.shape[0]}) != corpus size ({len(dois)})")
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    log.info("Embeddings: %s", emb.shape)

    # ---- abstracts, aligned to canonical order ----
    df_abs = pd.read_csv(ABSTRACTS)
    df_abs["doi"] = df_abs["doi"].astype(str).str.lower().str.strip()
    abs_by_doi = dict(zip(df_abs["doi"], df_abs["abstract"].astype(str)))
    docs = [abs_by_doi.get(d, "") for d in dois]
    n_empty = sum(1 for d in docs if not d.strip())
    log.info("Abstracts: %d aligned (%d empty)", len(docs), n_empty)

    # ---- citation edges, filtered to within-corpus ----
    # The edge list may key papers either by DOI or by OpenAlex work ID.
    # Build a key -> canonical-index map that accepts both.
    key_to_idx = dict(doi_to_idx)

    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        cdf = pd.read_csv(corpus_path)
        if args.corpus_id_col in cdf.columns and args.corpus_doi_col in cdf.columns:
            n_mapped = 0
            for wid, doi in zip(cdf[args.corpus_id_col], cdf[args.corpus_doi_col]):
                d = str(doi).lower().strip()
                if d in doi_to_idx:
                    w = str(wid).lower().strip()
                    key_to_idx[w] = doi_to_idx[d]
                    key_to_idx[w.rsplit("/", 1)[-1]] = doi_to_idx[d]  # bare W-id
                    n_mapped += 1
            log.info("OpenAlex-ID mapping: %d corpus papers", n_mapped)
        else:
            log.warning("Corpus file lacks '%s'/'%s'; relying on DOI keys only",
                        args.corpus_id_col, args.corpus_doi_col)
    else:
        log.warning("Corpus file %s not found; relying on DOI keys only", corpus_path)

    ed = pd.read_csv(args.edges)
    if args.source_col not in ed.columns or args.target_col not in ed.columns:
        raise ValueError(
            f"Edge file must contain '{args.source_col}' and '{args.target_col}'. "
            f"Found: {ed.columns.tolist()}"
        )
    log.info("Raw edge file: %d rows", len(ed))

    src_keys = ed[args.source_col].astype(str).str.lower().str.strip()
    tgt_keys = ed[args.target_col].astype(str).str.lower().str.strip()

    seen, edges_idx = set(), []
    for s_key, t_key in zip(src_keys, tgt_keys):
        i, j = key_to_idx.get(s_key), key_to_idx.get(t_key)
        if i is None or j is None or i == j:
            continue
        if (i, j) in seen:
            continue
        seen.add((i, j))
        edges_idx.append((i, j))

    log.info("Within-corpus citation edges: %d (both endpoints in corpus, deduplicated)",
             len(edges_idx))
    if len(edges_idx) < 100:
        raise ValueError(
            "Fewer than 100 within-corpus edges resolved. Check --corpus, "
            "--corpus-id-col and the edge column names."
        )

    # ---- sweep k ----
    rows = []
    for k in range(args.k_min, args.k_max + 1):
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine" if args.linkage != "ward" else "euclidean",
            linkage=args.linkage,
        )
        labels = model.fit_predict(emb)
        stats = compute_components(labels, docs, edges_idx)
        stats["k"] = k
        rows.append(stats)
        log.info("k=%2d  S_cross=%.4f  D_w=%.4f  CSC=%.4f  (%d/%d cross edges)",
                 k, stats["s_cross"], stats["d_w"] or float("nan"),
                 stats["csc"], stats["n_cross_edges"], stats["n_edges"])

    df = pd.DataFrame(rows)[
        ["k", "n_edges", "n_cross_edges", "s_cross", "d_w", "d_unweighted", "csc"]
    ]
    df.to_csv(OUT_DIR / "k_sensitivity.csv", index=False)
    (OUT_DIR / "k_sensitivity.json").write_text(
        json.dumps({"seed": SEED, "p_rbo": P_RBO, "top_n": TOP_N,
                    "linkage": args.linkage, "results": rows}, indent=2)
    )

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(7, 4.2))
        ax1.plot(df["k"], df["csc"], "o-", color="#1b4965", label="CSC")
        ax1.set_xlabel("Number of clusters (k)")
        ax1.set_ylabel("CSC-score", color="#1b4965")
        ax1.tick_params(axis="y", labelcolor="#1b4965")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(df["k"], df["s_cross"], "s--", color="#c1666b", label="S_cross")
        ax2.plot(df["k"], df["d_w"], "^--", color="#5fa8d3", label="D_w")
        ax2.set_ylabel("Component value")
        ax2.set_ylim(0, 1.05)

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
        plt.title("CSC-score sensitivity to the number of clusters")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "k_sensitivity.png", dpi=180)
        log.info("wrote %s", OUT_DIR / "k_sensitivity.png")
    except Exception as exc:
        log.warning("Plot skipped: %s", exc)

    print("\n" + "=" * 72)
    print("CSC SENSITIVITY TO k")
    print("=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)
    print(f"\nS_cross range: {df['s_cross'].min():.4f} - {df['s_cross'].max():.4f}")
    print(f"D_w range:     {df['d_w'].min():.4f} - {df['d_w'].max():.4f}")
    print(f"CSC range:     {df['csc'].min():.4f} - {df['csc'].max():.4f}")
    log.info("wrote %s", OUT_DIR / "k_sensitivity.csv")


if __name__ == "__main__":
    main()
