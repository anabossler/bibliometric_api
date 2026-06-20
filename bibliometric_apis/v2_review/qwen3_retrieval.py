"""
qwen3_retrieval.py
==================

Tests whether a dense encoder (Qwen3-Embedding-8B, 4096-dim, via the
OpenRouter embeddings endpoint) recovers cited cross-cluster papers better than
the MiniLM SBERT baseline (recall@50 = 0.338) under AWS vocabulary
fragmentation.

  - Ground truth = verified OpenAlex citation edges (697 cross-cluster pairs),
    independent of the encoder and of cluster membership.
  - The encoder change is the only manipulation; the evaluation protocol is
    byte-for-byte the HyGRAG phase4 protocol (cosine top-K over the same DOI
    order, same pairs, same K), so the Qwen3 column is directly comparable to
    the existing A_lexical / B_sbert / D_specter2 columns.
  - No held-out split is needed: dense retrieval does not train on the corpus;
    it encodes abstracts and ranks by cosine. The citation edges are never seen
    by the encoder.

PIPELINE (two phases, resumable)
--------------------------------
  Phase EMBED : read abstracts in the canonical phase0 DOI order, call the
                OpenRouter embeddings endpoint for Qwen3-8B in batches with a
                per-batch checkpoint, write qwen3_aligned.npy (N, 4096).
  Phase EVAL  : cosine top-K retrieval over the 697 citation pairs, recall@K,
                bootstrap CI, Wilcoxon vs the stored SBERT hit column.

The embed phase checkpoints after every batch to qwen3_partial.npy +
qwen3_progress.json. Re-running resumes from the last completed row, so an
interrupted run never loses credits or progress.

INPUTS
------
  runs/hygrag_v3/phase0_doi_order.csv          (canonical DOI order, col: doi)
  backup_recycled_a/full_corpus/abstracts_full.csv
  backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv
  runs/hygrag_v3/phase4_results.csv            (for the stored SBERT hit column)

OUTPUTS (runs/qwen3_retrieval/)
-------------------------------
  qwen3_aligned.npy           (N, 4096) float32, row i == phase0 DOI order row i
  qwen3_retrieval.csv         per-pair hits
  qwen3_summary.json          recall + CI + Wilcoxon vs SBERT

USAGE
-----
  python qwen3_retrieval.py --phase embed
  python qwen3_retrieval.py --phase eval
  python qwen3_retrieval.py --phase all     (default)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import wilcoxon

load_dotenv(Path.home() / "Desktop/openalex/.env")

OPENROUTER_API_KEY = os.getenv("YYYYY", "")
OPENROUTER_URL = "YYYYY"
MODEL = "qwen/qwen3-embedding-8b"

BASE_DIR = Path("backup_recycled_a/full_corpus")
ABSTRACTS = BASE_DIR / "abstracts_full.csv"
EDGES = BASE_DIR / "citation_edges_cross_cluster.csv"

HYGRAG_DIR = Path("runs/hygrag_v3")
DOI_ORDER = HYGRAG_DIR / "phase0_doi_order.csv"
PHASE4_RESULTS = HYGRAG_DIR / "phase4_results.csv"

OUT_DIR = Path("runs/qwen3_retrieval")
EMB_PATH = OUT_DIR / "qwen3_aligned.npy"
PARTIAL_PATH = OUT_DIR / "qwen3_partial.npy"
PROGRESS_PATH = OUT_DIR / "qwen3_progress.json"

TOP_K = 50
BATCH_SIZE = 32
MAX_RETRIES = 5
RETRY_SLEEP = 3.0
EMBED_DIM = 4096


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_doi_order() -> list[str]:
    """Canonical DOI order used by phase4; row i of every .npy maps to dois[i]."""
    df = pd.read_csv(DOI_ORDER)
    col = "doi" if "doi" in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def load_abstracts(dois: list[str]) -> list[str]:
    """Abstract text in the exact phase0 order; empty string if missing."""
    df = pd.read_csv(ABSTRACTS)
    text_col = "abstract" if "abstract" in df.columns else df.columns[-1]
    lookup = dict(zip(df["doi"].astype(str), df[text_col].astype(str)))
    out = []
    missing = 0
    for d in dois:
        t = lookup.get(d, "")
        if not isinstance(t, str) or t == "nan":
            t = ""
        if not t:
            missing += 1
        out.append(t)
    if missing:
        log(f"WARNING: {missing} DOIs have no abstract (encoded as empty).")
    return out


def embed_batch(texts: list[str]) -> np.ndarray:
    """Call the OpenRouter embeddings endpoint for one batch.

    Returns (len(texts), EMBED_DIM). Empty strings are sent as a single space
    so the API still returns a vector; those rows are later effectively inert
    (near-zero norm contributes nothing useful, and the corresponding DOI is
    typically not a citation endpoint anyway).
    """
    payload_texts = [t if t.strip() else " " for t in texts]
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {"model": MODEL, "input": payload_texts}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENROUTER_URL, json=body, headers=headers,
                                 timeout=120)
            resp.raise_for_status()
            data = resp.json()["data"]
            # API may not preserve order; sort by the returned index field.
            data_sorted = sorted(data, key=lambda d: d.get("index", 0))
            vecs = [d["embedding"] for d in data_sorted]
            arr = np.asarray(vecs, dtype=np.float32)
            if arr.shape[0] != len(texts):
                raise ValueError(
                    f"batch returned {arr.shape[0]} vecs for {len(texts)} inputs")
            return arr
        except Exception as e:
            log(f"  batch attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP * attempt)
    raise RuntimeError("unreachable")


def run_embed() -> None:
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set (.env or export).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dois = load_doi_order()
    n = len(dois)
    log(f"Canonical DOI order: {n} papers")

    if EMB_PATH.exists():
        existing = np.load(EMB_PATH)
        if existing.shape == (n, EMBED_DIM):
            log(f"{EMB_PATH} already complete ({existing.shape}). Skipping embed.")
            return
        log(f"{EMB_PATH} exists but shape {existing.shape} != ({n},{EMBED_DIM}); "
            f"recomputing.")

    texts = load_abstracts(dois)

    # Resume from checkpoint if present.
    start_row = 0
    emb = np.zeros((n, EMBED_DIM), dtype=np.float32)
    if PARTIAL_PATH.exists() and PROGRESS_PATH.exists():
        prog = json.loads(PROGRESS_PATH.read_text())
        if prog.get("n") == n and prog.get("model") == MODEL:
            partial = np.load(PARTIAL_PATH)
            done = int(prog.get("rows_done", 0))
            if partial.shape == (n, EMBED_DIM) and 0 < done <= n:
                emb = partial
                start_row = done
                log(f"Resuming from checkpoint at row {start_row}/{n}")

    t0 = time.time()
    for b0 in range(start_row, n, BATCH_SIZE):
        b1 = min(b0 + BATCH_SIZE, n)
        emb[b0:b1] = embed_batch(texts[b0:b1])

        np.save(PARTIAL_PATH, emb)
        PROGRESS_PATH.write_text(json.dumps(
            {"n": n, "model": MODEL, "rows_done": b1}, indent=2))

        rate = (b1 - start_row) / max(1e-6, time.time() - t0)
        eta = (n - b1) / max(1e-6, rate)
        log(f"  [{b1}/{n}] {rate:.1f} rows/s  ETA {eta/60:.1f} min")

    np.save(EMB_PATH, emb)
    PARTIAL_PATH.unlink(missing_ok=True)
    PROGRESS_PATH.unlink(missing_ok=True)
    log(f"Embeddings saved -> {EMB_PATH}  shape={emb.shape}")


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def bootstrap_ci(values, n_boot=1000, seed=42):
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def safe_wilcoxon(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if np.count_nonzero(d) < 10:
        return None, None
    w, p = wilcoxon(x, y, alternative="greater")
    return float(w), float(p)


def run_eval() -> None:
    if not EMB_PATH.exists():
        raise FileNotFoundError(
            f"{EMB_PATH} not found. Run --phase embed first.")

    dois = load_doi_order()
    doi_to_idx = {d: i for i, d in enumerate(dois)}
    n = len(dois)

    emb = np.load(EMB_PATH).astype(np.float32)
    if emb.shape[0] != n:
        raise ValueError(
            f"embedding rows {emb.shape[0]} != DOI order {n}; misaligned.")
    log(f"Loaded Qwen3 embeddings: {emb.shape}")
    emb = l2_normalize(emb)

    pairs = pd.read_csv(EDGES)
    log(f"{len(pairs)} citation pairs")

    # Stored SBERT per-pair hits from phase4, keyed by (source_doi, target_doi),
    # so the Wilcoxon comparison is paired on identical pairs.
    sbert_hit = {}
    if PHASE4_RESULTS.exists():
        p4 = pd.read_csv(PHASE4_RESULTS)
        if {"source_doi", "target_doi", "hit_B_sbert"}.issubset(p4.columns):
            for _, r in p4.iterrows():
                sbert_hit[(str(r["source_doi"]), str(r["target_doi"]))] = \
                    int(r["hit_B_sbert"])
            log(f"Loaded {len(sbert_hit)} stored SBERT hits from phase4.")
        else:
            log("phase4_results.csv lacks expected columns; "
                "SBERT comparison skipped.")
    else:
        log("phase4_results.csv not found; SBERT comparison skipped.")

    rows = []
    skipped = 0
    for _, row in pairs.iterrows():
        src_doi = str(row["source_doi"])
        tgt_doi = str(row["target_doi"])
        if src_doi not in doi_to_idx or tgt_doi not in doi_to_idx:
            skipped += 1
            continue
        src_idx = doi_to_idx[src_doi]
        tgt_idx = doi_to_idx[tgt_doi]

        # Cosine of source against all docs; exclude self; take top-K.
        sims = emb @ emb[src_idx]
        sims[src_idx] = -np.inf
        topk = np.argpartition(-sims, TOP_K)[:TOP_K]
        hit_qwen = int(tgt_idx in set(topk.tolist()))

        rec = {
            "source_doi": src_doi,
            "target_doi": tgt_doi,
            "source_cluster": int(row["source_cluster"]),
            "target_cluster": int(row["target_cluster"]),
            "hit_qwen3": hit_qwen,
        }
        key = (src_doi, tgt_doi)
        if key in sbert_hit:
            rec["hit_sbert"] = sbert_hit[key]
        rows.append(rec)

    df_r = pd.DataFrame(rows)
    df_r.to_csv(OUT_DIR / "qwen3_retrieval.csv", index=False)
    log(f"Evaluated {len(df_r)} pairs (skipped {skipped})")

    rec_qwen = float(df_r["hit_qwen3"].mean())
    ci_qwen = bootstrap_ci(df_r["hit_qwen3"].values)

    summary = {
        "model": MODEL,
        "embed_dim": EMBED_DIM,
        "n_pairs": int(len(df_r)),
        "n_skipped": int(skipped),
        "top_k": TOP_K,
        "recall_qwen3": round(rec_qwen, 4),
        "ci_qwen3": [round(ci_qwen[0], 4), round(ci_qwen[1], 4)],
    }

    print("\n" + "=" * 64)
    print("QWEN3-EMBEDDING-8B RETRIEVAL - plastic corpus (AWS)")
    print("=" * 64)
    print(f"Pairs evaluated: {len(df_r)}")
    print(f"\nRecall@{TOP_K}:")
    print(f"  Qwen3-8B (4096d) : {rec_qwen:.4f}  "
          f"CI[{ci_qwen[0]:.4f},{ci_qwen[1]:.4f}]")

    if "hit_sbert" in df_r.columns:
        paired = df_r.dropna(subset=["hit_sbert"])
        rec_sbert = float(paired["hit_sbert"].mean())
        ci_sbert = bootstrap_ci(paired["hit_sbert"].values)
        w_qs, p_qs = safe_wilcoxon(paired["hit_qwen3"].values,
                                   paired["hit_sbert"].values)
        w_sq, p_sq = safe_wilcoxon(paired["hit_sbert"].values,
                                   paired["hit_qwen3"].values)
        summary["paired_n"] = int(len(paired))
        summary["recall_sbert_paired"] = round(rec_sbert, 4)
        summary["ci_sbert_paired"] = [round(ci_sbert[0], 4),
                                      round(ci_sbert[1], 4)]
        summary["wilcoxon_qwen3_gt_sbert"] = {"W": w_qs, "p": p_qs}
        summary["wilcoxon_sbert_gt_qwen3"] = {"W": w_sq, "p": p_sq}

        print(f"  SBERT (paired)   : {rec_sbert:.4f}  "
              f"CI[{ci_sbert[0]:.4f},{ci_sbert[1]:.4f}]   (n={len(paired)})")
        print(f"\nWilcoxon (paired on identical citation pairs):")
        print(f"  Qwen3 > SBERT : W={w_qs}, p={p_qs}")
        print(f"  SBERT > Qwen3 : W={w_sq}, p={p_sq}")
        print()
        if p_qs is not None and p_qs < 0.05:
            print("  RESULT: Qwen3-8B best than SBERT.")
            print("          embedding is retrieval gain.")
        elif p_sq is not None and p_sq < 0.05:
            print("  RESULT: SBERT better than Qwen3-8B on this corpus.")
        else:
            print("  RESULT: Qwen3-8B and SBERT statistically indistinguishable.")

    (OUT_DIR / "qwen3_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutputs in {OUT_DIR}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["embed", "eval", "all"], default="all")
    args = ap.parse_args()

    if args.phase in ("embed", "all"):
        run_embed()
    if args.phase in ("eval", "all"):
        run_eval()


if __name__ == "__main__":
    main()
