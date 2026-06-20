"""
semantic_tmo_retrieval.py
=========================

Semantic retrieval over LLM-extracted TMO representations (technique + method +
objective), as a final ablation for the AWS paper revision.

Two new conditions:
  P. Qwen3 dense over TMO string per paper
  Q. RRF best-of-class with TMO included
     = RRF(Lexical, SBERT, SPECTER2, Qwen3-abstract, Qwen3-TMO, OpenAlex-topics)

Pipeline:
  Phase A — Embed TMO strings via Qwen3 OpenRouter API (~30 min, ~$0.10)
  Phase B — Evaluate as retrieval against the 656 cited cross-cluster pairs

Aligned with paper_tmo.csv order (3043 papers) → matches phase0_doi_order.csv.

Output:
  runs/semantic_tmo/qwen3_tmo.npy                (3043, 4096) embeddings
  runs/semantic_tmo/retrieval_results.csv        per-pair hits
  runs/semantic_tmo/summary.json                 aggregated table

Usage:
  python semantic_tmo_retrieval.py --phase embed   # ~30 min, ~$0.10
  python semantic_tmo_retrieval.py --phase eval    # 2 min
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("semtmo")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOI_ORDER       = Path("runs/hygrag_v3/phase0_doi_order.csv")
TMO_PATH        = Path("runs/paper_tmo/paper_tmo.csv")
ABSTRACTS       = Path("backup_recycled_a/full_corpus/abstracts_full.csv")
EDGES           = Path("backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv")

OUT_DIR         = Path("runs/semantic_tmo")
TMO_EMB         = OUT_DIR / "qwen3_tmo.npy"
RESULTS_CSV     = OUT_DIR / "retrieval_results.csv"
SUMMARY_JSON    = OUT_DIR / "summary.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_URL  = "https://openrouter.ai/api/v1/embeddings"
MODEL           = "qwen/qwen3-embedding-8b"
BATCH_SIZE      = 32
MAX_RETRIES     = 5
TOP_K           = 50


# ===========================================================================
# Helpers
# ===========================================================================

def load_api_key():
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY not in .env")
    return key


def build_tmo_text(row) -> str:
    """Concatenate techniques + methods + objectives into a single string."""
    parts = []
    for col, label in [("techniques", "Techniques"),
                       ("methods",    "Methods"),
                       ("objectives", "Objectives")]:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            # Replace pipes with commas for natural text
            clean = " ".join(val.replace("|", ",").split())
            parts.append(f"{label}: {clean}.")
    if not parts:
        return "(no TMO extracted)"
    return " ".join(parts)


# ===========================================================================
# PHASE A — Embed TMO strings via Qwen3
# ===========================================================================

def phase_embed():
    log.info("PHASE A — Embedding TMO strings via Qwen3-8B")

    api_key = load_api_key()
    order = pd.read_csv(DOI_ORDER)
    tmo = pd.read_csv(TMO_PATH)
    tmo_ok = tmo[tmo["status"] == "ok"].set_index("doi")

    # Build text in canonical order
    texts = []
    for doi in order["doi"]:
        if doi in tmo_ok.index:
            texts.append(build_tmo_text(tmo_ok.loc[doi]))
        else:
            texts.append("(no TMO extracted)")
    log.info("  Built %d TMO strings", len(texts))

    # Sanity: show a sample
    log.info("  Example: %s", texts[0][:200])

    # Resume support: load partial embeddings if present
    embs = None
    start_idx = 0
    if TMO_EMB.exists():
        embs = np.load(TMO_EMB)
        if embs.shape[0] == len(texts):
            log.info("  Already complete; nothing to do.")
            return
        # Detect first all-zero row for resume
        norms = np.linalg.norm(embs, axis=1)
        zero_idx = np.where(norms == 0)[0]
        if len(zero_idx) > 0:
            start_idx = int(zero_idx[0])
            log.info("  Resume from index %d", start_idx)
        else:
            log.info("  Partial file size mismatch; will recompute")
            embs = None

    if embs is None:
        embs = np.zeros((len(texts), 4096), dtype=np.float32)

    n = len(texts)
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    t0 = time.time()

    for i in range(start_idx, n, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        payload = {"model": MODEL, "input": batch}

        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(OPENROUTER_URL, json=payload,
                                  headers=headers, timeout=60)
                if r.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                r.raise_for_status()
                data = r.json()
                for j, item in enumerate(data["data"]):
                    v = np.asarray(item["embedding"], dtype=np.float32)
                    n_ = np.linalg.norm(v)
                    if n_ > 0:
                        v = v / n_
                    embs[i + j] = v
                break
            except Exception as ex:
                if attempt == MAX_RETRIES - 1:
                    log.warning("  batch %d failed after %d retries: %s",
                                i, MAX_RETRIES, ex)
                else:
                    log.info("  batch attempt %d/%d failed: %s",
                             attempt + 1, MAX_RETRIES, ex)
                    time.sleep(2 ** (attempt + 1))

        # Checkpoint every 320 rows
        if (i + BATCH_SIZE) % 320 == 0 or (i + BATCH_SIZE) >= n:
            np.save(TMO_EMB, embs)
            elapsed = time.time() - t0
            done = min(i + BATCH_SIZE, n) - start_idx
            rate = done / max(elapsed, 1e-9)
            eta = (n - (i + BATCH_SIZE)) / max(rate, 1e-9) / 60
            log.info("  [%d/%d] %.1f rows/s | ETA %.1f min",
                     min(i + BATCH_SIZE, n), n, rate, eta)

    np.save(TMO_EMB, embs)
    log.info("PHASE A done — embeddings saved → %s shape=%s",
             TMO_EMB, embs.shape)


# ===========================================================================
# PHASE B — Retrieval eval
# ===========================================================================

STOPWORDS = set(
    "the and for that this with from are was were been have has had not but "
    "its can will also into than more which their these they would could "
    "should about each between through during after before other some such "
    "only over how our who what when where doi org https http www "
    "available online published all any both did does done either "
    "however just much most must same several since still very while "
    "using used use results show study paper present".split()
)


def tokenize(text: str) -> set:
    if not isinstance(text, str):
        return set()
    words = re.findall(r"[a-z]+", text.lower())
    words = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    return set(words) | {f"{words[i]} {words[i+1]}"
                         for i in range(len(words) - 1)}


def phase_eval():
    log.info("PHASE B — Retrieval evaluation (conditions P, Q)")

    order = pd.read_csv(DOI_ORDER)
    doi_to_idx = {d: i for i, d in enumerate(order["doi"])}

    # Load all embeddings
    log.info("  Loading embeddings...")
    sbert = np.load("runs/hygrag_v3/phase0_embeddings/sbert.npy")
    specter2 = np.load("runs/hygrag_v3/phase0_embeddings/specter2.npy")
    qwen3 = np.load("runs/hygrag_v3/phase0_embeddings/qwen3.npy")
    qwen3_tmo = np.load(TMO_EMB)
    log.info("    sbert: %s | specter2: %s", sbert.shape, specter2.shape)
    log.info("    qwen3_abstract: %s | qwen3_tmo: %s",
             qwen3.shape, qwen3_tmo.shape)

    # Topics (sparse)
    topic_M = sparse.load_npz("runs/openalex_topics/topic_vectors.npz")
    log.info("    topics: %s nnz=%d", topic_M.shape, topic_M.nnz)

    # Lexical doc terms
    df_abs = pd.read_csv(ABSTRACTS)
    abstr_by_doi = dict(zip(df_abs["doi"], df_abs["abstract"]))
    doc_terms = [tokenize(abstr_by_doi.get(d, "")) for d in order["doi"]]

    def score_lexical(query, exclude, top_k):
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

    def score_dense(X, si, top_k):
        sims = X @ X[si]
        sims[si] = -np.inf
        return list(np.argsort(-sims)[:top_k])

    def score_topics(si, top_k):
        v = topic_M[si].toarray().ravel()
        if v.sum() == 0:
            return []
        sims = np.asarray(topic_M @ v.reshape(-1, 1)).ravel()
        sims[si] = -np.inf
        return list(np.argsort(-sims)[:top_k])

    def rrf(rankings, top_k, k_const=60):
        scores = defaultdict(float)
        for rl in rankings:
            for r, idx in enumerate(rl):
                scores[idx] += 1.0 / (k_const + r + 1)
        return sorted(scores, key=lambda x: -scores[x])[:top_k]

    # Citation pairs
    pairs = pd.read_csv(EDGES)
    log.info("  Evaluating %d pairs...", len(pairs))

    rows = []
    skipped = 0
    t0 = time.time()
    for ix, row in pairs.iterrows():
        src_doi, tgt_doi = row["source_doi"], row["target_doi"]
        if src_doi not in doi_to_idx or tgt_doi not in doi_to_idx:
            skipped += 1
            continue
        si, ti = doi_to_idx[src_doi], doi_to_idx[tgt_doi]
        src_lex = doc_terms[si]
        if not src_lex:
            skipped += 1
            continue

        top_a = score_lexical(src_lex, si, TOP_K)           # lexical
        top_b = score_dense(sbert, si, TOP_K)               # SBERT
        top_d = score_dense(specter2, si, TOP_K)            # SPECTER2
        top_l = score_dense(qwen3, si, TOP_K)               # Qwen3 abstract
        top_n = score_topics(si, TOP_K)                     # OpenAlex topics
        top_p = score_dense(qwen3_tmo, si, TOP_K)           # NEW: Qwen3 TMO

        # Q. Full best-of-class RRF
        top_q = rrf([top_a, top_b, top_d, top_l, top_n, top_p], TOP_K)

        rows.append({
            "source_doi": src_doi, "target_doi": tgt_doi,
            "hit_A_lexical":      int(ti in top_a),
            "hit_B_sbert":        int(ti in top_b),
            "hit_D_specter2":     int(ti in top_d),
            "hit_L_qwen3":        int(ti in top_l),
            "hit_N_topics":       int(ti in top_n),
            "hit_P_qwen3_tmo":    int(ti in top_p),
            "hit_Q_rrf_full":     int(ti in top_q),
        })

        if (ix + 1) % 100 == 0:
            log.info("    pair %d/%d   elapsed=%.0fs",
                     ix + 1, len(pairs), time.time() - t0)

    df_r = pd.DataFrame(rows)
    df_r.to_csv(RESULTS_CSV, index=False)
    log.info("  Evaluated %d pairs (skipped %d)", len(df_r), skipped)

    # Aggregate
    cols = [c for c in df_r.columns if c.startswith("hit_")]

    def bootstrap_ci(values, n_boot=1000, seed=42):
        if len(values) == 0:
            return (0.0, 0.0)
        rng = np.random.default_rng(seed)
        arr = np.asarray(values, dtype=float)
        boots = [rng.choice(arr, size=len(arr), replace=True).mean()
                 for _ in range(n_boot)]
        return (round(float(np.percentile(boots, 2.5)), 4),
                round(float(np.percentile(boots, 97.5)), 4))

    rec = {c.replace("hit_", ""): round(df_r[c].mean(), 4) for c in cols}
    cis = {c.replace("hit_", ""): list(bootstrap_ci(df_r[c].values))
           for c in cols}

    base = df_r["hit_A_lexical"].values
    wilcox = {}
    for c in cols:
        if c == "hit_A_lexical":
            continue
        d = df_r[c].values - base
        if np.count_nonzero(d) < 10:
            wilcox[f"{c}_vs_A"] = {"W": None, "p": None}
        else:
            w, p = wilcoxon(df_r[c].values, base, alternative="greater")
            wilcox[f"{c}_vs_A"] = {"W": float(w), "p": float(p)}

    # Critical comparisons
    # P vs L (does TMO-semantic beat abstract-semantic?)
    w_pl, p_pl = wilcoxon(df_r["hit_P_qwen3_tmo"].values,
                          df_r["hit_L_qwen3"].values,
                          alternative="greater")
    # Q vs L (does full RRF beat Qwen3 alone?)
    w_ql, p_ql = wilcoxon(df_r["hit_Q_rrf_full"].values,
                          df_r["hit_L_qwen3"].values,
                          alternative="greater")

    summary = {
        "n_pairs": len(df_r),
        "n_skipped": skipped,
        "top_k": TOP_K,
        "recall": rec,
        "bootstrap_ci": cis,
        "wilcoxon_vs_lexical": wilcox,
        "wilcoxon_P_vs_L_qwen3": {"W": float(w_pl), "p": float(p_pl)},
        "wilcoxon_Q_vs_L_qwen3": {"W": float(w_ql), "p": float(p_ql)},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("SEMANTIC TMO RETRIEVAL — final ablation")
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
    print(f"\nKey comparisons:")
    star = " ***" if p_pl < 0.001 else (" **" if p_pl < 0.01 else
            (" *" if p_pl < 0.05 else ""))
    print(f"  P (Qwen3-TMO) vs L (Qwen3-abstract):  "
          f"W={w_pl:.0f}, p={p_pl:.3e}{star}")
    print(f"    → If significant: TMO representation beats abstract.")
    star = " ***" if p_ql < 0.001 else (" **" if p_ql < 0.01 else
            (" *" if p_ql < 0.05 else ""))
    print(f"  Q (Full RRF) vs L (Qwen3-abstract):   "
          f"W={w_ql:.0f}, p={p_ql:.3e}{star}")
    print(f"    → If significant: combining everything beats best individual.")
    print(f"\nOutputs in {OUT_DIR}/")


# ===========================================================================
# Entry
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["embed", "eval", "all"], default="all")
    args = ap.parse_args()
    if args.phase in ("embed", "all"):
        phase_embed()
    if args.phase in ("eval", "all"):
        phase_eval()


if __name__ == "__main__":
    main()
