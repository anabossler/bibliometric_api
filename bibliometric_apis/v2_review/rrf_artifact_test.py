"""
rrf_artifact_test.py
====================


Question: Is the RRF fusion improvement a real signal
combination, or merely an artifact of mixing many rankings?

Strategy: replace each component of the best RRF (Q, 6-signal) by a random
top-50 ranking and observe the drop. If the drop is large and proportional
to the original signal strength, the fusion respects signal quality
(no artifact). If the drop is small or absent, the signal was noise.

Conditions evaluated:
  Q_real    — actual Q from final_table (6 real signals)
  Q_all_rand — all 6 signals replaced by random rankings (negative control)
  Q_minus_X — replace one signal X by random, keep the other 5 real

Each random ranking is sampled WITHOUT replacement from the candidate pool,
excluding the source paper. We run 10 random seeds for each "_minus" condition
and report the mean recall.

Outputs:
  runs/rrf_robustness/results.csv
  runs/rrf_robustness/summary.json
"""

from __future__ import annotations
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("artifact")

DOI_ORDER   = Path("runs/hygrag_v3/phase0_doi_order.csv")
EMB_DIR     = Path("runs/hygrag_v3/phase0_embeddings")
ABSTRACTS   = Path("backup_recycled_a/full_corpus/abstracts_full.csv")
EDGES       = Path("backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv")
TOPIC_VEC   = Path("runs/openalex_topics/topic_vectors.npz")
TMO_EMB     = Path("runs/semantic_tmo/qwen3_tmo.npy")

OUT_DIR     = Path("runs/rrf_robustness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K       = 50
N_SEEDS     = 10

STOPWORDS = set(
    "the and for that this with from are was were been have has had not but "
    "its can will also into than more which their these they would could "
    "should about each between through during after before other some such "
    "only over how our who what when where doi org https http www "
    "available online published all any both did does done either "
    "however just much most must same several since still very while "
    "using used use results show study paper present".split()
)


def tokenize(text):
    if not isinstance(text, str):
        return set()
    w = re.findall(r"[a-z]+", text.lower())
    w = [x for x in w if len(x) >= 3 and x not in STOPWORDS]
    return set(w) | {f"{w[i]} {w[i+1]}" for i in range(len(w) - 1)}


def l2_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def rrf(rankings, top_k, k_const=60):
    s = defaultdict(float)
    for rl in rankings:
        for r, idx in enumerate(rl):
            s[idx] += 1.0 / (k_const + r + 1)
    return sorted(s, key=lambda x: -s[x])[:top_k]


def main():
    log.info("Loading data...")
    order = pd.read_csv(DOI_ORDER)
    doi_to_idx = {d: i for i, d in enumerate(order["doi"])}
    N = len(order)

    sbert     = l2_normalize(np.load(EMB_DIR / "sbert.npy"))
    specter2  = l2_normalize(np.load(EMB_DIR / "specter2.npy"))
    qwen3     = l2_normalize(np.load(EMB_DIR / "qwen3.npy"))
    qwen3_tmo = l2_normalize(np.load(TMO_EMB))
    topic_M   = sparse.load_npz(TOPIC_VEC)

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

    def score_random(si, top_k, rng):
        # Random top-50 from anything except si
        candidates = np.setdiff1d(np.arange(N), [si], assume_unique=True)
        return list(rng.choice(candidates, size=top_k, replace=False))

    pairs = pd.read_csv(EDGES)
    log.info("Evaluating %d pairs...", len(pairs))

    # Component names in canonical order
    components = ["lex", "sbert", "specter2", "qwen3", "topics", "qwen3_tmo"]
    # Conditions: Q_real, Q_all_rand, Q_minus_<c> for each c
    seed_results = {f"Q_minus_{c}": [] for c in components}
    real_hits = []
    all_rand_hits = {seed: [] for seed in range(N_SEEDS)}

    rng_global = np.random.default_rng(42)

    skipped = 0
    t0 = time.time()
    for ix, row in pairs.iterrows():
        src_doi, tgt_doi = row["source_doi"], row["target_doi"]
        if src_doi not in doi_to_idx or tgt_doi not in doi_to_idx:
            skipped += 1
            continue
        si, ti = doi_to_idx[src_doi], doi_to_idx[tgt_doi]
        if not doc_terms[si]:
            skipped += 1
            continue

        # Real rankings
        real = {
            "lex":       score_lexical(doc_terms[si], si, TOP_K),
            "sbert":     score_dense(sbert, si, TOP_K),
            "specter2":  score_dense(specter2, si, TOP_K),
            "qwen3":     score_dense(qwen3, si, TOP_K),
            "topics":    score_topics(si, TOP_K),
            "qwen3_tmo": score_dense(qwen3_tmo, si, TOP_K),
        }

        # Q_real (replicates Q from final_table.py)
        top_q = rrf(list(real.values()), TOP_K)
        real_hits.append(int(ti in top_q))

        # For each seed: Q_all_rand (all 6 replaced by random)
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 1000 + ix)
            rand_rankings = [score_random(si, TOP_K, rng) for _ in components]
            top_rand = rrf(rand_rankings, TOP_K)
            all_rand_hits[seed].append(int(ti in top_rand))

        # Q_minus_X: replace ONLY component X by random (avg over N_SEEDS)
        for c in components:
            hits = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed * 1000 + ix)
                rand_ranking = score_random(si, TOP_K, rng)
                modified = []
                for cc in components:
                    if cc == c:
                        modified.append(rand_ranking)
                    else:
                        modified.append(real[cc])
                top_m = rrf(modified, TOP_K)
                hits.append(int(ti in top_m))
            # Store mean hit for this pair (over seeds)
            seed_results[f"Q_minus_{c}"].append(np.mean(hits))

        if (ix + 1) % 100 == 0:
            log.info("    pair %d/%d   elapsed=%.0fs",
                     ix + 1, len(pairs), time.time() - t0)

    n_pairs = len(real_hits)
    log.info("Evaluated %d pairs (skipped %d)", n_pairs, skipped)

    # Aggregate
    rec_real = float(np.mean(real_hits))
    rec_all_rand = float(np.mean(
        [np.mean(all_rand_hits[s]) for s in range(N_SEEDS)]
    ))
    rec_minus = {c: float(np.mean(seed_results[f"Q_minus_{c}"]))
                 for c in components}

    # Wilcoxon: Q_real vs each Q_minus_c (paired by pair; we use per-pair means)
    real_arr = np.array(real_hits, dtype=float)
    wilcox = {}
    for c in components:
        minus_arr = np.array(seed_results[f"Q_minus_{c}"], dtype=float)
        d = real_arr - minus_arr
        if np.count_nonzero(d) < 10:
            wilcox[c] = {"W": None, "p": None}
        else:
            w, p = wilcoxon(real_arr, minus_arr, alternative="greater")
            wilcox[c] = {"W": float(w), "p": float(p)}

    # Drop per component
    drops = {c: rec_real - rec_minus[c] for c in components}

    summary = {
        "n_pairs": n_pairs,
        "n_seeds_per_random_condition": N_SEEDS,
        "top_k": TOP_K,
        "Q_real_recall": round(rec_real, 4),
        "Q_all_random_recall": round(rec_all_rand, 4),
        "Q_minus_recall": {k: round(v, 4) for k, v in rec_minus.items()},
        "drop_when_replaced_by_random": {
            k: round(v, 4) for k, v in drops.items()
        },
        "wilcoxon_real_greater_than_minus": wilcox,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print
    print("\n" + "=" * 72)
    print("RRF ARTIFACT TEST — anti-fusion-noise control")
    print("=" * 72)
    print(f"Pairs: {n_pairs}   Seeds per random condition: {N_SEEDS}")
    print()
    print(f"  Q_real         (6 real signals):       {rec_real*100:.2f}%")
    print(f"  Q_all_random   (6 random rankings):    {rec_all_rand*100:.2f}%  "
          f"<-- expected ~1.5% (50/3043)")
    print(f"  Theoretical chance baseline:           "
          f"{TOP_K/N*100:.2f}%")
    print()
    print(f"  Replace ONE component by random "
          f"(avg over {N_SEEDS} seeds, others kept):")
    print(f"  {'Component':<15} {'Recall':>10} {'Drop':>10} "
          f"{'W':>8} {'p (real>minus)':>16}")
    print("  " + "-" * 60)
    # Sort by drop descending
    for c, drop in sorted(drops.items(), key=lambda x: -x[1]):
        rmin = rec_minus[c]
        w = wilcox[c]
        p = w["p"]
        if p is None:
            pstr = "  --"
        elif p < 0.001:
            pstr = f"{p:.2e} ***"
        elif p < 0.01:
            pstr = f"{p:.2e} **"
        elif p < 0.05:
            pstr = f"{p:.2e} *"
        else:
            pstr = f"{p:.2e}"
        wstr = f"{w['W']:.0f}" if w["W"] is not None else "--"
        print(f"  {c:<15} {rmin*100:>9.2f}% {drop*100:>+8.2f}pp "
              f"{wstr:>8} {pstr:>16}")
    print()
    print("Interpretation:")
    print("  - Large positive drop + significant Wilcoxon => component is real signal")
    print("  - Drop near 0 + non-significant => component was redundant noise")
    print("  - Q_all_random near chance baseline => RRF respects signal quality")
    print(f"\nOutputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
