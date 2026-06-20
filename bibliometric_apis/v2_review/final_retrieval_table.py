"""
final_retrieval_table.py
========================


Loads ALL precomputed embeddings and evaluates 17 conditions in a single pass:

  Lexical
  A.  TF-IDF / boolean overlap baseline                            (no model)

  Dense (single representation)
  B.  SBERT MiniLM (22M)                  abstract                 [text-only]
  C.  ChemBERTa (77M)                     abstract                 [text-only]
  D.  SPECTER2 (110M)                     abstract                 [citation-aware]
  L.  Qwen3-Embedding-8B                  abstract                 [SOTA general]
  P.  Qwen3-Embedding-8B                  TMO string               [structured]
  R.  SemCSE (CLAUSE-Bielefeld)           abstract                 [LLM-sem]

  External
  N.  OpenAlex topics                     cosine over topic vector [external KG]

  Fusions
  M.  RRF{Lex, SBERT, SPECTER2, Qwen3-abs}                        [4-signal]
  O.  RRF{Lex, SBERT, SPECTER2, Qwen3-abs, Topics}                [5-signal]
  Q.  RRF{Lex, SBERT, SPECTER2, Qwen3-abs, Topics, Qwen3-TMO}     [6-signal]
  S.  RRF{Lex, SBERT, SPECTER2, Qwen3-abs, Topics, Qwen3-TMO,
          SemCSE}                                                   [7-signal]

All conditions evaluated against the 656 cross-cluster citation pairs.
Bootstrap CI95 and Wilcoxon vs A_lexical for every condition.
Two additional critical comparisons:
  - R (SemCSE) vs D (SPECTER2): does content-aware beat citation-aware?
  - S (full RRF) vs L (Qwen3 alone): does combining everything help?

Outputs:
  runs/final_table/results.csv          per-pair hits
  runs/final_table/summary.json         aggregated table + tests
  runs/final_table/table_for_latex.tex  formatted LaTeX table

Usage:
  python final_retrieval_table.py
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
log = logging.getLogger("final")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOI_ORDER     = Path("runs/hygrag_v3/phase0_doi_order.csv")
EMB_DIR       = Path("runs/hygrag_v3/phase0_embeddings")
ABSTRACTS     = Path("backup_recycled_a/full_corpus/abstracts_full.csv")
EDGES         = Path("backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv")
TOPIC_VEC     = Path("runs/openalex_topics/topic_vectors.npz")
TMO_EMB       = Path("runs/semantic_tmo/qwen3_tmo.npy")

OUT_DIR       = Path("runs/final_table")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K         = 50


# ---------------------------------------------------------------------------
# Helpers
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


def tokenize(text: str) -> set:
    if not isinstance(text, str):
        return set()
    words = re.findall(r"[a-z]+", text.lower())
    words = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    return set(words) | {f"{words[i]} {words[i+1]}"
                         for i in range(len(words) - 1)}


def l2_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def bootstrap_ci(values, n_boot=1000, seed=42):
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return (round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    log.info("Loading DOI order and embeddings...")
    order = pd.read_csv(DOI_ORDER)
    doi_to_idx = {d: i for i, d in enumerate(order["doi"])}
    log.info("  Corpus: %d papers in canonical order", len(order))

    # Load dense embeddings
    sbert     = l2_normalize(np.load(EMB_DIR / "sbert.npy"))
    chemberta = l2_normalize(np.load(EMB_DIR / "chemberta.npy"))
    specter2  = l2_normalize(np.load(EMB_DIR / "specter2.npy"))
    qwen3     = l2_normalize(np.load(EMB_DIR / "qwen3.npy"))
    semcse    = l2_normalize(np.load(EMB_DIR / "semcse.npy"))
    qwen3_tmo = l2_normalize(np.load(TMO_EMB))
    log.info("    sbert     %s", sbert.shape)
    log.info("    chemberta %s", chemberta.shape)
    log.info("    specter2  %s", specter2.shape)
    log.info("    qwen3     %s", qwen3.shape)
    log.info("    semcse    %s", semcse.shape)
    log.info("    qwen3_tmo %s", qwen3_tmo.shape)

    # Sparse topics
    topic_M = sparse.load_npz(TOPIC_VEC)
    log.info("    topics    %s  nnz=%d", topic_M.shape, topic_M.nnz)

    # Lexical
    df_abs = pd.read_csv(ABSTRACTS)
    abstr_by_doi = dict(zip(df_abs["doi"], df_abs["abstract"]))
    doc_terms = [tokenize(abstr_by_doi.get(d, "")) for d in order["doi"]]

    # Scoring helpers
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
    log.info("  Evaluating %d citation pairs @ K=%d...", len(pairs), TOP_K)

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

        top_a = score_lexical(src_lex, si, TOP_K)        # A. Lexical
        top_b = score_dense(sbert, si, TOP_K)            # B. SBERT
        top_c = score_dense(chemberta, si, TOP_K)        # C. ChemBERTa
        top_d = score_dense(specter2, si, TOP_K)         # D. SPECTER2
        top_l = score_dense(qwen3, si, TOP_K)            # L. Qwen3 abstract
        top_p = score_dense(qwen3_tmo, si, TOP_K)        # P. Qwen3 TMO
        top_r = score_dense(semcse, si, TOP_K)           # R. SemCSE (NEW)
        top_n = score_topics(si, TOP_K)                  # N. OpenAlex topics

        # Fusions
        top_m = rrf([top_a, top_b, top_d, top_l], TOP_K)
        top_o = rrf([top_a, top_b, top_d, top_l, top_n], TOP_K)
        top_q = rrf([top_a, top_b, top_d, top_l, top_n, top_p], TOP_K)
        top_s = rrf([top_a, top_b, top_d, top_l, top_n, top_p, top_r], TOP_K)

        rows.append({
            "source_doi": src_doi, "target_doi": tgt_doi,
            "hit_A_lexical":     int(ti in top_a),
            "hit_B_sbert":       int(ti in top_b),
            "hit_C_chemberta":   int(ti in top_c),
            "hit_D_specter2":    int(ti in top_d),
            "hit_L_qwen3":       int(ti in top_l),
            "hit_N_topics":      int(ti in top_n),
            "hit_P_qwen3_tmo":   int(ti in top_p),
            "hit_R_semcse":      int(ti in top_r),
            "hit_M_rrf4":        int(ti in top_m),
            "hit_O_rrf5":        int(ti in top_o),
            "hit_Q_rrf6":        int(ti in top_q),
            "hit_S_rrf7_full":   int(ti in top_s),
        })

        if (ix + 1) % 100 == 0:
            log.info("    pair %d/%d   elapsed=%.0fs",
                     ix + 1, len(pairs), time.time() - t0)

    df_r = pd.DataFrame(rows)
    df_r.to_csv(OUT_DIR / "results.csv", index=False)
    log.info("  Evaluated %d pairs (skipped %d)", len(df_r), skipped)

    # ---------- Aggregate ----------
    cols = [c for c in df_r.columns if c.startswith("hit_")]
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
    def safe_w(x, y, alt="greater"):
        d = np.asarray(x) - np.asarray(y)
        if np.count_nonzero(d) < 10:
            return None, None
        w, p = wilcoxon(x, y, alternative=alt)
        return float(w), float(p)

    w_rd, p_rd = safe_w(df_r["hit_R_semcse"].values,   df_r["hit_D_specter2"].values)
    w_rl, p_rl = safe_w(df_r["hit_R_semcse"].values,   df_r["hit_L_qwen3"].values)
    w_sl, p_sl = safe_w(df_r["hit_S_rrf7_full"].values, df_r["hit_L_qwen3"].values)
    w_sq, p_sq = safe_w(df_r["hit_S_rrf7_full"].values, df_r["hit_Q_rrf6"].values)

    # Oracle upper bound (union of all individual conditions)
    individual_cols = ["hit_A_lexical", "hit_B_sbert", "hit_D_specter2",
                       "hit_L_qwen3", "hit_N_topics", "hit_P_qwen3_tmo",
                       "hit_R_semcse"]
    union_all = df_r[individual_cols].max(axis=1).mean()

    summary = {
        "n_pairs": len(df_r),
        "n_skipped": skipped,
        "top_k": TOP_K,
        "recall": rec,
        "bootstrap_ci": cis,
        "wilcoxon_vs_lexical": wilcox,
        "critical_tests": {
            "R_semcse_vs_D_specter2":      {"W": w_rd, "p": p_rd},
            "R_semcse_vs_L_qwen3":         {"W": w_rl, "p": p_rl},
            "S_rrf7full_vs_L_qwen3":       {"W": w_sl, "p": p_sl},
            "S_rrf7full_vs_Q_rrf6":        {"W": w_sq, "p": p_sq},
        },
        "oracle_upper_bound_union_of_all_individual": round(union_all, 4),
        "unrecoverable_by_any_method":   round(1 - union_all, 4),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- Print ----------
    label_map = {
        "A_lexical":     ("A", "Lexical (boolean overlap)"),
        "B_sbert":       ("B", "SBERT MiniLM (22M)"),
        "C_chemberta":   ("C", "ChemBERTa (77M)"),
        "D_specter2":    ("D", "SPECTER2 (110M, citation)"),
        "L_qwen3":       ("L", "Qwen3-Embedding-8B"),
        "N_topics":      ("N", "OpenAlex topics"),
        "P_qwen3_tmo":   ("P", "Qwen3-8B on TMO string"),
        "R_semcse":      ("R", "SemCSE (sci-semantic)"),
        "M_rrf4":        ("M", "RRF{Lex,SB,SP,Qw}"),
        "O_rrf5":        ("O", "RRF{...,Topics}"),
        "Q_rrf6":        ("Q", "RRF{...,Qw-TMO}"),
        "S_rrf7_full":   ("S", "RRF{...,SemCSE} full"),
    }

    print("\n" + "=" * 78)
    print("FINAL CONSOLIDATED RETRIEVAL TABLE — AWS paper revision")
    print("=" * 78)
    print(f"Pairs evaluated: {len(df_r)}\n")
    print(f"{'#':<3} {'Condition':<32} {'Recall@50':>10}   {'CI95':<18} {'vs A (p)':<12}")
    print("-" * 78)
    for key, (letter, desc) in label_map.items():
        r = rec[key]
        c = cis[key]
        wkey = f"hit_{key}_vs_A"
        wt = wilcox.get(wkey, {})
        p = wt.get("p")
        if p is None:
            pstr = "  --"
        elif p < 0.001:
            pstr = f"{p:.1e} ***"
        elif p < 0.01:
            pstr = f"{p:.1e} **"
        elif p < 0.05:
            pstr = f"{p:.1e} *"
        else:
            pstr = f"{p:.1e}"
        print(f"{letter:<3} {desc:<32} {r:>10.4f}   "
              f"[{c[0]:.3f},{c[1]:.3f}]   {pstr}")

    print("\n" + "-" * 78)
    print("CRITICAL COMPARISONS")
    print("-" * 78)
    for name, (w, p) in [
        ("R (SemCSE) > D (SPECTER2)",     (w_rd, p_rd)),
        ("R (SemCSE) > L (Qwen3-8B)",     (w_rl, p_rl)),
        ("S (full RRF) > L (Qwen3-8B)",   (w_sl, p_sl)),
        ("S (full RRF) > Q (RRF-6 prev)", (w_sq, p_sq)),
    ]:
        if w is None:
            print(f"  {name:<35s}: insufficient differences")
        else:
            star = ""
            if p < 0.001:   star = " ***"
            elif p < 0.01:  star = " **"
            elif p < 0.05:  star = " *"
            print(f"  {name:<35s}: W={w:>6.0f}, p={p:.3e}{star}")

    print("\n" + "-" * 78)
    print("ORACLE BOUND")
    print("-" * 78)
    print(f"  Union of all 7 individual conditions:   {union_all*100:.1f}%")
    print(f"  Pairs unrecoverable by ANY method:      {(1-union_all)*100:.1f}%")
    print(f"  Achieved by best fusion (S):            {rec['S_rrf7_full']*100:.1f}%")
    print(f"  Gap fusion vs oracle:                   "
          f"{(union_all - rec['S_rrf7_full'])*100:+.1f} pp")

    # ---------- LaTeX table ----------
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Retrieval evaluation against 656 verified cross-cluster citation pairs (recall@50, 95\% bootstrap CI). Wilcoxon one-sided test against lexical baseline. *** $p<0.001$, ** $p<0.01$, * $p<0.05$.}",
        r"\label{tab:retrieval}",
        r"\small",
        r"\begin{tabular}{cllr@{ }lc}",
        r"\toprule",
        r"\# & Condition & Type & \multicolumn{2}{c}{Recall@50} & vs Lexical \\",
        r"\midrule",
    ]
    type_map = {
        "A": "lexical", "B": "dense", "C": "dense", "D": "citation-aware",
        "L": "dense-SOTA", "N": "external-KG", "P": "structured-LLM",
        "R": "semantic-sci",
        "M": "fusion", "O": "fusion", "Q": "fusion", "S": "fusion",
    }
    for key, (letter, desc) in label_map.items():
        r = rec[key]
        c = cis[key]
        wkey = f"hit_{key}_vs_A"
        wt = wilcox.get(wkey, {})
        p = wt.get("p")
        if p is None:
            sig = "--"
        elif p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = ""
        desc_tex = desc.replace("&", r"\&")
        tex_lines.append(
            f"{letter} & {desc_tex} & {type_map[letter]} & "
            f"{r*100:.1f}\\% & [{c[0]*100:.1f},{c[1]*100:.1f}] & {sig} \\\\")
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    (OUT_DIR / "table_for_latex.tex").write_text("\n".join(tex_lines))
    log.info("\n  Outputs in %s/", OUT_DIR)


if __name__ == "__main__":
    main()
