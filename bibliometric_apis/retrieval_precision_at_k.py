"""
retrieval_precision_at_k.py
===========================
New retrieval experiment for AWS paper.

Metric: precision@50 = fraction of top-50 results belonging to target cluster.
Fixed denominator — result can go DOWN if expansion adds noise.


INPUTS:
    backup_recycled_a/full_corpus/abstracts_full.csv    (doi, abstract)
    backup_recycled_a/full_corpus/paper_topics.csv      (doi, cluster, ...)
    backup_recycled_a/full_corpus/alignment_raw.csv     (cluster_a, cluster_b,
                                                         term_a, term_b, confidence)
OUTPUTS (in backup_recycled_a/full_corpus/):
    retrieval_precision_results.csv
    retrieval_precision_pair.csv
    retrieval_precision_summary.csv
    retrieval_precision_report.txt
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

np.random.seed(42)
rng = np.random.default_rng(42)

BASE_DIR   = Path("backup_recycled_a/full_corpus")
ABSTRACTS  = BASE_DIR / "abstracts_full.csv"
TOPICS     = BASE_DIR / "paper_topics.csv"
ALIGNMENTS = BASE_DIR / "alignment_raw.csv"
OUT_DIR    = BASE_DIR

TOP_K          = 50
N_TERMS_Q      = 3
N_TERMS_EXP    = 3
N_QUERIES_PAIR = 33
MIN_CONFIDENCE = 3

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
    """Unigrams + bigrams, min length 3, stopwords removed."""
    if not isinstance(text, str):
        return set()
    words = re.findall(r"[a-z]+", text.lower())
    words = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    unigrams = set(words)
    bigrams  = {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}
    return unigrams | bigrams


# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------

print("Loading abstracts + cluster assignments...")
df_abs    = pd.read_csv(ABSTRACTS)
df_topics = pd.read_csv(TOPICS)[["doi", "cluster"]].dropna()
df_topics["cluster"] = df_topics["cluster"].astype(int)

# Merge on doi
df = df_abs.merge(df_topics, on="doi", how="inner")
df = df[df["abstract"].notna()].reset_index(drop=True)
print(f"  {len(df)} documents with abstracts and cluster assignments")
print(f"  Clusters: {sorted(df['cluster'].unique())}")
print(f"  Per cluster: {df['cluster'].value_counts().sort_index().to_dict()}")

doc_terms    : list[set[str]] = []
doc_clusters : list[int]      = []

for _, row in df.iterrows():
    doc_terms.append(tokenize(row["abstract"]))
    doc_clusters.append(int(row["cluster"]))

print("  Vocabulary built.")

# ---------------------------------------------------------------------------
# Load alignments
# ---------------------------------------------------------------------------

print("Loading alignments...")
df_align = pd.read_csv(ALIGNMENTS)
df_align = df_align[df_align["term_a"].notna() & df_align["term_b"].notna()]
df_align["confidence"] = pd.to_numeric(df_align["confidence"], errors="coerce")

# Per-cluster vocabulary from all alignment terms (any confidence)
cluster_vocab: dict[int, list[str]] = defaultdict(list)
for _, row in df_align.iterrows():
    ca = int(row["cluster_a"])
    cb = int(row["cluster_b"])
    ta = str(row["term_a"]).strip().lower()
    tb = str(row["term_b"]).strip().lower()
    if ta not in cluster_vocab[ca]:
        cluster_vocab[ca].append(ta)
    if tb not in cluster_vocab[cb]:
        cluster_vocab[cb].append(tb)

# Filter to terms that actually appear in corpus (sanity check)
print("  Checking term coverage in corpus...")
for c in sorted(cluster_vocab.keys()):
    terms    = cluster_vocab[c]
    covered  = [t for t in terms
                if any(t in doc_terms[i]
                       for i in range(len(doc_terms))
                       if doc_clusters[i] == c)]
    print(f"    C{c}: {len(covered)}/{len(terms)} terms found in abstracts")
    cluster_vocab[c] = covered  # keep only terms that appear

# LLM alignments above confidence threshold
alignments: dict[tuple[int, int], list[tuple[str, str]]] = defaultdict(list)
df_strong = df_align[df_align["confidence"] >= MIN_CONFIDENCE]
for _, row in df_strong.iterrows():
    ca = int(row["cluster_a"])
    cb = int(row["cluster_b"])
    ta = str(row["term_a"]).strip().lower()
    tb = str(row["term_b"]).strip().lower()
    if (ta, tb) not in alignments[(ca, cb)]:
        alignments[(ca, cb)].append((ta, tb))
    if (tb, ta) not in alignments[(cb, ca)]:
        alignments[(cb, ca)].append((tb, ta))

print(f"  Directed alignment pairs (conf>={MIN_CONFIDENCE}): {len(alignments)}")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def score_docs(query_terms: set[str]) -> list[tuple[int, int]]:
    scored = []
    for i, terms in enumerate(doc_terms):
        overlap = len(query_terms & terms)
        if overlap > 0:
            scored.append((i, overlap))
    scored.sort(key=lambda x: -x[1])
    return scored[:TOP_K]


def precision_at_k(top_k: list[tuple[int, int]], target_cluster: int) -> float:
    if not top_k:
        return 0.0
    hits = sum(1 for idx, _ in top_k if doc_clusters[idx] == target_cluster)
    return hits / len(top_k)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

print(f"\nRunning retrieval experiment...")
print(f"  TOP_K={TOP_K}, {N_QUERIES_PAIR} queries per directed pair")

clusters = sorted(cluster_vocab.keys())
directed_pairs = (
    [(ca, cb) for ca, cb in combinations(clusters, 2)] +
    [(cb, ca) for ca, cb in combinations(clusters, 2)]
)

rows = []
qid  = 0

for ca, cb in directed_pairs:
    terms_a = cluster_vocab.get(ca, [])
    aligned = alignments.get((ca, cb), [])

    # Filter aligned terms to those that exist in corpus
    aligned_valid = [(ta, tb) for ta, tb in aligned
                     if any(tb in doc_terms[i]
                            for i in range(len(doc_terms))
                            if doc_clusters[i] == cb)]

    if len(terms_a) < N_TERMS_Q:
        print(f"  Skipping C{ca}→C{cb}: only {len(terms_a)} source terms")
        continue

    for _ in range(N_QUERIES_PAIR):
        q_source = set(rng.choice(terms_a, size=N_TERMS_Q, replace=False))

        top_base = score_docs(q_source)
        p_base   = precision_at_k(top_base, cb)

        exp_terms = set(tb for _, tb in aligned_valid[:N_TERMS_EXP])
        q_exp     = q_source | exp_terms
        top_exp   = score_docs(q_exp)
        p_exp     = precision_at_k(top_exp, cb)

        rows.append({
            "query_id":        qid,
            "source_cluster":  ca,
            "target_cluster":  cb,
            "has_alignment":   len(aligned_valid) > 0,
            "n_aligned_valid": len(aligned_valid),
            "query_terms":     "; ".join(sorted(q_source)),
            "expansion_terms": "; ".join(sorted(exp_terms)),
            "n_results_base":  len(top_base),
            "n_results_exp":   len(top_exp),
            "precision_base":  round(p_base, 4),
            "precision_exp":   round(p_exp,  4),
            "delta_precision": round(p_exp - p_base, 4),
        })
        qid += 1

df_results = pd.DataFrame(rows)
print(f"  Generated {len(df_results)} queries")
print(f"  Non-zero precision_base: {(df_results['precision_base'] > 0).sum()}")
print(f"  Non-zero precision_exp:  {(df_results['precision_exp'] > 0).sum()}")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

diffs    = df_results["delta_precision"].values
nz       = diffs != 0
improved = (diffs > 0).sum()
degraded = (diffs < 0).sum()
neutral  = (diffs == 0).sum()

W, p_w = None, None
if nz.sum() >= 10:
    W, p_w = wilcoxon(
        df_results["precision_exp"].values[nz],
        df_results["precision_base"].values[nz],
        alternative="greater",
    )

t_stat, t_p = ttest_rel(df_results["precision_exp"], df_results["precision_base"])

pair_rows = []
for (ca, cb), grp in df_results.groupby(["source_cluster", "target_cluster"]):
    pair_rows.append({
        "source_cluster":      ca,
        "target_cluster":      cb,
        "n_queries":           len(grp),
        "has_alignment":       grp["has_alignment"].any(),
        "mean_precision_base": round(grp["precision_base"].mean(), 4),
        "mean_precision_exp":  round(grp["precision_exp"].mean(), 4),
        "mean_delta":          round(grp["delta_precision"].mean(), 4),
        "pct_improved":        round((grp["delta_precision"] > 0).mean() * 100, 1),
        "pct_degraded":        round((grp["delta_precision"] < 0).mean() * 100, 1),
    })
pair_df = pd.DataFrame(pair_rows)

# ---------------------------------------------------------------------------
# Save + Report
# ---------------------------------------------------------------------------

df_results.to_csv(OUT_DIR / "retrieval_precision_results.csv", index=False)
pair_df.to_csv(OUT_DIR / "retrieval_precision_pair.csv", index=False)

summary = {
    "n_queries":           len(df_results),
    "n_documents":         len(df),
    "top_k":               TOP_K,
    "mean_precision_base": round(df_results["precision_base"].mean(), 4),
    "mean_precision_exp":  round(df_results["precision_exp"].mean(), 4),
    "mean_delta":          round(df_results["delta_precision"].mean(), 4),
    "median_delta":        round(df_results["delta_precision"].median(), 4),
    "pct_improved":        round(improved / len(df_results) * 100, 1),
    "pct_degraded":        round(degraded / len(df_results) * 100, 1),
    "pct_neutral":         round(neutral  / len(df_results) * 100, 1),
    "wilcoxon_W":          W,
    "wilcoxon_p":          p_w,
    "n_nonzero":           int(nz.sum()),
    "ttest_t":             round(float(t_stat), 3),
    "ttest_p":             round(float(t_p), 6),
}
pd.DataFrame([summary]).to_csv(OUT_DIR / "retrieval_precision_summary.csv", index=False)

report_lines = [
    "=" * 70,
    "RETRIEVAL PRECISION@K — AWS paper (plastic recycling corpus)",
    "=" * 70,
    f"Documents: {len(df)} (full corpus with abstracts)",
    f"Queries:   {len(df_results)} ({N_QUERIES_PAIR} per directed pair)",
    f"TOP_K:     {TOP_K} (fixed denominator — metric can go negative)",
    f"Min LLM confidence for expansion: {MIN_CONFIDENCE}/5",
    "",
    "AGGREGATE",
    f"  Precision@{TOP_K} baseline:  {summary['mean_precision_base']:.4f}",
    f"  Precision@{TOP_K} expanded:  {summary['mean_precision_exp']:.4f}",
    f"  Mean delta:                 {summary['mean_delta']:+.4f}",
    f"  Median delta:               {summary['median_delta']:+.4f}",
    f"  Queries improved:           {summary['pct_improved']:.1f}%",
    f"  Queries degraded:           {summary['pct_degraded']:.1f}%",
    f"  Queries unchanged:          {summary['pct_neutral']:.1f}%",
    "",
    "SIGNIFICANCE",
]

if W is not None:
    report_lines += [
        f"  Wilcoxon signed-rank (one-sided, H1: expanded > baseline):",
        f"  W={W:.0f}, p={p_w:.3e}, non-zero={int(nz.sum())}/{len(df_results)}",
    ]
else:
    report_lines.append("  Wilcoxon: insufficient non-zero differences")

report_lines += [
    f"  Paired t-test: t={t_stat:.3f}, p={t_p:.3e}",
    "",
    "PER-PAIR SUMMARY",
    pair_df.to_string(index=False),
    "",
    "INTERPRETATION",
    "  delta > 0: LLM expansion retrieved MORE target-cluster docs in top-k.",
    "  delta < 0: expansion pushed target docs out of top-k (noise added).",
    "  delta = 0: no change in cluster purity of top-k results.",
    "=" * 70,
]

report = "\n".join(report_lines)
print("\n" + report)
(OUT_DIR / "retrieval_precision_report.txt").write_text(report)
print(f"\nOutputs saved to {OUT_DIR}/")
