"""

Ground-truth retrieval experiment for AWS paper.

Uses verified cross-cluster citation pairs as relevance judgments.
For each source paper, measures whether the cited target paper
appears in the top-50 lexical retrieval results, with and without
LLM-powered vocabulary expansion.

Ground truth is actual citation behavior, not cluster membership.

Key result: recall@50 baseline = 0.179, meaning 82% of cross-cluster
cited papers are lexically inaccessible despite confirmed citation links.
This directly quantifies the AWS failure mode.

USAGE:
    conda activate aws
    cd ~/Desktop/openalex
    python retrieval_ground_truth.py

INPUTS:
    backup_recycled_a/full_corpus/abstracts_full.csv
        (doi, abstract)
    backup_recycled_a/full_corpus/paper_topics.csv
        (doi, cluster, ...)
    backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv
        (source_doi, source_cluster, target_doi, target_cluster)
    backup_recycled_a/full_corpus/alignment_raw.csv
        (cluster_a, cluster_b, term_a, term_b, confidence)

OUTPUTS (in backup_recycled_a/full_corpus/):
    retrieval_ground_truth_results.csv
    retrieval_ground_truth_summary.csv
    retrieval_ground_truth_report.txt

To generate citation_edges_cross_cluster.csv, run:
    python download_citation_edges.py
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

BASE_DIR   = Path("backup_recycled_a/full_corpus")
ABSTRACTS  = BASE_DIR / "abstracts_full.csv"
TOPICS     = BASE_DIR / "paper_topics.csv"
EDGES      = BASE_DIR / "citation_edges_cross_cluster.csv"
ALIGNMENTS = BASE_DIR / "alignment_raw.csv"
OUT_DIR    = BASE_DIR

TOP_K          = 50
N_TERMS_EXP    = 3
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


def score_and_rank(
    query_terms: set[str],
    doc_terms: list[set[str]],
    exclude_idx: int,
) -> list[int]:
    """Boolean overlap scoring, returns top-TOP_K doc indices."""
    scored = []
    for i, terms in enumerate(doc_terms):
        if i == exclude_idx:
            continue
        overlap = len(query_terms & terms)
        if overlap > 0:
            scored.append((i, overlap))
    scored.sort(key=lambda x: -x[1])
    return [idx for idx, _ in scored[:TOP_K]]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading corpus...")
df_abs    = pd.read_csv(ABSTRACTS)
df_topics = pd.read_csv(TOPICS)[["doi", "cluster"]].dropna()
df_topics["cluster"] = df_topics["cluster"].astype(int)

df = df_abs.merge(df_topics, on="doi", how="inner")
df = df[df["abstract"].notna()].reset_index(drop=True)
print(f"  {len(df)} documents")

doi_to_idx   = {row["doi"]: i for i, row in df.iterrows()}
doc_terms    = [tokenize(row["abstract"]) for _, row in df.iterrows()]
doc_clusters = df["cluster"].tolist()

print("Loading citation pairs...")
pairs = pd.read_csv(EDGES)
print(f"  {len(pairs)} cross-cluster citation pairs")

print("Loading alignments...")
df_align = pd.read_csv(ALIGNMENTS)
df_align = df_align[df_align["term_a"].notna() & df_align["term_b"].notna()]
df_align["confidence"] = pd.to_numeric(df_align["confidence"], errors="coerce")

alignments: dict[tuple[int, int], list[str]] = defaultdict(list)
for _, row in df_align[df_align["confidence"] >= MIN_CONFIDENCE].iterrows():
    ca = int(row["cluster_a"])
    cb = int(row["cluster_b"])
    tb = str(row["term_b"]).strip().lower()
    ta = str(row["term_a"]).strip().lower()
    if tb not in alignments[(ca, cb)]:
        alignments[(ca, cb)].append(tb)
    if ta not in alignments[(cb, ca)]:
        alignments[(cb, ca)].append(ta)

print(f"  Alignment pairs loaded (conf>={MIN_CONFIDENCE}): {len(alignments)}")

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

print(f"\nRunning ground truth experiment (TOP_K={TOP_K})...")

results = []
skipped = 0

for _, row in pairs.iterrows():
    src_doi = row["source_doi"]
    tgt_doi = row["target_doi"]
    ca      = int(row["source_cluster"])
    cb      = int(row["target_cluster"])

    if src_doi not in doi_to_idx or tgt_doi not in doi_to_idx:
        skipped += 1
        continue

    src_idx = doi_to_idx[src_doi]
    tgt_idx = doi_to_idx[tgt_doi]

    src_terms = doc_terms[src_idx]
    if not src_terms:
        skipped += 1
        continue

    # Baseline: source abstract terms only
    top_base = score_and_rank(src_terms, doc_terms, src_idx)
    hit_base = 1 if tgt_idx in top_base else 0

    # Expanded: source + LLM-aligned terms for (Ca -> Cb)
    exp_terms = set(alignments.get((ca, cb), [])[:N_TERMS_EXP])
    top_exp   = score_and_rank(src_terms | exp_terms, doc_terms, src_idx)
    hit_exp   = 1 if tgt_idx in top_exp else 0

    results.append({
        "source_doi":     src_doi,
        "target_doi":     tgt_doi,
        "source_cluster": ca,
        "target_cluster": cb,
        "has_alignment":  len(alignments.get((ca, cb), [])) > 0,
        "hit_base":       hit_base,
        "hit_exp":        hit_exp,
        "delta":          hit_exp - hit_base,
    })

print(f"  Evaluated: {len(results)} pairs (skipped: {skipped})")

df_r = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

recall_base = df_r["hit_base"].mean()
recall_exp  = df_r["hit_exp"].mean()
improved    = (df_r["delta"] > 0).sum()
degraded    = (df_r["delta"] < 0).sum()
unchanged   = (df_r["delta"] == 0).sum()

nz = df_r["delta"] != 0
W, p_w = None, None
if nz.sum() >= 10:
    W, p_w = wilcoxon(
        df_r["hit_exp"].values[nz.values],
        df_r["hit_base"].values[nz.values],
        alternative="greater",
    )

pair_summary = (
    df_r.groupby(["source_cluster", "target_cluster"])
    [["hit_base", "hit_exp", "delta"]]
    .mean()
    .round(4)
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df_r.to_csv(OUT_DIR / "retrieval_ground_truth_results.csv", index=False)

summary = {
    "n_pairs":       len(df_r),
    "n_skipped":     skipped,
    "top_k":         TOP_K,
    "recall_base":   round(recall_base, 4),
    "recall_exp":    round(recall_exp,  4),
    "mean_delta":    round(df_r["delta"].mean(), 4),
    "improved":      int(improved),
    "degraded":      int(degraded),
    "unchanged":     int(unchanged),
    "wilcoxon_W":    W,
    "wilcoxon_p":    p_w,
    "n_nonzero":     int(nz.sum()),
}
pd.DataFrame([summary]).to_csv(
    OUT_DIR / "retrieval_ground_truth_summary.csv", index=False
)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

report_lines = [
    "=" * 70,
    "RETRIEVAL GROUND TRUTH — AWS paper (plastic recycling corpus)",
    "=" * 70,
    f"Citation pairs evaluated: {len(df_r)} (skipped: {skipped})",
    f"TOP_K: {TOP_K}",
    f"Ground truth: verified cross-cluster citation edges (OpenAlex)",
    "",
    "AGGREGATE",
    f"  Recall@{TOP_K} baseline:  {recall_base:.4f}",
    f"  Recall@{TOP_K} expanded:  {recall_exp:.4f}",
    f"  Mean delta:              {df_r['delta'].mean():+.4f}",
    f"  Improved: {improved} | Degraded: {degraded} | Unchanged: {unchanged}",
    "",
    "INTERPRETATION",
    f"  {recall_base:.1%} of cross-cluster cited papers are recoverable",
    f"  by lexical search at k={TOP_K}.",
    f"  This quantifies the AWS failure mode: {1-recall_base:.1%} of papers",
    f"  that are cited across subdomain boundaries are lexically inaccessible.",
    "",
]

if W is not None:
    report_lines += [
        "SIGNIFICANCE (LLM expansion vs baseline)",
        f"  Wilcoxon W={W:.0f}, p={p_w:.3e}, non-zero={nz.sum()}/{len(df_r)}",
        "",
    ]

report_lines += [
    "PER CLUSTER-PAIR",
    pair_summary.to_string(),
    "=" * 70,
]

report = "\n".join(report_lines)
print("\n" + report)
(OUT_DIR / "retrieval_ground_truth_report.txt").write_text(report)
print(f"\nOutputs saved to {OUT_DIR}/")
