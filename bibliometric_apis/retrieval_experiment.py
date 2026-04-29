"""
retrieval_experiment.py
=======================
Retrieval-based fragmentation diagnostic for scholarly knowledge graphs.

Measures whether cross-cluster vocabulary alignment improves document
retrieval across semantic boundaries. A positive gain indicates that
clusters use distinct terminology for overlapping concepts — a marker
of epistemic fragmentation.

Pipeline
--------
1. Load cluster assignments (paper_topics.csv) and abstracts (corpus CSV).
2. Build per-cluster c-TF-IDF vocabularies from abstract text.
3. Generate query pairs: source-only vs. source + aligned (cross-cluster).
4. Measure recall gain; test significance with Wilcoxon signed-rank test.
5. Report Cohen's d effect size and per-pair mean gain.
6. Benchmark against two null models: random and high-frequency vocabulary.
7. (Optional) Correlate per-pair gain with semantic distance D_ij from AWS.

Usage
-----
    python retrieval_experiment.py

Outputs (written to OUTPUT_DIR)
--------------------------------
    retrieval_results.csv  — per-query statistics
    retrieval_summary.csv  — aggregate metrics, effect size, null-model z-scores
    pair_stats.csv         — per cluster-pair mean gain (+ D_ij if available)
    retrieval_report.txt   — human-readable summary

Dependencies
------------
    numpy, pandas, scipy
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from scipy.stats import wilcoxon, spearmanr

np.random.seed(42)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR  = "results/retrieval"
TOPICS_PATH = "data/paper_topics.csv"       # columns: doi, cluster (int, k=6)
CORPUS_PATH = "data/corpus.csv"             # columns: doi, abstract
AWS_PAIRWISE_PATH = "results/aws_pairwise.csv"  # columns: cluster_a, cluster_b, D_ij
                                                # set to None to skip D_ij correlation

N_QUERIES  = 500   # total queries generated across all cluster pairs
N_TERMS    = 3     # terms per query component (source or aligned)
TOP_K      = 30    # top c-TF-IDF terms retained per cluster vocabulary
N_NULL     = 200   # null-model iterations for z-score estimation

# Minimum non-zero paired differences required for a valid Wilcoxon test.
# Fewer than this threshold makes the test unreliable.
MIN_NONZERO_WILCOXON = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Stop-word list (generic English + bibliographic / publisher noise)
# ---------------------------------------------------------------------------

STOPWORDS = set(
    "the and for that this with from are was were been have has had "
    "not but its can will also been into than more which their these "
    "they would could should about each between through during after "
    "before other some such only over also how our who what when where "
    "doi org https http www abstract article journal vol issue pages "
    "elsevier springer wiley science available online published".split()
)


# ---------------------------------------------------------------------------
# Step 1: Load and merge data
# ---------------------------------------------------------------------------

print("=" * 65)
print("RETRIEVAL FRAGMENTATION — canonical corpus k=6")
print("=" * 65)

# Cluster assignments
topics = pd.read_csv(TOPICS_PATH, on_bad_lines="skip")
topics["cluster"] = pd.to_numeric(topics["cluster"], errors="coerce")
topics = topics.dropna(subset=["cluster"])
topics["cluster"] = topics["cluster"].astype(int)
print(f"Topics   : {TOPICS_PATH} "
      f"({len(topics)} papers, clusters: {sorted(topics['cluster'].unique())})")

# Abstracts
corpus = pd.read_csv(CORPUS_PATH, on_bad_lines="skip")
if "abstract" not in corpus.columns:
    sys.exit("ERROR: 'abstract' column not found in corpus CSV.")
corpus = corpus[["doi", "abstract"]].dropna()
print(f"Abstracts: {CORPUS_PATH} ({len(corpus)} rows)")

# Primary merge on DOI
merged = topics.merge(corpus, on="doi", how="inner")
print(f"Merged (exact DOI match): {len(merged)} papers")

# Fallback: case-normalised DOI matching for unmatched rows
unmatched = topics[~topics["doi"].isin(merged["doi"])]
if len(unmatched) > 0:
    print(f"Unmatched DOIs: {len(unmatched)} — retrying with case normalisation")
    corpus_norm            = corpus.copy()
    corpus_norm["doi_key"] = corpus_norm["doi"].str.lower().str.strip()
    unmatched_norm         = unmatched.copy()
    unmatched_norm["doi_key"] = unmatched_norm["doi"].str.lower().str.strip()

    recovered = unmatched_norm.merge(
        corpus_norm[["doi_key", "abstract"]], on="doi_key", how="inner"
    ).drop(columns=["doi_key"])

    if len(recovered) > 0:
        merged = pd.concat([merged, recovered[merged.columns]], ignore_index=True)
        print(f"Recovered: {len(recovered)} papers via normalised DOI")

print(f"Total merged: {len(merged)} papers")

clusters = sorted(merged["cluster"].unique())
k = len(clusters)
print(f"Clusters : {clusters} (k={k})")


# ---------------------------------------------------------------------------
# Step 2: Tokeniser
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """
    Lowercase, strip punctuation and digits, remove short tokens and
    stop-words. Returns a list of content tokens.
    """
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Step 3: Build per-cluster c-TF-IDF vocabularies
# ---------------------------------------------------------------------------
# Approximation of c-TF-IDF: cluster document frequency x log(N / corpus DF).
# Binary within-document counting (via set) avoids document-length bias.

print("\nBuilding cluster vocabularies (c-TF-IDF)...")

cluster_vocab = {}   # {cluster_id: Counter(term -> cluster_doc_freq)}
corpus_df = Counter()  # term -> corpus-wide document frequency

for c in clusters:
    abstracts = merged[merged["cluster"] == c]["abstract"].tolist()
    term_freq = Counter()
    for doc in abstracts:
        term_freq.update(set(tokenize(doc)))  # binary: one count per document
    cluster_vocab[c] = term_freq
    corpus_df.update(term_freq.keys())

N_docs = len(merged)
cluster_top_terms = {}  # {cluster_id: [top_term, ...]}

for c in clusters:
    scores = {}
    for term, count in cluster_vocab[c].items():
        df_val = corpus_df[term]
        if df_val < 3:                          # discard very rare terms
            continue
        idf = np.log(N_docs / (df_val + 1))    # smoothed IDF
        scores[term] = count * idf

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]
    cluster_top_terms[c] = [t for t, _ in ranked]
    top5 = [t for t, _ in ranked[:5]]
    n_c  = len(merged[merged["cluster"] == c])
    print(f"  C{c} (n={n_c}): top-5 terms: {top5}")


# ---------------------------------------------------------------------------
# Step 4: Build in-memory search index
# ---------------------------------------------------------------------------

print("\nBuilding search index...")

# Store each document as a set of tokens for fast intersection testing.
doc_tokens   = []   # list of set[str]
doc_clusters = []   # list of int, parallel to doc_tokens

for _, row in merged.iterrows():
    doc_tokens.append(set(tokenize(row["abstract"])))
    doc_clusters.append(row["cluster"])


def search(query_terms: list) -> list:
    """
    Boolean OR search: returns indices of all documents that contain
    at least one term from query_terms.
    """
    query_set = set(query_terms)
    return [i for i, tokens in enumerate(doc_tokens) if query_set & tokens]


# ---------------------------------------------------------------------------
# Step 5: Generate queries and measure recall gain
# ---------------------------------------------------------------------------
# For each cluster pair (ca, cb):
#   - Source-only query    : N_TERMS terms from ca vocabulary
#   - Aligned query        : source terms + N_TERMS terms from cb vocabulary
# Gain = |aligned hits| - |source hits| measures cross-cluster retrieval uplift.

n_pairs = len(list(combinations(clusters, 2)))
print(f"\nGenerating {N_QUERIES} queries across {n_pairs} cluster pairs...")

pairs            = list(combinations(clusters, 2))
queries_per_pair = max(1, N_QUERIES // len(pairs))
extra_queries    = N_QUERIES - queries_per_pair * len(pairs)

results  = []
query_id = 0

for pi, (ca, cb) in enumerate(pairs):
    n_q     = queries_per_pair + (1 if pi < extra_queries else 0)
    terms_a = cluster_top_terms.get(ca, [])
    terms_b = cluster_top_terms.get(cb, [])

    if len(terms_a) < N_TERMS or len(terms_b) < N_TERMS:
        continue

    for _ in range(n_q):
        # Source-only query
        q_source    = list(np.random.choice(terms_a, size=N_TERMS, replace=False))
        hits_source = search(q_source)
        n_source    = len(hits_source)
        hits_in_target_before = sum(1 for i in hits_source if doc_clusters[i] == cb)

        # Aligned query (source + cross-cluster terms)
        q_aligned_terms      = list(np.random.choice(terms_b, size=N_TERMS, replace=False))
        q_aligned            = q_source + q_aligned_terms
        hits_aligned         = search(q_aligned)
        n_aligned            = len(hits_aligned)
        hits_in_target_after = sum(1 for i in hits_aligned if doc_clusters[i] == cb)

        gain     = n_aligned - n_source
        gain_pct = (gain / max(n_source, 1)) * 100

        results.append({
            "query_id":              query_id,
            "source_cluster":        ca,
            "target_cluster":        cb,
            "query_source":          " ".join(q_source),
            "query_aligned":         " ".join(q_aligned),
            "recall_source":         n_source,
            "recall_aligned":        n_aligned,
            "hits_in_target_before": hits_in_target_before,
            "hits_in_target_after":  hits_in_target_after,
            "gain":                  gain,
            "gain_pct":              gain_pct,
        })
        query_id += 1

df_results = pd.DataFrame(results)
print(f"Generated {len(df_results)} queries")


# ---------------------------------------------------------------------------
# Step 6: Significance test — Wilcoxon signed-rank (one-sided: aligned > source)
# ---------------------------------------------------------------------------
# Guard: the test is only run when there are enough non-zero differences.
# Below MIN_NONZERO_WILCOXON the test is statistically unreliable.

recall_source  = df_results["recall_source"].values
recall_aligned = df_results["recall_aligned"].values
diffs          = recall_aligned - recall_source
nonzero_mask   = diffs != 0

if nonzero_mask.sum() >= MIN_NONZERO_WILCOXON:
    stat, p_wilcoxon = wilcoxon(
        recall_aligned[nonzero_mask],
        recall_source[nonzero_mask],
        alternative="greater",
    )
else:
    print(
        f"WARNING: only {nonzero_mask.sum()} non-zero differences — "
        f"Wilcoxon test skipped (minimum required: {MIN_NONZERO_WILCOXON})."
    )
    stat, p_wilcoxon = 0.0, 1.0

mean_gain     = df_results["gain"].mean()
std_gain      = df_results["gain"].std()
median_gain   = df_results["gain"].median()
mean_gain_pct = df_results["gain_pct"].mean()
ci95          = 1.96 * std_gain / np.sqrt(len(df_results))

print(f"\nMean gain  : {mean_gain:.1f} +/- {ci95:.1f} papers  ({mean_gain_pct:.1f}%)")
print(f"Median gain: {median_gain:.1f} papers")
print(f"Std gain   : {std_gain:.2f}")
print(f"Wilcoxon   : W={stat:.0f}, p={p_wilcoxon:.2e}  "
      f"(non-zero pairs: {nonzero_mask.sum()}/{len(diffs)})")


# ---------------------------------------------------------------------------
# Step 7: Effect size — Cohen's d
# ---------------------------------------------------------------------------
# Cohen's d quantifies practical magnitude independently of sample size.
# Reported alongside p-value following Q1 reviewer conventions.
# Relative gain (mean_gain_pct) provides the narrative figure for the paper:
# "vocabulary alignment yields a ~X% increase in retrieval coverage."

cohens_d   = mean_gain / (std_gain + 1e-10)
effect_pct = mean_gain_pct

print(f"Cohen's d  : {cohens_d:.3f}")
print(f"Relative gain: {effect_pct:.1f}%")


# ---------------------------------------------------------------------------
# Step 8: Per-pair statistics
# ---------------------------------------------------------------------------

pair_stats = (
    df_results
    .groupby(["source_cluster", "target_cluster"])["gain"]
    .agg(mean_gain="mean", std_gain="std", n_queries="count")
    .reset_index()
)

# Optional: Spearman correlation between per-pair mean gain and D_ij.
# D_ij is the cross-cluster semantic distance from the AWS metric.
# A positive correlation would confirm that semantically distant clusters
# benefit more from vocabulary alignment — directly linking retrieval
# fragmentation to the AWS diagnostic.
rho_dij, p_dij = None, None

if AWS_PAIRWISE_PATH and os.path.exists(AWS_PAIRWISE_PATH):
    aws_pw = pd.read_csv(AWS_PAIRWISE_PATH)
    for col in ["cluster_a", "cluster_b"]:
        if col in aws_pw.columns:
            aws_pw[col] = pd.to_numeric(aws_pw[col], errors="coerce")

    pair_stats = pair_stats.merge(
        aws_pw[["cluster_a", "cluster_b", "D_ij"]],
        left_on=["source_cluster", "target_cluster"],
        right_on=["cluster_a", "cluster_b"],
        how="left",
    ).drop(columns=["cluster_a", "cluster_b"])

    valid = pair_stats["D_ij"].notna() & pair_stats["mean_gain"].notna()
    if valid.sum() >= 3:
        rho_dij, p_dij = spearmanr(
            pair_stats.loc[valid, "D_ij"],
            pair_stats.loc[valid, "mean_gain"],
        )
        print(f"\nSpearman D_ij x mean gain: rho={rho_dij:.3f}, p={p_dij:.3f} "
              f"(n={valid.sum()} pairs)")
    else:
        print("Insufficient matched pairs for D_ij x gain correlation.")
else:
    print(f"\nAWS pairwise file not found at '{AWS_PAIRWISE_PATH}' — "
          "D_ij correlation skipped.")


# ---------------------------------------------------------------------------
# Step 9: Null models
# ---------------------------------------------------------------------------
# Two null baselines isolate the contribution of cross-cluster alignment:
#
# Null 1 — Random   : aligned terms drawn uniformly from the full vocabulary.
# Null 2 — Frequency: aligned terms drawn from the top corpus-frequency words.
#
# If observed gain >> both null means, retrieval uplift is specifically due
# to semantically meaningful cross-cluster term alignment, not term frequency.

all_vocab  = list({t for terms in cluster_top_terms.values() for t in terms})
freq_terms = [
    t for t, _ in corpus_df.most_common(TOP_K * 2)
    if t not in STOPWORDS and len(t) > 3
][:TOP_K]


def run_null_model(term_pool: list) -> list:
    """
    Run N_NULL iterations. In each iteration, replace the aligned component
    of every query with N_TERMS terms randomly sampled from term_pool.
    Returns a list of per-iteration mean gains.
    """
    iteration_means = []
    for _ in range(N_NULL):
        gains = []
        for _, row in df_results.iterrows():
            null_terms = list(np.random.choice(term_pool, size=N_TERMS, replace=False))
            q_null = row["query_source"].split() + null_terms
            gains.append(len(search(q_null)) - row["recall_source"])
        iteration_means.append(np.mean(gains))
    return iteration_means


print("\nRunning null models...")
null_random_gains = run_null_model(all_vocab)
null_freq_gains   = run_null_model(freq_terms)

mu_r, sigma_r = np.mean(null_random_gains), np.std(null_random_gains)
mu_f, sigma_f = np.mean(null_freq_gains),   np.std(null_freq_gains)
Z_r = (mean_gain - mu_r) / (sigma_r + 1e-10)
Z_f = (mean_gain - mu_f) / (sigma_f + 1e-10)

print(f"Null random : mu={mu_r:.1f}, Z={Z_r:.1f}")
print(f"Null freq   : mu={mu_f:.1f}, Z={Z_f:.1f}")


# ---------------------------------------------------------------------------
# Step 10: Save outputs
# ---------------------------------------------------------------------------

df_results.to_csv(os.path.join(OUTPUT_DIR, "retrieval_results.csv"), index=False)
pair_stats.to_csv(os.path.join(OUTPUT_DIR, "pair_stats.csv"), index=False)

summary_row = {
    "n_queries":          len(df_results),
    "n_papers":           len(merged),
    "k":                  k,
    "mean_gain":          round(mean_gain, 3),
    "std_gain":           round(std_gain, 3),
    "ci95":               round(ci95, 3),
    "median_gain":        round(median_gain, 3),
    "mean_gain_pct":      round(mean_gain_pct, 3),
    "cohens_d":           round(cohens_d, 3),
    "wilcoxon_W":         stat,
    "wilcoxon_p":         p_wilcoxon,
    "n_nonzero_pairs":    int(nonzero_mask.sum()),
    "null_random_Z":      round(Z_r, 3),
    "null_freq_Z":        round(Z_f, 3),
    "spearman_rho_dij":   round(rho_dij, 3) if rho_dij is not None else None,
    "spearman_p_dij":     round(p_dij,   3) if p_dij   is not None else None,
}
pd.DataFrame([summary_row]).to_csv(
    os.path.join(OUTPUT_DIR, "retrieval_summary.csv"), index=False
)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

dij_line = (
    f"Spearman D_ij x gain : rho={rho_dij:.3f}, p={p_dij:.3f}"
    if rho_dij is not None
    else "Spearman D_ij x gain : not computed (AWS pairwise file missing)"
)

report = f"""
{'=' * 65}
RETRIEVAL FRAGMENTATION — canonical corpus k=6
{'=' * 65}
Papers  : {len(merged)}
Clusters: {k}  ({clusters[0]}-{clusters[-1]})
Queries : {len(df_results)}

RECALL
  Source-only  : {recall_source.mean():.1f} +/- {recall_source.std():.1f}
  Aligned      : {recall_aligned.mean():.1f} +/- {recall_aligned.std():.1f}
  Mean gain    : {mean_gain:.1f} +/- {ci95:.1f} (95% CI)  ({mean_gain_pct:.1f}%)
  Median gain  : {median_gain:.1f} papers
  Std gain     : {std_gain:.2f}

EFFECT SIZE
  Cohen's d    : {cohens_d:.3f}
  Relative gain: {mean_gain_pct:.1f}%

SIGNIFICANCE
  Wilcoxon signed-rank (one-sided, greater)
  W={stat:.0f}, p={p_wilcoxon:.2e}  (non-zero: {nonzero_mask.sum()}/{len(diffs)})

CORRELATION WITH SEMANTIC DISTANCE
  {dij_line}

NULL MODELS
  Null 1 — Random   : mu={mu_r:.1f}, Z={Z_r:.1f}
  Null 2 — Frequency: mu={mu_f:.1f}, Z={Z_f:.1f}
{'=' * 65}
"""

print(report)
with open(os.path.join(OUTPUT_DIR, "retrieval_report.txt"), "w") as f:
    f.write(report)

print(f"All outputs written to: {OUTPUT_DIR}/")
