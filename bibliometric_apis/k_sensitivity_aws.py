"""
k_sensitivity_aws.py
====================
Computes AWS-score components (S_cross, D_w, AWS) across k=4..12
using multi-k cluster assignments and document abstracts.
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR  = "results/k_sensitivity"
CORPUS_PATH = "data/corpus.csv"                  # columns: doi, abstract
TOPICS_PATH = "data/paper_topics_multik.csv"     # columns: doi, cluster[, k]
TOPICS_PATH_K6 = "data/paper_topics_k6.csv"      # fallback: single k=6 run
RBO_PATH    = "data/rbo_fragmentation.csv"        # fallback: precomputed RBO

TOP_N = 50
P_RBO = 0.9

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Stop-word list (generic English + bibliographic / publisher noise)
# ---------------------------------------------------------------------------

STOPWORDS = set(
    "the and for that this with from are was were been have has had "
    "not but its can will also into than more which their these they "
    "would could should about each between through during after before "
    "other some such only over how our who what when where doi org "
    "https http www abstract article journal vol issue pages "
    "elsevier springer wiley science available online published".split()
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rbo(l1: list, l2: list, p: float = 0.9) -> float:
    """
    Rank-Biased Overlap between two ranked lists.
    Returns a value in [0, 1]; higher means more overlap.
    """
    if not l1 or not l2:
        return 0.0
    sl, ll = (l1, l2) if len(l1) <= len(l2) else (l2, l1)
    s, d = set(), 0.0
    for i, v in enumerate(sl, 1):
        s.add(v)
        d += (len(s & set(ll[:i])) / i) * p ** (i - 1)
    return (1 - p) * d


def tokenize(text: str) -> list:
    """
    Lowercase, strip punctuation, remove short tokens and stop-words.
    """
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

topics_all = pd.read_csv(TOPICS_PATH, on_bad_lines="skip")
corpus     = pd.read_csv(CORPUS_PATH, on_bad_lines="skip")
corpus     = corpus[["doi", "abstract"]].dropna()

print("Topics columns:", topics_all.columns.tolist())
print("Sample clusters:", topics_all["cluster"].unique()[:10])
print("Total rows:", len(topics_all))

# Detect whether the file contains a multi-k column
if "k" in topics_all.columns:
    k_values = sorted(topics_all["k"].unique())
    print("k values found:", k_values)
    use_k_col = True
else:
    print("No 'k' column — treating as single-run file")
    use_k_col = False

results = []

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

if use_k_col:
    k_range = [k for k in k_values if 4 <= k <= 12]

else:
    # Fallback: load precomputed k=6 assignments and RBO data.
    # S_cross is k-invariant; D_w cannot be recomputed without re-clustering.
    backup_topics = pd.read_csv(TOPICS_PATH_K6, on_bad_lines="skip")
    backup_topics["cluster"] = pd.to_numeric(
        backup_topics["cluster"].astype(str).str.replace('"', ""), errors="coerce"
    )
    backup_topics = backup_topics.dropna(subset=["cluster"])
    backup_topics["cluster"] = backup_topics["cluster"].astype(int)

    merged6 = backup_topics.merge(corpus, on="doi", how="inner")

    rbo_df = pd.read_csv(RBO_PATH, on_bad_lines="skip")
    print("\nRBO file columns:", rbo_df.columns.tolist())
    print(rbo_df.head())

    print("\nTopics sample (multi-k file):")
    print(topics_all.head(10).to_string())
    k_range = []

print("\nDone. Check output above to decide next step.")
