"""

Downloads cross-cluster citation edges from OpenAlex for the
plastic recycling corpus.

For each paper in the corpus, fetches its reference list from OpenAlex
and filters to references that are also in the corpus but belong to
a different cluster. These pairs are the ground truth for the
retrieval experiment in retrieval_ground_truth.py.


INPUTS:
    backup_recycled_a/full_corpus/paper_topics.csv
        (doi, cluster, ...)

OUTPUTS:
    backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv
        (source_doi, source_cluster, target_doi, target_cluster)

"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR   = Path("backup_recycled_a/full_corpus")
TOPICS     = BASE_DIR / "paper_topics.csv"
OUT_PATH   = BASE_DIR / "citation_edges_cross_cluster.csv"

SLEEP_PER_PAPER = 0.1
SLEEP_PER_REF   = 0.05
TIMEOUT         = 15

topics = pd.read_csv(TOPICS)[["doi", "cluster"]].dropna()
topics["cluster"] = topics["cluster"].astype(int)

corpus_dois    = set(topics["doi"].tolist())
doi_to_cluster = dict(zip(topics["doi"], topics["cluster"]))

print(f"Corpus: {len(corpus_dois)} papers")

# ---------------------------------------------------------------------------
# Resume support: load already-found pairs
# ---------------------------------------------------------------------------

already_done: set[str] = set()
pairs: list[dict] = []

if OUT_PATH.exists():
    existing = pd.read_csv(OUT_PATH)
    pairs = existing.to_dict("records")
    already_done = set(existing["source_doi"].tolist())
    print(f"Resuming: {len(already_done)} papers already processed, "
          f"{len(pairs)} pairs found")

errors = 0

for i, (_, row) in enumerate(topics.iterrows()):
    doi            = row["doi"]
    source_cluster = int(row["cluster"])

    if doi in already_done:
        continue

    url = (
        f"https://api.openalex.org/works/https://doi.org/{doi}"
        f"?select=doi,referenced_works"
    )
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            refs = r.json().get("referenced_works", [])
            for ref_url in refs:
                work_id = ref_url.split("/")[-1]
                r2 = requests.get(
                    f"https://api.openalex.org/works/{work_id}?select=doi",
                    timeout=TIMEOUT,
                )
                if r2.status_code == 200:
                    ref_doi_full  = r2.json().get("doi", "")
                    ref_doi_clean = ref_doi_full.replace(
                        "https://doi.org/", ""
                    )
                    if ref_doi_clean in corpus_dois:
                        target_cluster = doi_to_cluster[ref_doi_clean]
                        if target_cluster != source_cluster:
                            pairs.append({
                                "source_doi":     doi,
                                "source_cluster": source_cluster,
                                "target_doi":     ref_doi_clean,
                                "target_cluster": target_cluster,
                            })
                time.sleep(SLEEP_PER_REF)
    except Exception:
        errors += 1

    already_done.add(doi)

    if (i + 1) % 100 == 0:
        pd.DataFrame(pairs).to_csv(OUT_PATH, index=False)
        print(
            f"  {i+1}/{len(topics)} papers — "
            f"{len(pairs)} cross-cluster pairs — "
            f"errors: {errors}"
        )

    time.sleep(SLEEP_PER_PAPER)


df_pairs = pd.DataFrame(pairs)
df_pairs.to_csv(OUT_PATH, index=False)

print(f"\nDone. {len(pairs)} cross-cluster citation pairs saved to {OUT_PATH}")
print(
    df_pairs
    .groupby(["source_cluster", "target_cluster"])
    .size()
    .to_string()
)
