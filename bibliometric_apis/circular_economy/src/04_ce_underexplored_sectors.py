"""

Counts publications in the corpus that mention each underexplored sector
(healthcare, defence, education, ICT, transport beyond EVs) so the
"Underexplored territories" section of the paper has real numbers, not
placeholders.

Run:
  python ce_underexplored_sectors.py \\
      --corpus ./corpus_circular_economy.csv \\
      --topics ./results_circular_economy/full_corpus/paper_topics_clean.csv \\
      --out results/ce_review/underexplored_sectors.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# Each sector defined by a regex disjunction. The patterns are deliberately
# inclusive: any abstract mentioning any of these terms is counted as
# touching the sector. Hits do not imply the paper is *about* that sector,
# only that the sector appears in the abstract.

SECTORS = {
    "Healthcare & pharmaceuticals": [
        r"\bhealthcare\b",
        r"\bhealth[- ]care\b",
        r"\bhospital[s]?\b",
        r"\bmedical (device|equipment|waste|packaging)\b",
        r"\bpharmaceutical[s]?\b",
        r"\bdrug (manufacturing|packaging)\b",
        r"\bbiomedical (waste|device)\b",
        r"\bclinical waste\b",
    ],
    "Defence & military": [
        r"\bmilitary\b",
        r"\bdefence\b",
        r"\bdefense\b",
        r"\barmed forces\b",
        r"\bdecommission\w*\b",
        r"\bweapon[s]? (recycl|disposal)\w*\b",
        r"\bammunition\b",
    ],
    "Education sector (own material flows)": [
        r"\beducation (sector|institution[s]?|building[s]?)\b",
        r"\bschool[s]? (building[s]?|infrastructure|waste)\b",
        r"\buniversity (campus|building[s]?|waste)\b",
        r"\beducational material[s]?\b",
        r"\btextbook recycl\w+\b",
    ],
    "ICT & data centres": [
        r"\bdata cent(re|er)[s]?\b",
        r"\bcloud computing\b",
        r"\bserver (farm|infrastructure)\b",
        r"\binformation and communication technolog\w*\b",
        r"\bICT (sector|infrastructure|equipment)\b",
        r"\bnetwork equipment\b",
        r"\btelecommunications (equipment|infrastructure)\b",
    ],
    "Transport (non-EV)": [
        r"\baviation\b",
        r"\baircraft (recycl|decommission|end-of-life)\w*\b",
        r"\bairline[s]?\b",
        r"\bmaritime (transport|shipping|sector|recycl)\w*\b",
        r"\bship (recycl|decommission|breaking|dismantling)\w*\b",
        r"\brail (sector|transport|recycl|infrastructure)\w*\b",
        r"\brailway[s]?\b",
        r"\bport (infrastructure|operations)\b",
    ],
}


def compile_patterns(sectors: dict[str, list[str]]) -> dict[str, re.Pattern]:
    return {
        name: re.compile("|".join(patterns), re.IGNORECASE)
        for name, patterns in sectors.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="circular_economy_corpus.csv with doi, abstract")
    ap.add_argument("--topics", default=None,
                    help="optional paper_topics_clean.csv to break down by mega-area")
    ap.add_argument("--out", default="underexplored_sectors.csv")
    args = ap.parse_args()

    print("Loading corpus...")
    corpus = pd.read_csv(args.corpus, on_bad_lines="skip")
    if "abstract" not in corpus.columns or "doi" not in corpus.columns:
        sys.exit("corpus must contain doi and abstract columns")
    corpus = corpus[["doi", "abstract"]].dropna()
    corpus["abstract"] = corpus["abstract"].astype(str)
    print(f"  {len(corpus)} abstracts loaded")

    patterns = compile_patterns(SECTORS)

    rows = []
    for sector, regex in patterns.items():
        mask = corpus["abstract"].str.contains(regex, na=False)
        n_hit = int(mask.sum())
        pct = round(100 * n_hit / len(corpus), 2)
        rows.append({
            "sector": sector,
            "n_publications": n_hit,
            "pct_of_corpus": pct,
        })
        print(f"  {sector}: {n_hit} ({pct}%)")

    df = pd.DataFrame(rows).sort_values("n_publications")
    df.to_csv(args.out, index=False)
    print(f"\nWritten: {args.out}")
    print()
    print(df.to_string(index=False))

    # Optional: per-mega-area breakdown
    if args.topics:
        print("\n--- Breakdown by mega-area (papers in topics file) ---")
        topics = pd.read_csv(args.topics, on_bad_lines="skip")
        topics["cluster"] = pd.to_numeric(topics["cluster"], errors="coerce")
        topics = topics.dropna(subset=["cluster"])
        topics["cluster"] = topics["cluster"].astype(int)

        MEGA = {
            "Materials & Technical Recycling": [10, 11],
            "Industrial Sectors & Applied Cases": [7, 8, 9],
            "Energy & Resource Systems": [2, 14],
            "Business, Policy & Governance": [4, 6],
            "Sustainability Framing & Society": [3, 5],
        }
        cluster_to_mega = {c: m for m, cs in MEGA.items() for c in cs}
        topics["mega_area"] = topics["cluster"].map(cluster_to_mega)
        topics = topics.dropna(subset=["mega_area"])

        merged = topics.merge(corpus, on="doi", how="inner")
        for sector, regex in patterns.items():
            merged[f"hit_{sector}"] = merged["abstract"].str.contains(regex, na=False)

        for sector in patterns:
            print(f"\n{sector}:")
            print(merged.groupby("mega_area")[f"hit_{sector}"].sum().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
