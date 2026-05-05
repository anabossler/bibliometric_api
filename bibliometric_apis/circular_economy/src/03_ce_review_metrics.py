"""

Extracts the four empirical metrics:
"Circular Economy as concept vs. practice: a critical mapping of 25,823 publications"

Inputs (CSV paths overridable via CLI):
  - paper_topics_clean.csv : doi, title, year, cluster, topic_label
  - corpus.csv             : doi, abstract  (for CE-talk vs CE-practice analysis)
  - paper_authorships.csv  : doi, institution_id   (OPTIONAL - for geography)
  - institutions.csv       : openalex_id, name, country, type

Outputs (under --out):
  - mega_area_summary.csv         : per mega-area, descriptive stats
  - temporal_growth.csv           : papers per year per mega-area
  - ce_talk_vs_practice.csv       : per mega-area, CE-talk and CE-practice frequencies
  - geography_by_area.csv         : top countries per mega-area  (if authorships available)
  - top_cited_per_area.csv        : top 10 papers per mega-area  (if citations available)
  - underexplored_topics.csv      : topics with low CE coverage (manual gap analysis seed)

Mapping of 11 thematic clusters to 5 mega-areas:
  Materials & Technical Recycling     : C10, C11
  Industrial Sectors & Applied Cases  : C7, C9, C8
  Energy & Resource Systems           : C2, C14
  Business, Policy & Governance       : C4, C6
  Sustainability Framing & Society    : C3, C5

Run:
  python ce_review_metrics.py \\
      --topics paper_topics_clean.csv \\
      --corpus circular_economy_corpus.csv \\
      --out results/ce_review
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------------
# Configuration: cluster-to-mega-area mapping
# ----------------------------------------------------------------------------

MEGA_AREAS = {
    "Materials & Technical Recycling": [10, 11],
    "Industrial Sectors & Applied Cases": [7, 8, 9],
    "Energy & Resource Systems": [2, 14],
    "Business, Policy & Governance": [4, 6],
    "Sustainability Framing & Society": [3, 5],
}

# Reverse lookup
CLUSTER_TO_MEGA = {c: mega for mega, cs in MEGA_AREAS.items() for c in cs}


# ----------------------------------------------------------------------------
# Vocabularies for CE-talk vs CE-practice analysis
# ----------------------------------------------------------------------------

# CE-TALK: explicit invocation of the circular economy paradigm
CE_TALK_TERMS = [
    r"\bcircular econom\w*\b",
    r"\bcircularity\b",
    r"\bclosed-?loop\b",
    r"\bsustainab\w+ paradigm\b",
    r"\bsustainability transition\b",
    r"\bCE framework\b",
    r"\bCE strateg\w+\b",
    r"\bcircular business model\w*\b",
]

# CE-PRACTICE: concrete CE actions (the "R-strategies" + LCA + concrete operations)
CE_PRACTICE_TERMS = [
    r"\brecycl\w+\b",            # recycling, recycled, recyclable
    r"\bremanufactur\w+\b",
    r"\brefurbish\w+\b",
    r"\brepair\w*\b",
    r"\breus\w+\b",               # reuse, reused, reusing
    r"\breduc\w+ waste\b",
    r"\bwaste reduction\b",
    r"\bupcycl\w+\b",
    r"\bdownsycl\w+\b",
    r"\blife[- ]cycle assessment\b",
    r"\b\bLCA\b",
    r"\bend[- ]of[- ]life\b",
    r"\bmaterial recovery\b",
    r"\benergy recovery\b",
    r"\breverse logistics\b",
    r"\bmechanical recycling\b",
    r"\bchemical recycling\b",
    r"\bbiodegrad\w+\b",
    r"\bcompost\w+\b",
]

CE_TALK_RE = re.compile("|".join(CE_TALK_TERMS), re.IGNORECASE)
CE_PRACTICE_RE = re.compile("|".join(CE_PRACTICE_TERMS), re.IGNORECASE)


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def load_topics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip", dtype={"cluster": "Int64"})
    df = df.dropna(subset=["cluster"])
    df["cluster"] = df["cluster"].astype(int)
    df["mega_area"] = df["cluster"].map(CLUSTER_TO_MEGA)
    df = df.dropna(subset=["mega_area"])
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def load_corpus(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    if "abstract" not in df.columns:
        sys.exit("corpus.csv must contain an 'abstract' column")
    return df[["doi", "abstract"]].dropna()


def load_institutions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    return df[["openalex_id", "country"]].dropna()


# ----------------------------------------------------------------------------
# Analysis 1 — Mega-area summary
# ----------------------------------------------------------------------------

def mega_area_summary(topics: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for mega, sub in topics.groupby("mega_area"):
        rows.append({
            "mega_area": mega,
            "clusters": ", ".join(f"C{c}" for c in sorted(sub["cluster"].unique())),
            "n_papers": len(sub),
            "year_min": int(sub["year"].min()) if sub["year"].notna().any() else None,
            "year_max": int(sub["year"].max()) if sub["year"].notna().any() else None,
            "year_median": int(sub["year"].median()) if sub["year"].notna().any() else None,
        })
    df = pd.DataFrame(rows).sort_values("n_papers", ascending=False)
    df.to_csv(out_path, index=False)
    return df


# ----------------------------------------------------------------------------
# Analysis 2 — Temporal growth
# ----------------------------------------------------------------------------

def temporal_growth(topics: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    sub = topics.dropna(subset=["year"]).copy()
    sub["year"] = sub["year"].astype(int)
    grp = (
        sub.groupby(["mega_area", "year"])
        .size()
        .reset_index(name="n_papers")
        .sort_values(["mega_area", "year"])
    )
    grp.to_csv(out_path, index=False)

    # Decade buckets for the table that goes in the paper
    sub["decade"] = (sub["year"] // 10) * 10
    decade_table = (
        sub.groupby(["mega_area", "decade"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    decade_table.to_csv(
        out_path.with_name(out_path.stem + "_by_decade.csv"), index=False
    )
    return grp


# ----------------------------------------------------------------------------
# Analysis 3 — CE-talk vs CE-practice (the core novel contribution)
# ----------------------------------------------------------------------------

def ce_talk_vs_practice(
    topics: pd.DataFrame, corpus: pd.DataFrame, out_path: Path
) -> pd.DataFrame:
    """
    For each abstract, count CE-talk hits and CE-practice hits.
    Aggregate per mega-area and report:
      - % of papers mentioning at least one CE-talk term
      - % of papers mentioning at least one CE-practice term
      - mean count of each
      - ratio practice/talk  (>1 = field practices CE; <1 = field labels CE)
    """
    df = topics.merge(corpus, on="doi", how="inner")
    print(f"  CE-talk/practice: {len(df)} abstracts merged")

    df["ce_talk_count"] = df["abstract"].str.count(CE_TALK_RE)
    df["ce_practice_count"] = df["abstract"].str.count(CE_PRACTICE_RE)
    df["has_talk"] = df["ce_talk_count"] > 0
    df["has_practice"] = df["ce_practice_count"] > 0

    rows = []
    for mega, sub in df.groupby("mega_area"):
        n = len(sub)
        rows.append({
            "mega_area": mega,
            "n_abstracts": n,
            "pct_with_ce_talk": round(100 * sub["has_talk"].mean(), 1),
            "pct_with_ce_practice": round(100 * sub["has_practice"].mean(), 1),
            "mean_ce_talk_count": round(sub["ce_talk_count"].mean(), 2),
            "mean_ce_practice_count": round(sub["ce_practice_count"].mean(), 2),
            "practice_to_talk_ratio": round(
                sub["ce_practice_count"].mean()
                / max(sub["ce_talk_count"].mean(), 0.01),
                2,
            ),
            "interpretation": _interpret_ratio(
                sub["ce_practice_count"].mean(), sub["ce_talk_count"].mean()
            ),
        })

    out = pd.DataFrame(rows).sort_values("practice_to_talk_ratio", ascending=False)
    out.to_csv(out_path, index=False)
    return out


def _interpret_ratio(practice_mean: float, talk_mean: float) -> str:
    if talk_mean < 0.1:
        return "CE absent / implicit field"
    ratio = practice_mean / max(talk_mean, 0.01)
    if ratio >= 3:
        return "CE-practice dominant (field acts on CE)"
    if ratio >= 1.5:
        return "CE-practice oriented"
    if ratio >= 0.7:
        return "CE-talk and CE-practice balanced"
    return "CE-talk dominant (field labels rather than acts)"


# ----------------------------------------------------------------------------
# Analysis 4 — Top cited per area
# ----------------------------------------------------------------------------

def top_cited_per_area(
    topics: pd.DataFrame,
    citations_col: str | None,
    out_path: Path,
    top_n: int = 10,
) -> pd.DataFrame | None:
    """If topics has a citation count column, return top-N papers per mega-area."""
    if citations_col is None or citations_col not in topics.columns:
        print(f"  (skipped: column '{citations_col}' not found)")
        return None
    rows = []
    for mega, sub in topics.groupby("mega_area"):
        top = sub.nlargest(top_n, citations_col)
        for _, r in top.iterrows():
            rows.append({
                "mega_area": mega,
                "doi": r.get("doi"),
                "title": r.get("title"),
                "year": r.get("year"),
                "citations": r[citations_col],
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


# ----------------------------------------------------------------------------
# Analysis 5 — Geography
# ----------------------------------------------------------------------------

def geography_by_area(
    topics: pd.DataFrame,
    authorships_path: Path | None,
    institutions: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame | None:
    """
    Requires an authorships file with columns [doi, institution_id].
    Cross-tabs mega-area x country and returns top countries per area.
    """
    if authorships_path is None or not authorships_path.exists():
        print("  (skipped: authorships file not provided)")
        return None

    auth = pd.read_csv(authorships_path, on_bad_lines="skip")
    if "institution_id" not in auth.columns or "doi" not in auth.columns:
        print("  (skipped: authorships file must have doi, institution_id)")
        return None

    inst_country = dict(zip(institutions["openalex_id"], institutions["country"]))
    auth["country"] = auth["institution_id"].map(inst_country)
    auth = auth.dropna(subset=["country"])

    df = topics[["doi", "mega_area"]].merge(auth, on="doi", how="inner")
    counts = (
        df.groupby(["mega_area", "country"])
        .size()
        .reset_index(name="n_papers")
        .sort_values(["mega_area", "n_papers"], ascending=[True, False])
    )

    # Top 10 per mega-area
    top = counts.groupby("mega_area").head(10)
    top.to_csv(out_path, index=False)
    return top


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True, help="paper_topics_clean.csv")
    ap.add_argument("--corpus", required=True, help="abstracts CSV (doi, abstract)")
    ap.add_argument("--authorships", default=None, help="optional doi,institution_id CSV")
    ap.add_argument("--institutions", default=None, help="optional institutions.csv")
    ap.add_argument("--citations-col", default=None,
                    help="column name with citation counts in --topics, e.g. 'cited_by_count'")
    ap.add_argument("--out", default="results/ce_review")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    topics = load_topics(Path(args.topics))
    corpus = load_corpus(Path(args.corpus))
    print(f"  topics: {len(topics)} papers across {topics['mega_area'].nunique()} mega-areas")
    print(f"  corpus: {len(corpus)} abstracts")

    print("\n[1/5] Mega-area summary")
    s1 = mega_area_summary(topics, out / "mega_area_summary.csv")
    print(s1.to_string(index=False))

    print("\n[2/5] Temporal growth")
    temporal_growth(topics, out / "temporal_growth.csv")
    print("  written: temporal_growth.csv, temporal_growth_by_decade.csv")

    print("\n[3/5] CE-talk vs CE-practice")
    s3 = ce_talk_vs_practice(topics, corpus, out / "ce_talk_vs_practice.csv")
    print(s3.to_string(index=False))

    print("\n[4/5] Top cited per mega-area")
    top_cited_per_area(
        topics, args.citations_col, out / "top_cited_per_area.csv"
    )

    print("\n[5/5] Geography")
    if args.institutions:
        institutions = load_institutions(Path(args.institutions))
        geography_by_area(
            topics,
            Path(args.authorships) if args.authorships else None,
            institutions,
            out / "geography_by_area.csv",
        )
    else:
        print("  (skipped: institutions.csv not provided)")

    print(f"\nDONE -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
