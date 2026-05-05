"""
Computes geographic distribution and top-cited papers per mega-area for the
Circular Economy review paper, using authorship data fetched directly from the
OpenAlex API (output of fetch_ce_affiliations.py) rather than the partial
figshare dataset.

Pipeline:
  topics (doi -> mega_area)
      <- join via ce_papers_meta.csv (doi <-> openalex_id) ->
  ce_authorships.csv (paper_id, author_id)
      <- join via author_id ->
  ce_affiliations.csv (author_id, institution_id)
      <- join via institution_id ->
  ce_institutions.csv (openalex_id, country)

Each paper gets all unique countries of its authors' affiliations. Country
attribution is paper-level (a paper with US + Brazil authors counts for both),
following standard scientometric collaboration analysis convention.

UN classification (UNDESA M49 + UN Statistics Division 2022):
  Global North = Europe, Northern America, Australia/NZ, Japan, South Korea,
                 Israel.
  Global South = everything else.

Inputs (under --base by default):
  ce_openalex/ce_papers_meta.csv
  ce_openalex/ce_authorships.csv
  ce_openalex/ce_affiliations.csv
  ce_openalex/ce_institutions.csv

Plus the topic file:
  --topics paper_topics_clean.csv

Outputs (under --out):
  geography_by_area_v2.csv         country counts per mega-area (top 15)
  geography_global_split_v2.csv    North vs South share per mega-area
  top_cited_per_area_v2.csv        top 10 most-cited papers per mega-area
  geography_coverage_v2.csv        coverage stats per mega-area

Run:
  python ce_geography_v2.py \\
      --topics ./results_circular_economy/full_corpus/paper_topics_clean.csv \\
      --base ./ce_openalex/ \\
      --out results/ce_review/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEGA_AREAS = {
    "Materials & Technical Recycling": [10, 11],
    "Industrial Sectors & Applied Cases": [7, 8, 9],
    "Energy & Resource Systems": [2, 14],
    "Business, Policy & Governance": [4, 6],
    "Sustainability Framing & Society": [3, 5],
}
CLUSTER_TO_MEGA = {c: m for m, cs in MEGA_AREAS.items() for c in cs}


GLOBAL_NORTH = {
    # Europe
    "AD", "AL", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK",
    "EE", "ES", "FI", "FO", "FR", "GB", "GG", "GI", "GR", "HR", "HU", "IE",
    "IM", "IS", "IT", "JE", "LI", "LT", "LU", "LV", "MC", "MD", "ME", "MK",
    "MT", "NL", "NO", "PL", "PT", "RO", "RS", "RU", "SE", "SI", "SK", "SM",
    "UA", "VA", "XK",
    # Northern America
    "CA", "US",
    # Oceania (developed)
    "AU", "NZ",
    # Asia (developed by UN)
    "JP", "KR",
    # Middle East (developed)
    "IL",
}


def classify(country: str | None) -> str:
    if country is None or pd.isna(country):
        return "Unknown"
    return "Global North" if str(country).upper() in GLOBAL_NORTH else "Global South"


def normalise_doi(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    )


def load_topics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip", dtype={"cluster": "Int64"})
    df = df.dropna(subset=["cluster"])
    df["cluster"] = df["cluster"].astype(int)
    df["mega_area"] = df["cluster"].map(CLUSTER_TO_MEGA)
    df = df.dropna(subset=["mega_area"])
    return df[["doi", "cluster", "mega_area"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--base", default="ce_openalex/",
                    help="folder with ce_papers_meta.csv etc.")
    ap.add_argument("--out", default="results/ce_review/")
    ap.add_argument("--top-cited-n", type=int, default=10)
    ap.add_argument("--top-countries-n", type=int, default=15)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)

    # ---- Load
    print("Loading topics + OpenAlex data...")
    topics = load_topics(Path(args.topics))
    papers = pd.read_csv(base / "ce_papers_meta.csv")
    auth = pd.read_csv(base / "ce_authorships.csv")
    aff = pd.read_csv(base / "ce_affiliations.csv")
    inst = pd.read_csv(base / "ce_institutions.csv")
    print(f"  topics: {len(topics):,} | papers: {len(papers):,} | "
          f"authorships: {len(auth):,} | affiliations: {len(aff):,} | "
          f"institutions: {len(inst):,}")

    # ---- Step 1: doi -> openalex_id
    print("\n[1/4] Joining topics to papers (doi -> openalex_id)...")
    topics["doi_norm"] = normalise_doi(topics["doi"])
    papers["doi_norm"] = normalise_doi(papers["doi"])
    paper_subset = papers[["openalex_id", "doi_norm", "title", "year",
                           "citations", "type", "is_oa"]]
    paper_subset = paper_subset.drop_duplicates(subset=["doi_norm"], keep="first")

    t = topics.merge(paper_subset, on="doi_norm", how="inner")
    t = t.rename(columns={"openalex_id": "paper_id"})
    print(f"  matched: {len(t):,} papers ({100*len(t)/len(topics):.1f}% of topics)")

    # ---- Step 2: paper_id -> author_id
    print("\n[2/4] Joining to authorships...")
    pa = auth[["paper_id", "author_id"]].merge(
        t[["paper_id", "mega_area", "title", "year", "citations", "is_oa"]],
        on="paper_id", how="inner",
    )
    print(f"  paper-author rows: {len(pa):,}")

    # ---- Step 3: author_id -> institution_id
    print("\n[3/4] Joining to affiliations...")
    pi = pa.merge(aff, on="author_id", how="inner")
    print(f"  paper-author-institution rows: {len(pi):,}")

    # ---- Step 4: institution_id -> country
    print("\n[4/4] Joining to institutions (country)...")
    inst_country = inst[["openalex_id", "country"]].rename(
        columns={"openalex_id": "institution_id"}
    )
    pic = pi.merge(inst_country, on="institution_id", how="inner")
    print(f"  paper-author-institution-country rows: {len(pic):,}")

    n_papers_with_country = pic["paper_id"].nunique()
    n_total = t["paper_id"].nunique()
    pct_cov = 100 * n_papers_with_country / max(n_total, 1)
    print(f"  papers with at least one country: {n_papers_with_country:,} "
          f"({pct_cov:.1f}% of analytical corpus)")

    # ---- Coverage by mega-area (for limitations section)
    print("\n=== Coverage by mega-area ===")
    cov = t[["paper_id", "mega_area"]].drop_duplicates()
    has_country = pic.groupby("mega_area")["paper_id"].nunique()
    total_per = cov.groupby("mega_area").size()
    cov_table = pd.DataFrame({
        "with_country": has_country,
        "total": total_per,
    }).fillna(0).astype(int)
    cov_table["pct_coverage"] = (100 * cov_table["with_country"] /
                                 cov_table["total"]).round(1)
    cov_table = cov_table.sort_values("pct_coverage", ascending=False)
    cov_table.to_csv(out / "geography_coverage_v2.csv")
    print(cov_table.to_string())

    # ---- De-duplicate at paper x country
    paper_country = (
        pic[["paper_id", "mega_area", "country"]]
        .drop_duplicates()
        .dropna(subset=["country"])
    )
    paper_country = paper_country[paper_country["country"].astype(str).str.len() > 0]
    paper_country["region"] = paper_country["country"].apply(classify)

    # ---- Top countries per mega-area
    print("\n=== Geography by mega-area (top countries) ===")
    counts = (
        paper_country.groupby(["mega_area", "country"])
        .size()
        .reset_index(name="n_papers")
        .sort_values(["mega_area", "n_papers"], ascending=[True, False])
    )
    top_countries = counts.groupby("mega_area").head(args.top_countries_n)
    top_countries.to_csv(out / "geography_by_area_v2.csv", index=False)
    print(top_countries.to_string(index=False))

    # ---- Global North vs South
    print("\n=== Global North vs South share by mega-area ===")
    region_counts = (
        paper_country.groupby(["mega_area", "region"])
        .size()
        .reset_index(name="n_paper_country_rows")
    )
    pivot = (
        region_counts.pivot(index="mega_area", columns="region",
                            values="n_paper_country_rows")
        .fillna(0)
    )
    for col in ("Global North", "Global South", "Unknown"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot["Global North"] + pivot["Global South"]
    pivot["pct_north"] = (100 * pivot["Global North"] / pivot["total"]).round(1)
    pivot["pct_south"] = (100 * pivot["Global South"] / pivot["total"]).round(1)
    pivot = pivot[["Global North", "Global South", "total",
                   "pct_north", "pct_south"]]
    pivot.to_csv(out / "geography_global_split_v2.csv")
    print(pivot.to_string())

    # ---- Top cited per mega-area
    print(f"\n=== Top-{args.top_cited_n} cited per mega-area ===")
    cited = t.dropna(subset=["citations"]).copy()
    cited["citations"] = pd.to_numeric(cited["citations"], errors="coerce")
    cited = cited.dropna(subset=["citations"])

    rows = []
    for mega, sub in cited.groupby("mega_area"):
        top = sub.nlargest(args.top_cited_n, "citations")
        for _, r in top.iterrows():
            rows.append({
                "mega_area": mega,
                "doi": r.get("doi"),
                "title": r.get("title"),
                "year": (int(float(r["year"]))
                         if pd.notna(r["year"]) and str(r["year"]).strip()
                         else None),
                "citations": int(r["citations"]),
            })
    top_cited = pd.DataFrame(rows)
    top_cited.to_csv(out / "top_cited_per_area_v2.csv", index=False)
    print(top_cited.head(20).to_string(index=False))

    print(f"\nDONE -> {out}")
    print("Files:")
    print("  geography_by_area_v2.csv")
    print("  geography_global_split_v2.csv")
    print("  geography_coverage_v2.csv")
    print("  top_cited_per_area_v2.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
