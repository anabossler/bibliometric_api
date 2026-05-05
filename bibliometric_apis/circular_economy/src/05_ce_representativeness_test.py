"""

Tests whether the subset of papers with country attribution (papers with at
least one author affiliation country code) is representative of the full
analytical corpus, in three observable variables:

  1. Year of publication
  2. Citation count
  3. Open Access status

This addresses a referee concern that the geographic analysis subset may be
systematically biased toward more recent / more cited / more Open Access
papers, which would distort Global North/South claims.

Methodology:
  - Year & citations: Mann-Whitney U test (non-parametric, robust to outliers)
  - Open Access: Chi-square test
  - Effect sizes: median difference (year, citations), proportion difference (OA)

Outputs (under --out):
  representativeness_test.csv  - test statistics and p-values
  representativeness_summary.txt - human-readable interpretation

Run:
  python ce_representativeness_test.py \\
      --topics ./results_circular_economy/full_corpus/paper_topics_clean.csv \\
      --base ./ce_openalex/ \\
      --out results/ce_review/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


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
    return df[["doi", "cluster"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", required=True)
    ap.add_argument("--base", default="ce_openalex/")
    ap.add_argument("--out", default="results/ce_review/")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.base)

    print("Loading data...")
    topics = load_topics(Path(args.topics))
    papers = pd.read_csv(base / "ce_papers_meta.csv")
    auth = pd.read_csv(base / "ce_authorships.csv")
    aff = pd.read_csv(base / "ce_affiliations.csv")
    inst = pd.read_csv(base / "ce_institutions.csv")

    # Build the analytical corpus (papers in topics with metadata)
    topics["doi_norm"] = normalise_doi(topics["doi"])
    papers["doi_norm"] = normalise_doi(papers["doi"])
    paper_subset = papers[["openalex_id", "doi_norm", "year",
                           "citations", "is_oa"]]
    paper_subset = paper_subset.drop_duplicates(subset=["doi_norm"], keep="first")

    full = topics.merge(paper_subset, on="doi_norm", how="inner")
    full = full.rename(columns={"openalex_id": "paper_id"})
    full["citations"] = pd.to_numeric(full["citations"], errors="coerce")
    full["year"] = pd.to_numeric(full["year"], errors="coerce")

    # Determine which papers have a country
    inst_country = inst[["openalex_id", "country"]].rename(
        columns={"openalex_id": "institution_id"}
    )
    inst_country = inst_country.dropna(subset=["country"])
    inst_country = inst_country[inst_country["country"].astype(str).str.len() > 0]

    chain = (auth[["paper_id", "author_id"]]
             .merge(aff, on="author_id", how="inner")
             .merge(inst_country, on="institution_id", how="inner"))
    papers_with_country = set(chain["paper_id"].unique())

    full["has_country"] = full["paper_id"].isin(papers_with_country)

    n_total = len(full)
    n_with = int(full["has_country"].sum())
    n_without = n_total - n_with
    print(f"  Total analytical corpus: {n_total:,}")
    print(f"  With country attribution:    {n_with:,} ({100*n_with/n_total:.1f}%)")
    print(f"  Without country attribution: {n_without:,} ({100*n_without/n_total:.1f}%)")

    if n_without == 0:
        print("\nFull coverage; representativeness test is trivially passed.")
        return 0

    # =====================================================================
    # Test 1: Year of publication (Mann-Whitney U)
    # =====================================================================
    print("\n=== Test 1: Year of publication ===")
    g_with = full.loc[full["has_country"], "year"].dropna()
    g_without = full.loc[~full["has_country"], "year"].dropna()

    stat_year, p_year = stats.mannwhitneyu(g_with, g_without, alternative="two-sided")
    median_with_year = g_with.median()
    median_without_year = g_without.median()
    delta_year = median_with_year - median_without_year

    print(f"  With country median year:    {median_with_year:.1f}")
    print(f"  Without country median year: {median_without_year:.1f}")
    print(f"  Delta:                        {delta_year:+.1f} years")
    print(f"  Mann-Whitney U: stat={stat_year:.0f}, p={p_year:.4g}")

    # =====================================================================
    # Test 2: Citation count (Mann-Whitney U)
    # =====================================================================
    print("\n=== Test 2: Citation count ===")
    c_with = full.loc[full["has_country"], "citations"].dropna()
    c_without = full.loc[~full["has_country"], "citations"].dropna()

    stat_cit, p_cit = stats.mannwhitneyu(c_with, c_without, alternative="two-sided")
    median_with_cit = c_with.median()
    median_without_cit = c_without.median()
    delta_cit = median_with_cit - median_without_cit

    print(f"  With country median citations:    {median_with_cit:.1f}")
    print(f"  Without country median citations: {median_without_cit:.1f}")
    print(f"  Delta:                             {delta_cit:+.1f}")
    print(f"  Mann-Whitney U: stat={stat_cit:.0f}, p={p_cit:.4g}")

    # =====================================================================
    # Test 3: Open Access (chi-square)
    # =====================================================================
    print("\n=== Test 3: Open Access ===")
    full["is_oa_bool"] = full["is_oa"].apply(
        lambda x: True if str(x).lower() in ("true", "1", "yes") else
                  (False if str(x).lower() in ("false", "0", "no") else np.nan)
    )

    contingency = pd.crosstab(full["has_country"], full["is_oa_bool"])
    chi2, p_oa, dof, expected = stats.chi2_contingency(contingency)

    pct_oa_with = 100 * full.loc[full["has_country"], "is_oa_bool"].mean()
    pct_oa_without = 100 * full.loc[~full["has_country"], "is_oa_bool"].mean()
    delta_oa_pct = pct_oa_with - pct_oa_without

    print(f"  With country % OA:    {pct_oa_with:.1f}%")
    print(f"  Without country % OA: {pct_oa_without:.1f}%")
    print(f"  Delta:                 {delta_oa_pct:+.1f} pp")
    print(f"  Chi-square: stat={chi2:.2f}, dof={dof}, p={p_oa:.4g}")

    # =====================================================================
    # Save results
    # =====================================================================
    res = pd.DataFrame([
        {"variable": "Year",
         "median_with": median_with_year, "median_without": median_without_year,
         "delta": delta_year, "test": "Mann-Whitney U",
         "statistic": stat_year, "p_value": p_year},
        {"variable": "Citations",
         "median_with": median_with_cit, "median_without": median_without_cit,
         "delta": delta_cit, "test": "Mann-Whitney U",
         "statistic": stat_cit, "p_value": p_cit},
        {"variable": "Open Access (%)",
         "median_with": pct_oa_with, "median_without": pct_oa_without,
         "delta": delta_oa_pct, "test": "Chi-square",
         "statistic": chi2, "p_value": p_oa},
    ])
    res.to_csv(out / "representativeness_test.csv", index=False)

    # Human-readable interpretation
    summary_path = out / "representativeness_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Representativeness test: subset with country vs subset without\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total analytical corpus:   {n_total:,}\n")
        f.write(f"With country attribution:  {n_with:,} ({100*n_with/n_total:.1f}%)\n")
        f.write(f"Without:                   {n_without:,} "
                f"({100*n_without/n_total:.1f}%)\n\n")

        f.write(f"Test 1 -- Year of publication\n")
        f.write(f"  Median with:    {median_with_year:.1f}\n")
        f.write(f"  Median without: {median_without_year:.1f}\n")
        f.write(f"  Delta:          {delta_year:+.1f} years\n")
        f.write(f"  Mann-Whitney U: U={stat_year:.0f}, p={p_year:.4g}\n")
        if abs(delta_year) < 0.5:
            f.write(f"  Interpretation: negligible difference. PASS.\n\n")
        elif abs(delta_year) < 2:
            f.write(f"  Interpretation: small difference, likely tolerable.\n\n")
        else:
            f.write(f"  Interpretation: meaningful difference; declare in limitations.\n\n")

        f.write(f"Test 2 -- Citation count\n")
        f.write(f"  Median with:    {median_with_cit:.1f}\n")
        f.write(f"  Median without: {median_without_cit:.1f}\n")
        f.write(f"  Delta:          {delta_cit:+.1f}\n")
        f.write(f"  Mann-Whitney U: U={stat_cit:.0f}, p={p_cit:.4g}\n")
        if abs(delta_cit) < 1:
            f.write(f"  Interpretation: negligible difference. PASS.\n\n")
        elif abs(delta_cit) < 5:
            f.write(f"  Interpretation: small difference, likely tolerable.\n\n")
        else:
            f.write(f"  Interpretation: meaningful difference; declare in limitations.\n\n")

        f.write(f"Test 3 -- Open Access\n")
        f.write(f"  With country % OA:    {pct_oa_with:.1f}%\n")
        f.write(f"  Without country % OA: {pct_oa_without:.1f}%\n")
        f.write(f"  Delta:                {delta_oa_pct:+.1f} pp\n")
        f.write(f"  Chi-square: chi2={chi2:.2f}, dof={dof}, p={p_oa:.4g}\n")
        if abs(delta_oa_pct) < 2:
            f.write(f"  Interpretation: negligible difference. PASS.\n\n")
        elif abs(delta_oa_pct) < 5:
            f.write(f"  Interpretation: small difference, likely tolerable.\n\n")
        else:
            f.write(f"  Interpretation: meaningful difference; declare in limitations.\n\n")

        f.write("=" * 70 + "\n")
        f.write("Overall verdict\n")
        f.write("=" * 70 + "\n")
        problems = []
        if abs(delta_year) >= 2:
            problems.append(f"year delta {delta_year:+.1f}y")
        if abs(delta_cit) >= 5:
            problems.append(f"citation delta {delta_cit:+.1f}")
        if abs(delta_oa_pct) >= 5:
            problems.append(f"OA delta {delta_oa_pct:+.1f}pp")

        if not problems:
            f.write("\nThe subset with country attribution is comparable to the\n")
            f.write("subset without on all three observable variables. The geographic\n")
            f.write("analysis can be reported as representative of the analytical\n")
            f.write("corpus.\n")
        else:
            f.write("\nThe following differences should be acknowledged in limitations:\n")
            for p in problems:
                f.write(f"  - {p}\n")
            f.write("\nThis does not invalidate the analysis but readers should be\n")
            f.write("informed of the direction of bias.\n")

    print(f"\nDONE.")
    print(f"  Results: {out / 'representativeness_test.csv'}")
    print(f"  Summary: {summary_path}")
    print(f"\nSummary:")
    with open(summary_path) as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
