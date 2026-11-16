"""Repeat the alignment analysis on papers with at least one author in the EU-27.

Eurostat treatment refers to the EU-27, while the corpus has authors worldwide.
This check keeps the papers with at least one authorship affiliated in an EU-27
country (paper_countries.csv, from fetch_paper_countries.py), recomputes the
research shares of each family with Eurostat data, and reports the Spearman
correlations with the same seed, permutations and bootstrap as
analyze_alignment.py. Families with fewer than MIN_PAPERS EU papers are listed.

Usage:
    python check_eu_authors.py
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

from analyze_alignment import PAIRS, SEED
from build_family_tables import EU27, research, research_flags
from check_topical_relevance import FLAGS, correlation

HERE = Path(__file__).resolve().parent
MIN_PAPERS = 20


def main():
    eu_papers = set()
    with (HERE / "paper_countries.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["country_code"] in EU27:
                eu_papers.add(r["paper_id"])
    routes = research()
    all_papers = {pid for pid, _ in routes}
    print(f"papers {len(all_papers)} | with an EU-27 author {len(all_papers & eu_papers)} "
          f"({100 * len(all_papers & eu_papers) / len(all_papers):.2f}%)")

    with (HERE / "family_eu_table.csv").open(newline="", encoding="utf-8") as fh:
        eu = {r["family"]: r for r in csv.DictReader(fh)}
    by_family = defaultdict(list)
    for (pid, fam), labels in routes.items():
        if fam in eu and pid in eu_papers:
            by_family[fam].append(labels)

    print("\nfamily | all papers | EU papers | EU: % recycling / thermal / landfill (upper) / "
          "no Eurostat route")
    shares = {}
    for fam in sorted(eu, key=lambda f: -int(float(eu[f]["papers"]))):
        flags = [research_flags(l) for l in by_family[fam]]
        n = len(flags)
        pct = {k: 100 * sum(x[k] for x in flags) / n if n else float("nan") for k in FLAGS}
        shares[fam] = pct
        mark = "" if n >= MIN_PAPERS else f"  (fewer than {MIN_PAPERS})"
        print(f"{fam} | {int(float(eu[fam]['papers']))} | {n} | "
              + " / ".join(f"{pct[k]:.2f}" for k in FLAGS) + mark)

    rng = random.Random(SEED)
    fams = sorted(f for f in shares if by_family[f])
    print(f"\nEU-27 authors, {len(fams)} families with Eurostat data and EU papers")
    for name, eu_key, res_key in PAIRS:
        a = [float(eu[f][eu_key]) for f in fams]
        b = [shares[f][res_key.replace("papers_pct_", "")] for f in fams]
        rho, p, lo, hi = correlation(a, b, rng)
        print(f"{name}: rho {rho:.4f} | permutation p {p:.4f} | bootstrap 95% CI "
              f"{lo:.4f} to {hi:.4f}")


if __name__ == "__main__":
    main()
