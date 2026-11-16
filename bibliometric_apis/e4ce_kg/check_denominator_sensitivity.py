"""Recompute the EU-27 treatment shares on complete Eurostat cells only and
repeat the alignment correlations.

A country-category cell is complete when it reports the total treated (TRT)
and the six operations sum to that total within TOL (the "consistent" cells of
check_eurostat_denominator.py). Cells with operations but no total, or whose
operations fall short of the total because part of the treatment is reported
only as RCV_R_B or DSP_L_OTH or is not allocated, are dropped from numerator
and denominator. The research shares are those of family_eu_table.csv; the
correlations use the seed, permutations and bootstrap of analyze_alignment.py.
As a consistency check, the shares over all cells are recomputed and compared
with family_eu_table.csv.

Reads the password from the environment variable NEO4J_PW.

Usage:
    python check_denominator_sensitivity.py
"""
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from analyze_alignment import PAIRS, SEED
from build_family_tables import EU27, GROUPS, OPS, REF_YEAR
from check_eurostat_denominator import AGG, QUERY, TOL
from check_topical_relevance import correlation

HERE = Path(__file__).resolve().parent


def shares(t):
    total = sum(t[op] for op in OPS)
    if not total:
        return None
    return {g: 100 * sum(t[o] for o in ops) / total for g, ops in GROUPS.items()}


def main():
    from neo4j import GraphDatabase
    pw = os.environ.get("NEO4J_PW")
    if not pw:
        sys.exit("set NEO4J_PW")
    with GraphDatabase.driver("bolt://localhost:7691", auth=("neo4j", pw)) as drv, \
            drv.session(database="circulareconomy") as s:
        rows = list(s.run(QUERY, year=REF_YEAR, ops=["TRT"] + OPS + AGG))
    cells = defaultdict(dict)
    for r in rows:
        if r["country"] in EU27:
            cells[(r["family"], r["country"], r["category"])][r["op"]] = float(r["tonnes"] or 0)

    full = defaultdict(lambda: defaultdict(float))
    kept = defaultdict(lambda: defaultdict(float))
    size, dropped = defaultdict(float), defaultdict(float)
    for (f, _, _), v in cells.items():
        six = sum(v.get(op, 0.0) for op in OPS)
        trt = v.get("TRT")
        for op in OPS:
            full[f][op] += v.get(op, 0.0)
        size[f] += max(six, trt or 0.0)
        complete = (trt is not None and trt > 0 and any(op in v for op in OPS)
                    and abs(six - trt) <= TOL * trt)
        if complete:
            for op in OPS:
                kept[f][op] += v.get(op, 0.0)
        else:
            dropped[f] += max(six, trt or 0.0)

    with (HERE / "family_eu_table.csv").open(newline="", encoding="utf-8") as fh:
        eu = {r["family"]: r for r in csv.DictReader(fh)}
    fams = sorted(f for f in eu if shares(kept[f]))
    print(f"year {REF_YEAR} | EU-27 | tolerance {100 * TOL:.0f}% | {len(fams)} families")
    print("family | recycling: table / all cells / complete | thermal: table / all cells / "
          "complete | tonnes dropped (%)")
    worst = 0.0
    for f in sorted(fams, key=lambda x: -size[x]):
        a, k = shares(full[f]), shares(kept[f])
        worst = max(worst, abs(a["recycling"] - float(eu[f]["eu_pct_recycling"])),
                    abs(a["thermal"] - float(eu[f]["eu_pct_thermal"])))
        print(f"{f} | {float(eu[f]['eu_pct_recycling']):.2f} / {a['recycling']:.2f} / "
              f"{k['recycling']:.2f} | {float(eu[f]['eu_pct_thermal']):.2f} / "
              f"{a['thermal']:.2f} / {k['thermal']:.2f} | "
              f"{dropped[f]:.0f} ({100 * dropped[f] / size[f]:.2f})")
    print(f"largest difference between table and recomputed all-cell shares: {worst:.4f} points")

    rng = random.Random(SEED)
    for label, source in (("all cells", full), ("complete cells", kept)):
        for name, _, res_key in PAIRS:
            a = [shares(source[f])[name] for f in fams]
            b = [float(eu[f][res_key]) for f in fams]
            rho, p, lo, hi = correlation(a, b, rng)
            print(f"{label}, {name}: rho {rho:.4f} | permutation p {p:.4f} | "
                  f"bootstrap 95% CI {lo:.4f} to {hi:.4f}")


if __name__ == "__main__":
    main()
