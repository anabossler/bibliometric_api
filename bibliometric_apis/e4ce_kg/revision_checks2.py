"""Second set of checks of the alignment analysis, from existing data only.

1. Block leave-out: the correlations without the four biological families
   (sludge, manure, food waste, vegetal waste), where the audit found
   energy-recovery labels that the annotator read as recycling.
2. Biogas reclassification: the prompt counts anaerobic digestion as recycling.
   Pairs whose energy-recovery label rests only on quotes about biogas,
   biomethane or anaerobic digestion are moved from energy recovery to
   recycling, and the correlations are repeated.
3. Differential error: thermal labels are given the precision measured in the
   random sample (energy recovery, TP/(TP+FP)) or its lower Wilson bound in the
   families whose EU thermal share is above the median, and precision 1 in the
   others; the expected thermal shares are recomputed.
4. Route composition: research shares among the papers of a family that state
   at least one route with a Eurostat counterpart.
5. Tonnes not recycled with and without backfilling: share of the three
   largest families and their share of papers.
6. Attention by volume: share of papers and share of tonnes treated per
   family, and the Spearman correlation between papers and tonnes treated.

Usage:
    python revision_checks2.py annotations/audit_route_A.csv
"""
import csv
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from analyze_alignment import PAIRS, SEED
from build_family_tables import research, research_flags
from check_topical_relevance import correlation
from revision_checks import EUROSTAT_ROUTES, codes, read
from score_disposal_audit import THERMAL, wilson

HERE = Path(__file__).resolve().parent
BIO = {"sludge", "manure", "food_waste", "vegetal_waste"}
BIOGAS = re.compile(r"biogas|bio-?methane|anaerobic|digest", re.IGNORECASE)


def shares_from(routes, fams):
    flags = defaultdict(list)
    for (pid, fam), labels in routes.items():
        if fam in fams:
            flags[fam].append(research_flags(labels))
    return {f: {k: 100 * sum(x[k] for x in flags[f]) / len(flags[f])
                for k in ("recycling", "thermal")} for f in fams}


def report(label, eu, fams, shares, rng):
    for name, eu_key, _ in PAIRS:
        rho, p, lo, hi = correlation([float(eu[f][eu_key]) for f in fams],
                                     [shares[f][name] for f in fams], rng)
        print(f"{label}, {name}: rho {rho:.4f} | permutation p {p:.4f} | "
              f"bootstrap 95% CI {lo:.4f} to {hi:.4f}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    eu = {r["family"]: r for r in read("family_eu_table.csv")}
    fams = sorted(eu)
    routes = research()
    rng = random.Random(SEED)
    base = shares_from(routes, fams)

    print("== 1. without the biological families", sorted(BIO))
    keep = [f for f in fams if f not in BIO]
    report(f"{len(keep)} families", eu, keep, base, rng)

    print("\n== 2. biogas-only energy recovery moved to recycling")
    quotes = defaultdict(list)
    for r in read("route_labels_long.csv"):
        if r["route"] == "energy_recovery" and r["evidence_verified"] == "True":
            quotes[(r["paper_id"], r["material"])].append(r["evidence"])
    moved, per_fam = set(), defaultdict(int)
    pair_family = {}
    for r in read("route_final.csv"):
        key = (r["paper_id"], r["material"])
        pair_family[key] = r["family"]
        if "energy_recovery" in codes(r["routes"]) and quotes[key] \
                and all(BIOGAS.search(q) for q in quotes[key]):
            moved.add(key)
            per_fam[r["family"]] += 1
    new_routes = defaultdict(set)
    for r in read("route_final.csv"):
        key = (r["paper_id"], r["material"])
        lab = codes(r["routes"])
        if key in moved:
            lab = (lab - {"energy_recovery"}) | {"recycling"}
        new_routes[(r["paper_id"], r["family"])] |= lab
    print(f"pairs moved: {len(moved)} | by family: {dict(sorted(per_fam.items(), key=lambda x: -x[1]))}")
    report("22 families", eu, fams, shares_from(new_routes, fams), rng)

    print("\n== 3. differential error in the thermal labels")
    human = {}
    with open(sys.argv[1], newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            human[r["item"]] = codes(r["routes_annotator"])
    key = {r["item"]: r for r in read("audit_route_key.csv") if r["item"] in human}
    tp = sum(bool(codes(k["final"]) & THERMAL) and bool(human[i] & THERMAL) for i, k in key.items())
    fp = sum(bool(codes(k["final"]) & THERMAL) and not human[i] & THERMAL for i, k in key.items())
    lo, _ = wilson(tp, tp + fp)
    median = statistics.median(float(eu[f]["eu_pct_thermal"]) for f in fams)
    high = [f for f in fams if float(eu[f]["eu_pct_thermal"]) > median]
    print(f"families above the median EU thermal share ({median:.2f}%): {high}")
    for label, prec in (("point estimate", tp / (tp + fp)), ("lower Wilson bound", lo)):
        adj = {f: {"recycling": base[f]["recycling"],
                   "thermal": base[f]["thermal"] * (prec if f in high else 1.0)} for f in fams}
        rho, p, clo, chi = correlation([float(eu[f]["eu_pct_thermal"]) for f in fams],
                                       [adj[f]["thermal"] for f in fams], rng)
        print(f"precision {prec:.4f} ({label}) in those families: thermal rho {rho:.4f} | "
              f"permutation p {p:.4f} | bootstrap 95% CI {clo:.4f} to {chi:.4f}")

    print("\n== 4. route composition among papers with a Eurostat route")
    cond = {(pid, fam): lab for (pid, fam), lab in routes.items() if lab & EUROSTAT_ROUTES}
    cshare = shares_from(cond, fams)
    for f in sorted(fams, key=lambda x: cshare[x]["recycling"]):
        n = sum(1 for (_, fam) in cond if fam == f)
        print(f"  {f} | {n} papers with a Eurostat route | recycling {cshare[f]['recycling']:.2f}% | "
              f"thermal {cshare[f]['thermal']:.2f}%")
    report("conditional shares", eu, fams, cshare, rng)

    print("\n== 5. tonnes not recycled, with and without backfilling")
    papers = {f: sum(1 for (_, fam) in routes if fam == f) for f in fams}
    total_papers = len({pid for (pid, fam) in routes if fam in eu})
    for label, excl in (("not recycled", ("recycling",)),
                        ("neither recycled nor backfilled", ("recycling", "backfilling"))):
        t = {f: float(eu[f]["tonnes_treated"]) * (1 - sum(float(eu[f][f"eu_pct_{k}"]) for k in excl) / 100)
             for f in fams}
        top = sorted(fams, key=lambda x: -t[x])[:3]
        share_t = 100 * sum(t[f] for f in top) / sum(t.values())
        top_papers = len({pid for (pid, fam) in routes if fam in top})
        print(f"{label}: total {sum(t.values()) / 1e6:.2f} Mt | top 3 {top} hold {share_t:.2f}% "
              f"and {top_papers} of {total_papers} papers ({100 * top_papers / total_papers:.2f}%)")

    print("\n== 6. attention by volume")
    tonnes = {f: float(eu[f]["tonnes_treated"]) for f in fams}
    tt, tp_ = sum(tonnes.values()), sum(papers.values())
    for f in sorted(fams, key=lambda x: -papers[x] / tp_ + tonnes[x] / tt):
        print(f"  {f} | {100 * papers[f] / tp_:.2f}% of paper-family pairs | "
              f"{100 * tonnes[f] / tt:.2f}% of tonnes treated")
    rho, p, clo, chi = correlation([tonnes[f] for f in fams], [papers[f] for f in fams], rng)
    print(f"papers vs tonnes treated: rho {rho:.4f} | permutation p {p:.4f} | "
          f"bootstrap 95% CI {clo:.4f} to {chi:.4f}")


if __name__ == "__main__":
    main()
