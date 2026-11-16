"""Additional checks of the alignment analysis, from existing data only.

1. Leave one family out: the correlations recomputed on 21 families, each
   family removed in turn.
2. Minimum detectable correlation with the number of families (two-sided
   alpha 0.05, power 0.80), from the Fisher z transform with the
   Fieller-Hartley-Pearson variance 1.06/(n-3) for Spearman's rho.
3. EU shares averaged over the panel years from 2018 instead of 2022 alone.
4. Effort per tonne: recycling papers per Mt not recycled, and the Spearman
   correlation between the number of recycling papers and the tonnes not
   recycled.
5. Count model: quasi-binomial regression of the number of papers of a family
   that study the route on the EU share of the route, with the dispersion
   estimated from the Pearson statistic.
6. Propagation of label error: per route class, the positive predictive value
   and the false omission rate are drawn from Beta posteriors (Beta(1,1) prior)
   of the random audit sample; every pair keeps or gains the label accordingly,
   the research shares are recomputed and the correlation is repeated over
   N_SIM draws. The audit sample has four pairs per family, so the rates are
   unweighted.
7. Agreement per family in the random audit sample, on the routes with a
   Eurostat counterpart (four pairs per family).

Usage:
    python revision_checks.py annotations/audit_route_A.csv
"""
import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from analyze_alignment import PAIRS, SEED, spearman
from build_family_tables import (PANEL_YEARS, load_ops, research, research_flags,
                                 treatment)
from check_topical_relevance import correlation
from score_disposal_audit import THERMAL

HERE = Path(__file__).resolve().parent
N_SIM = 1000
N_PERM_LOFO = 2000
EUROSTAT_ROUTES = {"recycling", "backfilling", "energy_recovery", "incineration",
                   "landfill", "other_disposal"}
CLASSES = {"recycling": {"recycling"}, "thermal": THERMAL}


def codes(cell):
    return {c.strip() for c in (cell or "").split(";") if c.strip()} - {"not_stated"}


def read(name):
    with (HERE / name).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def perm_p(a, b, rng, n_perm):
    rho = spearman(a, b)
    hits = sum(abs(spearman(a, rng.sample(b, len(b)))) >= abs(rho) - 1e-12
               for _ in range(n_perm))
    return rho, (hits + 1) / (n_perm + 1)


def quasi_binomial(y, n, x):
    """Logistic regression of y successes out of n on x by IRLS, quasi-binomial SE."""
    p0 = sum(y) / sum(n)
    a, b = math.log(p0 / (1 - p0)), 0.0
    for _ in range(100):
        s00 = s01 = s11 = r0 = r1 = 0.0
        for yi, ni, xi in zip(y, n, x):
            mu = 1 / (1 + math.exp(-(a + b * xi)))
            w = ni * mu * (1 - mu)
            z = a + b * xi + (yi / ni - mu) / (mu * (1 - mu))
            s00 += w
            s01 += w * xi
            s11 += w * xi * xi
            r0 += w * z
            r1 += w * xi * z
        det = s00 * s11 - s01 * s01
        na, nb = (s11 * r0 - s01 * r1) / det, (s00 * r1 - s01 * r0) / det
        done = abs(na - a) < 1e-10 and abs(nb - b) < 1e-10
        a, b = na, nb
        if done:
            break
    pearson = sum((yi - ni * m) ** 2 / (ni * m * (1 - m))
                  for yi, ni, xi in zip(y, n, x)
                  for m in [1 / (1 + math.exp(-(a + b * xi)))])
    phi = pearson / (len(y) - 2)
    se_b = math.sqrt(phi * s00 / det)
    p = 2 * (1 - statistics.NormalDist().cdf(abs(b / se_b)))
    return b, se_b, p, phi


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    eu = {r["family"]: r for r in read("family_eu_table.csv")}
    fams = sorted(eu)
    routes = research()
    flags = defaultdict(list)
    for (pid, fam), labels in routes.items():
        if fam in eu:
            flags[fam].append(research_flags(labels))
    count = {f: {k: sum(x[k] for x in flags[f]) for k in ("recycling", "thermal")} for f in fams}
    size = {f: len(flags[f]) for f in fams}
    share = {f: {k: 100 * count[f][k] / size[f] for k in count[f]} for f in fams}
    rng = random.Random(SEED)

    print("== 1. leave one family out")
    for name, eu_key, _ in PAIRS:
        runs = []
        for drop in fams:
            keep = [f for f in fams if f != drop]
            rho, p = perm_p([float(eu[f][eu_key]) for f in keep],
                            [share[f][name] for f in keep], rng, N_PERM_LOFO)
            runs.append((rho, p, drop))
        lo, hi = min(runs), max(runs)
        sig = sum(p < 0.05 for _, p, _ in runs)
        print(f"{name}: rho from {lo[0]:.4f} (without {lo[2]}) to {hi[0]:.4f} (without {hi[2]}) | "
              f"p < 0.05 in {sig} of {len(runs)} runs")

    n = len(fams)
    nd = statistics.NormalDist()
    z = (nd.inv_cdf(0.975) + nd.inv_cdf(0.80)) * math.sqrt(1.06 / (n - 3))
    print(f"\n== 2. minimum detectable |rho| with {n} families, alpha 0.05, power 0.80: "
          f"{math.tanh(z):.4f}")

    ops = load_ops()
    countries = sorted({c for (_, c, _, _) in ops})
    years = sorted({y for (_, _, y, _) in ops if y >= min(PANEL_YEARS)})
    print(f"\n== 3. EU shares averaged over {years}")
    multi = {}
    for f in fams:
        total, g = treatment(ops, f, countries, years)
        multi[f] = {k: 100 * g[k] / total for k in ("recycling", "thermal")}
    for name, eu_key, _ in PAIRS:
        rho, p, lo, hi = correlation([multi[f][name] for f in fams],
                                     [share[f][name] for f in fams], rng)
        moved = max(abs(multi[f][name] - float(eu[f][eu_key])) for f in fams)
        print(f"{name}: rho {rho:.4f} | permutation p {p:.4f} | bootstrap 95% CI {lo:.4f} to "
              f"{hi:.4f} | largest change of an EU share {moved:.2f} points")

    print("\n== 4. recycling papers per Mt not recycled")
    rows = []
    for f in fams:
        mt = float(eu[f]["tonnes_not_recycled"]) / 1e6
        rows.append((count[f]["recycling"] / mt if mt > 0 else float("inf"), f, mt))
    for ratio, f, mt in sorted(rows):
        print(f"  {f} | {count[f]['recycling']} recycling papers | {mt:.4f} Mt not recycled | "
              f"{ratio:.2f} papers per Mt")
    rho, p, lo, hi = correlation([float(eu[f]["tonnes_not_recycled"]) for f in fams],
                                 [count[f]["recycling"] for f in fams], rng)
    print(f"recycling papers vs tonnes not recycled: rho {rho:.4f} | permutation p {p:.4f} | "
          f"bootstrap 95% CI {lo:.4f} to {hi:.4f}")

    print("\n== 5. quasi-binomial count model (papers studying the route ~ EU share)")
    for name, eu_key, _ in PAIRS:
        b, se, p, phi = quasi_binomial([count[f][name] for f in fams], [size[f] for f in fams],
                                       [float(eu[f][eu_key]) / 100 for f in fams])
        print(f"{name}: slope per unit share {b:.4f} (SE {se:.4f}) | p {p:.4f} | "
              f"dispersion {phi:.2f}")

    print(f"\n== 6. propagation of label error ({N_SIM} draws)")
    human = {}
    with open(sys.argv[1], newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            human[r["item"]] = codes(r["routes_annotator"])
    key = {r["item"]: r for r in read("audit_route_key.csv") if r["item"] in human}
    final = read("route_final.csv")
    pairs = [(r["paper_id"], r["family"], codes(r["routes"])) for r in final if r["family"] in eu]
    for name, eu_key, _ in PAIRS:
        cls = CLASSES[name]
        tp = sum(bool(codes(k["final"]) & cls) and bool(human[i] & cls) for i, k in key.items())
        fp = sum(bool(codes(k["final"]) & cls) and not human[i] & cls for i, k in key.items())
        fn = sum(not codes(k["final"]) & cls and bool(human[i] & cls) for i, k in key.items())
        tn = len(key) - tp - fp - fn
        a = [float(eu[f][eu_key]) for f in fams]
        sims = []
        for _ in range(N_SIM):
            ppv = rng.betavariate(tp + 1, fp + 1)
            fom = rng.betavariate(fn + 1, tn + 1)
            has = {}
            for pid, fam, lab in pairs:
                hit = rng.random() < (ppv if lab & cls else fom)
                has[(pid, fam)] = has.get((pid, fam), False) or hit
            by_f = defaultdict(list)
            for (pid, fam), v in has.items():
                by_f[fam].append(v)
            sims.append(spearman(a, [100 * sum(by_f[f]) / len(by_f[f]) for f in fams]))
        sims.sort()
        print(f"{name}: audit TP {tp} FP {fp} FN {fn} TN {tn} | rho median "
              f"{statistics.median(sims):.4f} | 95% interval {sims[int(0.025 * N_SIM)]:.4f} to "
              f"{sims[int(0.975 * N_SIM) - 1]:.4f} | share of draws with rho > 0 "
              f"{100 * sum(s > 0 for s in sims) / N_SIM:.2f}%")

    print("\n== 7. agreement per family in the random audit sample (Eurostat routes)")
    per = defaultdict(lambda: [0, 0])
    for i, k in key.items():
        per[k["family"]][1] += 1
        per[k["family"]][0] += (codes(k["final"]) & EUROSTAT_ROUTES) == (human[i] & EUROSTAT_ROUTES)
    for f in sorted(per, key=lambda x: (per[x][0] / per[x][1], x)):
        print(f"  {f} | {per[f][0]}/{per[f][1]}")


if __name__ == "__main__":
    main()
