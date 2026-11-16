"""Thermal-treatment labels: precision evidence from the existing annotations
and robustness of the alignment correlations to the labelling rule.

1. Precision. No sample was drawn at random from all thermal labels. The
   script reports what the existing annotations give: the pairs of the random
   sample whose primary label contains energy_recovery or incineration, and the
   incineration-sampled pairs of the targeted sample. It also combines them in
   a stratified estimate over two strata of the corpus: pairs whose label
   contains incineration (random-sample and incineration-sampled pairs) and
   pairs with energy_recovery but no incineration (random-sample pairs only),
   weighted by the size of each stratum, with a bootstrap interval within
   strata. The random sample is stratified by family, so this estimate is
   approximate.
2. Robustness. The research shares are recomputed with three labelling rules,
   primary (two of three models with a verified quote, as in the paper),
   unanimous (all three models) and adjudicated, and the Spearman correlations
   with the EU shares of family_eu_table.csv are repeated with the seed,
   permutations and bootstrap of analyze_alignment.py.

Usage:
    python thermal_robustness.py annotations/audit_route_A.csv annotations/audit_disposal_A.csv
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

from analyze_alignment import PAIRS, SEED
from build_family_tables import research_flags
from check_topical_relevance import correlation
from score_disposal_audit import THERMAL, wilson

HERE = Path(__file__).resolve().parent
N_BOOT = 2000


def codes(cell):
    return {c.strip() for c in (cell or "").split(";") if c.strip()} - {"not_stated"}


def read(name):
    with (HERE / name).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def annotated(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {r["item"]: codes(r["routes_annotator"]) for r in csv.DictReader(fh)}


def precision(route_ann, disposal_ann, final):
    human_r, key_r = annotated(route_ann), {r["item"]: r for r in read("audit_route_key.csv")}
    human_d, key_d = annotated(disposal_ann), {r["item"]: r for r in read("audit_disposal_key.csv")}
    inc, enr = [], []
    print("== precision of thermal labels in the existing annotations")
    print("random sample, pairs with a thermal primary label:")
    for i, k in sorted(key_r.items(), key=lambda x: int(x[0])):
        if i not in human_r:
            continue
        label = codes(k["final"])
        if label & THERMAL:
            hit = bool(human_r[i] & THERMAL)
            (inc if "incineration" in label else enr).append(hit)
            print(f"  item {i} | {k['family']} | primary {';'.join(sorted(label))} | "
                  f"annotator {';'.join(sorted(human_r[i])) or 'not_stated'} | "
                  f"{'confirmed' if hit else 'not confirmed'}")
    n_r = len(inc) + len(enr)
    k_r = sum(inc) + sum(enr)
    lo, hi = wilson(k_r, n_r)
    print(f"random sample: {k_r}/{n_r} confirmed ({100 * lo:.2f}-{100 * hi:.2f})")
    targeted = [bool(human_d[i] & THERMAL) for i, k in key_d.items()
                if k["route_sampled"] == "incineration"]
    inc += targeted
    print(f"incineration-sampled pairs: {sum(targeted)}/{len(targeted)} confirmed")

    n_inc = sum("incineration" in codes(r["routes"]) for r in final)
    n_enr = sum("energy_recovery" in codes(r["routes"]) and "incineration" not in codes(r["routes"])
                for r in final)
    p_i = sum(inc) / len(inc) if inc else float("nan")
    p_e = sum(enr) / len(enr) if enr else float("nan")
    w_i = n_inc / (n_inc + n_enr)
    est = w_i * p_i + (1 - w_i) * p_e
    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOT):
        bi = [rng.choice(inc) for _ in inc]
        be = [rng.choice(enr) for _ in enr]
        boots.append(w_i * sum(bi) / len(bi) + (1 - w_i) * sum(be) / len(be))
    boots.sort()
    print(f"corpus strata: incineration {n_inc} pairs, energy recovery only {n_enr} pairs")
    print(f"stratum precision: incineration {sum(inc)}/{len(inc)} = {100 * p_i:.2f}% | "
          f"energy recovery only {sum(enr)}/{len(enr)} = {100 * p_e:.2f}%")
    print(f"stratified estimate: {100 * est:.2f}% (bootstrap 95% CI "
          f"{100 * boots[int(0.025 * N_BOOT)]:.2f}-{100 * boots[int(0.975 * N_BOOT) - 1]:.2f})")


def robustness(final):
    family = {(r["paper_id"], r["material"]): r["family"] for r in final}
    rules = {"primary": {}, "unanimous": {}, "adjudicated": {}}
    for r in final:
        key = (r["paper_id"], r["material"])
        rules["primary"][key] = codes(r["routes"])
        votes = dict(v.rsplit(":", 1) for v in r["votes"].split(";") if v)
        rules["unanimous"][key] = {x for x, n in votes.items()
                                   if int(n) == 3 and x != "not_stated"}
    for r in read("route_final_adjudicated__deepseek.csv"):
        rules["adjudicated"][(r["paper_id"], r["material"])] = codes(r["routes"])

    eu = {r["family"]: r for r in read("family_eu_table.csv")}
    fams = sorted(eu)
    print("\n== alignment correlations by labelling rule")
    rng = random.Random(SEED)
    for rule, labels in rules.items():
        routes = defaultdict(set)
        for key, lab in labels.items():
            if key in family:
                routes[(key[0], family[key])] |= lab
        by_fam = defaultdict(list)
        for (pid, fam), lab in routes.items():
            by_fam[fam].append(research_flags(lab))
        n_thermal = sum(bool(lab & THERMAL) for lab in labels.values())
        shares = {f: {k: 100 * sum(x[k] for x in by_fam[f]) / len(by_fam[f])
                      for k in ("recycling", "thermal")} for f in fams}
        if rule == "primary":
            diff = max(abs(shares[f][k] - float(eu[f][f"papers_pct_{k}"]))
                       for f in fams for k in ("recycling", "thermal"))
            print(f"check: largest difference from family_eu_table.csv {diff:.4f} points")
        print(f"{rule}: {n_thermal} pairs with a thermal label")
        for name, eu_key, _ in PAIRS:
            a = [float(eu[f][eu_key]) for f in fams]
            b = [shares[f][name] for f in fams]
            rho, p, lo, hi = correlation(a, b, rng)
            print(f"  {name}: rho {rho:.4f} | permutation p {p:.4f} | bootstrap 95% CI "
                  f"{lo:.4f} to {hi:.4f}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    final = read("route_final.csv")
    precision(sys.argv[1], sys.argv[2], final)
    robustness(final)


if __name__ == "__main__":
    main()
