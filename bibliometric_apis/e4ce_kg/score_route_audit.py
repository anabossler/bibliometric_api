"""Score the treatment-route labels against a human annotation.

Systems scored on the annotated paper-material pairs:
    gemini, gpt, qwen    verified routes of each extraction model
    primary              two of three models with a verified quote (route_final.csv)
    adjudicated          primary labels with disputed pairs resolved by the
                         adjudicator (route_final_adjudicated__<name>.csv)

Metrics:
    exact        the route set equals the annotator's set (empty set = not_stated)
    eurostat     the same, restricted to the six routes with a Eurostat operation
    per route    precision, recall and Cohen's kappa against the annotator
    weighted     exact agreement post-stratified by material family, with weights
                 equal to each family's share of all paper-material pairs and a
                 stratified bootstrap interval

Usage:
    python score_route_audit.py annotations/audit_route_A.csv [--adjudicator deepseek]
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

from validate_routes import MODELS, ROUTES, load

HERE = Path(__file__).resolve().parent
EUROSTAT = {"recycling", "backfilling", "energy_recovery", "incineration", "landfill",
            "other_disposal"}
CODES = set(ROUTES) | {"not_stated"}
SEED, N_BOOT, Z = 42, 2000, 1.959964


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def parse(cell):
    codes = {c.strip() for c in (cell or "").split(";") if c.strip()}
    unknown = codes - CODES
    if unknown:
        sys.exit(f"unknown codes {sorted(unknown)} in '{cell}'")
    return frozenset(codes - {"not_stated"})


def read_final(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["paper_id"], r["material"]): (parse(r["routes"]), r.get("family", ""))
                for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation")
    ap.add_argument("--adjudicator", default="deepseek")
    args = ap.parse_args()

    with open(args.annotation, newline="", encoding="utf-8-sig") as fh:
        ann = list(csv.DictReader(fh))
    missing = [r["item"] for r in ann if not (r["routes_annotator"] or "").strip()]
    if missing:
        sys.exit(f"items without annotation: {missing}")
    with (HERE / "audit_route_key.csv").open(newline="", encoding="utf-8") as fh:
        family_of = {r["item"]: r["family"] for r in csv.DictReader(fh)}

    papers = {}
    for line in (HERE / "route_input.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        papers[p["paper_id"]] = p
    systems = {m: {k: frozenset(v) for k, v in load(m, papers)[0].items()} for m in MODELS}
    primary = read_final(HERE / "route_final.csv")
    systems["primary"] = {k: v[0] for k, v in primary.items()}
    adj = read_final(HERE / f"route_final_adjudicated__{args.adjudicator}.csv")
    systems["adjudicated"] = {k: v[0] for k, v in adj.items()}

    population = Counter(fam for _, fam in primary.values())
    items = [(r["item"], (r["paper_id"], r["material"]), parse(r["routes_annotator"]),
              family_of[r["item"]]) for r in ann]
    for _, key, _, _ in items:
        for name, labels in systems.items():
            if key not in labels:
                sys.exit(f"{name} has no label for {key}")
    n = len(items)
    fams = sorted({f for *_, f in items})
    weights = {f: population[f] / sum(population[g] for g in fams) for f in fams}
    print(f"annotated pairs: {n} | families: {len(fams)} | human labels: "
          f"{Counter(';'.join(sorted(h)) or 'not_stated' for _, _, h, _ in items).most_common()}")

    def exact(name, key, human, restrict=None):
        got = systems[name][key]
        if restrict:
            return (got & restrict) == (human & restrict)
        return got == human

    rng = random.Random(SEED)
    by_family = defaultdict(list)
    for i, (_, _, _, f) in enumerate(items):
        by_family[f].append(i)

    print("\nsystem | exact (95% Wilson) | eurostat routes (95% Wilson) | "
          "family-weighted exact (95% bootstrap)")
    disagreements = []
    for name in [*MODELS, "primary", "adjudicated"]:
        hits = [exact(name, key, h) for _, key, h, _ in items]
        euro = [exact(name, key, h, EUROSTAT) for _, key, h, _ in items]
        point = sum(weights[f] * sum(hits[i] for i in idx) / len(idx)
                    for f, idx in by_family.items())
        boots = []
        for _ in range(N_BOOT):
            boots.append(sum(weights[f] * sum(hits[rng.choice(idx)] for _ in idx) / len(idx)
                             for f, idx in by_family.items()))
        boots.sort()
        lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
        e_lo, e_hi = wilson(sum(hits), n)
        u_lo, u_hi = wilson(sum(euro), n)
        print(f"{name} | {sum(hits)}/{n} = {100 * sum(hits) / n:.2f}% "
              f"({100 * e_lo:.2f}-{100 * e_hi:.2f}) | {sum(euro)}/{n} = "
              f"{100 * sum(euro) / n:.2f}% ({100 * u_lo:.2f}-{100 * u_hi:.2f}) | "
              f"{100 * point:.2f}% ({100 * lo:.2f}-{100 * hi:.2f})")
        if name in ("primary", "adjudicated"):
            for (item, key, h, f), ok in zip(items, hits):
                if not ok:
                    disagreements.append([name, item, key[0], key[1], f,
                                          ";".join(sorted(h)) or "not_stated",
                                          ";".join(sorted(systems[name][key])) or "not_stated"])

    print("\nroute | human n | system: precision / recall / kappa")
    for route in [*ROUTES, "not_stated"]:
        def has(labels):
            return (not labels) if route == "not_stated" else (route in labels)
        human = [int(has(h)) for _, _, h, _ in items]
        if not any(human) and not any(has(systems[s][k]) for s in systems for _, k, _, _ in items):
            continue
        cells = []
        for name in [*MODELS, "primary", "adjudicated"]:
            sysv = [int(has(systems[name][k])) for _, k, _, _ in items]
            tp = sum(a and b for a, b in zip(human, sysv))
            prec = tp / sum(sysv) if sum(sysv) else float("nan")
            rec = tp / sum(human) if sum(human) else float("nan")
            cells.append(f"{name} {prec:.2f}/{rec:.2f}/{kappa(human, sysv) + 0.0:.2f}")
        print(f"{route} | {sum(human)} | " + " | ".join(cells))

    out = HERE / "audit_route_disagreements.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "item", "paper_id", "material", "family", "human", "system_label"])
        w.writerows(disagreements)
    print(f"\nwrote {out.name}: {len(disagreements)} rows")


if __name__ == "__main__":
    main()
