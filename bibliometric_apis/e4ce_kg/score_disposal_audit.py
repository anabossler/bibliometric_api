"""Score the targeted disposal-route sample against a human annotation.

For each disposal route (landfill, incineration, backfilling, other_disposal)
the sample holds up to ten pairs whose primary label contains that route. The
precision of the route in the primary labels is the share of those pairs in
which the annotator also assigns the route, reported with a Wilson interval.
For the adjudicated labels the same share is computed on the sampled pairs
whose adjudicated label contains the route. The script also lists what the
annotator assigned when the route was not confirmed. Because the annotator
separates energy recovery from incineration, the incineration-sampled pairs
are also scored for thermal treatment (either route).

Usage:
    python score_disposal_audit.py annotations/audit_disposal_A.csv
"""
import csv
import sys
from collections import Counter
from math import sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTES = ["landfill", "incineration", "backfilling", "other_disposal"]
CODES = {"recycling", "backfilling", "energy_recovery", "incineration", "landfill",
         "other_disposal", "prevention_reuse", "not_stated"}
THERMAL = {"energy_recovery", "incineration"}
Z = 1.959964


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def parse(cell):
    codes = {c.strip() for c in (cell or "").split(";") if c.strip()}
    if not codes or codes - CODES:
        sys.exit(f"invalid or empty label '{cell}'")
    return frozenset(codes - {"not_stated"})


def main():
    with open(sys.argv[1], newline="", encoding="utf-8-sig") as fh:
        human = {r["item"]: parse(r["routes_annotator"]) for r in csv.DictReader(fh)}
    with (HERE / "audit_disposal_key.csv").open(newline="", encoding="utf-8") as fh:
        key = {r["item"]: r for r in csv.DictReader(fh)}
    if set(human) != set(key):
        sys.exit("items in the annotation and in the key differ")

    print("route | primary: confirmed / sampled (95% Wilson) | "
          "adjudicated: confirmed / labelled | annotator label when not confirmed")
    for route in ROUTES:
        items = [i for i, k in key.items() if k["route_sampled"] == route]
        hits = sum(route in human[i] for i in items)
        lo, hi = wilson(hits, len(items))
        adj = [i for i in items if route in parse(key[i]["adjudicated"] or "not_stated")]
        adj_hits = sum(route in human[i] for i in adj)
        missed = Counter(";".join(sorted(human[i])) or "not_stated"
                         for i in items if route not in human[i])
        share = 100 * hits / len(items) if items else float("nan")
        print(f"{route} | {hits}/{len(items)} = {share:.2f}% ({100 * lo:.2f}-{100 * hi:.2f}) | "
              f"{adj_hits}/{len(adj)} | {dict(missed.most_common())}")

    items = [i for i, k in key.items() if k["route_sampled"] == "incineration"]
    hits = sum(bool(human[i] & THERMAL) for i in items)
    lo, hi = wilson(hits, len(items))
    share = 100 * hits / len(items) if items else float("nan")
    print(f"thermal (incineration-sampled, energy_recovery or incineration) | "
          f"{hits}/{len(items)} = {share:.2f}% ({100 * lo:.2f}-{100 * hi:.2f})")

    for name in ("primary", "adjudicated"):
        exact = sum(parse(key[i][name] or "not_stated") == human[i] for i in key)
        print(f"{name}: identical route set in {exact}/{len(key)} sampled pairs")


if __name__ == "__main__":
    main()
