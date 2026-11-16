"""Audit the precision of the pooled thermal-treatment labels.

The disposal sample selects pairs by route, so its incineration pairs do not
estimate the precision of thermal treatment as the alignment analysis pools it
(energy recovery or incineration). This audit draws N pairs at random, with a
fixed seed, from all pairs whose primary label contains energy_recovery or
incineration, excluding the pairs of the route and disposal samples. A pair is
confirmed when the annotator assigns either route.

Commands:
    make    writes audit_thermal_blind.csv (same columns and guidelines as
            audit_route_blind.csv) and audit_thermal_key.csv
    score   scores an annotated copy of the blind sheet against the key

Usage:
    python thermal_audit.py make
    python thermal_audit.py score annotations/audit_thermal_A.csv
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from score_disposal_audit import THERMAL, parse, wilson

HERE = Path(__file__).resolve().parent
SEED, N = 42, 40
BLIND, KEY = HERE / "audit_thermal_blind.csv", HERE / "audit_thermal_key.csv"


def make():
    if BLIND.exists() or KEY.exists():
        sys.exit(f"{BLIND.name} or {KEY.name} exists; the sample is drawn once")
    papers = {}
    for line in (HERE / "route_input.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        papers[p["paper_id"]] = p
    seen = set()
    for name in ("audit_route_key.csv", "audit_disposal_key.csv"):
        with (HERE / name).open(newline="", encoding="utf-8") as fh:
            seen |= {(r["paper_id"], r["material"]) for r in csv.DictReader(fh)}
    with (HERE / "route_final_adjudicated__deepseek.csv").open(newline="", encoding="utf-8") as fh:
        adjudicated = {(r["paper_id"], r["material"]): r["routes"] for r in csv.DictReader(fh)}
    with (HERE / "route_final.csv").open(newline="", encoding="utf-8") as fh:
        pool = sorted((r for r in csv.DictReader(fh)
                       if THERMAL & set(r["routes"].split(";"))
                       and (r["paper_id"], r["material"]) not in seen),
                      key=lambda r: (r["paper_id"], r["material"]))
    sample = random.Random(SEED).sample(pool, min(N, len(pool)))
    with BLIND.open("w", newline="", encoding="utf-8") as fb, \
            KEY.open("w", newline="", encoding="utf-8") as fk:
        wb, wk = csv.writer(fb), csv.writer(fk)
        wb.writerow(["item", "paper_id", "title", "abstract", "material",
                     "routes_annotator", "notes"])
        wk.writerow(["item", "paper_id", "material", "family", "primary", "adjudicated"])
        for i, r in enumerate(sample, 1):
            p = papers[r["paper_id"]]
            wb.writerow([i, r["paper_id"], p["title"], p["abstract"], r["material"], "", ""])
            wk.writerow([i, r["paper_id"], r["material"], r["family"], r["routes"],
                         adjudicated.get((r["paper_id"], r["material"]), "")])
    print(f"{len(pool)} pairs with a thermal label outside the other samples; "
          f"{len(sample)} sampled | wrote {BLIND.name} and {KEY.name}")


def score(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        human = {r["item"]: parse(r["routes_annotator"]) for r in csv.DictReader(fh)}
    with KEY.open(newline="", encoding="utf-8") as fh:
        key = {r["item"]: r for r in csv.DictReader(fh)}
    if set(human) != set(key):
        sys.exit("items in the annotation and in the key differ")

    for name in ("primary", "adjudicated"):
        items = [i for i, k in key.items() if THERMAL & parse(k[name] or "not_stated")]
        hits = sum(bool(human[i] & THERMAL) for i in items)
        lo, hi = wilson(hits, len(items))
        share = 100 * hits / len(items) if items else float("nan")
        print(f"thermal precision, {name} labels | {hits}/{len(items)} = {share:.2f}% "
              f"(95% Wilson {100 * lo:.2f}-{100 * hi:.2f})")
    for route in sorted(THERMAL):
        items = [i for i, k in key.items() if route in parse(k["primary"])]
        hits = sum(route in human[i] for i in items)
        print(f"  {route} exact, primary | {hits}/{len(items)}")
    missed = Counter(";".join(sorted(human[i])) or "not_stated"
                     for i in key if not human[i] & THERMAL)
    print(f"annotator label when thermal not confirmed: {dict(missed.most_common())}")
    fams = Counter(key[i]["family"] for i in key)
    print(f"families in the sample: {dict(fams.most_common())}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "make":
        make()
    elif len(sys.argv) == 3 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        sys.exit(__doc__)
