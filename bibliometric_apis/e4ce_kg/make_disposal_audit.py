"""Draw a blind sample to validate the disposal routes of the primary labels.

The first audit sample contained no pair that the annotator labelled with
landfill, incineration, backfilling or other_disposal, so the precision of
those labels could not be measured. This sample takes, for each of those
routes, up to PER_ROUTE paper-material pairs whose primary label contains it,
excluding pairs already in the first sample. A pair is drawn for at most one
route.

Outputs:
    audit_disposal_blind.csv   sheet to annotate (same columns and guidelines
                               as audit_route_blind.csv)
    audit_disposal_key.csv     sampled route, family and system labels

Usage:
    python make_disposal_audit.py
"""
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTES = ["landfill", "incineration", "backfilling", "other_disposal"]
SEED, PER_ROUTE = 42, 10


def main():
    papers = {}
    for line in (HERE / "route_input.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        papers[p["paper_id"]] = p
    with (HERE / "audit_route_key.csv").open(newline="", encoding="utf-8") as fh:
        seen = {(r["paper_id"], r["material"]) for r in csv.DictReader(fh)}
    with (HERE / "route_final.csv").open(newline="", encoding="utf-8") as fh:
        final = [r for r in csv.DictReader(fh) if (r["paper_id"], r["material"]) not in seen]
    adj_path = HERE / "route_final_adjudicated__deepseek.csv"
    with adj_path.open(newline="", encoding="utf-8") as fh:
        adjudicated = {(r["paper_id"], r["material"]): r["routes"] for r in csv.DictReader(fh)}

    rng = random.Random(SEED)
    taken, sample = set(), []
    for route in ROUTES:
        pool = sorted((r for r in final if route in r["routes"].split(";")),
                      key=lambda r: (r["paper_id"], r["material"]))
        rng.shuffle(pool)
        picked = [r for r in pool if (r["paper_id"], r["material"]) not in taken][:PER_ROUTE]
        taken |= {(r["paper_id"], r["material"]) for r in picked}
        sample += [(route, r) for r in picked]
        print(f"{route}: {len(pool)} pairs with this route, {len(picked)} sampled")
    rng.shuffle(sample)

    with (HERE / "audit_disposal_blind.csv").open("w", newline="", encoding="utf-8") as fb, \
            (HERE / "audit_disposal_key.csv").open("w", newline="", encoding="utf-8") as fk:
        wb, wk = csv.writer(fb), csv.writer(fk)
        wb.writerow(["item", "paper_id", "title", "abstract", "material",
                     "routes_annotator", "notes"])
        wk.writerow(["item", "paper_id", "material", "family", "route_sampled",
                     "primary", "adjudicated"])
        for i, (route, r) in enumerate(sample, 1):
            p = papers[r["paper_id"]]
            wb.writerow([i, r["paper_id"], p["title"], p["abstract"], r["material"], "", ""])
            wk.writerow([i, r["paper_id"], r["material"], r["family"], route, r["routes"],
                         adjudicated.get((r["paper_id"], r["material"]), "")])
    print(f"wrote audit_disposal_blind.csv and audit_disposal_key.csv: {len(sample)} pairs")


if __name__ == "__main__":
    main()
