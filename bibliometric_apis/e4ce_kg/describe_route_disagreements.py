"""List the audited pairs that the annotator left without a route but the
aggregated (two-of-three) labels did not, with the verified quotes behind each
assigned route and the annotator's note.

Reads audit_route_disagreements.csv (from score_route_audit.py) and
route_labels_long.csv (from validate_routes.py). Writes
audit_route_unstated_quotes.csv and prints the same content.

Usage:
    python describe_route_disagreements.py annotations/audit_route_A.csv
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM = "primary"


def main():
    with open(sys.argv[1], newline="", encoding="utf-8-sig") as fh:
        ann = {r["item"]: r for r in csv.DictReader(fh)}
    with (HERE / "audit_route_disagreements.csv").open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["system"] == SYSTEM and r["human"] == "not_stated"]
    keys = {(r["paper_id"], r["material"]) for r in rows}

    quotes = defaultdict(list)
    with (HERE / "route_labels_long.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["paper_id"], r["material"])
            if key in keys and r["evidence_verified"] == "True":
                quotes[key, r["route"]].append((r["model"], r["evidence"]))

    print(f"{len(rows)} pairs: annotator not_stated, {SYSTEM} labels with a route")
    print(f"{SYSTEM} label: {dict(Counter(r['system_label'] for r in rows).most_common())}")
    out_rows = []
    for r in sorted(rows, key=lambda x: int(x["item"])):
        a = ann[r["item"]]
        print(f"\nitem {r['item']} | {r['family']} | {r['material']} | {SYSTEM}: "
              f"{r['system_label']} | annotator: {a.get('original_labels', '')} | "
              f"note: {a.get('notes', '')}")
        for route in r["system_label"].split(";"):
            for model, text in quotes[(r["paper_id"], r["material"]), route]:
                print(f"  {route} | {model}: {text}")
                out_rows.append([r["item"], r["family"], r["material"], route, model, text,
                                 a.get("notes", "")])
    out = HERE / "audit_route_unstated_quotes.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "family", "material", "route", "model", "evidence", "annotator_note"])
        w.writerows(out_rows)
    print(f"\nwrote {out.name}: {len(out_rows)} quotes")


if __name__ == "__main__":
    main()
