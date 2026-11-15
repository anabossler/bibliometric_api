"""Convert a route annotation written with free labels into the closed codes.

The label-to-code table is fixed in a CSV approved by the annotator. Labels
mapped to DECIDE are not converted: the row is listed for the annotator, who
writes the codes by hand in the column decision. A row is converted only when
all its labels are mapped; not_stated is dropped when the row has any other
code.

Usage:
    python convert_route_annotation.py RAW_SHEET LABEL_MAP OUT_SHEET
"""
import csv
import sys

CODES = {"recycling", "backfilling", "energy_recovery", "incineration", "landfill",
         "other_disposal", "prevention_reuse", "not_stated"}


def main():
    raw, label_map, out = sys.argv[1:4]
    with open(label_map, newline="", encoding="utf-8-sig") as fh:
        mapping = {r["label"]: r["code"] for r in csv.DictReader(fh)}
    bad = {c for c in mapping.values() if c not in CODES | {"DECIDE"}}
    if bad:
        sys.exit(f"invalid codes in the label map: {sorted(bad)}")
    with open(raw, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    unknown = sorted({l.strip() for r in rows for l in r["routes_annotator"].split(";")
                      if l.strip() and l.strip() not in mapping})
    if unknown:
        sys.exit(f"labels missing from the label map: {unknown}")

    pending = 0
    for r in rows:
        labels = [l.strip() for l in r["routes_annotator"].split(";") if l.strip()]
        codes = {mapping[l] for l in labels}
        r["original_labels"] = "; ".join(labels)
        if "DECIDE" in codes:
            r["routes_annotator"], r["decision"] = "", ""
            pending += 1
        else:
            if len(codes) > 1:
                codes.discard("not_stated")
            r["routes_annotator"], r["decision"] = ";".join(sorted(codes)), "n/a"
    fields = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows | converted {len(rows) - pending} | to decide by hand {pending}")
    for r in rows:
        if r["decision"] == "":
            print(f"  item {r['item']} | {r['material']} | {r['original_labels']} | {r['notes']}")


if __name__ == "__main__":
    main()
