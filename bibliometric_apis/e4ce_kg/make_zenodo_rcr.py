"""Assemble the files of the treatment-route study for deposit on Zenodo.

Copies the frozen input, prompts, raw model outputs, aggregated labels,
annotations, Eurostat extract and family tables into OUT (default zenodo_rcr/).
Every field named "abstract" (JSON key at any depth, or CSV column) is removed,
because abstracts are not redistributed; papers keep their OpenAlex ID. A check
then looks for the opening of 300 seeded abstracts in every output file and
stops if any is found. MANIFEST.csv lists the size and SHA-256 of each file.
The script stops if a listed file is missing or two files share a name.

Usage:
    python make_zenodo_rcr.py [OUT]
"""
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PATTERNS = [
    "route_input.jsonl", "route_raw__*.jsonl", "route_adjud_raw__*.jsonl",
    "route_final.csv", "route_final_adjudicated__*.csv", "route_labels_long.csv",
    "prompt_treatment_route.txt", "prompt_adjudicate_route.txt", "audit_route_guidelines.md",
    "audit_route_blind.csv", "audit_route_key.csv",
    "audit_disposal_blind.csv", "audit_disposal_key.csv", "audit_material_key.csv",
    "annotations/audit_route_A_raw.csv", "annotations/route_label_map_A.csv",
    "annotations/audit_route_A.csv", "annotations/audit_disposal_A.csv",
    "annotations/audit_material_A.csv",
    "tmo_material_classified.csv", "tmo_material_corpus_counts.csv",
    "eurostat_family_ops.csv", "family_eu_table.csv", "country_family_panel.csv",
    "paper_countries.csv", "alignment_stats.csv",
]
EXTRA = [REPO / "vocab" / "material_families.csv"]
DROP = "abstract"
N_CHECK, PREFIX, SEED = 300, 500, 42


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k.lower() != DROP}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def copy(src, dst):
    if src.suffix == ".jsonl":
        with src.open(encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
            for line in fi:
                if line.strip():
                    fo.write(json.dumps(strip(json.loads(line)), ensure_ascii=False) + "\n")
    elif src.suffix == ".csv":
        with src.open(newline="", encoding="utf-8-sig") as fi, \
                dst.open("w", newline="", encoding="utf-8") as fo:
            reader = csv.DictReader(fi)
            fields = [f for f in reader.fieldnames if f.lower() != DROP]
            w = csv.DictWriter(fo, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(reader)
    else:
        shutil.copyfile(src, dst)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "zenodo_rcr"
    sources = []
    for pattern in PATTERNS:
        found = sorted(HERE.glob(pattern))
        if not found:
            sys.exit(f"missing: {pattern}")
        sources += found
    for path in EXTRA:
        if not path.exists():
            sys.exit(f"missing: {path}")
        sources.append(path)
    names = [p.name for p in sources]
    if len(set(names)) != len(names):
        sys.exit(f"duplicate file names: {sorted(n for n in set(names) if names.count(n) > 1)}")
    if out.exists():
        sys.exit(f"{out} exists; remove it first")
    out.mkdir(parents=True)

    for src in sources:
        copy(src, out / src.name)

    abstracts = []
    for line in (HERE / "route_input.jsonl").open(encoding="utf-8"):
        text = (json.loads(line).get("abstract") or "").strip()
        if len(text) >= PREFIX:
            abstracts.append(text[:PREFIX])
    probe = random.Random(SEED).sample(abstracts, min(N_CHECK, len(abstracts)))
    for path in sorted(out.iterdir()):
        text = path.read_text(encoding="utf-8", errors="ignore")
        leaks = sum(p in text for p in probe)
        if leaks:
            shutil.rmtree(out)
            sys.exit(f"{path.name}: {leaks} of {len(probe)} sampled abstracts found; "
                     f"{out.name} removed, nothing packaged")

    with (out / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "bytes", "sha256", "source"])
        for src in sources:
            dst = out / src.name
            w.writerow([dst.name, dst.stat().st_size, sha256(dst), str(src.relative_to(REPO))])
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"{len(sources)} files + MANIFEST.csv in {out} | {total / 1e6:.2f} MB | "
          f"abstract check passed on {len(probe)} samples")


if __name__ == "__main__":
    main()
