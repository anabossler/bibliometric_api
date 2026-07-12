"""
diagnose_empty_recipes.py
=========================

Diagnose recipes that ended up with zero ingredients after Step 6
re-extraction.

The script maps a recipe_id such as:

    recipe::a_miscellany::00029

back to its global index in the original Relish JSON, then prints:

  - the original ingredients before re-extraction
  - the Step 6 LLM output for that recipe, if present
  - the audit metadata returned by Step 6, if present
  - the raw recipe text and translation that were available to the extractor

Typical usage
-------------

Inspect explicit recipe IDs:

    python diagnose_empty_recipes.py recipe::a_miscellany::00029 recipe::a_miscellany::00038

Inspect the first N zero-ingredient recipes from a source, using
layer_1_derived_fields.csv:

    python diagnose_empty_recipes.py --source a_miscellany --n 10

Custom paths:

    python diagnose_empty_recipes.py \
      --input-json data/relish_dataset.json \
      --reextracted-jsonl data/reextracted_ingredients.jsonl \
      --derived-csv data/layer_1_derived_fields.csv \
      --source a_miscellany --n 10

Notes
-----

The recipe_id mapping intentionally mirrors the per-source counter logic used
by build_graph_step1.py:

    recipe::<slugified_source_id>::<per_source_index_5_digits>

For compatibility with older Step 1 outputs, the mapper registers both the
current Unicode-normalized slug and a legacy ASCII-simple slug when they differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_JSON = Path("relish_dataset.json")
DEFAULT_REEXTRACTED_JSONL = Path("data/reextracted_ingredients.jsonl")
DEFAULT_DERIVED_CSV = Path("data/layer_1_derived_fields.csv")

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("diagnose_empty_recipes")


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Stable Unicode-normalized slug used for recipe IDs.

    This should match the publish-ready build_graph_step1.py behavior:
    normalize accents, lower-case, replace non-alphanumeric runs with "_".
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    s = ascii_text.strip().lower()
    s = _SLUG_RE.sub("_", s)
    return s.strip("_") or "unknown"


def legacy_slugify(text: str) -> str:
    """Older slug behavior used by early drafts of build_graph_step1.py.

    This is kept only as a compatibility fallback when diagnosing outputs from
    earlier runs.
    """
    s = text.strip().lower()
    s = _SLUG_RE.sub("_", s)
    return s.strip("_") or "unknown"


def recipe_id(source_id: str, per_source_index: int, *, legacy: bool = False) -> str:
    slug = legacy_slugify(source_id) if legacy else slugify(source_id)
    return f"recipe::{slug}::{per_source_index:05d}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")

    bad = sum(1 for x in data if not isinstance(x, dict))
    if bad:
        raise ValueError(f"Expected all records to be objects; found {bad} non-object records")

    log.info("Loaded %d records from %s", len(data), path)
    return data


def build_global_index_map(records: list[dict[str, Any]]) -> dict[str, int]:
    """Return recipe_id -> global JSON index.

    Replays the same per-source counter logic used by build_graph_step1.py.
    Registers both the current normalized slug and the legacy slug for robust
    diagnostics across older project outputs.
    """
    counter: Counter[str] = Counter()
    mapping: dict[str, int] = {}

    for global_index, rec in enumerate(records):
        source_id = str(rec.get("source_id") or "unknown_source")
        per_source_index = counter[source_id]
        counter[source_id] += 1

        rid = recipe_id(source_id, per_source_index)
        mapping[rid] = global_index

        legacy_rid = recipe_id(source_id, per_source_index, legacy=True)
        if legacy_rid != rid and legacy_rid not in mapping:
            mapping[legacy_rid] = global_index

    log.info("Built recipe_id map with %d IDs", len(mapping))
    return mapping


def load_reextracted_by_index(path: Path) -> dict[int, dict[str, Any]]:
    """Load Step 6 JSONL records keyed by global recipe index.

    Corrupt lines are skipped with warnings because Step 6 outputs may be
    append-only/resume-safe logs from interrupted runs.
    """
    if not path.exists():
        log.warning("Re-extracted JSONL not found: %s", path)
        return {}

    out: dict[int, dict[str, Any]] = {}
    skipped = 0

    with path.open(encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                idx = rec["index"]
                idx_int = int(idx)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                skipped += 1
                log.warning("Skipping malformed JSONL line %d in %s: %s", line_no, path, exc)
                continue

            # Keep the last record for a recipe index. This is intentional for
            # append-only logs where a later retry may supersede an earlier one.
            out[idx_int] = rec

    log.info("Loaded %d Step 6 records from %s", len(out), path)
    if skipped:
        log.warning("Skipped %d malformed Step 6 JSONL lines", skipped)
    return out


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def target_ids_from_source(derived_csv: Path, source_id: str, n: int) -> list[str]:
    """Return first N recipe IDs from source_id whose n_ingredients is zero."""
    if not derived_csv.exists():
        raise FileNotFoundError(f"Derived CSV not found: {derived_csv}")

    target_ids: list[str] = []

    with derived_csv.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"recipe_id", "source_id", "n_ingredients"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{derived_csv} is missing required columns: {sorted(missing)}")

        for row in reader:
            if row.get("source_id") != source_id:
                continue
            if safe_int(row.get("n_ingredients")) == 0:
                rid = row.get("recipe_id")
                if rid:
                    target_ids.append(rid)
            if len(target_ids) >= n:
                break

    log.info(
        "Found %d zero-ingredient recipes for source_id=%r in %s",
        len(target_ids),
        source_id,
        derived_csv,
    )
    return target_ids


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def ingredient_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("canonical_name") or item)
    return str(item)


def show_recipe(
    *,
    recipe_id_value: str,
    global_index: int,
    original_record: dict[str, Any],
    reextracted_record: dict[str, Any] | None,
    max_items: int,
    max_text_chars: int,
    full_text: bool,
) -> None:
    print("=" * 88)
    print(f"recipe_id    : {recipe_id_value}")
    print(f"global index : {global_index}")
    print(f"source_id    : {original_record.get('source_id')}")
    print(f"title        : {original_record.get('title')}")
    print("-" * 88)

    print("ORIGINAL ingredients before Step 6:")
    original_ingredients = original_record.get("ingredients") or []
    if not original_ingredients:
        print("  (already empty before Step 6)")
    else:
        for item in original_ingredients[:max_items]:
            print(f"  - {ingredient_name(item)}")
        if len(original_ingredients) > max_items:
            print(f"  ... {len(original_ingredients) - max_items} more")

    print("-" * 88)

    if reextracted_record is None:
        print("STEP 6 output:")
        print("  No Step 6 record for this global index.")
        print("  Possible causes: never targeted, failed before write, or different index scheme.")
    else:
        extracted = reextracted_record.get("ingredients") or []
        audit = reextracted_record.get("audit") or {}
        status = reextracted_record.get("status")
        error = reextracted_record.get("error")

        print("STEP 6 output:")
        print(f"  ingredients: {len(extracted)}")
        if status is not None:
            print(f"  status     : {status}")
        if audit:
            print(f"  audit      : {json.dumps(audit, ensure_ascii=False)}")
        if error:
            print(f"  error      : {error}")

        for item in extracted[:max_items]:
            if isinstance(item, dict):
                name = item.get("name")
                verbatim = item.get("verbatim_form")
                conf = item.get("confidence_score")
                extras = []
                if verbatim is not None:
                    extras.append(f"verbatim={verbatim!r}")
                if conf is not None:
                    extras.append(f"confidence={conf!r}")
                suffix = f" ({', '.join(extras)})" if extras else ""
                print(f"  - {name!r}{suffix}")
            else:
                print(f"  - {item!r}")

        if len(extracted) > max_items:
            print(f"  ... {len(extracted) - max_items} more")

    print("-" * 88)
    recipe_text = str(original_record.get("recipe_text") or "")
    translation = str(original_record.get("translation") or "")
    combined_text = (recipe_text + "\n\n" + translation).strip()

    print("RAW TEXT given/available to extractor:")
    if not combined_text:
        print("  (empty recipe_text and translation)")
    elif full_text:
        print(combined_text)
    else:
        preview = combined_text[:max_text_chars]
        print(preview)
        if len(combined_text) > max_text_chars:
            print(f"... [{len(combined_text) - max_text_chars} chars omitted; use --full-text]")
    print()


def write_json_report(
    out_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON report: %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "recipe_ids",
        nargs="*",
        help="Recipe IDs to inspect, e.g. recipe::a_miscellany::00029",
    )
    parser.add_argument(
        "--source",
        help="Instead of explicit IDs, inspect the first N zero-ingredient recipes from this source_id.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of zero-ingredient recipes to pull when --source is used.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help=f"Path to Relish base JSON. Default: {DEFAULT_INPUT_JSON}",
    )
    parser.add_argument(
        "--reextracted-jsonl",
        type=Path,
        default=DEFAULT_REEXTRACTED_JSONL,
        help=f"Path to Step 6 re-extraction JSONL. Default: {DEFAULT_REEXTRACTED_JSONL}",
    )
    parser.add_argument(
        "--derived-csv",
        type=Path,
        default=DEFAULT_DERIVED_CSV,
        help=f"Path to layer_1_derived_fields.csv. Default: {DEFAULT_DERIVED_CSV}",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=15,
        help="Maximum original/extracted ingredients printed per recipe.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=600,
        help="Maximum raw text characters printed unless --full-text is used.",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Print full recipe text instead of a preview.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Optional path to write a machine-readable JSON diagnostic report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    if args.n < 1:
        log.error("--n must be >= 1")
        return 2
    if args.max_items < 1:
        log.error("--max-items must be >= 1")
        return 2
    if args.max_text_chars < 1:
        log.error("--max-text-chars must be >= 1")
        return 2

    try:
        records = load_json_records(args.input_json)
        id_map = build_global_index_map(records)
        reextracted = load_reextracted_by_index(args.reextracted_jsonl)

        target_ids = list(args.recipe_ids)

        if args.source:
            source_targets = target_ids_from_source(args.derived_csv, args.source, args.n)
            target_ids.extend(source_targets)

        # Preserve order, remove duplicates.
        target_ids = list(dict.fromkeys(target_ids))

        if not target_ids:
            log.error("No recipe IDs given and no --source zero-ingredient matches found.")
            return 1

        json_rows: list[dict[str, Any]] = []

        for rid in target_ids:
            global_index = id_map.get(rid)
            if global_index is None:
                log.warning("Could not map %s to a global index. Check the recipe_id scheme.", rid)
                json_rows.append({
                    "recipe_id": rid,
                    "mapped": False,
                    "error": "recipe_id_not_found",
                })
                continue

            original_record = records[global_index]
            reextracted_record = reextracted.get(global_index)

            show_recipe(
                recipe_id_value=rid,
                global_index=global_index,
                original_record=original_record,
                reextracted_record=reextracted_record,
                max_items=args.max_items,
                max_text_chars=args.max_text_chars,
                full_text=args.full_text,
            )

            json_rows.append({
                "recipe_id": rid,
                "mapped": True,
                "global_index": global_index,
                "source_id": original_record.get("source_id"),
                "title": original_record.get("title"),
                "original_ingredient_count": len(original_record.get("ingredients") or []),
                "step6_record_present": reextracted_record is not None,
                "step6_ingredient_count": (
                    len(reextracted_record.get("ingredients") or [])
                    if reextracted_record is not None
                    else None
                ),
                "step6_audit": (
                    reextracted_record.get("audit")
                    if reextracted_record is not None
                    else None
                ),
                "step6_status": (
                    reextracted_record.get("status")
                    if reextracted_record is not None
                    else None
                ),
            })

        if args.out_json:
            write_json_report(args.out_json, json_rows)

        return 0

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
