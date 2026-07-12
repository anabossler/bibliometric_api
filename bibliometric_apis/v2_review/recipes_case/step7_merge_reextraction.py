"""
===========================

Merge Step 6 re-extracted ingredients back into the RELISH JSON dataset.

Why this step exists
--------------------

build_graph_step1.py already knows how to build the graph from a RELISH-schema
JSON list. Step 7 therefore only fixes the input dataset:

  - For recipes successfully processed by Step 6, replace the original
    `ingredients` field with the re-extracted, verbatim-validated ingredients.
  - For recipes not targeted by Step 6, leave ingredients unchanged.
  - For contaminated-source recipes that Step 6 targeted but did not finish,
    do not allow contaminated ingredients to pass silently. By default, their
    ingredients are set to [] and they are written to a residual CSV.

Step 6 schema
-------------

  {
    "name": "...",
    "verbatim_form": "..."
  }

RELISH schema output
--------------------

  {
    "name": "...",
    "confidence_score": 1.0,
    "specific_forms": ["<verbatim_form>"]
  }

By default, an extracted ingredient is kept only if its `verbatim_form` appears
literally in the recipe text or translation. Use --keep-unverified to retain
unverified forms with confidence_score=0.3.

Default inputs
--------------

  relish_dataset.json
  data/reextracted_ingredients.jsonl

Default outputs
---------------

  data/relish_dataset_reextracted.json
  data/step7_merge_audit.csv
  data/step7_residual_contaminated.csv
  data/step7_merge_stats.json

Usage
-----

  python step7_merge_reextraction.py

  python step7_merge_reextraction.py --keep-unverified

  python step7_merge_reextraction.py --on-residual keep

  python step7_merge_reextraction.py --dry-run

Next step
---------

  python build_graph_step1.py --input data/relish_dataset_reextracted.json --output-dir data/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_JSON = Path("relish_dataset.json")
DEFAULT_REEXTRACTED_JSONL = Path("data/reextracted_ingredients.jsonl")
DEFAULT_OUT_JSON = Path("data/relish_dataset_reextracted.json")
DEFAULT_AUDIT_CSV = Path("data/step7_merge_audit.csv")
DEFAULT_STATS_JSON = Path("data/step7_merge_stats.json")
DEFAULT_RESIDUAL_CSV = Path("data/step7_residual_contaminated.csv")

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step7_merge_reextraction")


# Must mirror Step 6's default contaminated-source list. The original Step 7
# script omitted sefardies_es even though Step 6 included it; keep it here.
DEFAULT_CONTAMINATED_SOURCES = {
    "apicius",
    "sefardies_es",
    "a_miscellany",
    "enseignements",
    "veneziano",
    "cuina_catalana",
    "fundacio_alicia_0",
    "fundacio_alicia_1",
    "fundacio_alicia_2",
    "fundacio_alicia_3",
    "fundacio_alicia_4",
    "fundacio_alicia_5",
    "fundacio_alicia_6",
    "fundacio_alicia_7",
    "fundacio_alicia_8",
    "fundacio_alicia_9",
    "fundacio_alicia_10",
    "fundacio_alicia_11",
    "fundacio_alicia_12",
    "fundacio_alicia_13",
    "fundacio_alicia_14",
    "fundacio_alicia_15",
    "fundacio_alicia_16",
    "harpestreng",
    "proper_newe",
    "viandier_1485",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return default
    return text if text else default


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def canonical_name(value: Any) -> str:
    text = safe_str(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,.;:()[]{}")
    return text


def build_recipe_text(recipe: dict[str, Any], *, text_fields: list[str]) -> str:
    parts: list[str] = []
    for field in text_fields:
        value = safe_str(recipe.get(field))
        if value:
            parts.append(value)
    return normalize_space(" ".join(parts))


def verbatim_in_text(verbatim: str, text: str) -> bool:
    verbatim_norm = normalize_space(verbatim).lower()
    text_norm = normalize_space(text).lower()
    return bool(verbatim_norm) and verbatim_norm in text_norm


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    log.info("Loading RELISH JSON: %s", path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    records = [row for row in data if isinstance(row, dict)]
    if len(records) != len(data):
        log.warning("Dropped %d non-dict records", len(data) - len(records))

    log.info("Loaded %d recipes", len(records))
    return records


def load_reextracted(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Index Step 6 JSONL by recipe index. Last duplicate wins."""
    if not path.exists():
        raise FileNotFoundError(f"Re-extracted JSONL not found: {path}")

    log.info("Loading re-extracted ingredients: %s", path)

    by_index: dict[int, dict[str, Any]] = {}
    n_lines = 0
    n_malformed = 0
    n_missing_index = 0
    n_duplicate = 0

    with path.open(encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            n_lines += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_malformed += 1
                log.warning("Skipping malformed JSONL line %d", line_no)
                continue

            if not isinstance(record, dict) or "index" not in record:
                n_missing_index += 1
                continue

            try:
                index = int(record["index"])
            except (TypeError, ValueError):
                n_missing_index += 1
                continue

            if index in by_index:
                n_duplicate += 1

            by_index[index] = record

    stats = {
        "jsonl_lines": n_lines,
        "indexed_recipes": len(by_index),
        "malformed_lines": n_malformed,
        "missing_or_bad_index": n_missing_index,
        "duplicate_indices": n_duplicate,
    }

    if n_duplicate:
        log.warning("%d duplicate indices in JSONL; kept last occurrence", n_duplicate)

    log.info("Indexed %d re-extracted recipes", len(by_index))
    return by_index, stats


def load_sources(default_sources: set[str], source_args: list[str], source_file: Path | None) -> set[str]:
    sources = set(default_sources)

    for item in source_args or []:
        if item.strip():
            sources.add(item.strip())

    if source_file is not None:
        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")

        for line in source_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sources.add(line)

    return sources


# ---------------------------------------------------------------------------
# Ingredient conversion
# ---------------------------------------------------------------------------

def convert_ingredients(
    raw_ingredients: Any,
    *,
    recipe_text: str,
    keep_unverified: bool,
    merge_duplicate_names: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Translate Step 6 ingredient schema into RELISH ingredient schema."""
    if not isinstance(raw_ingredients, list):
        raw_ingredients = []

    converted: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}

    stats = Counter()

    for item in raw_ingredients:
        if not isinstance(item, dict):
            stats["bad_item_skipped"] += 1
            continue

        name = canonical_name(item.get("name"))
        verbatim = safe_str(item.get("verbatim_form"))

        if not name:
            stats["missing_name_skipped"] += 1
            continue

        is_verified = verbatim_in_text(verbatim, recipe_text)

        if not is_verified:
            stats["unverified"] += 1
            if not keep_unverified:
                stats["unverified_dropped"] += 1
                continue

        output_item = {
            "name": name,
            "confidence_score": 1.0 if is_verified else 0.3,
            "specific_forms": [verbatim] if verbatim else [],
        }

        if merge_duplicate_names and name in by_name:
            existing = by_name[name]
            for form in output_item["specific_forms"]:
                if form and form not in existing["specific_forms"]:
                    existing["specific_forms"].append(form)

            # If any occurrence is verified, keep the higher confidence.
            existing["confidence_score"] = max(
                float(existing.get("confidence_score", 0.0)),
                float(output_item["confidence_score"]),
            )
            stats["duplicate_name_merged"] += 1
            continue

        converted.append(output_item)
        by_name[name] = output_item
        stats["kept"] += 1
        if is_verified:
            stats["verified_kept"] += 1
        else:
            stats["unverified_kept"] += 1

    return converted, {str(key): int(value) for key, value in stats.items()}


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_reextractions(
    records: list[dict[str, Any]],
    reextracted: dict[int, dict[str, Any]],
    *,
    contaminated_sources: set[str],
    keep_unverified: bool,
    on_residual: str,
    text_fields: list[str],
    min_text_chars: int,
    merge_duplicate_names: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []

    counters = Counter()
    source_before: Counter[str] = Counter()
    source_after: Counter[str] = Counter()
    source_replaced: Counter[str] = Counter()
    source_residual: Counter[str] = Counter()

    for index, recipe in enumerate(records):
        source_id = safe_str(recipe.get("source_id"), "unknown_source")
        original_ingredients = recipe.get("ingredients") or []
        if not isinstance(original_ingredients, list):
            original_ingredients = []

        n_before = len(original_ingredients)
        source_before[source_id] += n_before

        re_record = reextracted.get(index)

        if re_record is None:
            if source_id in contaminated_sources:
                counters["residual_contaminated"] += 1
                source_residual[source_id] += 1
                residual_rows.append({
                    "index": int(index),
                    "source_id": source_id,
                    "title": safe_str(recipe.get("title"))[:100],
                    "n_ingredients_original": int(n_before),
                    "residual_action": on_residual,
                    "reason": "source_in_contaminated_list_but_no_step6_record",
                })

                if on_residual == "drop":
                    recipe["ingredients"] = []
                    source_after[source_id] += 0
                else:
                    source_after[source_id] += n_before
                continue

            counters["unchanged_clean_source"] += 1
            source_after[source_id] += n_before
            continue

        text = build_recipe_text(recipe, text_fields=text_fields)
        if len(text) < min_text_chars:
            counters["skipped_short_text_kept_original"] += 1
            source_after[source_id] += n_before
            audit_rows.append({
                "index": int(index),
                "source_id": source_id,
                "title": safe_str(recipe.get("title"))[:100],
                "n_ingredients_before": int(n_before),
                "n_ingredients_after": int(n_before),
                "n_verified_kept": 0,
                "n_unverified_kept": 0,
                "n_unverified_dropped": 0,
                "n_bad_items_skipped": 0,
                "action": "kept_original_short_text",
            })
            continue

        converted, conversion_stats = convert_ingredients(
            re_record.get("ingredients", []),
            recipe_text=text,
            keep_unverified=keep_unverified,
            merge_duplicate_names=merge_duplicate_names,
        )

        recipe["ingredients"] = converted
        n_after = len(converted)

        source_after[source_id] += n_after
        source_replaced[source_id] += 1
        counters["recipes_replaced"] += 1
        counters["ingredients_before_replaced"] += n_before
        counters["ingredients_after_replaced"] += n_after
        counters["unverified_dropped_or_flagged"] += conversion_stats.get("unverified", 0)

        audit_rows.append({
            "index": int(index),
            "source_id": source_id,
            "title": safe_str(recipe.get("title"))[:100],
            "n_ingredients_before": int(n_before),
            "n_ingredients_after": int(n_after),
            "n_verified_kept": int(conversion_stats.get("verified_kept", 0)),
            "n_unverified_kept": int(conversion_stats.get("unverified_kept", 0)),
            "n_unverified_dropped": int(conversion_stats.get("unverified_dropped", 0)),
            "n_bad_items_skipped": int(conversion_stats.get("bad_item_skipped", 0)),
            "action": "replaced",
        })

    stats = {
        "counts": {str(key): int(value) for key, value in counters.items()},
        "source_before": {str(key): int(value) for key, value in source_before.items()},
        "source_after": {str(key): int(value) for key, value in source_after.items()},
        "source_replaced": {str(key): int(value) for key, value in source_replaced.items()},
        "source_residual": {str(key): int(value) for key, value in source_residual.items()},
    }

    return records, audit_rows, residual_rows, stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_json(path: Path, data: Any, *, indent: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
    log.info("Wrote JSON: %s", path)


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote CSV: %s (%d rows)", path, len(rows))


def write_stats(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("Wrote stats JSON: %s", path)


def print_source_changes(stats: dict[str, Any], contaminated_sources: set[str]) -> None:
    source_before = stats["source_before"]
    source_after = stats["source_after"]
    source_replaced = stats["source_replaced"]

    sources_to_print = sorted(
        source for source in contaminated_sources
        if source in source_before or source in source_after or source in source_replaced
    )

    if not sources_to_print:
        return

    log.info("Per-source ingredient count change for contaminated sources:")
    for source_id in sources_to_print:
        before = source_before.get(source_id, 0)
        after = source_after.get(source_id, 0)
        n_replaced = source_replaced.get(source_id, 0)
        log.info("  %-25s %5d -> %5d | recipes replaced=%d", source_id, before, after, n_replaced)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help=f"Base RELISH JSON. Default: {DEFAULT_INPUT_JSON}",
    )
    parser.add_argument(
        "--reextracted",
        type=Path,
        default=DEFAULT_REEXTRACTED_JSONL,
        help=f"Step 6 re-extracted JSONL. Default: {DEFAULT_REEXTRACTED_JSONL}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help=f"Merged output JSON. Default: {DEFAULT_OUT_JSON}",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=DEFAULT_AUDIT_CSV,
        help=f"Merge audit CSV. Default: {DEFAULT_AUDIT_CSV}",
    )
    parser.add_argument(
        "--residual-csv",
        type=Path,
        default=DEFAULT_RESIDUAL_CSV,
        help=f"Residual contaminated CSV. Default: {DEFAULT_RESIDUAL_CSV}",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=DEFAULT_STATS_JSON,
        help=f"Stats JSON. Default: {DEFAULT_STATS_JSON}",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Additional contaminated source_id. Can be repeated.",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        help="Optional file with one contaminated source_id per line.",
    )
    parser.add_argument(
        "--no-default-sources",
        dest="use_default_sources",
        action="store_false",
        help="Do not include built-in contaminated-source list.",
    )
    parser.set_defaults(use_default_sources=True)
    parser.add_argument(
        "--text-field",
        action="append",
        default=["recipe_text", "translation"],
        help="Recipe text field used for verbatim verification. Can be repeated.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=10,
        help="Minimum concatenated recipe text length required for replacement.",
    )
    parser.add_argument(
        "--keep-unverified",
        action="store_true",
        help=(
            "Keep ingredients whose verbatim_form was not found in text. "
            "They receive confidence_score=0.3."
        ),
    )
    parser.add_argument(
        "--no-merge-duplicate-names",
        dest="merge_duplicate_names",
        action="store_false",
        help="Do not merge duplicate canonical names from Step 6 output.",
    )
    parser.set_defaults(merge_duplicate_names=True)
    parser.add_argument(
        "--on-residual",
        choices=["drop", "keep"],
        default="drop",
        help=(
            "Action for contaminated-source recipes with no Step 6 output. "
            "'drop' sets ingredients=[]; 'keep' leaves original ingredients."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats but do not write files unless --write-dry-run-outputs is set.",
    )
    parser.add_argument(
        "--write-dry-run-outputs",
        action="store_true",
        help="With --dry-run, still write audit/residual/stats but not merged JSON.",
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

    if args.min_text_chars < 0:
        log.error("--min-text-chars must be >= 0")
        return 2
    if not args.text_field:
        log.error("At least one --text-field is required")
        return 2

    try:
        records = load_json_list(args.input)
        reextracted, reextract_stats = load_reextracted(args.reextracted)

        contaminated_sources = load_sources(
            DEFAULT_CONTAMINATED_SOURCES if args.use_default_sources else set(),
            args.source,
            args.source_file,
        )

        if not contaminated_sources:
            raise ValueError("No contaminated sources selected. Use --source or default sources.")

        records, audit_rows, residual_rows, merge_stats = merge_reextractions(
            records,
            reextracted,
            contaminated_sources=contaminated_sources,
            keep_unverified=args.keep_unverified,
            on_residual=args.on_residual,
            text_fields=args.text_field,
            min_text_chars=args.min_text_chars,
            merge_duplicate_names=args.merge_duplicate_names,
        )

        counts = merge_stats["counts"]
        log.info("Recipes replaced: %d", counts.get("recipes_replaced", 0))
        log.info("Residual contaminated recipes: %d", counts.get("residual_contaminated", 0))
        log.info("Unverified Step 6 ingredient entries dropped/flagged: %d", counts.get("unverified_dropped_or_flagged", 0))

        if residual_rows:
            action = "ingredients set to []" if args.on_residual == "drop" else "original ingredients kept"
            log.warning(
                "%d contaminated-source recipes have no Step 6 output -> %s",
                len(residual_rows),
                action,
            )
            log.warning("Review residual CSV and re-run Step 6 for these indices if needed.")

        print_source_changes(merge_stats, contaminated_sources)

        stats_payload = {
            "timestamp_utc": timestamp_utc(),
            "inputs": {
                "input_json": str(args.input),
                "reextracted_jsonl": str(args.reextracted),
            },
            "outputs": {
                "merged_json": None if args.dry_run else str(args.output),
                "audit_csv": str(args.audit_csv),
                "residual_csv": str(args.residual_csv),
                "stats_json": str(args.stats_json),
            },
            "parameters": {
                "keep_unverified": args.keep_unverified,
                "on_residual": args.on_residual,
                "text_fields": args.text_field,
                "min_text_chars": args.min_text_chars,
                "merge_duplicate_names": args.merge_duplicate_names,
                "dry_run": args.dry_run,
            },
            "contaminated_sources": sorted(contaminated_sources),
            "reextracted_jsonl_stats": reextract_stats,
            "n_recipes_total": len(records),
            "merge": merge_stats,
            "next_step": f"python build_graph_step1.py --input {args.output} --output-dir data/",
        }

        if args.dry_run and not args.write_dry_run_outputs:
            log.info("[dry-run] No files written.")
            print(json.dumps({
                "recipes_replaced": counts.get("recipes_replaced", 0),
                "residual_contaminated": counts.get("residual_contaminated", 0),
                "unverified_dropped_or_flagged": counts.get("unverified_dropped_or_flagged", 0),
            }, indent=2, ensure_ascii=False))
            return 0

        if not args.dry_run:
            write_json(args.output, records, indent=1)

        audit_fields = [
            "index",
            "source_id",
            "title",
            "n_ingredients_before",
            "n_ingredients_after",
            "n_verified_kept",
            "n_unverified_kept",
            "n_unverified_dropped",
            "n_bad_items_skipped",
            "action",
        ]
        residual_fields = [
            "index",
            "source_id",
            "title",
            "n_ingredients_original",
            "residual_action",
            "reason",
        ]

        write_csv(args.audit_csv, audit_rows, fieldnames=audit_fields)
        write_csv(args.residual_csv, residual_rows, fieldnames=residual_fields)
        write_stats(args.stats_json, stats_payload)

        log.info("NEXT STEP:")
        if residual_rows:
            log.info(
                "  Re-run step6_reextract_ingredients.py for indices listed in %s, then re-run this script.",
                args.residual_csv,
            )
        log.info("  python build_graph_step1.py --input %s --output-dir data/", args.output)

        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
        csv.Error,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
