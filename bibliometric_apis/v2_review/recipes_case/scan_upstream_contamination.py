"""
==============================

Scan the base for upstream data-integrity issues that predate later
graph-building or clustering steps.

The script looks for two main problems in the source JSON itself:

1. Placeholder / LLM meta-response contamination
   recipe_text or translation contains a literal assistant refusal or clarifying
   response, such as "Please provide the recipe text...", instead of recipe
   content.

2. Short-text / ingredient-field mismatch
   recipe_text and translation are suspiciously short, or original ingredients
   exist but none of their tokens appear in the current text fields.

Default input
-------------

  relish_dataset.json

Default outputs
---------------

  data/upstream_contamination_report.csv
  data/upstream_contamination_summary.json

Usage
-----

  python scan_upstream_contamination.py

  python scan_upstream_contamination.py \
    --input relish_dataset.json \
    --out-csv data/upstream_contamination_report.csv \
    --out-summary data/upstream_contamination_summary.json

  python scan_upstream_contamination.py \
    --short-text-word-threshold 20 \
    --min-token-length 3

  python scan_upstream_contamination.py \
    --marker "please provide the recipe text" \
    --marker "as an ai language model"

Notes
-----

This is a diagnostic script. It flags suspicious records; it does not repair or
delete anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_JSON = Path("relish_dataset.json")
DEFAULT_OUT_CSV = Path("data/upstream_contamination_report.csv")
DEFAULT_OUT_SUMMARY = Path("data/upstream_contamination_summary.json")

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("scan_upstream_contamination")


DEFAULT_PLACEHOLDER_MARKERS = [
    "please provide the recipe text",
    "please provide the text",
    "i need the text to be able to",
    "i need the content of the recipe",
    "i need the content to be able to",
    "just paste it here",
    "i will provide a modern english translation",
    "as an ai language model",
    "i'm sorry, but i cannot",
    "i cannot translate",
    "i can’t translate",
    "i cannot provide",
    "i can't provide",
    "i need you to provide",
    "please paste the recipe",
    "please share the recipe",
]

DEFAULT_STOP_TOKENS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "into",
    "onto",
    "that",
    "this",
    "then",
    "than",
    "them",
    "they",
    "you",
    "your",
    "recipe",
    "recipes",
    "ingredient",
    "ingredients",
    "preparation",
    "translation",
    "text",
    "water",
    "salt",
}


# ---------------------------------------------------------------------------
# Loading and normalization
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    log.info("Loading %s", path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of recipe records")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if isinstance(item, dict):
            records.append(item)
        else:
            log.warning("Skipping non-object record at index %d", index)

    log.info("Loaded %d recipe records", len(records))
    return records


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_fields(rec: dict[str, Any], fields: list[str]) -> dict[str, str]:
    return {
        field: safe_str(rec.get(field, "") or "")
        for field in fields
    }


def combined_text(rec: dict[str, Any], fields: list[str]) -> str:
    values = text_fields(rec, fields).values()
    return normalize_space(" ".join(value for value in values if value))


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

def normalize_marker(marker: str) -> str:
    return normalize_space(marker.lower())


def load_markers(marker_args: list[str], marker_file: Path | None) -> list[str]:
    markers = list(DEFAULT_PLACEHOLDER_MARKERS)

    markers.extend(marker_args or [])

    if marker_file is not None:
        if not marker_file.exists():
            raise FileNotFoundError(f"Marker file not found: {marker_file}")

        markers.extend(
            line.strip()
            for line in marker_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    # Deterministic, deduplicated order.
    seen: set[str] = set()
    out: list[str] = []
    for marker in markers:
        normalized = normalize_marker(marker)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)

    return out


def placeholder_marker_found(text: str, markers: list[str]) -> str | None:
    low = normalize_space(text.lower())
    for marker in markers:
        if marker in low:
            return marker
    return None


# ---------------------------------------------------------------------------
# Ingredient/token extraction
# ---------------------------------------------------------------------------

def tokens_of(
    text: str,
    *,
    min_token_length: int,
    stop_tokens: set[str],
) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{%d,}" % min_token_length, text)
    }
    return {token for token in tokens if token not in stop_tokens}


def ingredient_name_from_item(item: Any) -> str:
    """Extract a name-like string from common ingredient encodings."""
    if item is None:
        return ""

    if isinstance(item, dict):
        for key in ["name", "canonical_name", "raw", "text", "ingredient"]:
            value = item.get(key)
            if value:
                return safe_str(value)
        return ""

    return safe_str(item)


def ingredient_names(rec: dict[str, Any], field: str) -> list[str]:
    raw = rec.get(field) or []

    if isinstance(raw, list):
        names = [ingredient_name_from_item(item) for item in raw]
        return [name for name in names if name.strip()]

    if isinstance(raw, dict):
        name = ingredient_name_from_item(raw)
        return [name] if name else []

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        # Handle newline or semicolon-separated fallback strings.
        if "\n" in text:
            return [part.strip() for part in text.splitlines() if part.strip()]
        if ";" in text:
            return [part.strip() for part in text.split(";") if part.strip()]
        return [text]

    return []


def ingredient_token_overlap(
    text: str,
    ingredients: list[str],
    *,
    min_token_length: int,
    stop_tokens: set[str],
) -> tuple[set[str], set[str], set[str]]:
    text_tokens = tokens_of(text, min_token_length=min_token_length, stop_tokens=stop_tokens)

    ingredient_tokens: set[str] = set()
    for ingredient in ingredients:
        ingredient_tokens.update(
            tokens_of(ingredient, min_token_length=min_token_length, stop_tokens=stop_tokens)
        )

    overlap = text_tokens & ingredient_tokens
    return text_tokens, ingredient_tokens, overlap


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def scan_records(
    records: list[dict[str, Any]],
    *,
    text_field_names: list[str],
    ingredient_field: str,
    markers: list[str],
    short_text_word_threshold: int,
    min_token_length: int,
    stop_tokens: set[str],
    require_short_for_zero_overlap: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    placeholder_count: Counter[str] = Counter()
    short_text_count: Counter[str] = Counter()
    zero_overlap_count: Counter[str] = Counter()
    flagged_by_source: Counter[str] = Counter()
    flag_type_counts: Counter[str] = Counter()

    for index, rec in enumerate(records):
        source_id = safe_str(rec.get("source_id", "unknown_source")) or "unknown_source"
        title = safe_str(rec.get("title", ""))

        current_text = combined_text(rec, text_field_names)
        words = current_text.split()
        word_count = len(words)

        ingredients = ingredient_names(rec, ingredient_field)
        n_ingredients = len(ingredients)

        flags: list[str] = []

        marker = placeholder_marker_found(current_text, markers)
        if marker:
            flags.append(f"placeholder:{marker}")
            placeholder_count[source_id] += 1
            flag_type_counts["placeholder"] += 1

        is_short_text = word_count < short_text_word_threshold and n_ingredients > 0
        if is_short_text:
            flags.append(f"short_text({word_count}w)")
            short_text_count[source_id] += 1
            flag_type_counts["short_text"] += 1

        text_tokens: set[str] = set()
        ingredient_tokens: set[str] = set()
        overlap: set[str] = set()

        if n_ingredients > 0:
            text_tokens, ingredient_tokens, overlap = ingredient_token_overlap(
                current_text,
                ingredients,
                min_token_length=min_token_length,
                stop_tokens=stop_tokens,
            )

            should_check_zero_overlap = not require_short_for_zero_overlap or is_short_text
            if should_check_zero_overlap and ingredient_tokens and not overlap:
                flags.append("zero_token_overlap_with_original_ingredients")
                zero_overlap_count[source_id] += 1
                flag_type_counts["zero_token_overlap"] += 1

        if flags:
            flagged_by_source[source_id] += 1
            rows.append({
                "index": index,
                "source_id": source_id,
                "title": title[:120],
                "flags": "; ".join(flags),
                "word_count": word_count,
                "n_original_ingredients": n_ingredients,
                "n_text_tokens": len(text_tokens),
                "n_ingredient_tokens": len(ingredient_tokens),
                "n_overlap_tokens": len(overlap),
                "overlap_tokens_preview": ", ".join(sorted(overlap)[:20]),
                "ingredient_tokens_preview": ", ".join(sorted(ingredient_tokens)[:20]),
                "text_preview": current_text[:240].replace("\n", " "),
            })

    summary = {
        "n_recipes_total": len(records),
        "n_flagged_total": len(rows),
        "flag_type_counts": dict(flag_type_counts),
        "flagged_by_source": dict(flagged_by_source),
        "placeholder_contamination_by_source": dict(placeholder_count),
        "short_text_by_source": dict(short_text_count),
        "zero_overlap_by_source": dict(zero_overlap_count),
        "parameters": {
            "text_fields": text_field_names,
            "ingredient_field": ingredient_field,
            "short_text_word_threshold": short_text_word_threshold,
            "min_token_length": min_token_length,
            "require_short_for_zero_overlap": require_short_for_zero_overlap,
            "n_placeholder_markers": len(markers),
        },
    }

    return rows, summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "index",
    "source_id",
    "title",
    "flags",
    "word_count",
    "n_original_ingredients",
    "n_text_tokens",
    "n_ingredient_tokens",
    "n_overlap_tokens",
    "overlap_tokens_preview",
    "ingredient_tokens_preview",
    "text_preview",
]


def write_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote CSV report: %s (%d rows)", path, len(rows))


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log.info("Wrote summary JSON: %s", path)


def print_log_summary(summary: dict[str, Any]) -> None:
    log.info(
        "Flagged recipes: %d / %d",
        summary["n_flagged_total"],
        summary["n_recipes_total"],
    )

    sections = [
        ("Placeholder contamination by source", "placeholder_contamination_by_source"),
        ("Suspiciously short text by source", "short_text_by_source"),
        ("Zero token overlap by source", "zero_overlap_by_source"),
    ]

    for title, key in sections:
        counts = Counter(summary.get(key, {}))
        log.info("%s:", title)
        if not counts:
            log.info("  none")
            continue
        for source_id, count in counts.most_common():
            log.info("  %-30s %d", source_id, count)


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
        help=f"Input RELISH JSON. Default: {DEFAULT_INPUT_JSON}",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Output CSV report. Default: {DEFAULT_OUT_CSV}",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=DEFAULT_OUT_SUMMARY,
        help=f"Output summary JSON. Default: {DEFAULT_OUT_SUMMARY}",
    )
    parser.add_argument(
        "--text-field",
        action="append",
        default=[],
        help=(
            "Text field to scan. Can be repeated. "
            "Default: recipe_text and translation."
        ),
    )
    parser.add_argument(
        "--ingredient-field",
        default="ingredients",
        help="Ingredient field containing original ingredient names.",
    )
    parser.add_argument(
        "--short-text-word-threshold",
        type=int,
        default=12,
        help="Word count below which a text is suspicious when ingredients exist.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=3,
        help="Minimum token length for ingredient/text overlap checks.",
    )
    parser.add_argument(
        "--marker",
        action="append",
        default=[],
        help="Additional placeholder marker phrase. Can be repeated.",
    )
    parser.add_argument(
        "--marker-file",
        type=Path,
        help="Optional text file with one additional placeholder marker per line.",
    )
    parser.add_argument(
        "--stop-token",
        action="append",
        default=[],
        help="Additional stop token to ignore in overlap checks. Can be repeated.",
    )
    parser.add_argument(
        "--require-short-for-zero-overlap",
        action="store_true",
        help=(
            "Only flag zero ingredient-token overlap when the text is also short. "
            "Default flags zero overlap whenever original ingredients exist."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress source-level log summary.",
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

    if args.short_text_word_threshold < 1:
        log.error("--short-text-word-threshold must be >= 1")
        return 2
    if args.min_token_length < 1:
        log.error("--min-token-length must be >= 1")
        return 2

    try:
        records = load_records(args.input)
        markers = load_markers(args.marker, args.marker_file)
        stop_tokens = set(DEFAULT_STOP_TOKENS) | {token.lower() for token in args.stop_token}

        text_fields = args.text_field if args.text_field else ["recipe_text", "translation"]

        rows, summary = scan_records(
            records,
            text_field_names=text_fields,
            ingredient_field=args.ingredient_field,
            markers=markers,
            short_text_word_threshold=args.short_text_word_threshold,
            min_token_length=args.min_token_length,
            stop_tokens=stop_tokens,
            require_short_for_zero_overlap=args.require_short_for_zero_overlap,
        )

        write_report_csv(args.out_csv, rows)
        write_summary_json(args.out_summary, summary)

        if not args.quiet:
            print_log_summary(summary)

        return 0

    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
