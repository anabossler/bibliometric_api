"""
==============================

Flag upstream-corrupt Recipe nodes in the RELISH knowledge graph.

This step reads the contamination report produced by
scan_upstream_contamination.py and marks affected Recipe nodes in the graph:

  quality_flag   = "upstream_corrupt"
  corrupt_reason = "<semicolon-separated reasons>"

Recipes are NOT removed. Downstream analyses should filter them out explicitly:

  if G.nodes[n].get("quality_flag") == "upstream_corrupt":
      continue

Default inputs
--------------

  data/graph_step1.gpickle
  data/upstream_contamination_report.csv

Default outputs
---------------

  data/graph_step1_flagged.gpickle
  data/quality_flag_summary.json
  data/quality_flag_audit.csv
  data/quality_flag_missing.csv

Usage
-----

  python step8_flag_upstream_corrupt.py

  python step8_flag_upstream_corrupt.py --dry-run

  python step8_flag_upstream_corrupt.py --report-csv data/upstream_contamination_report.csv

  python step8_flag_upstream_corrupt.py --strict

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_GRAPH_IN = Path("data/graph_step1.gpickle")
DEFAULT_GRAPH_OUT = Path("data/graph_step1_flagged.gpickle")
DEFAULT_REPORT_CSV = Path("data/upstream_contamination_report.csv")
DEFAULT_SUMMARY_JSON = Path("data/quality_flag_summary.json")
DEFAULT_AUDIT_CSV = Path("data/quality_flag_audit.csv")
DEFAULT_MISSING_CSV = Path("data/quality_flag_missing.csv")

DEFAULT_QUALITY_FLAG = "upstream_corrupt"

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step8_flag_upstream_corrupt")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph not found: {path}. Run build_graph_step1.py first.")

    log.info("Loading graph: %s", path)
    with path.open("rb") as fh:
        graph = pickle.load(fh)

    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError(f"Object loaded from {path} does not look like a NetworkX graph")

    log.info("Graph loaded: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def load_report(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Contamination report not found: {path}. Run scan_upstream_contamination.py first."
        )

    log.info("Loading contamination report: %s", path)
    df = pd.read_csv(path)

    required = {"index", "source_id", "flags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Report CSV is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    bad_index = df["index"].isna().sum()
    if bad_index:
        log.warning("Dropping %d report rows with non-numeric index", bad_index)
        df = df[df["index"].notna()].copy()

    df["index"] = df["index"].astype(int)
    df["source_id"] = df["source_id"].fillna("").astype(str)
    df["flags"] = df["flags"].fillna("").astype(str)

    log.info("Report rows loaded: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Node matching
# ---------------------------------------------------------------------------

def recipe_id_from_index(global_index: int, source_id: str) -> str:
    """Reconstruct the default recipe node id used by build_graph_step1.py."""
    return f"recipe::{source_id}::{global_index:05d}"


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return default
    return text if text else default


def parse_flags(flags: Any) -> list[str]:
    text = safe_str(flags)
    if not text:
        return []
    return [
        item.strip()
        for item in re.split(r"[;,|]+", text)
        if item.strip()
    ]


def build_recipe_lookup(graph: Any) -> dict[tuple[str, int], str]:
    """Build fallback lookup from Recipe node attributes and node-id pattern."""
    lookup: dict[tuple[str, int], str] = {}

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Recipe":
            continue

        source_id = safe_str(attrs.get("source_id"))

        for key in ["index", "global_index", "dataset_index", "record_index", "json_index"]:
            value = attrs.get(key)
            try:
                if value is not None and source_id:
                    lookup[(source_id, int(value))] = str(node_id)
            except (TypeError, ValueError):
                pass

        # Default node-id pattern: recipe::<source_id>::<00000>
        node_text = str(node_id)
        match = re.match(r"^recipe::(.+)::(\d+)$", node_text)
        if match:
            lookup[(match.group(1), int(match.group(2)))] = node_text

    return lookup


def candidate_node_ids(row: pd.Series) -> list[str]:
    """Return possible node IDs from report row."""
    candidates: list[str] = []

    for column in ["recipe_id", "node_id"]:
        if column in row and safe_str(row[column]):
            candidates.append(safe_str(row[column]))

    source_id = safe_str(row["source_id"])
    global_index = int(row["index"])
    candidates.append(recipe_id_from_index(global_index, source_id))

    # Remove duplicates while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)

    return out


def resolve_recipe_node(
    *,
    graph: Any,
    row: pd.Series,
    lookup: dict[tuple[str, int], str],
) -> str | None:
    """Resolve report row to a Recipe node in graph."""
    for candidate in candidate_node_ids(row):
        if candidate in graph and graph.nodes[candidate].get("node_type") == "Recipe":
            return candidate

    source_id = safe_str(row["source_id"])
    global_index = int(row["index"])
    fallback = lookup.get((source_id, global_index))

    if fallback and fallback in graph and graph.nodes[fallback].get("node_type") == "Recipe":
        return fallback

    return None


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

def apply_flags(
    *,
    graph: Any,
    report_df: pd.DataFrame,
    quality_flag: str,
    preserve_existing_quality_flag: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    lookup = build_recipe_lookup(graph)

    audit_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for _, row in report_df.iterrows():
        flags = parse_flags(row["flags"])
        flags_text = "; ".join(flags) if flags else safe_str(row["flags"])

        node_id = resolve_recipe_node(graph=graph, row=row, lookup=lookup)

        if node_id is None:
            missing_rows.append({
                "index": int(row["index"]),
                "source_id": safe_str(row["source_id"]),
                "title": safe_str(row.get("title"))[:120],
                "flags": flags_text,
                "candidate_node_id": recipe_id_from_index(int(row["index"]), safe_str(row["source_id"])),
            })
            continue

        attrs = graph.nodes[node_id]

        previous_quality_flag = safe_str(attrs.get("quality_flag"))
        previous_corrupt_reason = safe_str(attrs.get("corrupt_reason"))

        if preserve_existing_quality_flag and previous_quality_flag and previous_quality_flag != quality_flag:
            audit_action = "skipped_existing_quality_flag"
        else:
            attrs["quality_flag"] = quality_flag
            attrs["corrupt_reason"] = flags_text
            attrs["quality_flag_source"] = "scan_upstream_contamination.py"
            audit_action = "flagged"

            for flag in flags:
                reason_counts[flag] += 1

        audit_rows.append({
            "node_id": node_id,
            "index": int(row["index"]),
            "source_id": safe_str(row["source_id"]),
            "title": safe_str(row.get("title") or attrs.get("title"))[:120],
            "flags": flags_text,
            "action": audit_action,
            "previous_quality_flag": previous_quality_flag,
            "previous_corrupt_reason": previous_corrupt_reason,
        })

    return audit_rows, missing_rows, reason_counts


def recipe_counts(graph: Any) -> tuple[int, Counter[str], Counter[str]]:
    total_recipes = 0
    quality_counts: Counter[str] = Counter()
    flagged_by_source: Counter[str] = Counter()

    for _, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Recipe":
            continue

        total_recipes += 1
        quality_flag = safe_str(attrs.get("quality_flag"), "unflagged")
        quality_counts[quality_flag] += 1

        if quality_flag == DEFAULT_QUALITY_FLAG:
            flagged_by_source[safe_str(attrs.get("source_id"), "unknown")] += 1

    return total_recipes, quality_counts, flagged_by_source


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_graph(graph: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Wrote flagged graph: %s", path)


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote CSV: %s (%d rows)", path, len(rows))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    log.info("Wrote JSON: %s", path)


def print_summary(
    *,
    audit_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    reason_counts: Counter[str],
    flagged_by_source: Counter[str],
    top_sources: int,
) -> None:
    n_flagged = sum(1 for row in audit_rows if row["action"] == "flagged")

    log.info("Recipes flagged: %d", n_flagged)
    if missing_rows:
        log.warning("Report rows missing from graph: %d", len(missing_rows))

    if reason_counts:
        log.info("Reason distribution:")
        for reason, count in reason_counts.most_common():
            log.info("  %-40s %d", reason, count)

    if flagged_by_source:
        log.info("Flagged recipes by source:")
        for source_id, count in flagged_by_source.most_common(top_sources):
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
        "--graph-in",
        type=Path,
        default=DEFAULT_GRAPH_IN,
        help=f"Input graph. Default: {DEFAULT_GRAPH_IN}",
    )
    parser.add_argument(
        "--graph-out",
        type=Path,
        default=DEFAULT_GRAPH_OUT,
        help=f"Output flagged graph. Default: {DEFAULT_GRAPH_OUT}",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=DEFAULT_REPORT_CSV,
        help=f"Upstream contamination report CSV. Default: {DEFAULT_REPORT_CSV}",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help=f"Summary JSON. Default: {DEFAULT_SUMMARY_JSON}",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=DEFAULT_AUDIT_CSV,
        help=f"Per-node audit CSV. Default: {DEFAULT_AUDIT_CSV}",
    )
    parser.add_argument(
        "--missing-csv",
        type=Path,
        default=DEFAULT_MISSING_CSV,
        help=f"Rows not matched to graph CSV. Default: {DEFAULT_MISSING_CSV}",
    )
    parser.add_argument(
        "--quality-flag",
        default=DEFAULT_QUALITY_FLAG,
        help=f"Quality flag value to set. Default: {DEFAULT_QUALITY_FLAG}",
    )
    parser.add_argument(
        "--preserve-existing-quality-flag",
        action="store_true",
        help="Do not overwrite an existing different quality_flag.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any report rows cannot be matched to Recipe nodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute flags but do not write graph.",
    )
    parser.add_argument(
        "--write-dry-run-outputs",
        action="store_true",
        help="With --dry-run, still write audit/missing/summary files.",
    )
    parser.add_argument(
        "--top-sources",
        type=int,
        default=15,
        help="Number of flagged sources to print.",
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

    if args.top_sources < 0:
        log.error("--top-sources must be >= 0")
        return 2

    try:
        graph = load_graph(args.graph_in)
        report_df = load_report(args.report_csv)

        audit_rows, missing_rows, reason_counts = apply_flags(
            graph=graph,
            report_df=report_df,
            quality_flag=args.quality_flag,
            preserve_existing_quality_flag=args.preserve_existing_quality_flag,
        )

        total_recipes, quality_counts, flagged_by_source = recipe_counts(graph)

        print_summary(
            audit_rows=audit_rows,
            missing_rows=missing_rows,
            reason_counts=reason_counts,
            flagged_by_source=flagged_by_source,
            top_sources=args.top_sources,
        )

        summary = {
            "inputs": {
                "graph_in": str(args.graph_in),
                "report_csv": str(args.report_csv),
            },
            "outputs": {
                "graph_out": None if args.dry_run else str(args.graph_out),
                "summary_json": str(args.summary_json),
                "audit_csv": str(args.audit_csv),
                "missing_csv": str(args.missing_csv),
            },
            "parameters": {
                "quality_flag": args.quality_flag,
                "preserve_existing_quality_flag": args.preserve_existing_quality_flag,
                "strict": args.strict,
                "dry_run": args.dry_run,
            },
            "total_recipes_in_graph": int(total_recipes),
            "report_rows": int(len(report_df)),
            "recipes_flagged": int(sum(1 for row in audit_rows if row["action"] == "flagged")),
            "recipes_skipped_existing_quality_flag": int(
                sum(1 for row in audit_rows if row["action"] == "skipped_existing_quality_flag")
            ),
            "missing_from_graph": int(len(missing_rows)),
            "quality_flag_counts": {str(key): int(value) for key, value in quality_counts.items()},
            "reasons": {str(key): int(value) for key, value in reason_counts.items()},
            "by_source": {str(key): int(value) for key, value in flagged_by_source.items()},
            "downstream_filter": "G.nodes[n].get('quality_flag') != 'upstream_corrupt'",
        }

        if args.dry_run and not args.write_dry_run_outputs:
            log.info("[dry-run] No files written.")
            print(json.dumps({
                "recipes_flagged": summary["recipes_flagged"],
                "missing_from_graph": summary["missing_from_graph"],
                "reasons": summary["reasons"],
            }, indent=2, ensure_ascii=False))
            return 1 if args.strict and missing_rows else 0

        audit_fields = [
            "node_id",
            "index",
            "source_id",
            "title",
            "flags",
            "action",
            "previous_quality_flag",
            "previous_corrupt_reason",
        ]
        missing_fields = [
            "index",
            "source_id",
            "title",
            "flags",
            "candidate_node_id",
        ]

        write_csv(args.audit_csv, audit_rows, fieldnames=audit_fields)
        write_csv(args.missing_csv, missing_rows, fieldnames=missing_fields)
        write_json(args.summary_json, summary)

        if not args.dry_run:
            write_graph(graph, args.graph_out)

        if args.strict and missing_rows:
            log.error("--strict enabled and %d report rows were not matched to graph", len(missing_rows))
            return 1

        log.info("Done.")
        log.info("Downstream filter:")
        log.info("  G.nodes[n].get('quality_flag') != 'upstream_corrupt'")
        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        pickle.PickleError,
        json.JSONDecodeError,
        OSError,
        pd.errors.ParserError,
        csv.Error,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
