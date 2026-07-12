"""
dump_cluster_examples.py
========================

Dump raw cluster contents without interpretation.

For each cluster, the script writes:

  - distribution of source_id
  - distribution of source_language
  - distribution of period_derived
  - top N example recipes with title, source, period, language, year, and quality flag

The output is factual only. It does not infer traditions, cuisines, labels, or
semantic meanings for clusters.

Reads
-----

  data/graph_step3_flagged.gpickle

The graph must contain Recipe nodes with a cluster_id attribute.

Writes
------

  data/cluster_examples.txt
      Human-readable per-cluster report.

  data/cluster_examples.csv
      One row per example recipe.

Optional:

  data/cluster_examples_summary.json
      Machine-readable summary if --out-json is supplied.

Usage
-----

  python dump_cluster_examples.py

  python dump_cluster_examples.py --n-examples 15

  python dump_cluster_examples.py \
    --graph data/graph_step3_flagged.gpickle \
    --out-txt data/cluster_examples.txt \
    --out-csv data/cluster_examples.csv \
    --out-json data/cluster_examples_summary.json

"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_GRAPH = Path("data/graph_step3_flagged.gpickle")
DEFAULT_OUT_TXT = Path("data/cluster_examples.txt")
DEFAULT_OUT_CSV = Path("data/cluster_examples.csv")

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("dump_cluster_examples")


# ---------------------------------------------------------------------------
# Loading and extraction
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> Any:
    """Load a trusted local NetworkX graph.

    Do not use with untrusted pickle/gpickle files.
    """
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    log.info("Loading graph: %s", path)
    with path.open("rb") as fh:
        graph = pickle.load(fh)

    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError(f"Object loaded from {path} does not look like a NetworkX graph")

    log.info("Graph loaded: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_cluster_id(value: Any) -> int | str | None:
    """Return a stable cluster id.

    Numeric cluster IDs are returned as int; non-numeric labels are kept as
    strings. Missing values return None.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    try:
        text = str(value).strip()
        if not text:
            return None
        as_float = float(text)
        if as_float.is_integer():
            return int(as_float)
    except (TypeError, ValueError):
        pass

    return str(value)


def sort_cluster_ids(values: list[int | str]) -> list[int | str]:
    """Sort numeric clusters before string clusters, deterministically."""
    return sorted(values, key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)))


def recipe_rows_from_graph(graph: Any, include_unclustered: bool = False) -> pd.DataFrame:
    """Extract one factual row per Recipe node."""
    rows: list[dict[str, Any]] = []
    recipe_count = 0
    skipped_unclustered = 0

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Recipe":
            continue

        recipe_count += 1
        cluster = normalize_cluster_id(attrs.get("cluster_id"))

        if cluster is None and not include_unclustered:
            skipped_unclustered += 1
            continue

        year = (
            attrs.get("source_year")
            if attrs.get("source_year") is not None
            else attrs.get("year")
        )

        rows.append({
            "recipe_id": str(node_id),
            "cluster_id": cluster if cluster is not None else "UNCLUSTERED",
            "quality_flag": safe_str(attrs.get("clustering_quality_flag"), "unknown"),
            "title": safe_str(attrs.get("title")),
            "source_id": safe_str(attrs.get("source_id")),
            "source_title": safe_str(attrs.get("source_title")),
            "source_author": safe_str(attrs.get("source_author")),
            "source_language": safe_str(attrs.get("source_language")),
            "period_derived": safe_str(attrs.get("period_derived")),
            "source_place": safe_str(attrs.get("source_place")),
            "year": safe_str(year),
        })

    log.info("Recipe nodes found: %d", recipe_count)
    log.info("Recipe nodes used: %d", len(rows))
    if skipped_unclustered:
        log.warning("Skipped %d Recipe nodes without cluster_id", skipped_unclustered)

    if not rows:
        return pd.DataFrame(columns=[
            "recipe_id",
            "cluster_id",
            "quality_flag",
            "title",
            "source_id",
            "source_title",
            "source_author",
            "source_language",
            "period_derived",
            "source_place",
            "year",
        ])

    df = pd.DataFrame(rows)

    # Deterministic ordering. This avoids run-to-run drift caused by graph
    # insertion order.
    df = df.sort_values(
        by=["cluster_id", "source_id", "period_derived", "source_language", "recipe_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Summaries and examples
# ---------------------------------------------------------------------------

def value_counts_rows(series: pd.Series, top_n: int) -> list[dict[str, Any]]:
    counts = series.fillna("").astype(str).replace("", "(missing)").value_counts()
    total = int(counts.sum())

    rows: list[dict[str, Any]] = []
    for value, count in counts.head(top_n).items():
        rows.append({
            "value": value,
            "count": int(count),
            "percent": round(100 * int(count) / total, 1) if total else 0.0,
        })
    return rows


def choose_diverse_examples(sub: pd.DataFrame, n_examples: int) -> pd.DataFrame:
    """Pick examples while avoiding domination by one source_id.

    The selection is deterministic and factual. It does not attempt to label or
    interpret cluster contents.
    """
    if sub.empty:
        return sub

    n_sources = max(1, sub["source_id"].nunique())
    max_per_source = max(2, n_examples // n_sources)

    selected_indices: list[int] = []
    seen_sources: Counter[str] = Counter()

    # First pass: diversify by source_id.
    for idx, row in sub.iterrows():
        source_id = str(row["source_id"])
        if seen_sources[source_id] >= max_per_source:
            continue
        selected_indices.append(idx)
        seen_sources[source_id] += 1
        if len(selected_indices) >= n_examples:
            break

    # Second pass: fill remaining slots.
    if len(selected_indices) < n_examples:
        selected_set = set(selected_indices)
        for idx, _ in sub.iterrows():
            if idx in selected_set:
                continue
            selected_indices.append(idx)
            if len(selected_indices) >= n_examples:
                break

    return sub.loc[selected_indices].copy()


def build_cluster_reports(
    df: pd.DataFrame,
    *,
    n_examples: int,
    top_sources: int,
    top_languages: int,
    top_periods: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Build text lines, CSV rows, and JSON summary."""
    lines: list[str] = []
    csv_rows: list[dict[str, Any]] = []
    clusters_json: list[dict[str, Any]] = []

    cluster_ids = sort_cluster_ids(list(df["cluster_id"].dropna().unique()))

    for cluster_id in cluster_ids:
        sub = df[df["cluster_id"] == cluster_id].copy()
        n = len(sub)

        quality_counts = {
            str(k): int(v)
            for k, v in sub["quality_flag"].value_counts(dropna=False).items()
        }

        source_dist = value_counts_rows(sub["source_id"], top_sources)
        language_dist = value_counts_rows(sub["source_language"], top_languages)
        period_dist = value_counts_rows(sub["period_derived"], top_periods)

        examples = choose_diverse_examples(sub, n_examples)

        lines.append("=" * 78)
        lines.append(f"CLUSTER {cluster_id}  ·  n = {n}  ·  quality flags: {quality_counts}")
        lines.append("=" * 78)
        lines.append("")

        lines.append(f"Source distribution (top {top_sources}):")
        for row in source_dist:
            lines.append(f"   {row['count']:4d} ({row['percent']:5.1f}%)  {row['value']}")
        lines.append("")

        lines.append(f"Language distribution (top {top_languages}):")
        for row in language_dist:
            lines.append(f"   {row['count']:4d} ({row['percent']:5.1f}%)  {row['value']}")
        lines.append("")

        lines.append(f"Period distribution (top {top_periods}):")
        for row in period_dist:
            lines.append(f"   {row['count']:4d} ({row['percent']:5.1f}%)  {row['value']}")
        lines.append("")

        lines.append(f"Example recipes (showing {len(examples)} of {n}):")

        example_rows: list[dict[str, Any]] = []
        for example_n, (_, row) in enumerate(examples.iterrows(), start=1):
            title = safe_str(row["title"], "(no title)")
            title_preview = title[:90]

            lines.append(
                f"   {example_n:2d}. "
                f"[{row['source_id']}] "
                f"[{row['source_language']}] "
                f"[{row['period_derived']}] "
                f"[{row['year']}]"
            )
            lines.append(f"       {title_preview}")

            out_row = {
                "cluster_id": cluster_id,
                "example_n": example_n,
                "recipe_id": row["recipe_id"],
                "title": row["title"],
                "source_id": row["source_id"],
                "source_title": row["source_title"],
                "source_author": row["source_author"],
                "source_language": row["source_language"],
                "period_derived": row["period_derived"],
                "source_place": row["source_place"],
                "year": row["year"],
                "quality_flag": row["quality_flag"],
            }
            csv_rows.append(out_row)
            example_rows.append(out_row)

        lines.append("")
        lines.append("")

        clusters_json.append({
            "cluster_id": cluster_id,
            "n_recipes": n,
            "quality_flags": quality_counts,
            "source_distribution": source_dist,
            "language_distribution": language_dist,
            "period_distribution": period_dist,
            "examples": example_rows,
        })

    summary = {
        "n_recipes_with_cluster": int(len(df)),
        "n_clusters": int(len(cluster_ids)),
        "clusters": clusters_json,
    }

    return lines, csv_rows, summary


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote text report: %s (%d lines)", path, len(lines))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote CSV: %s (%d rows)", path, len(rows))


def write_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON summary: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help=f"Path to graph gpickle. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--out-txt",
        type=Path,
        default=DEFAULT_OUT_TXT,
        help=f"Human-readable report path. Default: {DEFAULT_OUT_TXT}",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"CSV examples path. Default: {DEFAULT_OUT_CSV}",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Optional machine-readable JSON summary path.",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Number of example recipes per cluster.",
    )
    parser.add_argument(
        "--top-sources",
        type=int,
        default=8,
        help="Number of source_id values shown per cluster.",
    )
    parser.add_argument(
        "--top-languages",
        type=int,
        default=5,
        help="Number of source_language values shown per cluster.",
    )
    parser.add_argument(
        "--top-periods",
        type=int,
        default=5,
        help="Number of period_derived values shown per cluster.",
    )
    parser.add_argument(
        "--include-unclustered",
        action="store_true",
        help="Include Recipe nodes without cluster_id under cluster label UNCLUSTERED.",
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

    if args.n_examples < 1:
        log.error("--n-examples must be >= 1")
        return 2
    if args.top_sources < 1:
        log.error("--top-sources must be >= 1")
        return 2
    if args.top_languages < 1:
        log.error("--top-languages must be >= 1")
        return 2
    if args.top_periods < 1:
        log.error("--top-periods must be >= 1")
        return 2

    try:
        graph = load_graph(args.graph)
        df = recipe_rows_from_graph(graph, include_unclustered=args.include_unclustered)

        if df.empty:
            log.error("No Recipe nodes with cluster_id found in graph.")
            return 1

        n_clusters = df["cluster_id"].nunique()
        log.info("Total recipes in report: %d", len(df))
        log.info("Total clusters in report: %d", n_clusters)

        lines, csv_rows, summary = build_cluster_reports(
            df,
            n_examples=args.n_examples,
            top_sources=args.top_sources,
            top_languages=args.top_languages,
            top_periods=args.top_periods,
        )

        write_text(args.out_txt, lines)
        write_csv(args.out_csv, csv_rows)

        if args.out_json:
            write_json(args.out_json, summary)

        return 0

    except (FileNotFoundError, TypeError, ValueError, pickle.PickleError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
