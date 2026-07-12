"""
======================

Identify and re-type nutritional-metadata artifacts in the RELISH Ingredient
layer.

Some upstream NER passes may turn nutritional-table rows into Ingredient nodes,
for example:

  "0.33 cholesterol"
  "1.4 carbohydrate"
  "23 sodium"

These are not culinary ingredients. This script detects such nodes using their
surface forms and re-types them as NutritionalMetadata.

The graph is not destructively edited:

  - the node is preserved
  - all edges are preserved
  - original_node_type="Ingredient" is added
  - node_type is changed to "NutritionalMetadata"
  - contamination_reason records the diagnostic evidence

Downstream culinary analyses should filter with:

  node_type == "Ingredient"

Default input
-------------

  data/graph_step4_foodon.gpickle

Default outputs
---------------

  data/graph_step5_audited.gpickle
  data/contamination_audit.csv
  data/contamination_summary.json

Usage
-----

  python step5_quality_audit.py

  python step5_quality_audit.py --ratio-threshold 0.5 --min-forms 3

  python step5_quality_audit.py --dry-run

  python step5_quality_audit.py --write-dry-run-outputs

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_IN_GRAPH = Path("data/graph_step4_foodon.gpickle")
DEFAULT_OUT_GRAPH = Path("data/graph_step5_audited.gpickle")
DEFAULT_AUDIT_CSV = Path("data/contamination_audit.csv")
DEFAULT_SUMMARY_JSON = Path("data/contamination_summary.json")

DEFAULT_RATIO_THRESHOLD = 0.5
DEFAULT_MIN_FORMS = 3

# Examples matched:
#   0.33 cholesterol
#   1.4 carbohydrate
#   23 sodium
#   250 mg sodium
DEFAULT_NUTRI_PATTERN = r"^\d+(?:\.\d+)?\s+(?:[a-zA-Z_][\w-]*)(?:\s+[a-zA-Z_][\w-]*)?$"

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step5_quality_audit")


# ---------------------------------------------------------------------------
# Loading and normalization
# ---------------------------------------------------------------------------

def load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input graph not found: {path}")

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
    if text.lower() in {"nan", "none", "null"}:
        return default
    return text if text else default


def as_surface_forms(value: Any) -> list[str]:
    """Normalize a node's surface_forms/raw_variants/aliases field to strings."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, (list, tuple, set)):
        return [safe_str(item) for item in value if safe_str(item)]

    return [safe_str(value)] if safe_str(value) else []


def collect_surface_forms(attrs: dict[str, Any]) -> list[str]:
    forms: list[str] = []

    for key in ["surface_forms", "raw_variants", "specific_forms", "aliases"]:
        forms.extend(as_surface_forms(attrs.get(key)))

    # Include canonical/name/label as a weak fallback, but do not let a single
    # canonical name satisfy min_forms on its own.
    for key in ["canonical_name", "name", "label"]:
        value = safe_str(attrs.get(key))
        if value:
            forms.append(value)

    seen: set[str] = set()
    unique: list[str] = []
    for form in forms:
        normalized = re.sub(r"\s+", " ", form).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)

    return unique


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_node(
    attrs: dict[str, Any],
    *,
    pattern: re.Pattern[str],
    ratio_threshold: float,
    min_forms: int,
) -> tuple[bool, float, int, int, list[str]]:
    """Return contaminated?, ratio, matched count, total forms, examples."""
    forms = collect_surface_forms(attrs)

    if len(forms) < min_forms:
        return False, 0.0, 0, len(forms), []

    matched = [form for form in forms if pattern.match(form.strip())]
    ratio = len(matched) / len(forms)

    return ratio >= ratio_threshold, ratio, len(matched), len(forms), matched[:10]


def node_type_counts(graph: Any) -> dict[str, int]:
    counts = Counter()
    for _, attrs in graph.nodes(data=True):
        counts[str(attrs.get("node_type", "Unknown"))] += 1
    return dict(counts)


def edge_type_counts(graph: Any) -> dict[str, int]:
    counts = Counter()
    for _, _, attrs in graph.edges(data=True):
        counts[str(attrs.get("edge_type", "Unknown"))] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Audit and mutation
# ---------------------------------------------------------------------------

def audit_contamination(
    graph: Any,
    *,
    pattern: re.Pattern[str],
    ratio_threshold: float,
    min_forms: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    audit_rows: list[dict[str, Any]] = []
    retyped_nodes: list[str] = []

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Ingredient":
            continue

        is_contaminated, ratio, n_nutri, n_total, examples = classify_node(
            attrs,
            pattern=pattern,
            ratio_threshold=ratio_threshold,
            min_forms=min_forms,
        )

        if not is_contaminated:
            continue

        node_id_str = str(node_id)
        retyped_nodes.append(node_id_str)

        audit_rows.append({
            "node_id": node_id_str,
            "canonical_name": safe_str(
                attrs.get("canonical_name")
                or attrs.get("name")
                or attrs.get("label")
            ),
            "n_occurrences": attrs.get("n_occurrences", 0),
            "n_forms_total": n_total,
            "n_forms_nutri": n_nutri,
            "ratio_nutri": round(ratio, 6),
            "n_edges_affected": int(graph.degree(node_id)),
            "matched_examples": " | ".join(examples),
        })

    audit_rows.sort(
        key=lambda row: (
            -int(row.get("n_occurrences") or 0),
            -float(row.get("ratio_nutri") or 0),
            row["node_id"],
        )
    )

    return audit_rows, retyped_nodes


def apply_retyping(
    graph: Any,
    *,
    retyped_nodes: list[str],
    audit_rows_by_node: dict[str, dict[str, Any]],
) -> None:
    for node_id in retyped_nodes:
        if node_id not in graph:
            continue

        attrs = graph.nodes[node_id]
        row = audit_rows_by_node.get(node_id, {})

        attrs["original_node_type"] = attrs.get("node_type", "Ingredient")
        attrs["node_type"] = "NutritionalMetadata"
        attrs["contamination_reason"] = (
            f"surface_forms_nutri_ratio={float(row.get('ratio_nutri', 0.0)):.6f} "
            f"({int(row.get('n_forms_nutri', 0))}/{int(row.get('n_forms_total', 0))} "
            f"forms matched nutritional-table pattern)"
        )


def corpus_impact(
    graph: Any,
    *,
    retyped_nodes: set[str],
) -> tuple[Counter[str], Counter[str], int]:
    contaminated_by_source: Counter[str] = Counter()
    total_by_source: Counter[str] = Counter()

    for source, target, edge_attrs in graph.edges(data=True):
        if edge_attrs.get("edge_type") != "contains":
            continue
        if graph.nodes.get(source, {}).get("node_type") != "Recipe":
            continue

        source_id = safe_str(graph.nodes[source].get("source_id"), "unknown")
        total_by_source[source_id] += 1

        if str(target) in retyped_nodes:
            contaminated_by_source[source_id] += 1

    affected_edges = sum(contaminated_by_source.values())
    return contaminated_by_source, total_by_source, affected_edges


def build_summary(
    *,
    args: argparse.Namespace,
    graph_before_nodes: int,
    graph_before_edges: int,
    node_types_before: dict[str, int],
    edge_types_before: dict[str, int],
    graph: Any,
    audit_rows: list[dict[str, Any]],
    retyped_nodes: list[str],
    contaminated_by_source: Counter[str],
    total_by_source: Counter[str],
    affected_edges: int,
) -> dict[str, Any]:
    retyped_set = set(retyped_nodes)

    ingredients_before = node_types_before.get("Ingredient", 0)
    ingredients_after = node_type_counts(graph).get("Ingredient", 0)

    return {
        "inputs": {
            "graph_in": str(args.graph_in),
        },
        "outputs": {
            "graph_out": None if args.dry_run else str(args.graph_out),
            "audit_csv": str(args.audit_csv),
            "summary_json": str(args.summary_json),
        },
        "parameters": {
            "pattern": args.pattern,
            "ratio_threshold": args.ratio_threshold,
            "min_forms": args.min_forms,
            "dry_run": args.dry_run,
        },
        "before": {
            "nodes": int(graph_before_nodes),
            "edges": int(graph_before_edges),
            "node_types": node_types_before,
            "edge_types": edge_types_before,
            "ingredient_nodes": int(ingredients_before),
        },
        "after": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "node_types": node_type_counts(graph),
            "edge_types": edge_type_counts(graph),
            "ingredient_nodes": int(ingredients_after),
            "nutritional_metadata_nodes": int(node_type_counts(graph).get("NutritionalMetadata", 0)),
        },
        "totals": {
            "candidate_contaminated_nodes": int(len(audit_rows)),
            "retyped_nodes": int(len(retyped_nodes)),
            "recipe_contains_edges_affected": int(affected_edges),
            "ingredient_node_reduction": int(ingredients_before - ingredients_after),
            "pct_ingredient_nodes_retyped": round(
                100 * len(retyped_set) / max(1, ingredients_before),
                4,
            ),
        },
        "by_source": {
            source_id: {
                "edges_contaminated": int(contaminated_by_source[source_id]),
                "edges_total": int(total_by_source[source_id]),
                "pct_contaminated": round(
                    100 * contaminated_by_source[source_id] / max(1, total_by_source[source_id]),
                    4,
                ),
            }
            for source_id in sorted(
                contaminated_by_source,
                key=lambda item: (-contaminated_by_source[item], item),
            )
        },
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_graph(graph: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Wrote graph: %s", path)


def write_audit_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "node_id",
        "canonical_name",
        "n_occurrences",
        "n_forms_total",
        "n_forms_nutri",
        "ratio_nutri",
        "n_edges_affected",
        "matched_examples",
    ]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote audit CSV: %s (%d rows)", path, len(rows))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    log.info("Wrote JSON: %s", path)


def print_top_sources(contaminated_by_source: Counter[str], total_by_source: Counter[str], *, limit: int) -> None:
    if not contaminated_by_source:
        return

    log.info("Top %d corpora by contaminated Recipe->NutritionalMetadata edges:", limit)

    for source_id, n_contaminated in contaminated_by_source.most_common(limit):
        total = total_by_source[source_id]
        pct = 100 * n_contaminated / max(1, total)
        log.info("  %-35s %5d / %5d (%6.2f%%)", source_id, n_contaminated, total, pct)


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
        default=DEFAULT_IN_GRAPH,
        help=f"Input graph. Default: {DEFAULT_IN_GRAPH}",
    )
    parser.add_argument(
        "--graph-out",
        type=Path,
        default=DEFAULT_OUT_GRAPH,
        help=f"Output audited graph. Default: {DEFAULT_OUT_GRAPH}",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=DEFAULT_AUDIT_CSV,
        help=f"Per-node audit CSV. Default: {DEFAULT_AUDIT_CSV}",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help=f"Summary JSON. Default: {DEFAULT_SUMMARY_JSON}",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_NUTRI_PATTERN,
        help="Regex pattern used to detect nutritional-table surface forms.",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=DEFAULT_RATIO_THRESHOLD,
        help="Minimum matched-form ratio required to re-type a node.",
    )
    parser.add_argument(
        "--min-forms",
        type=int,
        default=DEFAULT_MIN_FORMS,
        help="Minimum number of surface forms required for classification.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute audit and stats but do not write graph.",
    )
    parser.add_argument(
        "--write-dry-run-outputs",
        action="store_true",
        help="With --dry-run, still write audit CSV and summary JSON.",
    )
    parser.add_argument(
        "--allow-zero",
        action="store_true",
        help="Return success even when no contaminated nodes are detected. Default now also returns success.",
    )
    parser.add_argument(
        "--top-sources",
        type=int,
        default=10,
        help="Number of top affected sources to print.",
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

    if not (0 <= args.ratio_threshold <= 1):
        log.error("--ratio-threshold must be in [0, 1]")
        return 2
    if args.min_forms < 1:
        log.error("--min-forms must be >= 1")
        return 2
    if args.top_sources < 0:
        log.error("--top-sources must be >= 0")
        return 2

    try:
        pattern = re.compile(args.pattern, re.IGNORECASE)

        graph = load_graph(args.graph_in)
        graph_before_nodes = graph.number_of_nodes()
        graph_before_edges = graph.number_of_edges()
        node_types_before = node_type_counts(graph)
        edge_types_before = edge_type_counts(graph)

        audit_rows, retyped_nodes = audit_contamination(
            graph,
            pattern=pattern,
            ratio_threshold=args.ratio_threshold,
            min_forms=args.min_forms,
        )

        log.info("Detected %d candidate contaminated Ingredient nodes", len(retyped_nodes))

        audit_rows_by_node = {row["node_id"]: row for row in audit_rows}

        if not args.dry_run:
            apply_retyping(
                graph,
                retyped_nodes=retyped_nodes,
                audit_rows_by_node=audit_rows_by_node,
            )
        else:
            # For dry-run stats, simulate node-type counts by applying to an
            # in-memory copy would be expensive for large graphs. Instead, keep
            # graph unchanged and report candidate counts explicitly.
            log.info("[dry-run] Graph node types are not mutated in memory.")

        contaminated_by_source, total_by_source, affected_edges = corpus_impact(
            graph,
            retyped_nodes=set(retyped_nodes),
        )

        summary = build_summary(
            args=args,
            graph_before_nodes=graph_before_nodes,
            graph_before_edges=graph_before_edges,
            node_types_before=node_types_before,
            edge_types_before=edge_types_before,
            graph=graph,
            audit_rows=audit_rows,
            retyped_nodes=retyped_nodes,
            contaminated_by_source=contaminated_by_source,
            total_by_source=total_by_source,
            affected_edges=affected_edges,
        )

        log.info("Recipe->NutritionalMetadata affected edges: %d", affected_edges)
        if args.top_sources:
            print_top_sources(contaminated_by_source, total_by_source, limit=args.top_sources)

        if not retyped_nodes:
            log.warning("No contamination detected with current thresholds.")

        if args.dry_run and not args.write_dry_run_outputs:
            log.info("[dry-run] No outputs written.")
            print(json.dumps({
                "candidate_contaminated_nodes": len(retyped_nodes),
                "affected_edges": affected_edges,
                "parameters": summary["parameters"],
            }, indent=2, ensure_ascii=False))
            return 0

        write_audit_csv(audit_rows, args.audit_csv)
        write_json(args.summary_json, summary)

        if not args.dry_run:
            write_graph(graph, args.graph_out)

        log.info("Done.")
        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        re.error,
        pickle.PickleError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
