"""

====================

Dump one Recipe node that has all structural edge types needed for a complete
diagram:

  - contains    -> Ingredient
  - uses_tool   -> Tool
  - performs    -> Action
  - origin      -> Place
  - dated       -> Period

The script prints the recipe attributes, its outgoing neighborhood grouped by
node type, and the recipes sharing the most ingredients with the selected recipe.

Default input
-------------

  data/graph_step2_canonical.gpickle

Usage
-----

  python dump_full_example.py

  python dump_full_example.py --graph data/graph_step2_canonical.gpickle

  python dump_full_example.py --recipe-id recipe::a_miscellany::00029

  python dump_full_example.py --allow-partial

  python dump_full_example.py --out-json data/full_example_report.json


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

import networkx as nx


DEFAULT_GRAPH = Path("data/graph_step2_canonical.gpickle")

REQUIRED_EDGE_TYPES = {"contains", "uses_tool", "performs", "origin", "dated"}
NODE_TYPE_ORDER = ["Ingredient", "Tool", "Action", "Place", "Period"]

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("dump_full_example")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: Path) -> nx.Graph:
    """Load a trusted local NetworkX graph.

    Do not use this function with untrusted pickle/gpickle files.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_str(value: Any, default: str = "?") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def node_label(node_id: str, attrs: dict[str, Any]) -> str:
    return safe_str(
        attrs.get("canonical_name")
        or attrs.get("canonical_verb")
        or attrs.get("label")
        or attrs.get("title")
        or node_id
    )


def format_confidence(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"conf={float(value):.2f}"
    except (TypeError, ValueError):
        return f"conf={value}"


def grouped_out_neighbors(graph: nx.Graph, recipe_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return outgoing recipe neighbors grouped by target node_type."""
    groups: dict[str, list[dict[str, Any]]] = {node_type: [] for node_type in NODE_TYPE_ORDER}

    for _, target, edge_attrs in graph.out_edges(recipe_id, data=True):
        target_attrs = graph.nodes.get(target, {})
        node_type = target_attrs.get("node_type")
        if node_type not in groups:
            continue

        groups[node_type].append({
            "node_id": str(target),
            "node_type": node_type,
            "label": node_label(str(target), target_attrs),
            "n_occurrences": target_attrs.get("n_occurrences"),
            "edge_type": edge_attrs.get("edge_type"),
            "confidence_score": edge_attrs.get("confidence_score"),
            "specific_forms_used": edge_attrs.get("specific_forms_used") or [],
            "edge_attrs": dict(edge_attrs),
        })

    for values in groups.values():
        values.sort(key=lambda row: (safe_str(row.get("edge_type")), safe_str(row.get("label"))))

    return groups


def edge_types_present(graph: nx.Graph, recipe_id: str) -> set[str]:
    return {
        str(edge_attrs.get("edge_type"))
        for _, _, edge_attrs in graph.out_edges(recipe_id, data=True)
        if edge_attrs.get("edge_type")
    }


def recipe_ingredient_set(graph: nx.Graph, recipe_id: str) -> set[str]:
    ingredients: set[str] = set()

    for _, target, edge_attrs in graph.out_edges(recipe_id, data=True):
        if edge_attrs.get("edge_type") == "contains":
            if graph.nodes.get(target, {}).get("node_type") == "Ingredient":
                ingredients.add(str(target))

    return ingredients


def recipe_summary_attrs(graph: nx.Graph, recipe_id: str) -> dict[str, Any]:
    attrs = graph.nodes[recipe_id]
    keys = [
        "title",
        "source_id",
        "source_title",
        "source_author",
        "source_year",
        "source_place",
        "source_language",
        "period_derived",
    ]
    return {key: attrs.get(key) for key in keys}


def score_recipe(graph: nx.Graph, recipe_id: str) -> float:
    """Score candidate recipes.

    Primary criterion: number of required edge types present.
    Secondary criterion: richer diagrams, especially more ingredients, tools,
    and actions.
    """
    present = edge_types_present(graph, recipe_id)
    groups = grouped_out_neighbors(graph, recipe_id)

    score = 100 * len(present & REQUIRED_EDGE_TYPES)
    score += min(len(groups["Ingredient"]), 50) * 0.20
    score += min(len(groups["Tool"]), 10) * 0.75
    score += min(len(groups["Action"]), 20) * 0.50
    return score


def find_best_recipe(graph: nx.Graph, allow_partial: bool = False) -> str | None:
    """Find the best Recipe node for a full diagram."""
    candidates: list[str] = []

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Recipe":
            continue

        present = edge_types_present(graph, str(node_id))
        has_all_required = REQUIRED_EDGE_TYPES <= present

        if has_all_required or allow_partial:
            candidates.append(str(node_id))

    if not candidates:
        return None

    candidates.sort(key=lambda rid: (-score_recipe(graph, rid), rid))
    return candidates[0]


def ingredient_overlap_report(
    graph: nx.Graph,
    recipe_id: str,
    *,
    top_n: int,
    shared_preview_limit: int,
) -> list[dict[str, Any]]:
    """Return recipes sharing the most ingredients with recipe_id."""
    my_ings = recipe_ingredient_set(graph, recipe_id)
    overlap_counts: Counter[str] = Counter()

    for ing_id in my_ings:
        if not graph.has_node(ing_id):
            continue
        for pred in graph.predecessors(ing_id):
            pred_attrs = graph.nodes.get(pred, {})
            if pred_attrs.get("node_type") == "Recipe" and str(pred) != recipe_id:
                overlap_counts[str(pred)] += 1

    rows: list[dict[str, Any]] = []
    for other_recipe_id, count in overlap_counts.most_common(top_n):
        attrs = graph.nodes[other_recipe_id]
        other_ings = recipe_ingredient_set(graph, other_recipe_id)
        shared = sorted(my_ings & other_ings)

        rows.append({
            "recipe_id": other_recipe_id,
            "title": attrs.get("title"),
            "source_place": attrs.get("source_place"),
            "source_year": attrs.get("source_year"),
            "period_derived": attrs.get("period_derived"),
            "n_shared": count,
            "n_query_ingredients": len(my_ings),
            "shared_ingredients": [x.replace("ing::", "") for x in shared],
            "shared_preview": [x.replace("ing::", "") for x in shared[:shared_preview_limit]],
            "n_shared_omitted_from_preview": max(0, len(shared) - shared_preview_limit),
        })

    return rows


def build_report(
    graph: nx.Graph,
    recipe_id: str,
    *,
    top_related: int,
    max_neighbors: int,
    shared_preview_limit: int,
) -> dict[str, Any]:
    if not graph.has_node(recipe_id):
        raise ValueError(f"Recipe node not found: {recipe_id}")

    attrs = graph.nodes[recipe_id]
    if attrs.get("node_type") != "Recipe":
        raise ValueError(f"Node exists but is not a Recipe: {recipe_id}")

    groups = grouped_out_neighbors(graph, recipe_id)
    present = edge_types_present(graph, recipe_id)
    missing = sorted(REQUIRED_EDGE_TYPES - present)

    displayed_groups: dict[str, list[dict[str, Any]]] = {}
    omitted_counts: dict[str, int] = {}

    for node_type, items in groups.items():
        if max_neighbors < 0:
            displayed_groups[node_type] = items
            omitted_counts[node_type] = 0
        else:
            displayed_groups[node_type] = items[:max_neighbors]
            omitted_counts[node_type] = max(0, len(items) - max_neighbors)

    return {
        "recipe_id": recipe_id,
        "recipe_attributes": recipe_summary_attrs(graph, recipe_id),
        "edge_types_present": sorted(present),
        "required_edge_types": sorted(REQUIRED_EDGE_TYPES),
        "missing_required_edge_types": missing,
        "has_all_required_edge_types": not missing,
        "neighbors": displayed_groups,
        "neighbor_counts": {node_type: len(items) for node_type, items in groups.items()},
        "neighbor_omitted_counts": omitted_counts,
        "related_recipes_by_ingredient_overlap": ingredient_overlap_report(
            graph,
            recipe_id,
            top_n=top_related,
            shared_preview_limit=shared_preview_limit,
        ),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(report: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"RECIPE NODE: {report['recipe_id']}")
    print("=" * 78)

    for key, value in report["recipe_attributes"].items():
        print(f"  {key}: {value}")

    print(f"  edge_types present: {report['edge_types_present']}")
    if report["missing_required_edge_types"]:
        print(f"  missing required edge_types: {report['missing_required_edge_types']}")

    for node_type in NODE_TYPE_ORDER:
        items = report["neighbors"].get(node_type, [])
        omitted = report["neighbor_omitted_counts"].get(node_type, 0)
        total = report["neighbor_counts"].get(node_type, len(items))

        print(f"\n--- {node_type} ({total}) ---")
        for item in items:
            extras: list[str] = []

            conf = format_confidence(item.get("confidence_score"))
            if conf:
                extras.append(conf)

            forms = item.get("specific_forms_used") or []
            if forms:
                extras.append(f"forms={forms[:2]}")

            n_occ = item.get("n_occurrences")
            if n_occ not in (None, ""):
                extras.append(f"n_occ={n_occ}")

            extra_text = f"  {'  '.join(extras)}" if extras else ""
            print(
                f"  {safe_str(item.get('edge_type')):12s} → "
                f"{safe_str(item.get('node_id')):45s} "
                f"({safe_str(item.get('label'))}){extra_text}"
            )

        if omitted:
            print(f"  ... +{omitted} more")

    print("\n--- RECIPES SHARING MOST INGREDIENTS ---")
    related = report["related_recipes_by_ingredient_overlap"]
    query_n = report["neighbor_counts"].get("Ingredient", 0)

    if not related:
        print("  No other Recipe nodes share ingredients with this recipe.")
    else:
        for row in related:
            print(
                f"  {row['n_shared']}/{query_n} shared with "
                f"{safe_str(row.get('title')):40s} "
                f"({safe_str(row.get('source_place'))}, {safe_str(row.get('source_year'))})"
            )
            print(f"    recipe_id: {row['recipe_id']}")
            preview = ", ".join(row["shared_preview"])
            print(f"    shared: {preview}")
            if row["n_shared_omitted_from_preview"]:
                print(f"    ... +{row['n_shared_omitted_from_preview']} more")

    print("\nDone.")


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON report: %s", path)


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
        "--recipe-id",
        help="Specific Recipe node id to dump. If omitted, the best full example is selected.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="If no recipe has all required edge types, allow the best partial recipe.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=15,
        help="Maximum neighbors per node type to print/include. Use -1 for all.",
    )
    parser.add_argument(
        "--top-related",
        type=int,
        default=5,
        help="Number of related recipes by shared ingredients to show.",
    )
    parser.add_argument(
        "--shared-preview-limit",
        type=int,
        default=8,
        help="Maximum shared ingredient labels printed per related recipe.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Optional path to write a machine-readable JSON report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the human-readable report; useful with --out-json.",
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

    if args.max_neighbors < -1:
        log.error("--max-neighbors must be >= -1")
        return 2
    if args.top_related < 0:
        log.error("--top-related must be >= 0")
        return 2
    if args.shared_preview_limit < 1:
        log.error("--shared-preview-limit must be >= 1")
        return 2

    try:
        graph = load_graph(args.graph)

        if args.recipe_id:
            selected_recipe_id = args.recipe_id
        else:
            selected_recipe_id = find_best_recipe(graph, allow_partial=args.allow_partial)
            if selected_recipe_id is None:
                log.error(
                    "No Recipe node found with required edge types %s. "
                    "Use --allow-partial to dump the best partial example.",
                    sorted(REQUIRED_EDGE_TYPES),
                )
                return 1

        report = build_report(
            graph,
            selected_recipe_id,
            top_related=args.top_related,
            max_neighbors=args.max_neighbors,
            shared_preview_limit=args.shared_preview_limit,
        )

        if report["missing_required_edge_types"] and not args.allow_partial and not args.recipe_id:
            log.error("Selected recipe is missing required edge types unexpectedly.")
            return 1

        if report["missing_required_edge_types"] and args.recipe_id and not args.allow_partial:
            log.warning(
                "Requested recipe is missing required edge types: %s",
                report["missing_required_edge_types"],
            )

        if args.out_json:
            write_json(args.out_json, report)

        if not args.quiet:
            print_report(report)

        return 0

    except (FileNotFoundError, TypeError, ValueError, pickle.PickleError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
