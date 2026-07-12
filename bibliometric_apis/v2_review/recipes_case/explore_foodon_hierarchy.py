"""

===========================

Explore the FoodOn hierarchy embedded in the recipe graph.

For each Ingredient node mapped to a FoodOnClass node, the script walks upward
through FoodOn `is_a` edges and reports ancestor paths. It also reports the most
frequently shared FoodOn ancestors across mapped ingredients. These shared
ancestors are ontology-derived semantic groupings.

Default input
-------------

  data/graph_step4_foodon.gpickle

Default output
--------------

  stdout only

Optional outputs
----------------

  --out-txt   Human-readable report
  --out-json  Machine-readable summary

Usage
-----

  python explore_foodon_hierarchy.py

  python explore_foodon_hierarchy.py \
    --graph data/graph_step4_foodon.gpickle \
    --top-ancestors 30 \
    --max-paths-per-ingredient 5

  python explore_foodon_hierarchy.py \
    --out-txt data/foodon_hierarchy_report.txt \
    --out-json data/foodon_hierarchy_summary.json

"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx


DEFAULT_GRAPH = Path("data/graph_step4_foodon.gpickle")

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("explore_foodon_hierarchy")


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


def label_for_node(graph: nx.Graph, node_id: str) -> str:
    attrs = graph.nodes.get(node_id, {})
    return safe_str(
        attrs.get("label")
        or attrs.get("canonical_name")
        or attrs.get("canonical_verb")
        or attrs.get("name")
        or node_id
    )


def compact_ingredient_name(graph: nx.Graph, node_id: str) -> str:
    attrs = graph.nodes.get(node_id, {})
    label = (
        attrs.get("canonical_name")
        or attrs.get("name")
        or attrs.get("label")
        or str(node_id).replace("ing::", "")
    )
    return safe_str(label)


def foodon_class_nodes(graph: nx.Graph) -> list[str]:
    return sorted(
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "FoodOnClass"
    )


def mapped_ingredients(graph: nx.Graph) -> dict[str, str]:
    """Return Ingredient node -> mapped FoodOnClass node.

    If an ingredient has multiple mapped_to_foodon edges, the first sorted target
    is used and the rest can still be inspected in the JSON output if needed.
    """
    mapping: dict[str, str] = {}
    multi: dict[str, list[str]] = defaultdict(list)

    for source, target, edge_attrs in graph.edges(data=True):
        if edge_attrs.get("edge_type") != "mapped_to_foodon":
            continue

        source_attrs = graph.nodes.get(source, {})
        target_attrs = graph.nodes.get(target, {})

        if source_attrs.get("node_type") != "Ingredient":
            continue
        if target_attrs.get("node_type") != "FoodOnClass":
            continue

        multi[str(source)].append(str(target))

    for ingredient, targets in multi.items():
        mapping[ingredient] = sorted(targets)[0]

    if any(len(v) > 1 for v in multi.values()):
        log.warning(
            "%d ingredients have multiple mapped_to_foodon targets; using sorted first target",
            sum(1 for v in multi.values() if len(v) > 1),
        )

    return dict(sorted(mapping.items(), key=lambda item: compact_ingredient_name(graph, item[0])))


def foodon_parents(graph: nx.Graph, foodon_node: str) -> list[str]:
    """Return direct FoodOn parents reached by outgoing is_a edges."""
    parents: list[str] = []

    if not graph.has_node(foodon_node):
        return parents

    for _, parent, edge_attrs in graph.out_edges(foodon_node, data=True):
        if edge_attrs.get("edge_type") != "is_a":
            continue
        if graph.nodes.get(parent, {}).get("node_type") == "FoodOnClass":
            parents.append(str(parent))

    return sorted(set(parents), key=lambda node: label_for_node(graph, node))


def ancestor_nodes(graph: nx.Graph, foodon_node: str, max_depth: int) -> list[str]:
    """Return all transitive FoodOn ancestors without duplicates."""
    seen: set[str] = set()
    ordered: list[str] = []
    stack: list[tuple[str, int]] = [(foodon_node, 0)]

    while stack:
        current, depth = stack.pop()

        if max_depth >= 0 and depth >= max_depth:
            continue

        for parent in reversed(foodon_parents(graph, current)):
            if parent in seen:
                continue
            seen.add(parent)
            ordered.append(parent)
            stack.append((parent, depth + 1))

    return ordered


def ancestor_paths(
    graph: nx.Graph,
    leaf_foodon: str,
    *,
    max_depth: int,
    max_paths: int,
) -> list[list[str]]:
    """Return leaf-to-root FoodOn paths.

    FoodOn is a DAG-like ontology and a class may have multiple parents. This
    function therefore returns multiple paths. Cycles are guarded against.
    """
    paths: list[list[str]] = []
    stack: list[tuple[str, list[str]]] = [(leaf_foodon, [leaf_foodon])]

    while stack:
        current, path = stack.pop()

        if max_depth >= 0 and len(path) - 1 >= max_depth:
            paths.append(path)
            continue

        parents = [p for p in foodon_parents(graph, current) if p not in path]

        if not parents:
            paths.append(path)
            continue

        for parent in reversed(parents):
            stack.append((parent, path + [parent]))

        if max_paths > 0 and len(paths) >= max_paths:
            break

    paths.sort(key=lambda p: [label_for_node(graph, n) for n in p])
    if max_paths > 0:
        return paths[:max_paths]
    return paths


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    graph: nx.Graph,
    *,
    top_ancestors: int,
    max_paths_per_ingredient: int,
    max_depth: int,
    max_ingredients_per_ancestor: int,
) -> dict[str, Any]:
    foodon_nodes = foodon_class_nodes(graph)
    ing_to_leaf = mapped_ingredients(graph)

    ancestor_counter: Counter[str] = Counter()
    ancestor_to_ingredients: dict[str, list[str]] = defaultdict(list)
    ingredient_rows: list[dict[str, Any]] = []

    for ingredient_node, leaf_node in ing_to_leaf.items():
        ingredient_name = compact_ingredient_name(graph, ingredient_node)
        leaf_label = label_for_node(graph, leaf_node)

        ancestors = ancestor_nodes(graph, leaf_node, max_depth=max_depth)

        for ancestor in ancestors:
            ancestor_counter[ancestor] += 1
            ancestor_to_ingredients[ancestor].append(ingredient_name)

        paths = ancestor_paths(
            graph,
            leaf_node,
            max_depth=max_depth,
            max_paths=max_paths_per_ingredient,
        )

        ingredient_rows.append({
            "ingredient_node": ingredient_node,
            "ingredient_label": ingredient_name,
            "foodon_leaf_node": leaf_node,
            "foodon_leaf_label": leaf_label,
            "n_ancestors": len(ancestors),
            "ancestors": [
                {
                    "node": ancestor,
                    "label": label_for_node(graph, ancestor),
                }
                for ancestor in ancestors
            ],
            "paths": [
                [
                    {
                        "node": node,
                        "label": label_for_node(graph, node),
                    }
                    for node in path
                ]
                for path in paths
            ],
        })

    top_rows: list[dict[str, Any]] = []
    for ancestor, count in ancestor_counter.most_common(top_ancestors):
        ingredient_names = sorted(set(ancestor_to_ingredients[ancestor]))
        top_rows.append({
            "ancestor_node": ancestor,
            "ancestor_label": label_for_node(graph, ancestor),
            "ingredient_coverage": int(count),
            "ingredients": ingredient_names[:max_ingredients_per_ancestor],
            "n_ingredients_omitted": max(0, len(ingredient_names) - max_ingredients_per_ancestor),
        })

    return {
        "counts": {
            "mapped_ingredients": len(ing_to_leaf),
            "foodon_class_nodes_total": len(foodon_nodes),
            "distinct_ancestors_used": len(ancestor_counter),
            "ancestors_shared_by_2_plus_ingredients": sum(1 for c in ancestor_counter.values() if c >= 2),
            "ancestors_shared_by_5_plus_ingredients": sum(1 for c in ancestor_counter.values() if c >= 5),
        },
        "parameters": {
            "top_ancestors": top_ancestors,
            "max_paths_per_ingredient": max_paths_per_ingredient,
            "max_depth": max_depth,
            "max_ingredients_per_ancestor": max_ingredients_per_ancestor,
        },
        "ingredients": ingredient_rows,
        "top_ancestors": top_rows,
    }


def report_lines(report: dict[str, Any], *, show_paths: bool) -> list[str]:
    lines: list[str] = []

    lines.append("=" * 78)
    lines.append("ANCESTOR PATHS PER MAPPED INGREDIENT")
    lines.append("=" * 78)

    for row in report["ingredients"]:
        lines.append("")
        lines.append(f"  {row['ingredient_label']:30s} -> {row['foodon_leaf_label']}")

        if show_paths:
            for path_i, path in enumerate(row["paths"], start=1):
                labels = [item["label"] for item in path]
                lines.append(f"        path {path_i}: " + "  ->  ".join(labels))
        else:
            for ancestor in row["ancestors"]:
                lines.append(f"        is_a -> {ancestor['label']}")

    lines.append("")
    lines.append("=" * 78)
    lines.append("TOP FoodOn ANCESTORS BY INGREDIENT COVERAGE")
    lines.append("=" * 78)
    lines.append("These are ontology-derived semantic groupings. A class appearing here")
    lines.append("means N mapped ingredients share this ancestor.")
    lines.append("")

    for row in report["top_ancestors"]:
        ingredients = ", ".join(row["ingredients"])
        if row["n_ingredients_omitted"]:
            ingredients += f" (+{row['n_ingredients_omitted']} more)"
        lines.append(f"  {row['ingredient_coverage']:3d} ingredients -> {row['ancestor_label']}")
        lines.append(f"        {ingredients}")

    counts = report["counts"]
    lines.append("")
    lines.append("=" * 78)
    lines.append("SUMMARY")
    lines.append("=" * 78)
    lines.append(f"  Mapped ingredients:                  {counts['mapped_ingredients']}")
    lines.append(f"  FoodOnClass nodes total:             {counts['foodon_class_nodes_total']}")
    lines.append(f"  Distinct ancestors used:             {counts['distinct_ancestors_used']}")
    lines.append(
        "  Ancestors shared by 2+ ingredients: "
        f"{counts['ancestors_shared_by_2_plus_ingredients']}"
    )
    lines.append(
        "  Ancestors shared by 5+ ingredients: "
        f"{counts['ancestors_shared_by_5_plus_ingredients']}"
    )

    return lines


def print_report(report: dict[str, Any], *, show_paths: bool) -> None:
    print("\n".join(report_lines(report, show_paths=show_paths)))


def write_text(path: Path, report: dict[str, Any], *, show_paths: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report_lines(report, show_paths=show_paths)), encoding="utf-8")
    log.info("Wrote text report: %s", path)


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
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
        "--top-ancestors",
        type=int,
        default=20,
        help="Number of shared FoodOn ancestors to display.",
    )
    parser.add_argument(
        "--max-paths-per-ingredient",
        type=int,
        default=3,
        help="Maximum leaf-to-root paths stored/displayed per mapped ingredient. Use 0 for all.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
        help="Maximum is_a depth to walk. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--max-ingredients-per-ancestor",
        type=int,
        default=8,
        help="Maximum ingredient names shown under each top ancestor.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print full leaf-to-root paths instead of a flat ancestor list.",
    )
    parser.add_argument(
        "--out-txt",
        type=Path,
        help="Optional path to write the human-readable report.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Optional path to write a machine-readable JSON summary.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print to stdout; useful with --out-txt or --out-json.",
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

    if args.top_ancestors < 1:
        log.error("--top-ancestors must be >= 1")
        return 2
    if args.max_paths_per_ingredient < 0:
        log.error("--max-paths-per-ingredient must be >= 0")
        return 2
    if args.max_depth < -1:
        log.error("--max-depth must be >= -1")
        return 2
    if args.max_ingredients_per_ancestor < 1:
        log.error("--max-ingredients-per-ancestor must be >= 1")
        return 2

    try:
        graph = load_graph(args.graph)

        report = build_report(
            graph,
            top_ancestors=args.top_ancestors,
            max_paths_per_ingredient=args.max_paths_per_ingredient,
            max_depth=args.max_depth,
            max_ingredients_per_ancestor=args.max_ingredients_per_ancestor,
        )

        if report["counts"]["mapped_ingredients"] == 0:
            log.warning("No Ingredient -> FoodOnClass mapped_to_foodon edges found.")

        if args.out_txt:
            write_text(args.out_txt, report, show_paths=args.show_paths)

        if args.out_json:
            write_json(args.out_json, report)

        if not args.quiet:
            print_report(report, show_paths=args.show_paths)

        return 0

    except (FileNotFoundError, TypeError, ValueError, pickle.PickleError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
