"""
==============

Find the strongest bridge between historical recipe graph recipes and
contemporary Wikibooks recipes.

The script:

1. Loads a trusted NetworkX recipe graph.
2. Loads contemporary Wikibooks recipes.
3. Maps Wikibooks ingredients to Ingredient nodes in the graph.
4. Finds the Wikibooks recipe with the most shared ingredient nodes.
5. Finds the historical graph recipe sharing the most ingredients with that
   Wikibooks recipe.
6. Prints all relevant node types for both sides so the bridge can be drawn
   accurately in a diagram.

Default inputs
--------------

  data/graph_step2_canonical.gpickle
  data/wikibooks_contemporary.json

Usage
-----

  python dump_bridge.py

  python dump_bridge.py \
    --graph data/graph_step2_canonical.gpickle \
    --wikibooks data/wikibooks_contemporary.json \
    --top-wikibooks 5 \
    --top-historical 10

  python dump_bridge.py --out-json data/best_bridge_report.json

"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx


DEFAULT_GRAPH = Path("data/graph_step2_canonical.gpickle")
DEFAULT_WIKIBOOKS = Path("data/wikibooks_contemporary.json")

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("dump_bridge")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Slug and display helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Stable ASCII slug compatible with the publish-ready Step 1 scripts."""
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    s = ascii_text.strip().lower()
    s = _SLUG_RE.sub("_", s)
    return s.strip("_") or "unknown"


def node_label(node_id: str, prefix: str) -> str:
    if node_id.startswith(prefix):
        return node_id[len(prefix):]
    return node_id


def safe_str(value: Any, default: str = "?") -> str:
    if value is None:
        return default
    value_str = str(value).strip()
    return value_str if value_str else default


def ingredient_node_id(name: str) -> str:
    return f"ing::{slugify(name)}"


def edge_type_to_node_type(edge_type: str) -> str | None:
    return {
        "contains": "Ingredient",
        "uses_tool": "Tool",
        "performs": "Action",
        "origin": "Place",
        "dated": "Period",
    }.get(edge_type)


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


def load_wikibooks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Wikibooks JSON not found: {path}")

    log.info("Loading Wikibooks recipes: %s", path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")

    bad = sum(1 for row in data if not isinstance(row, dict))
    if bad:
        raise ValueError(f"Expected recipe objects in {path}; found {bad} non-object records")

    log.info("Loaded %d contemporary recipes", len(data))
    return data


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def ingredient_nodes(graph: nx.Graph) -> set[str]:
    nodes = {
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "Ingredient"
    }
    if not nodes:
        raise ValueError("No Ingredient nodes found in graph")
    return nodes


def extract_wikibooks_ingredient_names(recipe: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in recipe.get("ingredients") or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("canonical_name")
        else:
            name = item
        if name is not None and str(name).strip():
            names.append(str(name).strip())
    return names


def recipe_ingredient_set(graph: nx.Graph, recipe_id: str) -> set[str]:
    ingredients: set[str] = set()

    if not graph.has_node(recipe_id):
        return ingredients

    for _, target, edge_attrs in graph.out_edges(recipe_id, data=True):
        if edge_attrs.get("edge_type") == "contains":
            if graph.nodes.get(target, {}).get("node_type") == "Ingredient":
                ingredients.add(str(target))

    return ingredients


def grouped_out_neighbors(graph: nx.Graph, recipe_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return outgoing recipe neighbors grouped by node_type."""
    groups: dict[str, list[dict[str, Any]]] = {
        "Ingredient": [],
        "Tool": [],
        "Action": [],
        "Place": [],
        "Period": [],
    }

    if not graph.has_node(recipe_id):
        return groups

    for _, target, edge_attrs in graph.out_edges(recipe_id, data=True):
        target_attrs = graph.nodes.get(target, {})
        node_type = target_attrs.get("node_type") or edge_type_to_node_type(
            str(edge_attrs.get("edge_type", ""))
        )

        if node_type not in groups:
            continue

        label = (
            target_attrs.get("canonical_name")
            or target_attrs.get("canonical_verb")
            or target_attrs.get("label")
            or str(target)
        )

        groups[node_type].append({
            "node_id": str(target),
            "label": label,
            "n_occurrences": target_attrs.get("n_occurrences"),
            "edge_type": edge_attrs.get("edge_type"),
            "edge_attrs": dict(edge_attrs),
        })

    for values in groups.values():
        values.sort(key=lambda row: safe_str(row.get("label")))

    return groups


def historical_recipes_sharing(graph: nx.Graph, ingredient_ids: set[str]) -> Counter[str]:
    """Count historical Recipe nodes sharing each ingredient."""
    overlap: Counter[str] = Counter()

    for ing_id in ingredient_ids:
        if not graph.has_node(ing_id):
            continue

        for pred in graph.predecessors(ing_id):
            attrs = graph.nodes.get(pred, {})
            if attrs.get("node_type") == "Recipe":
                overlap[str(pred)] += 1

    return overlap


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_wikibooks_recipes(
    wikibooks: list[dict[str, Any]],
    graph_ingredients: set[str],
) -> list[dict[str, Any]]:
    """Find graph ingredient overlap for every Wikibooks recipe."""
    matches: list[dict[str, Any]] = []

    for index, recipe in enumerate(wikibooks):
        shared: set[str] = set()

        for name in extract_wikibooks_ingredient_names(recipe):
            node_id = ingredient_node_id(name)
            if node_id in graph_ingredients:
                shared.add(node_id)

        if shared:
            matches.append({
                "wikibooks_index": index,
                "recipe": recipe,
                "shared_ingredients": shared,
            })

    matches.sort(
        key=lambda row: (
            -len(row["shared_ingredients"]),
            safe_str(row["recipe"].get("title")),
        )
    )
    return matches


def build_report(
    *,
    graph: nx.Graph,
    wikibooks_matches: list[dict[str, Any]],
    graph_ingredients: set[str],
    top_wikibooks: int,
    top_historical: int,
    neighbor_limit: int,
) -> dict[str, Any]:
    """Build a machine-readable report for the strongest bridge."""
    if not wikibooks_matches:
        return {
            "counts": {
                "graph_ingredients": len(graph_ingredients),
                "wikibooks_matches": 0,
            },
            "top_wikibooks_matches": [],
            "best_bridge": None,
        }

    top_rows: list[dict[str, Any]] = []
    for match in wikibooks_matches[:top_wikibooks]:
        recipe = match["recipe"]
        shared = match["shared_ingredients"]
        top_rows.append({
            "wikibooks_index": match["wikibooks_index"],
            "title": recipe.get("title"),
            "source_place": recipe.get("source_place"),
            "source_year": recipe.get("source_year"),
            "n_shared_ingredients": len(shared),
            "shared_ingredients": sorted(node_label(x, "ing::") for x in shared),
        })

    best_match = wikibooks_matches[0]
    best_wiki = best_match["recipe"]
    best_wiki_ings = set(best_match["shared_ingredients"])

    overlap = historical_recipes_sharing(graph, best_wiki_ings)

    historical_rows: list[dict[str, Any]] = []
    for hist_recipe_id, count in overlap.most_common(top_historical):
        attrs = graph.nodes[hist_recipe_id]
        hist_ings = recipe_ingredient_set(graph, hist_recipe_id)
        shared = best_wiki_ings & hist_ings

        historical_rows.append({
            "recipe_id": hist_recipe_id,
            "title": attrs.get("title"),
            "source_id": attrs.get("source_id"),
            "source_title": attrs.get("source_title"),
            "source_author": attrs.get("source_author"),
            "source_year": attrs.get("source_year"),
            "source_place": attrs.get("source_place"),
            "source_language": attrs.get("source_language"),
            "period_derived": attrs.get("period_derived"),
            "n_shared_ingredients": count,
            "shared_ingredients": sorted(node_label(x, "ing::") for x in shared),
        })

    best_historical: dict[str, Any] | None = None
    if overlap:
        best_hist_id, best_hist_count = overlap.most_common(1)[0]
        attrs = graph.nodes[best_hist_id]
        hist_ings = recipe_ingredient_set(graph, best_hist_id)
        shared = best_wiki_ings & hist_ings
        neighbor_groups = grouped_out_neighbors(graph, best_hist_id)

        if neighbor_limit >= 0:
            neighbor_groups = {
                group: values[:neighbor_limit]
                for group, values in neighbor_groups.items()
            }

        best_historical = {
            "recipe_id": best_hist_id,
            "n_shared_ingredients": best_hist_count,
            "shared_ingredients": sorted(node_label(x, "ing::") for x in shared),
            "attributes": {
                "title": attrs.get("title"),
                "source_id": attrs.get("source_id"),
                "source_title": attrs.get("source_title"),
                "source_author": attrs.get("source_author"),
                "source_year": attrs.get("source_year"),
                "source_place": attrs.get("source_place"),
                "source_language": attrs.get("source_language"),
                "period_derived": attrs.get("period_derived"),
            },
            "neighbors": neighbor_groups,
        }

    wiki_ingredient_statuses: list[dict[str, Any]] = []
    for name in extract_wikibooks_ingredient_names(best_wiki):
        node_id = ingredient_node_id(name)
        node_attrs = graph.nodes.get(node_id, {}) if graph.has_node(node_id) else {}
        wiki_ingredient_statuses.append({
            "name": name,
            "node_id": node_id,
            "in_graph": node_id in graph_ingredients,
            "n_occurrences": node_attrs.get("n_occurrences"),
        })

    return {
        "counts": {
            "graph_ingredients": len(graph_ingredients),
            "wikibooks_matches": len(wikibooks_matches),
        },
        "top_wikibooks_matches": top_rows,
        "best_bridge": {
            "wikibooks_recipe": {
                "wikibooks_index": best_match["wikibooks_index"],
                "title": best_wiki.get("title"),
                "source_place": best_wiki.get("source_place"),
                "source_year": best_wiki.get("source_year"),
                "n_shared_ingredients": len(best_wiki_ings),
                "shared_ingredients": sorted(node_label(x, "ing::") for x in best_wiki_ings),
                "ingredients": wiki_ingredient_statuses,
            },
            "top_historical_matches": historical_rows,
            "best_historical_recipe": best_historical,
        },
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(report: dict[str, Any]) -> None:
    counts = report["counts"]

    print(f"\nIngredient nodes in graph: {counts['graph_ingredients']}")
    print(f"Wikibooks recipes with ≥1 shared ingredient: {counts['wikibooks_matches']}")

    if not report["top_wikibooks_matches"]:
        print("\nNo Wikibooks recipe shares ingredients with the graph.")
        return

    print("\nTOP WIKIBOOKS RECIPES BY SHARED INGREDIENT COUNT:")
    for row in report["top_wikibooks_matches"]:
        title = safe_str(row.get("title"))
        place = safe_str(row.get("source_place"))
        shared_preview = ", ".join(row["shared_ingredients"][:6])
        if len(row["shared_ingredients"]) > 6:
            shared_preview += f", ... +{len(row['shared_ingredients']) - 6}"
        print(
            f"  {title:35s} ({place}) — "
            f"{row['n_shared_ingredients']} shared: {shared_preview}"
        )

    bridge = report.get("best_bridge")
    if not bridge:
        return

    wiki = bridge["wikibooks_recipe"]
    print(f"\n{'=' * 78}")
    print(f"BEST WIKIBOOKS RECIPE: {safe_str(wiki.get('title'))}")
    print(f"{'=' * 78}")
    print(f"  source_place: {safe_str(wiki.get('source_place'))}")
    print(f"  source_year : {safe_str(wiki.get('source_year'))}")
    print("  ALL ingredients:")
    for item in wiki["ingredients"]:
        status = "IN GRAPH" if item["in_graph"] else "NEW"
        occ = item.get("n_occurrences")
        occ_text = f" n_occ={occ}" if occ is not None else ""
        print(f"    {item['node_id']:45s} [{status}]{occ_text}")

    print(
        f"\nTOP HISTORICAL RECIPES SHARING INGREDIENTS WITH "
        f"{safe_str(wiki.get('title'))!r}:"
    )
    for row in bridge["top_historical_matches"]:
        print(f"\n  {row['recipe_id']}")
        print(f"    title : {safe_str(row.get('title'))}")
        print(f"    source: {safe_str(row.get('source_title'))} ({safe_str(row.get('source_author'))})")
        print(f"    place : {safe_str(row.get('source_place'))}")
        print(
            f"    year  : {safe_str(row.get('source_year'))} "
            f"period: {safe_str(row.get('period_derived'))}"
        )
        print(
            f"    shared ({row['n_shared_ingredients']}/{wiki['n_shared_ingredients']}): "
            f"{', '.join(row['shared_ingredients'])}"
        )

    hist = bridge.get("best_historical_recipe")
    if not hist:
        print("\nNo historical recipe match found for the best Wikibooks recipe.")
        return

    print(f"\n{'=' * 78}")
    print(f"BEST HISTORICAL MATCH: {hist['recipe_id']}")
    print(f"{'=' * 78}")
    attrs = hist["attributes"]
    for key in [
        "title",
        "source_id",
        "source_title",
        "source_author",
        "source_year",
        "source_place",
        "source_language",
        "period_derived",
    ]:
        print(f"  {key}: {attrs.get(key)}")

    shared_ids = {
        f"ing::{x}" if not x.startswith("ing::") else x
        for x in hist["shared_ingredients"]
    }

    for node_type in ["Ingredient", "Tool", "Action", "Place", "Period"]:
        items = hist["neighbors"].get(node_type, [])
        print(f"\n  --- {node_type} ({len(items)}) ---")
        for item in items:
            node_id = item["node_id"]
            marker = " *** SHARED WITH WIKIBOOKS ***" if node_id in shared_ids else ""
            occ = item.get("n_occurrences")
            occ_text = f"n_occ={occ}" if occ is not None else "n_occ="
            print(f"    {node_id:45s} ({occ_text}){marker}")

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
        "--wikibooks",
        type=Path,
        default=DEFAULT_WIKIBOOKS,
        help=f"Path to Wikibooks JSON. Default: {DEFAULT_WIKIBOOKS}",
    )
    parser.add_argument(
        "--top-wikibooks",
        type=int,
        default=5,
        help="Number of top Wikibooks recipes to print.",
    )
    parser.add_argument(
        "--top-historical",
        type=int,
        default=10,
        help="Number of historical recipe matches to print.",
    )
    parser.add_argument(
        "--neighbor-limit",
        type=int,
        default=20,
        help=(
            "Maximum neighbors per node type to include for the best historical recipe. "
            "Use -1 for all."
        ),
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

    if args.top_wikibooks < 1:
        log.error("--top-wikibooks must be >= 1")
        return 2
    if args.top_historical < 1:
        log.error("--top-historical must be >= 1")
        return 2
    if args.neighbor_limit < -1:
        log.error("--neighbor-limit must be >= -1")
        return 2

    try:
        graph = load_graph(args.graph)
        wikibooks = load_wikibooks(args.wikibooks)

        graph_ings = ingredient_nodes(graph)
        matches = match_wikibooks_recipes(wikibooks, graph_ings)

        report = build_report(
            graph=graph,
            wikibooks_matches=matches,
            graph_ingredients=graph_ings,
            top_wikibooks=args.top_wikibooks,
            top_historical=args.top_historical,
            neighbor_limit=args.neighbor_limit,
        )

        if args.out_json:
            write_json(args.out_json, report)

        if not args.quiet:
            print_report(report)

        return 0

    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
