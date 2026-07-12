"""

Find meaningful Wikibooks ↔ historical recipe bridges through distinctive
ingredients.

The script compares contemporary Wikibooks recipes against a recipe graph and
filters out "universal" ingredients such as egg, sugar, butter, salt, flour, etc.
Universal ingredients are defined by frequency in the graph. By default, the top
30 most frequent ingredients are excluded.

Inputs
------

  data/graph_step2_canonical.gpickle
      NetworkX graph with Recipe and Ingredient nodes.

  data/wikibooks_contemporary.json
      List of contemporary Wikibooks recipes with ingredient lists.

Outputs
-------

By default, results are printed to stdout. Optionally, pass --out-json to write a
machine-readable report.

Usage
-----

  python dump_bridge_distinctive.py

  python dump_bridge_distinctive.py \
    --graph data/graph_step2_canonical.gpickle \
    --wikibooks data/wikibooks_contemporary.json \
    --top-universal 30 \
    --top-wikibooks 10 \
    --top-historical 10

  python dump_bridge_distinctive.py --out-json data/bridge_distinctive_report.json

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
log = logging.getLogger("bridge_distinctive")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Slug and display helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Stable ASCII slug compatible with publish-ready Step 1 scripts."""
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    s = ascii_text.strip().lower()
    s = _SLUG_RE.sub("_", s)
    return s.strip("_") or "unknown"


def node_label(node_id: str, prefix: str) -> str:
    """Remove a graph node prefix and make labels easier to read."""
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

    bad = sum(1 for item in data if not isinstance(item, dict))
    if bad:
        raise ValueError(f"Expected recipe objects in {path}; found {bad} non-object records")

    log.info("Loaded %d Wikibooks recipes", len(data))
    return data


# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------

def ingredient_frequencies(graph: nx.Graph) -> dict[str, int]:
    """Return Ingredient node -> n_occurrences."""
    freqs: dict[str, int] = {}

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Ingredient":
            continue

        raw_count = attrs.get("n_occurrences", 0)
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 0

        freqs[str(node_id)] = count

    if not freqs:
        raise ValueError("No Ingredient nodes found in graph")

    return freqs


def universal_ingredients(freqs: dict[str, int], top_n: int) -> set[str]:
    """Return top-N most frequent ingredients."""
    if top_n < 0:
        raise ValueError("--top-universal must be >= 0")
    return {node_id for node_id, _ in sorted(freqs.items(), key=lambda x: (-x[1], x[0]))[:top_n]}


def recipe_ingredient_set(graph: nx.Graph, recipe_id: str) -> set[str]:
    """Return Ingredient targets connected by contains edges from a recipe."""
    ingredients: set[str] = set()

    if not graph.has_node(recipe_id):
        return ingredients

    for _, target, edge_attrs in graph.out_edges(recipe_id, data=True):
        if edge_attrs.get("edge_type") == "contains":
            if graph.nodes.get(target, {}).get("node_type") == "Ingredient":
                ingredients.add(str(target))

    return ingredients


def recipe_tools(graph: nx.Graph, recipe_id: str, limit: int) -> list[str]:
    out: list[str] = []
    for _, target, edge_attrs in graph.out_edges(recipe_id, data=True):
        if edge_attrs.get("edge_type") == "uses_tool":
            out.append(node_label(str(target), "tool::"))
    return out[:limit]


def recipe_actions(graph: nx.Graph, recipe_id: str, limit: int) -> list[str]:
    out: list[str] = []
    for _, target, edge_attrs in graph.out_edges(recipe_id, data=True):
        if edge_attrs.get("edge_type") == "performs":
            out.append(node_label(str(target), "act::"))
    return out[:limit]


def historical_recipes_sharing(
    graph: nx.Graph,
    distinctive_ingredients: set[str],
) -> Counter[str]:
    """Count historical recipe nodes sharing each distinctive ingredient."""
    overlap: Counter[str] = Counter()

    for ing_id in distinctive_ingredients:
        if not graph.has_node(ing_id):
            continue

        for pred in graph.predecessors(ing_id):
            pred_attrs = graph.nodes.get(pred, {})
            if pred_attrs.get("node_type") == "Recipe":
                overlap[str(pred)] += 1

    return overlap


# ---------------------------------------------------------------------------
# Wikibooks matching
# ---------------------------------------------------------------------------

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


def match_wikibooks_recipes(
    wikibooks: list[dict[str, Any]],
    graph_ingredients: set[str],
    distinctive_ingredients: set[str],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []

    for index, recipe in enumerate(wikibooks):
        shared_all: set[str] = set()
        shared_distinctive: set[str] = set()

        for name in extract_wikibooks_ingredient_names(recipe):
            node_id = ingredient_node_id(name)
            if node_id in graph_ingredients:
                shared_all.add(node_id)
                if node_id in distinctive_ingredients:
                    shared_distinctive.add(node_id)

        if shared_distinctive:
            matches.append({
                "wikibooks_index": index,
                "recipe": recipe,
                "shared_distinctive": shared_distinctive,
                "shared_all": shared_all,
            })

    matches.sort(
        key=lambda row: (
            -len(row["shared_distinctive"]),
            -len(row["shared_all"]),
            safe_str(row["recipe"].get("title")),
        )
    )
    return matches


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def make_report(
    *,
    graph: nx.Graph,
    freqs: dict[str, int],
    universal: set[str],
    wikibooks_matches: list[dict[str, Any]],
    top_wikibooks: int,
    top_historical: int,
    detail_limit: int,
) -> dict[str, Any]:
    graph_ingredients = set(freqs)
    distinctive = graph_ingredients - universal

    top_universal_rows = [
        {
            "ingredient_id": node_id,
            "label": node_label(node_id, "ing::"),
            "n_occurrences": freqs[node_id],
        }
        for node_id in sorted(universal, key=lambda x: (-freqs[x], x))
    ]

    top_wikibooks_rows: list[dict[str, Any]] = []

    for match in wikibooks_matches[:top_wikibooks]:
        recipe = match["recipe"]
        shared_distinctive = match["shared_distinctive"]
        shared_all = match["shared_all"]

        top_wikibooks_rows.append({
            "wikibooks_index": match["wikibooks_index"],
            "title": recipe.get("title"),
            "source_place": recipe.get("source_place"),
            "n_distinctive_shared": len(shared_distinctive),
            "n_total_shared": len(shared_all),
            "distinctive_shared": sorted(node_label(x, "ing::") for x in shared_distinctive),
        })

    best: dict[str, Any] | None = None

    if wikibooks_matches:
        best_match = wikibooks_matches[0]
        best_recipe = best_match["recipe"]
        best_distinctive = set(best_match["shared_distinctive"])
        best_all = set(best_match["shared_all"])

        ingredient_statuses: list[dict[str, Any]] = []
        for name in extract_wikibooks_ingredient_names(best_recipe):
            node_id = ingredient_node_id(name)
            if node_id in universal:
                status = "universal_filtered"
            elif node_id in graph_ingredients:
                status = "distinctive"
            else:
                status = "new_not_in_graph"

            ingredient_statuses.append({
                "name": name,
                "node_id": node_id,
                "status": status,
                "n_occurrences": freqs.get(node_id),
            })

        overlap = historical_recipes_sharing(graph, best_distinctive)
        historical_rows: list[dict[str, Any]] = []

        for hist_recipe_id, count in overlap.most_common(top_historical):
            attrs = graph.nodes[hist_recipe_id]
            hist_ings = recipe_ingredient_set(graph, hist_recipe_id)
            shared = best_distinctive & hist_ings

            historical_rows.append({
                "recipe_id": hist_recipe_id,
                "title": attrs.get("title"),
                "source_title": attrs.get("source_title"),
                "source_place": attrs.get("source_place"),
                "source_year": attrs.get("source_year"),
                "period_derived": attrs.get("period_derived"),
                "n_distinctive_shared": count,
                "distinctive_shared": sorted(node_label(x, "ing::") for x in shared),
                "tools": recipe_tools(graph, hist_recipe_id, detail_limit),
                "actions": recipe_actions(graph, hist_recipe_id, detail_limit),
            })

        best = {
            "wikibooks_index": best_match["wikibooks_index"],
            "title": best_recipe.get("title"),
            "source_place": best_recipe.get("source_place"),
            "n_distinctive_shared": len(best_distinctive),
            "n_total_shared": len(best_all),
            "distinctive_shared": sorted(node_label(x, "ing::") for x in best_distinctive),
            "ingredients": ingredient_statuses,
            "historical_matches": historical_rows,
        }

    return {
        "method": {
            "universal_definition": "top_N_most_frequent_ingredients",
            "top_universal_n": len(universal),
            "top_wikibooks_n": top_wikibooks,
            "top_historical_n": top_historical,
        },
        "counts": {
            "graph_ingredients": len(graph_ingredients),
            "universal_filtered": len(universal),
            "distinctive_ingredients": len(distinctive),
            "wikibooks_matches_with_distinctive_overlap": len(wikibooks_matches),
        },
        "universal_ingredients": top_universal_rows,
        "top_wikibooks_matches": top_wikibooks_rows,
        "best_match": best,
    }


def print_report(report: dict[str, Any]) -> None:
    counts = report["counts"]

    print("\nFILTERED OUT: top universal ingredients")
    print(f"Universal filtered: {counts['universal_filtered']}")
    for row in report["universal_ingredients"]:
        print(f"  {row['ingredient_id']:40s} n_occ={row['n_occurrences']}")

    print(
        "\nMatching through "
        f"{counts['distinctive_ingredients']} distinctive ingredients "
        f"(excluding {counts['universal_filtered']} universal)"
    )

    print("\nTOP WIKIBOOKS RECIPES BY DISTINCTIVE SHARED INGREDIENTS:")
    for row in report["top_wikibooks_matches"]:
        title = safe_str(row.get("title"))
        place = safe_str(row.get("source_place"))
        print(f"\n  {title:40s} ({place})")
        print(
            f"    distinctive shared: {row['n_distinctive_shared']} "
            f"total shared: {row['n_total_shared']}"
        )
        print(f"    distinctive: {', '.join(row['distinctive_shared'])}")

    best = report.get("best_match")
    if not best:
        print("\nNo Wikibooks recipe shared any distinctive ingredient.")
        return

    print(f"\n{'=' * 78}")
    print(f"BEST: {safe_str(best.get('title'))} ({safe_str(best.get('source_place'))})")
    print(f"{'=' * 78}")

    print("ALL ingredients:")
    for item in best["ingredients"]:
        occ = item.get("n_occurrences")
        occ_text = f", n_occ={occ}" if occ is not None else ""
        print(f"  {item['node_id']:45s} [{item['status']}{occ_text}]")

    print("\nHistorical recipes sharing DISTINCTIVE ingredients:")
    for row in best["historical_matches"]:
        print(f"\n  {row['recipe_id']}")
        print(f"    title: {safe_str(row.get('title'))}")
        print(f"    source: {safe_str(row.get('source_title'))}")
        print(f"    place: {safe_str(row.get('source_place'))}")
        print(
            f"    year: {safe_str(row.get('source_year'))} "
            f"period: {safe_str(row.get('period_derived'))}"
        )
        print(
            f"    distinctive shared ({row['n_distinctive_shared']}): "
            f"{', '.join(row['distinctive_shared'])}"
        )
        if row["tools"]:
            print(f"    tools: {', '.join(row['tools'])}")
        if row["actions"]:
            print(f"    actions: {', '.join(row['actions'])}")

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
        "--top-universal",
        type=int,
        default=30,
        help="Number of most frequent ingredients to filter as universal.",
    )
    parser.add_argument(
        "--top-wikibooks",
        type=int,
        default=10,
        help="Number of top Wikibooks matches to print.",
    )
    parser.add_argument(
        "--top-historical",
        type=int,
        default=10,
        help="Number of historical matches to print for the best Wikibooks recipe.",
    )
    parser.add_argument(
        "--detail-limit",
        type=int,
        default=5,
        help="Maximum tools/actions to print for each historical recipe.",
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

    if args.top_universal < 0:
        log.error("--top-universal must be >= 0")
        return 2
    if args.top_wikibooks < 1:
        log.error("--top-wikibooks must be >= 1")
        return 2
    if args.top_historical < 1:
        log.error("--top-historical must be >= 1")
        return 2
    if args.detail_limit < 0:
        log.error("--detail-limit must be >= 0")
        return 2

    try:
        graph = load_graph(args.graph)
        wikibooks = load_wikibooks(args.wikibooks)

        freqs = ingredient_frequencies(graph)
        universal = universal_ingredients(freqs, args.top_universal)
        graph_ingredients = set(freqs)
        distinctive = graph_ingredients - universal

        matches = match_wikibooks_recipes(
            wikibooks=wikibooks,
            graph_ingredients=graph_ingredients,
            distinctive_ingredients=distinctive,
        )

        report = make_report(
            graph=graph,
            freqs=freqs,
            universal=universal,
            wikibooks_matches=matches,
            top_wikibooks=args.top_wikibooks,
            top_historical=args.top_historical,
            detail_limit=args.detail_limit,
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
