"""
Step 1: Build the heterogeneous Relish graph base.

Reads:
  data/relish_dataset.json

Writes:
  layer_1_derived_fields.csv
      Per-recipe derived fields: recipe_id, period_derived, counts, etc.

  graph_step1.gpickle
      NetworkX DiGraph with node and structural edge layers.

  graph_step1_stats.json
      Schema counters, quality flags, NER-noise flags, and collision reports.


Node types created here:
  Recipe, Ingredient, Tool, Action, Place, Period

Edge types created here:
  Recipe --contains--> Ingredient
  Recipe --uses_tool--> Tool
  Recipe --performs--> Action
  Recipe --origin--> Place
  Recipe --dated--> Period

Usage:
  python build_graph_step1.py --input data/relish_dataset.json --output-dir data/

Notes:
  - IDs are deterministic.
  - Unicode text is normalized for stable public node IDs.
  - The graph is saved as pickle because later project steps expect .gpickle.
  
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("relish_step1")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Period derivation
# ---------------------------------------------------------------------------

# Buckets are inclusive on both ends. "ancient" covers everything up to 500 CE
# to absorb early sources such as Apicius without creating sparse buckets.
PERIOD_BUCKETS: list[tuple[str, int, int]] = [
    ("ancient", -10_000, 500),
    ("medieval_early", 501, 1199),
    ("13c", 1200, 1299),
    ("14c", 1300, 1399),
    ("15c", 1400, 1499),
    ("16c", 1500, 1599),
    ("17c", 1600, 1699),
    ("18c", 1700, 1799),
    ("19c", 1800, 1899),
    ("20c", 1900, 1999),
    ("21c", 2000, 2099),
]

_PERIOD_LOOKUP = {label: (lo, hi) for label, lo, hi in PERIOD_BUCKETS}
_YEAR_RE = re.compile(r"-?\d{1,5}")


def parse_year(value: Any) -> int | None:
    """Parse a year-like value into an int.

    Handles ints, floats, and simple strings such as "1520", "c. 1520",
    or "1520-1530" by taking the first year-like integer. Returns None for
    missing or unparseable values.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)

    text = str(value).strip()
    if not text:
        return None
    match = _YEAR_RE.search(text.replace("–", "-").replace("—", "-"))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def derive_period(year: int | None) -> str | None:
    """Map a year to a period bucket label."""
    if year is None:
        return None
    for label, lo, hi in PERIOD_BUCKETS:
        if lo <= year <= hi:
            return label
    return None


# ---------------------------------------------------------------------------
# Slug / ID helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: Any) -> str:
    """Return a stable ASCII slug for node IDs.

    Accents are normalized. Fully non-ASCII strings fall back to a short hash
    instead of collapsing to an ambiguous "unknown" ID.
    """
    raw = str(text).strip().lower()
    if not raw:
        return "unknown"

    normalized = (
        unicodedata.normalize("NFKD", raw)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = _SLUG_RE.sub("_", normalized).strip("_")
    if slug:
        return slug

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"x_{digest}"


def recipe_id(source_id: Any, index: int) -> str:
    return f"recipe::{slugify(source_id)}::{index:05d}"


def ingredient_id(key: str) -> str:
    return f"ing::{key}"


def tool_id(key: str) -> str:
    return f"tool::{key}"


def action_id(key: str) -> str:
    return f"act::{key}"


def place_id(canonical: Any) -> str:
    return f"place::{slugify(canonical)}"


def period_id(label: str) -> str:
    return f"period::{label}"


# ---------------------------------------------------------------------------
# Small coercion helpers
# ---------------------------------------------------------------------------


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def sorted_strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(v).strip() for v in values if str(v).strip()})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_relish_base(path: Path) -> list[dict[str, Any]]:
    log.info("Loading Relish base JSON: %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")
    if not all(isinstance(row, dict) for row in data):
        bad = sum(1 for row in data if not isinstance(row, dict))
        raise ValueError(f"Expected JSON objects; found {bad} invalid records")

    log.info("Loaded %d recipe records", len(data))
    return data


# ---------------------------------------------------------------------------
# Entity aggregation
# ---------------------------------------------------------------------------


def entity_key(name: Any) -> str:
    return slugify(name)


def make_entity_pool() -> dict[str, dict[str, Any]]:
    return defaultdict(
        lambda: {
            "surface_forms": set(),
            "raw_names": set(),
            "confidences": [],
            "n_occurrences": 0,
        }
    )


def update_entity_pool(pool: dict[str, dict[str, Any]], name: Any, entity: dict[str, Any]) -> None:
    raw_name = clean_text(name)
    if raw_name is None:
        return

    key = entity_key(raw_name)
    entry = pool[key]
    entry["surface_forms"].add(raw_name)
    entry["raw_names"].add(raw_name.lower())
    entry["confidences"].append(as_float(entity.get("confidence_score"), 0.0))
    entry["n_occurrences"] += 1

    for form in as_list(entity.get("specific_forms")):
        form_text = clean_text(form)
        if form_text:
            entry["surface_forms"].add(form_text)


def collect_entity_pools(
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, set[str]]],
    dict[str, int],
]:
    """Aggregate Ingredient, Tool, and Action surface forms/confidences."""
    ingredients = make_entity_pool()
    tools = make_entity_pool()
    actions = make_entity_pool()
    skipped: Counter[str] = Counter()

    for rec in records:
        for ing in as_list(rec.get("ingredients")):
            if not isinstance(ing, dict) or clean_text(ing.get("name")) is None:
                skipped["bad_ingredient_records"] += 1
                continue
            update_entity_pool(ingredients, ing.get("name"), ing)

        for tool in as_list(rec.get("tools")):
            if not isinstance(tool, dict) or clean_text(tool.get("name")) is None:
                skipped["bad_tool_records"] += 1
                continue
            update_entity_pool(tools, tool.get("name"), tool)

        for action in as_list(rec.get("actions")):
            if not isinstance(action, dict) or clean_text(action.get("verb")) is None:
                skipped["bad_action_records"] += 1
                continue
            update_entity_pool(actions, action.get("verb"), action)

    collisions = {
        "Ingredient": {k: set(v["raw_names"]) for k, v in ingredients.items() if len(v["raw_names"]) > 1},
        "Tool": {k: set(v["raw_names"]) for k, v in tools.items() if len(v["raw_names"]) > 1},
        "Action": {k: set(v["raw_names"]) for k, v in actions.items() if len(v["raw_names"]) > 1},
    }

    log.info(
        "Entity pools: %d ingredients, %d tools, %d actions",
        len(ingredients), len(tools), len(actions),
    )
    log.info(
        "Collisions: %d ingredients, %d tools, %d actions",
        len(collisions["Ingredient"]), len(collisions["Tool"]), len(collisions["Action"]),
    )
    if skipped:
        log.warning("Skipped malformed entity records: %s", dict(skipped))

    return dict(ingredients), dict(tools), dict(actions), collisions, dict(skipped)


def collect_places(records: list[dict[str, Any]]) -> list[str]:
    places = sorted({
        str(rec["source_place"]).strip()
        for rec in records
        if clean_text(rec.get("source_place")) is not None
    })
    log.info("Distinct places: %d", len(places))
    return places


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def looks_like_ner_noise(raw_names: set[str]) -> bool:
    """Flag likely nutritional-table/OCR/NER junk, without removing it."""
    noise_markers = ("%", "|", "°")
    return any(any(marker in raw for marker in noise_markers) for raw in raw_names)


def add_entity_nodes(G: nx.DiGraph, pool: dict[str, dict[str, Any]], node_type: str) -> None:
    for key in sorted(pool):
        agg = pool[key]
        confs = [as_float(c, 0.0) for c in agg["confidences"]]
        mean_confidence = sum(confs) / len(confs) if confs else 0.0
        common = {
            "node_type": node_type,
            "layer": "relish_base",
            "surface_forms": sorted_strings(agg["surface_forms"]),
            "raw_variants": sorted_strings(agg["raw_names"]),
            "n_occurrences": int(agg["n_occurrences"]),
            "mean_confidence": mean_confidence,
            "ner_noise_flag": looks_like_ner_noise(set(agg["raw_names"])),
        }

        if node_type == "Ingredient":
            G.add_node(ingredient_id(key), **common, canonical_name=key, ingredient_category=None)
        elif node_type == "Tool":
            G.add_node(tool_id(key), **common, canonical_name=key)
        elif node_type == "Action":
            G.add_node(action_id(key), **common, canonical_verb=key)
        else:
            raise ValueError(f"Unknown entity node_type: {node_type}")


def add_or_update_edge(G: nx.DiGraph, source: str, target: str, edge_type: str, **attrs: Any) -> None:
    """Add an edge, preserving duplicate mentions as n_mentions.

    NetworkX DiGraph stores one edge per source-target pair. If the same
    relation appears more than once, this keeps a single edge and increments
    n_mentions. The maximum confidence is retained as confidence_score and all
    observed confidences are stored in confidence_scores.
    """
    confidence = attrs.pop("confidence_score", None)

    if G.has_edge(source, target):
        edge = G[source][target]
        if edge.get("edge_type") != edge_type:
            edge["edge_type_conflict"] = sorted({edge.get("edge_type", "UNKNOWN"), edge_type})
        edge["n_mentions"] = int(edge.get("n_mentions", 1)) + 1

        if confidence is not None:
            scores = edge.setdefault("confidence_scores", [])
            scores.append(as_float(confidence, 0.0))
            edge["confidence_score"] = max(scores)

        if "specific_forms_used" in attrs:
            edge.setdefault("specific_forms_used", []).extend(as_list(attrs["specific_forms_used"]))
        return

    edge_attrs = {"edge_type": edge_type, "n_mentions": 1, **attrs}
    if confidence is not None:
        score = as_float(confidence, 0.0)
        edge_attrs["confidence_score"] = score
        edge_attrs["confidence_scores"] = [score]
    G.add_edge(source, target, **edge_attrs)


def build_graph(records: list[dict[str, Any]]) -> tuple[nx.DiGraph, pd.DataFrame, dict[str, dict[str, set[str]]]]:
    G = nx.DiGraph()
    quality_flags: Counter[str] = Counter()

    ingredients, tools, actions, collisions, skipped = collect_entity_pools(records)
    quality_flags.update(skipped)
    places = collect_places(records)

    add_entity_nodes(G, ingredients, "Ingredient")
    add_entity_nodes(G, tools, "Tool")
    add_entity_nodes(G, actions, "Action")

    for place in places:
        G.add_node(
            place_id(place),
            node_type="Place",
            layer="relish_base",
            canonical_name=place,
            region_macro=None,
        )

    source_counter: Counter[str] = Counter()
    derived_rows: list[dict[str, Any]] = []
    used_periods: set[str] = set()

    for rec_idx, rec in enumerate(records):
        src = clean_text(rec.get("source_id")) or "unknown_source"
        idx = source_counter[src]
        source_counter[src] += 1
        rid = recipe_id(src, idx)

        year = parse_year(rec.get("source_year"))
        period_label = derive_period(year)
        if rec.get("source_year") is not None and year is None:
            quality_flags["unparseable_source_year"] += 1
        if year is not None and period_label is None:
            quality_flags["year_outside_period_buckets"] += 1

        recipe_text = clean_text(rec.get("recipe_text"))
        translation = clean_text(rec.get("translation"))

        G.add_node(
            rid,
            node_type="Recipe",
            layer="relish_base",
            title=rec.get("title"),
            recipe_text=recipe_text,
            translation_en=translation or recipe_text,
            source_id=src,
            source_title=rec.get("source_title"),
            source_author=rec.get("source_author"),
            source_year=year,
            source_year_raw=rec.get("source_year"),
            source_place=rec.get("source_place"),
            source_language=rec.get("source_language"),
            source_record_index=rec_idx,
            period_derived=period_label,
            embedding_vec=None,
            tradition=None,
            dietary_system=None,
        )

        n_ing = n_tools = n_actions = 0

        for ing in as_list(rec.get("ingredients")):
            if not isinstance(ing, dict) or clean_text(ing.get("name")) is None:
                continue
            key = entity_key(ing["name"])
            add_or_update_edge(
                G,
                rid,
                ingredient_id(key),
                "contains",
                confidence_score=ing.get("confidence_score", 0.0),
                specific_forms_used=as_list(ing.get("specific_forms")),
            )
            n_ing += 1

        for tool in as_list(rec.get("tools")):
            if not isinstance(tool, dict) or clean_text(tool.get("name")) is None:
                continue
            key = entity_key(tool["name"])
            add_or_update_edge(
                G,
                rid,
                tool_id(key),
                "uses_tool",
                confidence_score=tool.get("confidence_score", 0.0),
            )
            n_tools += 1

        for action in as_list(rec.get("actions")):
            if not isinstance(action, dict) or clean_text(action.get("verb")) is None:
                continue
            key = entity_key(action["verb"])
            add_or_update_edge(
                G,
                rid,
                action_id(key),
                "performs",
                confidence_score=action.get("confidence_score", 0.0),
            )
            n_actions += 1

        source_place = clean_text(rec.get("source_place"))
        if source_place:
            add_or_update_edge(G, rid, place_id(source_place), "origin")

        if period_label is not None:
            if period_label not in used_periods:
                lo, hi = _PERIOD_LOOKUP[period_label]
                G.add_node(
                    period_id(period_label),
                    node_type="Period",
                    layer="derived",
                    label=period_label,
                    year_start=lo,
                    year_end=hi,
                )
                used_periods.add(period_label)
            add_or_update_edge(G, rid, period_id(period_label), "dated")

        if n_ing == 0:
            quality_flags["recipes_without_ingredients"] += 1

        derived_rows.append(
            {
                "recipe_id": rid,
                "source_id": src,
                "source_year": year,
                "source_year_raw": rec.get("source_year"),
                "period_derived": period_label,
                "source_place": rec.get("source_place"),
                "source_language": rec.get("source_language"),
                "n_ingredients": n_ing,
                "n_tools": n_tools,
                "n_actions": n_actions,
            }
        )

    G.graph["quality_flags"] = dict(quality_flags)
    G.graph["period_buckets"] = PERIOD_BUCKETS
    G.graph["builder"] = "build_graph_step1.py"

    log.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    if quality_flags:
        log.warning("Quality flags: %s", dict(quality_flags))

    return G, pd.DataFrame(derived_rows), collisions


# ---------------------------------------------------------------------------
# Invariants and reports
# ---------------------------------------------------------------------------


def check_invariants(G: nx.DiGraph) -> dict[str, Any]:
    by_type = Counter(d.get("node_type", "UNKNOWN") for _, d in G.nodes(data=True))
    edges_by_type = Counter(d.get("edge_type", "UNKNOWN") for _, _, d in G.edges(data=True))

    recipes_without_ingredients: list[str] = []
    recipes_missing_period: list[str] = []
    bad_entity_counts: list[str] = []

    for node_id, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type")
        if node_type == "Recipe":
            has_contains = any(
                edge_attrs.get("edge_type") == "contains"
                for _, _, edge_attrs in G.out_edges(node_id, data=True)
            )
            if not has_contains:
                recipes_without_ingredients.append(node_id)
            if attrs.get("source_year") is not None and attrs.get("period_derived") is None:
                recipes_missing_period.append(node_id)

        if node_type in {"Ingredient", "Tool", "Action"}:
            if int(attrs.get("n_occurrences", 0)) < 1:
                bad_entity_counts.append(node_id)

    ingredient_names = [
        attrs.get("canonical_name")
        for _, attrs in G.nodes(data=True)
        if attrs.get("node_type") == "Ingredient"
    ]
    duplicate_ingredients = [
        name for name, count in Counter(ingredient_names).items()
        if name is not None and count > 1
    ]
    ner_noise_count = sum(
        1
        for _, attrs in G.nodes(data=True)
        if attrs.get("node_type") in {"Ingredient", "Tool", "Action"}
        and bool(attrs.get("ner_noise_flag"))
    )
    edge_type_conflicts = sum(
        1 for _, _, attrs in G.edges(data=True) if attrs.get("edge_type_conflict")
    )

    return {
        "nodes_by_type": dict(by_type),
        "edges_by_type": dict(edges_by_type),
        "recipes_without_ingredients": len(recipes_without_ingredients),
        "recipes_without_ingredients_examples": recipes_without_ingredients[:20],
        "entities_with_zero_occurrences": len(bad_entity_counts),
        "entities_with_zero_occurrences_examples": bad_entity_counts[:20],
        "recipes_with_year_but_no_period": len(recipes_missing_period),
        "recipes_with_year_but_no_period_examples": recipes_missing_period[:20],
        "duplicate_ingredient_canonical_names": len(duplicate_ingredients),
        "duplicate_ingredient_canonical_names_examples": duplicate_ingredients[:20],
        "ner_noise_flagged_entities": ner_noise_count,
        "edge_type_conflicts": edge_type_conflicts,
        "quality_flags": G.graph.get("quality_flags", {}),
    }


def invariant_report_has_errors(report: dict[str, Any]) -> bool:
    return any(
        int(report.get(key, 0)) > 0
        for key in [
            "entities_with_zero_occurrences",
            "recipes_with_year_but_no_period",
            "duplicate_ingredient_canonical_names",
            "edge_type_conflicts",
        ]
    )


def serializable_collisions(
    collisions: dict[str, dict[str, set[str]]],
    max_examples_per_type: int | None = None,
) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for entity_type, cols in collisions.items():
        items = sorted(cols.items(), key=lambda item: item[0])
        if max_examples_per_type is not None:
            items = items[:max_examples_per_type]
        out[entity_type] = {key: sorted(values) for key, values in items}
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_graph(G: nx.DiGraph, path: Path) -> None:
    log.info("Saving graph to %s", path)
    with path.open("wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)


def save_derived(df: pd.DataFrame, path: Path) -> None:
    log.info("Saving derived fields CSV to %s", path)
    df.to_csv(path, index=False)


def save_stats(report: dict[str, Any], path: Path) -> None:
    log.info("Saving stats to %s", path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to Relish base JSON, layer 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/"),
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if schema invariants fail.",
    )
    parser.add_argument(
        "--collision-report-limit",
        type=int,
        default=None,
        help="Limit collisions saved per entity type. Default: save all.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        records = load_relish_base(args.input)
        G, derived_df, collisions = build_graph(records)

        report = check_invariants(G)
        report["collisions"] = serializable_collisions(
            collisions,
            max_examples_per_type=args.collision_report_limit,
        )

        log.info(
            "Invariants report: nodes=%s edges=%s ner_noise=%d",
            report["nodes_by_type"],
            report["edges_by_type"],
            report["ner_noise_flagged_entities"],
        )

        save_graph(G, args.output_dir / "graph_step1.gpickle")
        save_derived(derived_df, args.output_dir / "layer_1_derived_fields.csv")
        save_stats(report, args.output_dir / "graph_step1_stats.json")

        if args.strict and invariant_report_has_errors(report):
            log.error("Strict mode failed. See graph_step1_stats.json for details.")
            return 2

        log.info("Done.")
        return 0

    except Exception as exc:
        log.exception("Step 1 failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
