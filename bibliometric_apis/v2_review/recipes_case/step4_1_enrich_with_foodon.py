"""
=============================

Enrich the recipe graph with validated FoodOn mappings.

This step reads the human-reviewed FoodOn coverage CSV from Step 4.0, keeps
only rows where `validated_keep` is TRUE, fetches FoodOn ancestor chains from
EBI OLS, and injects a FoodOn hierarchy slice into the graph.

What it adds
------------

  - FoodOnClass nodes for each validated FoodOn term and its ancestors.
  - Ingredient --mapped_to_foodon--> FoodOnClass edges for validated leaf terms.
  - FoodOnClass --is_a--> FoodOnClass edges for ontology hierarchy.

Original Ingredient nodes are kept unchanged. Downstream analyses that do not
use ontology nodes continue to work normally.

Default inputs
--------------

  data/graph_step3_flagged.gpickle
  data/foodon_coverage.csv

Default outputs
---------------

  data/graph_step4_foodon.gpickle
  data/foodon_enrichment_audit.csv
  data/foodon_ancestor_cache.jsonl
  data/foodon_enrichment_stats.json

Usage
-----

  python step4_1_enrich_with_foodon.py

  python step4_1_enrich_with_foodon.py --max-depth 3

  python step4_1_enrich_with_foodon.py --dry-run

  python step4_1_enrich_with_foodon.py --refresh-cache


"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_GRAPH_IN = Path("data/graph_step3_flagged.gpickle")
DEFAULT_GRAPH_OUT = Path("data/graph_step4_foodon.gpickle")
DEFAULT_COVERAGE_CSV = Path("data/foodon_coverage.csv")
DEFAULT_DATA_DIR = Path("data")

DEFAULT_ONTOLOGY = "foodon"
DEFAULT_OLS_ONTOLOGY_BASE = "https://www.ebi.ac.uk/ols4/api/ontologies/{ontology}"
DEFAULT_MAX_DEPTH = 5
DEFAULT_SLEEP_SECONDS = 0.15
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_USER_AGENT = "relish-recipes-research/1.0 (academic; contact: project-local)"

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step4_1_enrich_with_foodon")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def is_true(value: Any) -> bool:
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "si",
        "sí",
        "keep",
        "validated",
        "accept",
        "accepted",
    }


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return default
    return text if text else default


def normalize_foodon_id(foodon_id: str) -> str:
    """Normalize FOODON_03301710 / FOODON:03301710 into FOODON:03301710."""
    text = safe_str(foodon_id).upper().replace("_", ":")

    # Recover IDs embedded in URLs or opaque strings.
    match = re.search(r"FOODON[:_](\d+)", text)
    if match:
        return f"FOODON:{match.group(1)}"

    return text


def foodon_iri(foodon_id: str) -> str:
    obo_id = normalize_foodon_id(foodon_id)
    return f"http://purl.obolibrary.org/obo/{obo_id.replace(':', '_')}"


def foodon_node_key(foodon_id: str) -> str:
    return f"foodon::{normalize_foodon_id(foodon_id)}"


def ingredient_display_label(graph: Any, ingredient_id: str) -> str:
    attrs = graph.nodes.get(ingredient_id, {})
    return safe_str(
        attrs.get("canonical_name")
        or attrs.get("name")
        or attrs.get("label")
        or ingredient_id.replace("ing::", "")
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    log.info("Loading graph: %s", path)
    with path.open("rb") as fh:
        graph = pickle.load(fh)

    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError(f"Object loaded from {path} does not look like a NetworkX graph")

    log.info("Graph loaded: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def load_validated_coverage(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Coverage CSV not found: {path}. Run Step 4.0 first.")

    log.info("Loading FoodOn coverage CSV: %s", path)
    df = pd.read_csv(path)

    required = {"ingredient_id", "foodon_id", "validated_keep"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coverage CSV is missing required columns: {sorted(missing)}")

    n_total = len(df)
    df = df[df["validated_keep"].apply(is_true)].copy()
    log.info("%d/%d rows marked validated_keep=TRUE", len(df), n_total)

    if df.empty:
        raise ValueError(
            "No validated mappings to inject. Fill validated_keep=TRUE/FALSE in the coverage CSV."
        )

    df["ingredient_id"] = df["ingredient_id"].astype(str).str.strip()
    df["foodon_id"] = df["foodon_id"].apply(lambda x: normalize_foodon_id(safe_str(x)))
    df = df[(df["ingredient_id"] != "") & (df["foodon_id"] != "")].copy()

    if df.empty:
        raise ValueError("Validated rows exist, but none have non-empty ingredient_id and foodon_id.")

    # De-duplicate repeated Ingredient -> FoodOn mappings.
    before = len(df)
    df = df.drop_duplicates(subset=["ingredient_id", "foodon_id"]).copy()
    if len(df) != before:
        log.info("Dropped %d duplicate validated mapping rows", before - len(df))

    log.info("%d validated Ingredient -> FoodOn mappings will be processed", len(df))
    return df


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_key(*, foodon_id: str, ontology: str) -> str:
    return json.dumps(
        {"foodon_id": normalize_foodon_id(foodon_id), "ontology": ontology},
        sort_keys=True,
    )


def load_cache(path: Path, *, enabled: bool) -> dict[str, list[dict[str, Any]]]:
    if not enabled or not path.exists():
        return {}

    cache: dict[str, list[dict[str, Any]]] = {}

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)

                if "key" in row:
                    key = str(row["key"])
                else:
                    key = cache_key(
                        foodon_id=safe_str(row.get("foodon_id")),
                        ontology=safe_str(row.get("ontology"), DEFAULT_ONTOLOGY),
                    )

                parents = row.get("parents", [])
                if isinstance(parents, list):
                    cache[key] = parents
            except (json.JSONDecodeError, TypeError):
                log.warning("Skipping malformed cache line %d in %s", line_no, path)

    log.info("Loaded %d cached ancestor lookups", len(cache))
    return cache


def append_cache(
    path: Path,
    *,
    key: str,
    foodon_id: str,
    ontology: str,
    parents: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "key": key,
        "foodon_id": normalize_foodon_id(foodon_id),
        "ontology": ontology,
        "parents": parents,
    }

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# OLS API
# ---------------------------------------------------------------------------

def ontology_base_url(template: str, ontology: str) -> str:
    if "{ontology}" in template:
        return template.format(ontology=ontology)
    return template.rstrip("/")


def fetch_parents(
    *,
    foodon_id: str,
    ontology: str,
    ols_base_template: str,
    timeout_s: int,
    user_agent: str,
    max_retries: int,
    retry_backoff_s: float,
) -> list[dict[str, Any]]:
    """Fetch direct parents of a FoodOn term from EBI OLS."""
    obo_id = normalize_foodon_id(foodon_id)
    iri = foodon_iri(obo_id)
    base = ontology_base_url(ols_base_template, ontology)
    url = f"{base}/parents?id={quote_plus(iri)}"

    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))

            terms = (data.get("_embedded") or {}).get("terms") or []
            parents: list[dict[str, Any]] = []

            for term in terms:
                parent_id = normalize_foodon_id(
                    safe_str(term.get("obo_id") or term.get("short_form"))
                )
                if not parent_id:
                    continue

                parents.append({
                    "obo_id": parent_id,
                    "label": safe_str(term.get("label")),
                    "iri": safe_str(term.get("iri")) or foodon_iri(parent_id),
                    "is_obsolete": bool(term.get("is_obsolete")),
                })

            return parents

        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                log.warning("Non-retryable HTTP error for %s: %s", obo_id, exc.code)
                return []

            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "HTTP error for %s attempt %d/%d: %s. Sleeping %.1fs",
                obo_id,
                attempt + 1,
                max_retries,
                exc.code,
                wait,
            )
            time.sleep(wait)

        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "OLS error for %s attempt %d/%d: %s. Sleeping %.1fs",
                obo_id,
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    log.warning("Failed to fetch parents of %s after %d retries: %s", obo_id, max_retries, last_error)
    return []


def get_parents_cached(
    *,
    foodon_id: str,
    ontology: str,
    cache: dict[str, list[dict[str, Any]]],
    cache_path: Path,
    use_cache: bool,
    refresh_cache: bool,
    write_cache: bool,
    ols_base_template: str,
    timeout_s: int,
    user_agent: str,
    max_retries: int,
    retry_backoff_s: float,
    sleep_s: float,
) -> tuple[list[dict[str, Any]], bool]:
    key = cache_key(foodon_id=foodon_id, ontology=ontology)

    if use_cache and not refresh_cache and key in cache:
        return cache[key], False

    parents = fetch_parents(
        foodon_id=foodon_id,
        ontology=ontology,
        ols_base_template=ols_base_template,
        timeout_s=timeout_s,
        user_agent=user_agent,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )

    cache[key] = parents

    if write_cache:
        append_cache(
            cache_path,
            key=key,
            foodon_id=foodon_id,
            ontology=ontology,
            parents=parents,
        )

    if sleep_s > 0:
        time.sleep(sleep_s)

    return parents, True


# ---------------------------------------------------------------------------
# Ancestor traversal
# ---------------------------------------------------------------------------

def register_foodon_term(
    terms: dict[str, dict[str, Any]],
    *,
    foodon_id: str,
    label: str = "",
    iri: str = "",
    source: str,
) -> None:
    obo_id = normalize_foodon_id(foodon_id)
    if not obo_id:
        return

    existing = terms.get(obo_id, {})
    terms[obo_id] = {
        "foodon_id": obo_id,
        "label": safe_str(existing.get("label")) or safe_str(label),
        "iri": safe_str(existing.get("iri")) or safe_str(iri) or foodon_iri(obo_id),
        "sources": sorted(set(existing.get("sources", [])) | {source}),
    }


def collect_foodon_hierarchy(
    *,
    coverage: pd.DataFrame,
    ontology: str,
    cache: dict[str, list[dict[str, Any]]],
    cache_path: Path,
    use_cache: bool,
    refresh_cache: bool,
    write_cache: bool,
    ols_base_template: str,
    max_depth: int,
    sleep_s: float,
    timeout_s: int,
    user_agent: str,
    max_retries: int,
    retry_backoff_s: float,
) -> tuple[
    dict[str, dict[str, Any]],
    set[tuple[str, str]],
    list[tuple[str, str]],
    int,
    pd.DataFrame,
]:
    """Return FoodOn terms, is_a edges, leaf mappings, new lookup count, traversal audit."""
    foodon_terms: dict[str, dict[str, Any]] = {}
    is_a_edges: set[tuple[str, str]] = set()
    leaf_mappings: list[tuple[str, str]] = []
    traversal_rows: list[dict[str, Any]] = []
    new_lookups = 0

    for _, row in coverage.iterrows():
        ingredient_id = safe_str(row["ingredient_id"])
        leaf_id = normalize_foodon_id(row["foodon_id"])
        leaf_label = safe_str(row.get("foodon_label") or row.get("matched_text") or "")
        leaf_iri = safe_str(row.get("foodon_url"))

        register_foodon_term(
            foodon_terms,
            foodon_id=leaf_id,
            label=leaf_label,
            iri=leaf_iri,
            source="validated_leaf",
        )
        leaf_mappings.append((ingredient_id, leaf_id))

        frontier = deque([(leaf_id, 0)])
        visited = {leaf_id}

        while frontier:
            child_id, depth = frontier.popleft()
            if depth >= max_depth:
                continue

            parents, was_new = get_parents_cached(
                foodon_id=child_id,
                ontology=ontology,
                cache=cache,
                cache_path=cache_path,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                write_cache=write_cache,
                ols_base_template=ols_base_template,
                timeout_s=timeout_s,
                user_agent=user_agent,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                sleep_s=sleep_s,
            )
            if was_new:
                new_lookups += 1

            for parent in parents:
                if parent.get("is_obsolete"):
                    continue

                parent_id = normalize_foodon_id(safe_str(parent.get("obo_id")))
                if not parent_id:
                    continue

                register_foodon_term(
                    foodon_terms,
                    foodon_id=parent_id,
                    label=safe_str(parent.get("label")),
                    iri=safe_str(parent.get("iri")),
                    source="ancestor",
                )

                is_a_edges.add((child_id, parent_id))

                traversal_rows.append({
                    "ingredient_id": ingredient_id,
                    "leaf_foodon_id": leaf_id,
                    "child_foodon_id": child_id,
                    "parent_foodon_id": parent_id,
                    "parent_label": safe_str(parent.get("label")),
                    "depth": int(depth + 1),
                })

                if parent_id not in visited:
                    visited.add(parent_id)
                    frontier.append((parent_id, depth + 1))

    traversal_df = pd.DataFrame(traversal_rows)
    return foodon_terms, is_a_edges, leaf_mappings, new_lookups, traversal_df


# ---------------------------------------------------------------------------
# Graph injection
# ---------------------------------------------------------------------------

def inject_foodon(
    *,
    graph: Any,
    coverage: pd.DataFrame,
    foodon_terms: dict[str, dict[str, Any]],
    is_a_edges: set[tuple[str, str]],
    leaf_mappings: list[tuple[str, str]],
) -> tuple[dict[str, int], pd.DataFrame]:
    stats = Counter()
    audit_rows: list[dict[str, Any]] = []

    for foodon_id, info in sorted(foodon_terms.items()):
        node_key = foodon_node_key(foodon_id)

        if node_key not in graph:
            graph.add_node(
                node_key,
                node_type="FoodOnClass",
                foodon_id=foodon_id,
                label=safe_str(info.get("label")),
                iri=safe_str(info.get("iri")) or foodon_iri(foodon_id),
                ontology="foodon",
                source="EBI OLS",
            )
            stats["foodon_nodes_added"] += 1
        else:
            graph.nodes[node_key].setdefault("node_type", "FoodOnClass")
            graph.nodes[node_key].setdefault("foodon_id", foodon_id)
            if safe_str(info.get("label")) and not safe_str(graph.nodes[node_key].get("label")):
                graph.nodes[node_key]["label"] = safe_str(info.get("label"))
            if safe_str(info.get("iri")) and not safe_str(graph.nodes[node_key].get("iri")):
                graph.nodes[node_key]["iri"] = safe_str(info.get("iri"))
            stats["foodon_nodes_existing"] += 1

    coverage_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in coverage.iterrows():
        coverage_by_pair[(safe_str(row["ingredient_id"]), normalize_foodon_id(row["foodon_id"]))] = dict(row)

    for ingredient_id, foodon_id in sorted(set(leaf_mappings)):
        if ingredient_id not in graph:
            stats["missing_ingredient_skipped"] += 1
            continue

        target_key = foodon_node_key(foodon_id)
        if target_key not in graph:
            stats["missing_foodon_target_skipped"] += 1
            continue

        if not graph.has_edge(ingredient_id, target_key):
            graph.add_edge(
                ingredient_id,
                target_key,
                edge_type="mapped_to_foodon",
                mapping_source="foodon_coverage_validated",
            )
            stats["mapped_to_foodon_edges_added"] += 1
        else:
            graph.edges[ingredient_id, target_key].setdefault("edge_type", "mapped_to_foodon")
            stats["mapped_to_foodon_edges_existing"] += 1

        coverage_row = coverage_by_pair.get((ingredient_id, foodon_id), {})
        audit_rows.append({
            "ingredient_id": ingredient_id,
            "ingredient_label": ingredient_display_label(graph, ingredient_id),
            "foodon_id": foodon_id,
            "foodon_label": safe_str(foodon_terms.get(foodon_id, {}).get("label")),
            "foodon_node": target_key,
            "match_type": safe_str(coverage_row.get("match_type")),
            "confidence": coverage_row.get("confidence", ""),
            "validated_keep": coverage_row.get("validated_keep", ""),
        })

    for child_id, parent_id in sorted(is_a_edges):
        child_key = foodon_node_key(child_id)
        parent_key = foodon_node_key(parent_id)

        if child_key not in graph or parent_key not in graph:
            stats["is_a_missing_node_skipped"] += 1
            continue

        if not graph.has_edge(child_key, parent_key):
            graph.add_edge(
                child_key,
                parent_key,
                edge_type="is_a",
                ontology="foodon",
                source="EBI OLS",
            )
            stats["is_a_edges_added"] += 1
        else:
            graph.edges[child_key, parent_key].setdefault("edge_type", "is_a")
            stats["is_a_edges_existing"] += 1

    return dict(stats), pd.DataFrame(audit_rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

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


def write_graph(graph: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Wrote enriched graph: %s", path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote CSV: %s (%d rows)", path, len(df))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON: %s", path)


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
        help=f"Output graph. Default: {DEFAULT_GRAPH_OUT}",
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=DEFAULT_COVERAGE_CSV,
        help=f"Validated coverage CSV. Default: {DEFAULT_COVERAGE_CSV}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        help="Audit CSV path. Default: data-dir/foodon_enrichment_audit.csv.",
    )
    parser.add_argument(
        "--traversal-out",
        type=Path,
        help="Ancestor traversal CSV path. Default: data-dir/foodon_ancestor_traversal.csv.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        help="Stats JSON path. Default: data-dir/foodon_enrichment_stats.json.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Ancestor cache JSONL path. Default: data-dir/foodon_ancestor_cache.jsonl.",
    )
    parser.add_argument(
        "--ontology",
        default=DEFAULT_ONTOLOGY,
        help=f"OLS ontology slug. Default: {DEFAULT_ONTOLOGY}",
    )
    parser.add_argument(
        "--ols-base",
        default=DEFAULT_OLS_ONTOLOGY_BASE,
        help=(
            "OLS ontology API base URL or template. "
            "Default: https://www.ebi.ac.uk/ols4/api/ontologies/{ontology}"
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum ancestor levels to traverse.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds to sleep between uncached API calls.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries per OLS lookup.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Initial exponential backoff in seconds.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read existing ancestor cache.",
    )
    parser.add_argument(
        "--no-write-cache",
        action="store_true",
        help="Do not append new ancestor lookups to cache.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached hits and re-query all terms.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch/compute additions but do not save graph.",
    )
    parser.add_argument(
        "--write-dry-run-outputs",
        action="store_true",
        help="With --dry-run, still write audit/stats/traversal outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    args.audit_out = args.audit_out or args.data_dir / "foodon_enrichment_audit.csv"
    args.traversal_out = args.traversal_out or args.data_dir / "foodon_ancestor_traversal.csv"
    args.stats_out = args.stats_out or args.data_dir / "foodon_enrichment_stats.json"
    args.cache = args.cache or args.data_dir / "foodon_ancestor_cache.jsonl"

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    if args.max_depth < 0:
        log.error("--max-depth must be >= 0")
        return 2
    if args.sleep < 0:
        log.error("--sleep must be >= 0")
        return 2
    if args.timeout < 1:
        log.error("--timeout must be >= 1")
        return 2
    if args.max_retries < 1:
        log.error("--max-retries must be >= 1")
        return 2
    if args.retry_backoff < 0:
        log.error("--retry-backoff must be >= 0")
        return 2

    try:
        args.data_dir.mkdir(parents=True, exist_ok=True)

        coverage = load_validated_coverage(args.coverage_csv)
        graph = load_graph(args.graph_in)

        nodes_before = graph.number_of_nodes()
        edges_before = graph.number_of_edges()
        node_types_before = node_type_counts(graph)
        edge_types_before = edge_type_counts(graph)

        cache = load_cache(
            args.cache,
            enabled=not args.no_cache and not args.refresh_cache,
        )

        foodon_terms, is_a_edges, leaf_mappings, new_lookups, traversal_df = collect_foodon_hierarchy(
            coverage=coverage,
            ontology=args.ontology,
            cache=cache,
            cache_path=args.cache,
            use_cache=not args.no_cache,
            refresh_cache=args.refresh_cache,
            write_cache=not args.no_write_cache,
            ols_base_template=args.ols_base,
            max_depth=args.max_depth,
            sleep_s=args.sleep,
            timeout_s=args.timeout,
            user_agent=args.user_agent,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff,
        )

        log.info("Collected %d unique FoodOn classes", len(foodon_terms))
        log.info("Collected %d is_a edges", len(is_a_edges))
        log.info("New OLS lookups this run: %d", new_lookups)

        injection_stats, audit_df = inject_foodon(
            graph=graph,
            coverage=coverage,
            foodon_terms=foodon_terms,
            is_a_edges=is_a_edges,
            leaf_mappings=leaf_mappings,
        )

        stats = {
            "inputs": {
                "graph_in": str(args.graph_in),
                "coverage_csv": str(args.coverage_csv),
            },
            "outputs": {
                "graph_out": None if args.dry_run else str(args.graph_out),
                "audit_csv": str(args.audit_out),
                "traversal_csv": str(args.traversal_out),
                "ancestor_cache_jsonl": str(args.cache),
                "stats_json": str(args.stats_out),
            },
            "parameters": {
                "ontology": args.ontology,
                "max_depth": args.max_depth,
                "sleep": args.sleep,
                "timeout": args.timeout,
                "max_retries": args.max_retries,
                "use_cache": not args.no_cache,
                "refresh_cache": args.refresh_cache,
                "write_cache": not args.no_write_cache,
                "dry_run": args.dry_run,
            },
            "coverage": {
                "n_validated_rows": int(len(coverage)),
                "n_leaf_mappings": int(len(set(leaf_mappings))),
            },
            "hierarchy": {
                "n_foodon_classes_collected": int(len(foodon_terms)),
                "n_is_a_edges_collected": int(len(is_a_edges)),
                "new_ols_lookups": int(new_lookups),
            },
            "graph_before": {
                "nodes": int(nodes_before),
                "edges": int(edges_before),
                "node_types": node_types_before,
                "edge_types": edge_types_before,
            },
            "graph_after": {
                "nodes": int(graph.number_of_nodes()),
                "edges": int(graph.number_of_edges()),
                "node_types": node_type_counts(graph),
                "edge_types": edge_type_counts(graph),
            },
            "injection_stats": injection_stats,
        }

        if args.dry_run and not args.write_dry_run_outputs:
            log.info("[dry-run] No outputs written.")
            print(json.dumps({
                "n_validated_rows": len(coverage),
                "n_foodon_classes_collected": len(foodon_terms),
                "n_is_a_edges_collected": len(is_a_edges),
                "new_ols_lookups": new_lookups,
                "injection_stats": injection_stats,
            }, indent=2, ensure_ascii=False))
            return 0

        write_csv(audit_df, args.audit_out)
        write_csv(traversal_df, args.traversal_out)
        write_json(args.stats_out, stats)

        if not args.dry_run:
            write_graph(graph, args.graph_out)

        log.info("Done.")
        log.info(
            "FoodOnClass nodes added: %s; mapped_to_foodon edges added: %s; is_a edges added: %s",
            injection_stats.get("foodon_nodes_added", 0),
            injection_stats.get("mapped_to_foodon_edges_added", 0),
            injection_stats.get("is_a_edges_added", 0),
        )
        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        pickle.PickleError,
        json.JSONDecodeError,
        OSError,
        pd.errors.ParserError,
    ) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
