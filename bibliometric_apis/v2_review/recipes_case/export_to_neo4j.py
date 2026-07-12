"""
==================

Export the graph to Neo4j.

The script reads a trusted NetworkX gpickle graph and creates typed Neo4j nodes
and relationships using batched MERGE operations. It is idempotent: re-running
the script updates existing nodes/relationships rather than duplicating them.

Default input
-------------

  data/graph_step4_foodon.gpickle

Environment variables
---------------------

  NEO4J_URI          bolt://localhost:7690
  NEO4J_RECIPES_DB   budget
  NEO4J_USER         neo4j
  NEO4J_PASSWORD     <password>
  NEO4J_NO_AUTH      true/false
  ALLOWED_NEO4J_DBS  budget             # comma-separated allow-list

Optional .env loading
---------------------

  python export_to_neo4j.py --dotenv .env

Usage
-----

  python export_to_neo4j.py --graph data/graph_step4_foodon.gpickle

  python export_to_neo4j.py \
    --graph data/graph_step4_foodon.gpickle \
    --uri bolt://localhost:7690 \
    --db budget

  python export_to_neo4j.py --graph data/graph_step4_foodon.gpickle --wipe

  python export_to_neo4j.py --dry-run

Neo4j setup
-----------

In Neo4j Browser, create the target database first if needed:

  CREATE DATABASE budget IF NOT EXISTS

"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

from neo4j import GraphDatabase


DEFAULT_GRAPH = Path("data/graph_step4_foodon.gpickle")
DEFAULT_URI = "bolt://localhost:7690"
DEFAULT_DB = "budget"

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("export_to_neo4j")


# ---------------------------------------------------------------------------
# Graph schema mapping
# ---------------------------------------------------------------------------

# Map graph node_type -> Neo4j label.
LABELS = {
    "Recipe": "Recipe",
    "Ingredient": "Ingredient",
    "Tool": "Tool",
    "Action": "Action",
    "Place": "Place",
    "Period": "Period",
    "FoodOnClass": "FoodOnClass",
}

# Map graph edge_type -> Neo4j relationship type.
RELS = {
    "contains": "CONTAINS",
    "uses_tool": "USES_TOOL",
    "performs": "PERFORMS",
    "origin": "ORIGIN",
    "dated": "DATED",
    "is_a": "IS_A",
    "mapped_to_foodon": "MAPPED_TO_FOODON",
}

# Node attributes to export. Large text fields such as recipe_text and
# translation_en are intentionally excluded to keep the Neo4j graph light.
NODE_PROPS = {
    "title",
    "source_id",
    "source_title",
    "source_author",
    "source_year",
    "source_place",
    "source_language",
    "period_derived",
    "canonical_name",
    "canonical_verb",
    "name",
    "label",
    "diaspora_branch",
    "ingredient_category",
    "region_macro",
    "year_start",
    "year_end",
    "n_occurrences",
    "mean_confidence",
    "ner_noise_flag",
    "foodon_id",
    "iri",
    "definition",
}

# Relationship attributes to export.
REL_PROPS = {
    "confidence_score",
    "specific_forms_used",
    "mapping_score",
    "mapping_method",
    "source",
}


# ---------------------------------------------------------------------------
# Environment and loading
# ---------------------------------------------------------------------------

def load_dotenv_file(path: Path | None) -> None:
    """Load KEY=VALUE pairs from a .env file without requiring python-dotenv.

    Existing environment variables are not overwritten.
    """
    if path is None:
        return

    if not path.exists():
        raise FileNotFoundError(f".env file not found: {path}")

    with path.open(encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key.isidentifier() and key not in os.environ:
                os.environ[key] = value

    log.info("Loaded environment variables from %s", path)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "none", "disabled"}


def parse_allowed_dbs(raw: str | None) -> set[str]:
    if not raw:
        return {DEFAULT_DB}
    return {item.strip() for item in raw.split(",") if item.strip()}


def load_graph(path: Path) -> Any:
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
# Property cleaning
# ---------------------------------------------------------------------------

def is_neo4j_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def clean_value(value: Any) -> Any:
    """Return a Neo4j-safe value or None.

    Neo4j properties can be scalars or homogeneous-ish lists of scalar values.
    Dicts and nested structures are dropped.
    """
    if value is None:
        return None

    if is_neo4j_scalar(value):
        return value

    if isinstance(value, (list, tuple, set)):
        cleaned = [x for x in value if is_neo4j_scalar(x)]
        return cleaned if cleaned else None

    return None


def clean_props(attrs: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    props: dict[str, Any] = {}

    for key in sorted(allowed):
        if key not in attrs:
            continue

        value = clean_value(attrs.get(key))
        if value is not None:
            props[key] = value

    return props


def batched(items: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []

    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Export preparation
# ---------------------------------------------------------------------------

def rows_by_label(graph: Any) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    skipped = 0

    for node_id, attrs in graph.nodes(data=True):
        label = LABELS.get(attrs.get("node_type"))
        if not label:
            skipped += 1
            continue

        row = {
            "id": str(node_id),
            **clean_props(dict(attrs), NODE_PROPS),
        }
        grouped.setdefault(label, []).append(row)

    if skipped:
        log.warning("Skipped %d nodes with unmapped node_type", skipped)

    for rows in grouped.values():
        rows.sort(key=lambda row: row["id"])

    return grouped


def rows_by_relationship_type(graph: Any) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    skipped = 0

    for source, target, attrs in graph.edges(data=True):
        rel_type = RELS.get(attrs.get("edge_type"))
        if not rel_type:
            skipped += 1
            continue

        row = {
            "u": str(source),
            "v": str(target),
            "props": clean_props(dict(attrs), REL_PROPS),
        }
        grouped.setdefault(rel_type, []).append(row)

    if skipped:
        log.warning("Skipped %d edges with unmapped edge_type", skipped)

    for rows in grouped.values():
        rows.sort(key=lambda row: (row["u"], row["v"]))

    return grouped


# ---------------------------------------------------------------------------
# Neo4j operations
# ---------------------------------------------------------------------------

def create_constraints(session: Any) -> None:
    """Create uniqueness constraints for id on each exported label."""
    for label in sorted(set(LABELS.values())):
        session.run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
            "REQUIRE n.id IS UNIQUE"
        )


def export_nodes(session: Any, grouped_nodes: dict[str, list[dict[str, Any]]], batch_size: int) -> None:
    for label, rows in sorted(grouped_nodes.items()):
        log.info("%s: %d nodes", label, len(rows))

        for batch in batched(rows, batch_size):
            session.run(
                f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{id: row.id}})
                SET n += row
                """,
                rows=batch,
            )


def export_relationships(
    session: Any,
    grouped_relationships: dict[str, list[dict[str, Any]]],
    batch_size: int,
) -> None:
    for rel_type, rows in sorted(grouped_relationships.items()):
        log.info("%s: %d relationships", rel_type, len(rows))

        for batch in batched(rows, batch_size):
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (a {{id: row.u}})
                MATCH (b {{id: row.v}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += row.props
                """,
                rows=batch,
            )


def make_driver(uri: str, user: str, password: str, no_auth: bool):
    if no_auth:
        log.info("Connecting to Neo4j with auth disabled")
        return GraphDatabase.driver(uri, auth=None)

    log.info("Connecting to Neo4j as user %r", user)
    return GraphDatabase.driver(uri, auth=(user, password))


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
        "--dotenv",
        type=Path,
        help="Optional .env file to load. Existing env vars are not overwritten.",
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", DEFAULT_URI),
        help=f"Neo4j Bolt URI. Default: env NEO4J_URI or {DEFAULT_URI}",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("NEO4J_RECIPES_DB", DEFAULT_DB),
        help=f"Target Neo4j database. Default: env NEO4J_RECIPES_DB or {DEFAULT_DB}",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username. Default: env NEO4J_USER or neo4j.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD", ""),
        help="Neo4j password. Prefer env NEO4J_PASSWORD over CLI for security.",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        default=truthy(os.environ.get("NEO4J_NO_AUTH")),
        help="Connect with auth=None. Can also set NEO4J_NO_AUTH=true.",
    )
    parser.add_argument(
        "--allowed-db",
        action="append",
        help=(
            "Allowed database name. Can be repeated. "
            "Defaults to env ALLOWED_NEO4J_DBS or budget."
        ),
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete all nodes/relationships in the target database before export.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1000,
        help="Batch size for Neo4j writes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load graph and print export counts without writing to Neo4j.",
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

    if args.batch < 1:
        log.error("--batch must be >= 1")
        return 2

    try:
        load_dotenv_file(args.dotenv)

        # Re-read env after optional .env loading, unless explicit CLI values were supplied
        # before .env. This keeps simple usage intuitive:
        #
        #   python export_to_neo4j.py --dotenv .env
        #
        uri = args.uri or os.environ.get("NEO4J_URI", DEFAULT_URI)
        db = args.db or os.environ.get("NEO4J_RECIPES_DB", DEFAULT_DB)
        user = args.user or os.environ.get("NEO4J_USER", "neo4j")
        password = args.password if args.password != "" else os.environ.get("NEO4J_PASSWORD", "")
        no_auth = bool(args.no_auth or truthy(os.environ.get("NEO4J_NO_AUTH")))

        allowed_dbs = (
            set(args.allowed_db)
            if args.allowed_db
            else parse_allowed_dbs(os.environ.get("ALLOWED_NEO4J_DBS"))
        )

        if db not in allowed_dbs:
            log.error("Refusing to run against database %r. Allowed databases: %s", db, sorted(allowed_dbs))
            log.error("Set NEO4J_RECIPES_DB=%s or ALLOWED_NEO4J_DBS=%s if this is intentional.", DEFAULT_DB, db)
            return 1

        graph = load_graph(args.graph)
        grouped_nodes = rows_by_label(graph)
        grouped_relationships = rows_by_relationship_type(graph)

        total_nodes = sum(len(rows) for rows in grouped_nodes.values())
        total_rels = sum(len(rows) for rows in grouped_relationships.values())

        log.info("Prepared %d exportable nodes across %d labels", total_nodes, len(grouped_nodes))
        log.info(
            "Prepared %d exportable relationships across %d types",
            total_rels,
            len(grouped_relationships),
        )

        if args.dry_run:
            print("\nDRY RUN")
            print(f"  graph: {args.graph}")
            print(f"  uri:   {uri}")
            print(f"  db:    {db}")
            print("\nNodes:")
            for label, rows in sorted(grouped_nodes.items()):
                print(f"  {label:15s} {len(rows):8d}")
            print("\nRelationships:")
            for rel_type, rows in sorted(grouped_relationships.items()):
                print(f"  {rel_type:20s} {len(rows):8d}")
            return 0

        driver = make_driver(uri, user, password, no_auth=no_auth)

        try:
            with driver.session(database=db) as session:
                if args.wipe:
                    log.warning("Wiping database %r with MATCH (n) DETACH DELETE n", db)
                    session.run("MATCH (n) DETACH DELETE n")

                create_constraints(session)
                export_nodes(session, grouped_nodes, args.batch)
                export_relationships(session, grouped_relationships, args.batch)
        finally:
            driver.close()

        print("\nDone. Open Neo4j Browser and try:")
        print(f"  :use {db}")
        print("  MATCH (r:Recipe)-[:CONTAINS]->(i:Ingredient) RETURN r,i LIMIT 100")
        print("  MATCH (i:Ingredient)-[:MAPPED_TO_FOODON]->(f:FoodOnClass) RETURN i,f LIMIT 100")

        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        pickle.PickleError,
    ) as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:
        log.error("Neo4j export failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
