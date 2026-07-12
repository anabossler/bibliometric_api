"""

=================================

Apply reviewed ingredient-canonicalization decisions to the RELISH graph.

This script reads decision files produced by Steps 2.5 / 2.5b / 2.5c and
creates a canonicalized graph. It is conservative by default:

  - auto-merge pairs are applied
  - accepted_ids from the human-review CSV are applied
  - expert correction files are applied when present
  - deletion-candidate files are applied only for rows explicitly accepted,
    unless --delete-all-candidates is used
  - raw LLM suggestions are NOT applied unless --use-llm-suggestions is used

Conflict rules
--------------

1. Later / more explicit decisions override earlier ones.
2. Deletion wins over merge.
3. KEEP_SEPARATE protection can block merges when the same node is judged
   distinct in another hub.
4. Transitive closure is applied: if A -> B and B -> C, then A -> C.
5. Cycles are detected and skipped rather than silently producing bad mappings.

Default inputs
--------------

  data/graph_step1.gpickle
  data/canonicalization_auto_merge.csv
  data/canonicalization_needs_review_llm.csv
  data/llm_review_corrections_filled.csv
  data/canonicalization_deletion_candidates.csv

Default outputs
---------------

  data/graph_step2_canonical.gpickle
  data/canonicalization_mapping.csv
  data/graph_step2_stats.json

Usage
-----

  python step2_6_apply_canonicalization.py

  python step2_6_apply_canonicalization.py --dry-run

  python step2_6_apply_canonicalization.py \
    --graph-in data/graph_step1.gpickle \
    --graph-out data/graph_step2_canonical.gpickle

  python step2_6_apply_canonicalization.py \
    --use-llm-suggestions \
    --delete-all-candidates


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
import pandas as pd


DEFAULT_DATA_DIR = Path("data")
DEFAULT_GRAPH_IN = DEFAULT_DATA_DIR / "graph_step1.gpickle"
DEFAULT_GRAPH_OUT = DEFAULT_DATA_DIR / "graph_step2_canonical.gpickle"

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step2_6_apply_canonicalization")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def parse_id_list(value: Any) -> list[str]:
    """Parse comma/semicolon/pipe-separated ids into a clean list."""
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    # Support common separators used in manually edited CSVs.
    pieces = []
    for chunk in text.replace(";", ",").replace("|", ",").split(","):
        item = chunk.strip()
        if item:
            pieces.append(item)

    return pieces


def truthy(value: Any) -> bool:
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass

    return str(value).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "accepted",
        "accept",
        "delete",
        "merge",
    }


def read_csv_optional(path: Path, *, required_columns: set[str], required: bool = False) -> pd.DataFrame:
    """Read CSV if present. Return empty DataFrame if optional and missing."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required CSV not found: {path}")
        log.warning("Optional CSV not found; skipping: %s", path)
        return pd.DataFrame(columns=sorted(required_columns))

    log.info("Loading CSV: %s", path)
    df = pd.read_csv(path)

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    log.info("Loaded %d rows from %s", len(df), path)
    return df


def load_graph(path: Path) -> nx.DiGraph:
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    log.info("Loading graph: %s", path)
    with path.open("rb") as fh:
        graph = pickle.load(fh)

    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError(f"Object loaded from {path} does not look like a NetworkX graph")

    log.info("Loaded graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def ingredient_node_ids(graph: nx.Graph) -> set[str]:
    return {
        str(node_id)
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "Ingredient"
    }


# ---------------------------------------------------------------------------
# Decision collection
# ---------------------------------------------------------------------------

def add_merge(
    merge_decisions: dict[str, tuple[str, str]],
    *,
    source_id: str,
    target_id: str,
    tag: str,
) -> None:
    source_id = str(source_id).strip()
    target_id = str(target_id).strip()

    if not source_id or not target_id or source_id == target_id:
        return

    merge_decisions[source_id] = (tag, target_id)


def collect_auto_merges(path: Path) -> dict[str, tuple[str, str]]:
    df = read_csv_optional(
        path,
        required_columns={"canonical_root_id", "merged_id"},
    )

    merge_decisions: dict[str, tuple[str, str]] = {}

    for _, row in df.iterrows():
        tag = str(row.get("category", row.get("suggested_action", "auto_merge"))).strip() or "auto_merge"
        add_merge(
            merge_decisions,
            source_id=row["merged_id"],
            target_id=row["canonical_root_id"],
            tag=tag,
        )

    log.info("Collected %d auto-merge decisions", len(merge_decisions))
    return merge_decisions


def collect_review_decisions(
    path: Path,
    *,
    use_llm_suggestions: bool,
) -> tuple[dict[str, tuple[str, str]], set[str], dict[str, str]]:
    """Return merges, keep_separate_protected, deletes from review CSV.

    Conservative default:
      - accepted_ids drives merges
      - LLM suggestions are ignored unless --use-llm-suggestions is set
    """
    df = read_csv_optional(
        path,
        required_columns={"hub_id"},
    )

    merges: dict[str, tuple[str, str]] = {}
    keep_separate: set[str] = set()
    deletes: dict[str, str] = {}

    for _, row in df.iterrows():
        hub = str(row["hub_id"]).strip()
        if not hub:
            continue

        accepted_ids = parse_id_list(row.get("accepted_ids"))

        for candidate in accepted_ids:
            add_merge(
                merges,
                source_id=candidate,
                target_id=hub,
                tag="human_review_accepted",
            )

        # LLM suggestions remain suggestions unless explicitly enabled.
        if use_llm_suggestions:
            for candidate in parse_id_list(row.get("llm_merge_ids")):
                if candidate not in accepted_ids:
                    add_merge(
                        merges,
                        source_id=candidate,
                        target_id=hub,
                        tag="llm_review_suggestion",
                    )

            for candidate in parse_id_list(row.get("llm_delete_ids")):
                deletes[candidate] = "llm_review_delete"

            keep_separate.update(parse_id_list(row.get("llm_keep_separate_ids")))

        # If the human-edited file has explicit keep/delete columns, honor them.
        keep_separate.update(parse_id_list(row.get("accepted_keep_separate_ids")))
        keep_separate.update(parse_id_list(row.get("keep_separate_ids")))

        for candidate in parse_id_list(row.get("accepted_delete_ids")):
            deletes[candidate] = "human_review_delete"

    log.info(
        "Collected review decisions: %d merges, %d keep-separate protections, %d deletes",
        len(merges),
        len(keep_separate),
        len(deletes),
    )

    return merges, keep_separate, deletes


def collect_expert_corrections(path: Path) -> tuple[dict[str, tuple[str, str]], set[str], set[tuple[str, str]]]:
    """Return expert merges, keep-separate protections, and LLM merge reverts."""
    if not path.exists():
        log.warning("Expert corrections file not found; skipping: %s", path)
        return {}, set(), set()

    log.info("Loading expert corrections: %s", path)
    df = pd.read_csv(path)
    log.info("Loaded %d expert correction rows", len(df))

    merges: dict[str, tuple[str, str]] = {}
    keep_separate: set[str] = set()
    reverts: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        hub = str(row.get("hub_id", "")).strip()
        if not hub:
            continue

        for candidate in parse_id_list(row.get("original_llm_merge")):
            reverts.add((candidate, hub))

        for candidate in parse_id_list(row.get("expert_decision_merge")):
            add_merge(
                merges,
                source_id=candidate,
                target_id=hub,
                tag="expert_correction",
            )

        keep_separate.update(parse_id_list(row.get("expert_decision_keep_separate")))
        keep_separate.update(parse_id_list(row.get("expert_keep_separate")))
        keep_separate.update(parse_id_list(row.get("keep_separate_ids")))

    log.info(
        "Collected expert corrections: %d merges, %d keep-separate protections, %d reverts",
        len(merges),
        len(keep_separate),
        len(reverts),
    )

    return merges, keep_separate, reverts


def collect_deletions(path: Path, *, delete_all_candidates: bool) -> dict[str, str]:
    df = read_csv_optional(
        path,
        required_columns={"node_id"},
    )

    deletions: dict[str, str] = {}

    for _, row in df.iterrows():
        node_id = str(row["node_id"]).strip()
        if not node_id:
            continue

        accepted_columns = [
            "accepted_for_deletion",
            "accepted",
            "delete",
            "confirmed",
        ]
        explicit_accept = any(
            column in df.columns and truthy(row.get(column))
            for column in accepted_columns
        )

        if not delete_all_candidates and not explicit_accept:
            continue

        reason = str(row.get("reason", "deletion_candidate")).strip() or "deletion_candidate"
        deletions[node_id] = reason

    mode = "all candidates" if delete_all_candidates else "accepted candidates only"
    log.info("Collected %d deletions from %s (%s)", len(deletions), path, mode)
    return deletions


def collect_all_decisions(args: argparse.Namespace) -> tuple[
    dict[str, tuple[str, str]],
    dict[str, str],
    set[str],
    dict[str, Any],
]:
    """Collect and resolve merge/deletion/keep-separate decisions."""
    merge_decisions: dict[str, tuple[str, str]] = {}
    deletion_decisions: dict[str, str] = {}
    keep_separate_protected: set[str] = set()

    decision_stats: dict[str, Any] = {}

    # 1. Auto merges, lowest priority.
    auto_merges = collect_auto_merges(args.auto_merge)
    merge_decisions.update(auto_merges)
    decision_stats["auto_merges_loaded"] = len(auto_merges)

    # 2. Review file.
    review_merges, review_keep, review_deletes = collect_review_decisions(
        args.review,
        use_llm_suggestions=args.use_llm_suggestions,
    )
    merge_decisions.update(review_merges)
    keep_separate_protected.update(review_keep)
    deletion_decisions.update(review_deletes)
    decision_stats["review_merges_loaded"] = len(review_merges)
    decision_stats["review_deletes_loaded"] = len(review_deletes)
    decision_stats["review_keep_separate_loaded"] = len(review_keep)

    # 3. Expert corrections override review decisions.
    expert_merges, expert_keep, expert_reverts = collect_expert_corrections(args.expert_corrections)

    n_reverted = 0
    for source_id, target_id in expert_reverts:
        existing = merge_decisions.get(source_id)
        if existing is not None and existing[1] == target_id and existing[0].startswith("llm"):
            del merge_decisions[source_id]
            n_reverted += 1

    merge_decisions.update(expert_merges)
    keep_separate_protected.update(expert_keep)

    decision_stats["expert_merges_loaded"] = len(expert_merges)
    decision_stats["expert_keep_separate_loaded"] = len(expert_keep)
    decision_stats["expert_reverts_applied"] = n_reverted

    # 4. Explicit deletion file.
    deletion_file_decisions = collect_deletions(
        args.deletions,
        delete_all_candidates=args.delete_all_candidates,
    )
    deletion_decisions.update(deletion_file_decisions)
    decision_stats["deletion_file_loaded"] = len(deletion_file_decisions)

    # 5. Deletion wins over merge.
    deletion_merge_conflicts = 0
    for node_id in list(deletion_decisions):
        if node_id in merge_decisions:
            del merge_decisions[node_id]
            deletion_merge_conflicts += 1

    decision_stats["deletion_merge_conflicts_removed"] = deletion_merge_conflicts

    # 6. KEEP_SEPARATE protection wins over merge if enabled.
    keep_conflicts = 0
    if args.keep_separate_wins:
        for node_id in list(keep_separate_protected):
            if node_id in merge_decisions:
                del merge_decisions[node_id]
                keep_conflicts += 1

    decision_stats["keep_separate_conflicts_removed"] = keep_conflicts

    return merge_decisions, deletion_decisions, keep_separate_protected, decision_stats


# ---------------------------------------------------------------------------
# Transitive closure and conflict checks
# ---------------------------------------------------------------------------

def transitive_closure(
    merge_decisions: dict[str, tuple[str, str]],
) -> tuple[dict[str, tuple[str, str]], list[list[str]]]:
    """Resolve A->B and B->C to A->C. Cycles are reported and skipped."""
    closed: dict[str, tuple[str, str]] = {}
    cycles: list[list[str]] = []

    for source, (tag, target) in merge_decisions.items():
        visited_order = [source]
        visited = {source}
        current = target

        while current in merge_decisions:
            if current in visited:
                cycle_start = visited_order.index(current) if current in visited_order else 0
                cycles.append(visited_order[cycle_start:] + [current])
                current = target
                break

            visited.add(current)
            visited_order.append(current)
            current = merge_decisions[current][1]

        closed[source] = (tag, current)

    if cycles:
        cycle_sources = {node for cycle in cycles for node in cycle}
        closed = {
            source: decision
            for source, decision in closed.items()
            if source not in cycle_sources
        }
        log.warning("Detected %d merge cycles; skipped nodes involved in cycles", len(cycles))

    n_changed = sum(
        1
        for source in closed
        if source in merge_decisions and merge_decisions[source][1] != closed[source][1]
    )
    if n_changed:
        log.info("Transitive closure rewrote %d merge targets", n_changed)

    return closed, cycles


def filter_invalid_decisions(
    graph: nx.Graph,
    merge_decisions: dict[str, tuple[str, str]],
    deletion_decisions: dict[str, str],
) -> tuple[dict[str, tuple[str, str]], dict[str, str], dict[str, int]]:
    """Drop decisions for missing/non-Ingredient nodes or missing targets."""
    ing_ids = ingredient_node_ids(graph)
    stats = Counter()

    filtered_merges: dict[str, tuple[str, str]] = {}
    for source, (tag, target) in merge_decisions.items():
        if source not in graph:
            stats["merge_source_missing"] += 1
            continue
        if target not in graph:
            stats["merge_target_missing"] += 1
            continue
        if source not in ing_ids:
            stats["merge_source_not_ingredient"] += 1
            continue
        if target not in ing_ids:
            stats["merge_target_not_ingredient"] += 1
            continue
        if source == target:
            stats["merge_self"] += 1
            continue

        filtered_merges[source] = (tag, target)

    filtered_deletions: dict[str, str] = {}
    for node_id, reason in deletion_decisions.items():
        if node_id not in graph:
            stats["delete_node_missing"] += 1
            continue
        if node_id not in ing_ids:
            stats["delete_node_not_ingredient"] += 1
            continue

        filtered_deletions[node_id] = reason

    return filtered_merges, filtered_deletions, dict(stats)


# ---------------------------------------------------------------------------
# Graph mutation
# ---------------------------------------------------------------------------

def merge_edge_attributes(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge edge attributes conservatively."""
    out = dict(existing)

    # Keep highest confidence if present.
    if "confidence_score" in incoming:
        try:
            incoming_conf = float(incoming["confidence_score"])
            existing_conf = float(out.get("confidence_score", float("-inf")))
            if incoming_conf > existing_conf:
                out["confidence_score"] = incoming["confidence_score"]
        except (TypeError, ValueError):
            out.setdefault("confidence_score", incoming["confidence_score"])

    # Preserve specific forms / evidence-like list fields.
    for key in ["specific_forms", "forms", "evidence", "raw_variants"]:
        merged = set()
        for value in [out.get(key), incoming.get(key)]:
            if isinstance(value, list):
                merged.update(str(x) for x in value)
            elif value:
                merged.add(str(value))
        if merged:
            out[key] = sorted(merged)

    # Fill missing values from incoming.
    for key, value in incoming.items():
        if key not in out or out[key] in {None, ""}:
            out[key] = value

    return out


def merge_node_attributes(target_attrs: dict[str, Any], source_attrs: dict[str, Any], source_id: str) -> None:
    """Merge selected Ingredient attributes into the canonical target node."""
    try:
        target_attrs["n_occurrences"] = int(target_attrs.get("n_occurrences", 0) or 0) + int(
            source_attrs.get("n_occurrences", 0) or 0
        )
    except (TypeError, ValueError):
        pass

    for key in ["surface_forms", "raw_variants", "aliases", "merged_from"]:
        merged = set()

        target_value = target_attrs.get(key)
        source_value = source_attrs.get(key)

        if isinstance(target_value, list):
            merged.update(str(x) for x in target_value)
        elif target_value:
            merged.add(str(target_value))

        if isinstance(source_value, list):
            merged.update(str(x) for x in source_value)
        elif source_value:
            merged.add(str(source_value))

        if key == "merged_from":
            merged.add(source_id)

        if merged:
            target_attrs[key] = sorted(merged)

    if source_attrs.get("ner_noise_flag"):
        target_attrs["ner_noise_flag"] = bool(target_attrs.get("ner_noise_flag")) or True


def add_or_merge_edge(graph: nx.DiGraph, source: str, target: str, attrs: dict[str, Any]) -> None:
    if source == target:
        return

    if graph.has_edge(source, target):
        merged_attrs = merge_edge_attributes(dict(graph.edges[source, target]), attrs)
        graph.edges[source, target].update(merged_attrs)
    else:
        graph.add_edge(source, target, **attrs)


def merge_node_into(graph: nx.DiGraph, source: str, target: str) -> bool:
    """Redirect incident edges of source to target, merge attributes, remove source."""
    if source not in graph or target not in graph or source == target:
        return False

    source_attrs = dict(graph.nodes[source])
    target_attrs = graph.nodes[target]
    merge_node_attributes(target_attrs, source_attrs, source)

    for predecessor, _, attrs in list(graph.in_edges(source, data=True)):
        if predecessor == target:
            continue
        add_or_merge_edge(graph, predecessor, target, dict(attrs))

    for _, successor, attrs in list(graph.out_edges(source, data=True)):
        if successor == target:
            continue
        add_or_merge_edge(graph, target, successor, dict(attrs))

    graph.remove_node(source)
    return True


def apply_decisions(
    graph: nx.DiGraph,
    merge_decisions: dict[str, tuple[str, str]],
    deletion_decisions: dict[str, str],
    *,
    dry_run: bool,
) -> dict[str, int]:
    stats = Counter()

    # Apply deletions first.
    for node_id in sorted(deletion_decisions):
        if node_id not in graph:
            stats["delete_skipped_missing"] += 1
            continue

        if not dry_run:
            graph.remove_node(node_id)

        stats["deleted"] += 1

    # Apply merges deterministically by source id.
    for source in sorted(merge_decisions):
        _, target = merge_decisions[source]

        if source not in graph:
            stats["merge_skipped_source_missing"] += 1
            continue
        if target not in graph:
            stats["merge_skipped_target_missing"] += 1
            continue

        if not dry_run:
            did_merge = merge_node_into(graph, source, target)
        else:
            did_merge = source != target

        if did_merge:
            stats["merged"] += 1
        else:
            stats["merge_skipped"] += 1

    return dict(stats)


# ---------------------------------------------------------------------------
# Audit trail and outputs
# ---------------------------------------------------------------------------

def build_audit_trail(
    original_ingredient_ids: set[str],
    merge_decisions: dict[str, tuple[str, str]],
    deletion_decisions: dict[str, str],
    canonical_graph: nx.Graph,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for node_id in sorted(original_ingredient_ids):
        if node_id in deletion_decisions:
            rows.append({
                "original_id": node_id,
                "canonical_id": "",
                "decision_source": "deletion",
                "action": "deleted",
                "reason": deletion_decisions[node_id],
            })
        elif node_id in merge_decisions:
            tag, target = merge_decisions[node_id]
            rows.append({
                "original_id": node_id,
                "canonical_id": target,
                "decision_source": tag,
                "action": "merged",
                "reason": "",
            })
        elif node_id in canonical_graph:
            rows.append({
                "original_id": node_id,
                "canonical_id": node_id,
                "decision_source": "kept_as_canonical",
                "action": "kept",
                "reason": "",
            })
        else:
            rows.append({
                "original_id": node_id,
                "canonical_id": "",
                "decision_source": "unknown",
                "action": "lost",
                "reason": "node disappeared with no recorded decision",
            })

    return pd.DataFrame(rows)


def node_type_counts(graph: nx.Graph) -> dict[str, int]:
    counts = Counter()
    for _, attrs in graph.nodes(data=True):
        counts[str(attrs.get("node_type", "Unknown"))] += 1
    return dict(counts)


def edge_type_counts(graph: nx.Graph) -> dict[str, int]:
    counts = Counter()
    for _, _, attrs in graph.edges(data=True):
        counts[str(attrs.get("edge_type", "Unknown"))] += 1
    return dict(counts)


def write_pickle_graph(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Wrote canonical graph: %s", path)


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
        help=f"Output canonicalized graph. Default: {DEFAULT_GRAPH_OUT}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Default directory for decision files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--auto-merge",
        type=Path,
        help="Path to canonicalization_auto_merge.csv.",
    )
    parser.add_argument(
        "--review",
        type=Path,
        help="Path to canonicalization_needs_review_llm.csv or human review CSV.",
    )
    parser.add_argument(
        "--expert-corrections",
        type=Path,
        help="Path to llm_review_corrections_filled.csv.",
    )
    parser.add_argument(
        "--deletions",
        type=Path,
        help="Path to canonicalization_deletion_candidates.csv.",
    )
    parser.add_argument(
        "--mapping-out",
        type=Path,
        help="Path to canonicalization_mapping.csv.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        help="Path to graph_step2_stats.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute decisions and counts but do not write graph/mapping/stats.",
    )
    parser.add_argument(
        "--write-dry-run-outputs",
        action="store_true",
        help="With --dry-run, still write mapping/stats previews.",
    )
    parser.add_argument(
        "--use-llm-suggestions",
        action="store_true",
        help=(
            "Apply llm_merge_ids / llm_delete_ids from the review file. "
            "Default only applies accepted_ids."
        ),
    )
    parser.add_argument(
        "--delete-all-candidates",
        action="store_true",
        help=(
            "Delete all rows in canonicalization_deletion_candidates.csv. "
            "Default only deletes rows explicitly accepted/confirmed."
        ),
    )
    parser.add_argument(
        "--no-keep-separate-wins",
        dest="keep_separate_wins",
        action="store_false",
        help="Do not let KEEP_SEPARATE protections override merges.",
    )
    parser.set_defaults(keep_separate_wins=True)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    args.auto_merge = args.auto_merge or args.data_dir / "canonicalization_auto_merge.csv"
    args.review = args.review or args.data_dir / "canonicalization_needs_review_llm.csv"
    args.expert_corrections = args.expert_corrections or args.data_dir / "llm_review_corrections_filled.csv"
    args.deletions = args.deletions or args.data_dir / "canonicalization_deletion_candidates.csv"
    args.mapping_out = args.mapping_out or args.data_dir / "canonicalization_mapping.csv"
    args.stats_out = args.stats_out or args.data_dir / "graph_step2_stats.json"

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    try:
        graph = load_graph(args.graph_in)

        original_node_count = graph.number_of_nodes()
        original_edge_count = graph.number_of_edges()
        original_type_counts = node_type_counts(graph)
        original_edge_type_counts = edge_type_counts(graph)
        original_ingredients = ingredient_node_ids(graph)

        merge_decisions, deletion_decisions, keep_separate, decision_stats = collect_all_decisions(args)
        merge_decisions, cycles = transitive_closure(merge_decisions)

        merge_decisions, deletion_decisions, validation_stats = filter_invalid_decisions(
            graph,
            merge_decisions,
            deletion_decisions,
        )

        log.info(
            "Final valid decisions: %d merges, %d deletions, %d keep-separate protections",
            len(merge_decisions),
            len(deletion_decisions),
            len(keep_separate),
        )

        mutation_stats = apply_decisions(
            graph,
            merge_decisions,
            deletion_decisions,
            dry_run=args.dry_run,
        )

        log.info("Mutation stats: %s", mutation_stats)
        log.info("Graph now: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

        audit_df = build_audit_trail(
            original_ingredients,
            merge_decisions,
            deletion_decisions,
            graph,
        )

        decision_source_counts = Counter(tag for tag, _ in merge_decisions.values())
        deletion_reason_counts = Counter(reason.split(":", 1)[0] for reason in deletion_decisions.values())

        stats = {
            "inputs": {
                "graph_in": str(args.graph_in),
                "auto_merge": str(args.auto_merge),
                "review": str(args.review),
                "expert_corrections": str(args.expert_corrections),
                "deletions": str(args.deletions),
            },
            "outputs": {
                "graph_out": str(args.graph_out),
                "mapping_out": str(args.mapping_out),
                "stats_out": str(args.stats_out),
            },
            "parameters": {
                "dry_run": args.dry_run,
                "use_llm_suggestions": args.use_llm_suggestions,
                "delete_all_candidates": args.delete_all_candidates,
                "keep_separate_wins": args.keep_separate_wins,
            },
            "before": {
                "nodes": original_node_count,
                "edges": original_edge_count,
                "node_types": original_type_counts,
                "edge_types": original_edge_type_counts,
                "ingredient_nodes": len(original_ingredients),
            },
            "after": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "node_types": node_type_counts(graph),
                "edge_types": edge_type_counts(graph),
                "ingredient_nodes": node_type_counts(graph).get("Ingredient", 0),
            },
            "decisions": {
                "merge_decisions": len(merge_decisions),
                "deletion_decisions": len(deletion_decisions),
                "keep_separate_protections": len(keep_separate),
                "decision_source_counts": dict(decision_source_counts),
                "deletion_reason_counts": dict(deletion_reason_counts),
                "decision_collection_stats": decision_stats,
                "validation_stats": validation_stats,
                "merge_cycles_detected": cycles,
            },
            "mutation_stats": mutation_stats,
            "ingredient_reduction": {
                "before": len(original_ingredients),
                "after": node_type_counts(graph).get("Ingredient", 0),
                "absolute_reduction": len(original_ingredients) - node_type_counts(graph).get("Ingredient", 0),
                "pct_reduction": round(
                    100 * (
                        1 - node_type_counts(graph).get("Ingredient", 0) / max(1, len(original_ingredients))
                    ),
                    2,
                ),
            },
        }

        if args.dry_run and not args.write_dry_run_outputs:
            log.info("[dry-run] No outputs written.")
            print(json.dumps({
                "final_valid_merges": len(merge_decisions),
                "final_valid_deletions": len(deletion_decisions),
                "mutation_stats": mutation_stats,
                "ingredient_reduction": stats["ingredient_reduction"],
            }, indent=2, ensure_ascii=False))
            return 0

        if not args.dry_run:
            write_pickle_graph(graph, args.graph_out)

        write_csv(audit_df, args.mapping_out)
        write_json(args.stats_out, stats)

        log.info("Done.")
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
