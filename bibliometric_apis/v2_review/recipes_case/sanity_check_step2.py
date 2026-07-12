"""
=====================

Quick sanity check for Step 2 embeddings.

Checks
------

1. The .npz file loads correctly.
2. Required arrays are present and node/vector counts match.
3. Vector dimensions are consistent.
4. Vectors contain no NaN or Inf.
5. Vector norms are in a reasonable range.
6. Optional count checks against expected counts and/or the Step 1 graph.
7. Nearest neighbours for selected probe nodes are semantically inspectable.

Default inputs
--------------

  data/layer_2_embeddings.npz
  data/graph_step1.gpickle

Usage
-----

  python sanity_check_step2.py

  python sanity_check_step2.py \
    --embeddings data/layer_2_embeddings.npz \
    --graph data/graph_step1.gpickle

  python sanity_check_step2.py --no-expected-counts

  python sanity_check_step2.py \
    --probe ing::wine \
    --probe ing::salt \
    --probe act::boil

  python sanity_check_step2.py --out-json data/layer_2_embeddings_sanity.json

Exit codes
----------

  0  all checks passed
  1  file/load/schema error
  2  sanity checks ran but one or more checks failed


"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EMBEDDINGS = Path("data/layer_2_embeddings.npz")
DEFAULT_GRAPH = Path("data/graph_step1.gpickle")

DEFAULT_EXPECTED_COUNTS = {
    "recipes": 6024,
    "ingredients": 5537,
    "actions": 599,
}

DEFAULT_PROBES = [
    "ing::wine",
    "ing::salt",
    "ing::garlic",
    "ing::pepper",
    "ing::almond",
]

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("sanity_check_step2")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    log.info("Loading embeddings: %s", path)
    with np.load(path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}

    if not data:
        raise ValueError(f"No arrays found in {path}")

    return data


def load_graph_counts(path: Path) -> dict[str, int]:
    """Load Step 1 graph and count node types.

    Do not use with untrusted pickle/gpickle files.
    """
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    log.info("Loading graph for count comparison: %s", path)
    with path.open("rb") as fh:
        graph = pickle.load(fh)

    if not hasattr(graph, "nodes"):
        raise TypeError(f"Object loaded from {path} does not look like a NetworkX graph")

    node_type_to_type_key = {
        "Recipe": "recipes",
        "Ingredient": "ingredients",
        "Action": "actions",
    }

    counts = {key: 0 for key in node_type_to_type_key.values()}

    for _, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type")
        type_key = node_type_to_type_key.get(node_type)
        if type_key:
            counts[type_key] += 1

    return counts


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def available_type_keys(data: dict[str, np.ndarray]) -> list[str]:
    """Return type keys such as recipes/ingredients/actions from NPZ arrays."""
    keys: list[str] = []

    for array_name in data:
        if not array_name.endswith("_node_ids"):
            continue

        type_key = array_name.removesuffix("_node_ids")
        if f"{type_key}_vectors" in data:
            keys.append(type_key)

    return sorted(keys)


def validate_schema(data: dict[str, np.ndarray], required_types: list[str] | None = None) -> list[str]:
    """Validate NPZ schema and return available type keys."""
    type_keys = available_type_keys(data)

    if not type_keys:
        raise ValueError(
            "No complete '<type>_node_ids' + '<type>_vectors' pairs found in NPZ"
        )

    missing: list[str] = []
    if required_types:
        for type_key in required_types:
            if type_key not in type_keys:
                missing.append(type_key)

    if missing:
        raise ValueError(f"Required embedding types missing from NPZ: {missing}")

    for type_key in type_keys:
        ids = data[f"{type_key}_node_ids"]
        vectors = data[f"{type_key}_vectors"]

        if vectors.ndim != 2:
            raise ValueError(f"{type_key}_vectors must be 2D, got shape {vectors.shape}")

        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"{type_key}: node_ids length {len(ids)} does not match "
                f"vectors rows {vectors.shape[0]}"
            )

    return type_keys


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def parse_expected_counts(items: list[str]) -> dict[str, int]:
    """Parse CLI items like recipes=6024 ingredients=5537."""
    out: dict[str, int] = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --expected-count {item!r}; use type=count")

        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(f"Invalid --expected-count {item!r}; missing type key")

        try:
            count = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid count in --expected-count {item!r}") from exc

        if count < 0:
            raise ValueError(f"Expected count must be >= 0 in {item!r}")

        out[key] = count

    return out


def count_validation(
    data: dict[str, np.ndarray],
    type_keys: list[str],
    expected_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    ok = True

    for type_key in type_keys:
        actual = int(len(data[f"{type_key}_node_ids"]))
        expected = expected_counts.get(type_key)

        if expected is None:
            status = "not_checked"
            matches = None
        else:
            matches = actual == expected
            status = "ok" if matches else "mismatch"
            if not matches:
                ok = False

        rows.append({
            "type": type_key,
            "actual": actual,
            "expected": expected,
            "status": status,
        })

    return rows, ok


def vector_health(
    data: dict[str, np.ndarray],
    type_keys: list[str],
    *,
    min_norm: float,
    max_norm: float,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    ok = True

    for type_key in type_keys:
        vectors = np.asarray(data[f"{type_key}_vectors"])
        norms = np.linalg.norm(vectors, axis=1)

        finite_mask = np.isfinite(vectors).all(axis=1)
        norm_mask = (norms >= min_norm) & (norms <= max_norm)

        n_nan = int(np.isnan(vectors).any(axis=1).sum())
        n_inf = int(np.isinf(vectors).any(axis=1).sum())
        n_zero = int((norms < 1e-12).sum())
        n_bad_norm = int((~norm_mask).sum())
        n_nonfinite = int((~finite_mask).sum())

        if n_nan or n_inf or n_zero or n_nonfinite or n_bad_norm:
            ok = False

        rows.append({
            "type": type_key,
            "shape": list(vectors.shape),
            "dtype": str(vectors.dtype),
            "norm_min": float(norms.min()) if len(norms) else None,
            "norm_max": float(norms.max()) if len(norms) else None,
            "norm_mean": float(norms.mean()) if len(norms) else None,
            "n_nan_vectors": n_nan,
            "n_inf_vectors": n_inf,
            "n_nonfinite_vectors": n_nonfinite,
            "n_near_zero_vectors": n_zero,
            "n_out_of_norm_range": n_bad_norm,
            "norm_range_checked": [min_norm, max_norm],
        })

    return rows, ok


# ---------------------------------------------------------------------------
# Nearest neighbours
# ---------------------------------------------------------------------------

def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine_top_k(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    ids: np.ndarray,
    *,
    query_id: str,
    k: int,
) -> list[dict[str, Any]]:
    """Return top-k most similar ids by cosine similarity, excluding self."""
    if matrix.shape[0] == 0:
        return []

    normalized_matrix = normalize_matrix(matrix)
    query = np.asarray(query_vec, dtype=np.float32)
    query = query / max(float(np.linalg.norm(query)), 1e-12)

    similarities = normalized_matrix @ query
    order = np.argsort(-similarities)

    rows: list[dict[str, Any]] = []
    for index in order:
        node_id = str(ids[index])
        if node_id == query_id:
            continue

        rows.append({
            "node_id": node_id,
            "similarity": float(similarities[index]),
        })

        if len(rows) >= k:
            break

    return rows


def infer_type_key_from_node_id(node_id: str) -> str | None:
    if node_id.startswith("recipe::"):
        return "recipes"
    if node_id.startswith("ing::"):
        return "ingredients"
    if node_id.startswith("act::"):
        return "actions"
    return None


def semantic_probe_checks(
    data: dict[str, np.ndarray],
    probes: list[str],
    *,
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for probe in probes:
        type_key = infer_type_key_from_node_id(probe)

        if type_key is None:
            rows.append({
                "probe": probe,
                "status": "skipped_unknown_prefix",
                "neighbours": [],
            })
            continue

        ids_key = f"{type_key}_node_ids"
        vecs_key = f"{type_key}_vectors"

        if ids_key not in data or vecs_key not in data:
            rows.append({
                "probe": probe,
                "type": type_key,
                "status": "skipped_type_missing",
                "neighbours": [],
            })
            continue

        ids = data[ids_key].astype(str)
        vectors = data[vecs_key]

        matches = np.where(ids == probe)[0]
        if len(matches) == 0:
            rows.append({
                "probe": probe,
                "type": type_key,
                "status": "not_found",
                "neighbours": [],
            })
            continue

        index = int(matches[0])
        neighbours = cosine_top_k(
            vectors[index],
            vectors,
            ids,
            query_id=probe,
            k=k,
        )

        rows.append({
            "probe": probe,
            "type": type_key,
            "status": "ok",
            "neighbours": neighbours,
        })

    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("FILES IN NPZ")
    print("=" * 72)
    for item in report["arrays"]:
        print(f"  {item['name']}: shape={tuple(item['shape'])}, dtype={item['dtype']}")

    print()
    print("=" * 72)
    print("COUNT VALIDATION")
    print("=" * 72)
    for row in report["count_validation"]:
        expected = row["expected"] if row["expected"] is not None else "not checked"
        print(f"  {row['type']}: {row['actual']} expected={expected} [{row['status']}]")

    print()
    print("=" * 72)
    print("VECTOR HEALTH")
    print("=" * 72)
    for row in report["vector_health"]:
        print(f"  {row['type']}:")
        print(
            "    shape="
            f"{tuple(row['shape'])}, dtype={row['dtype']}"
        )
        print(
            "    norm range: "
            f"[{row['norm_min']:.4f}, {row['norm_max']:.4f}], "
            f"mean={row['norm_mean']:.4f}"
        )
        print(
            "    NaN vectors: "
            f"{row['n_nan_vectors']}, Inf vectors: {row['n_inf_vectors']}, "
            f"near-zero: {row['n_near_zero_vectors']}, "
            f"out-of-range: {row['n_out_of_norm_range']}"
        )

    print()
    print("=" * 72)
    print("SEMANTIC SANITY CHECK: nearest neighbours")
    print("=" * 72)
    for row in report["semantic_probes"]:
        probe = row["probe"]
        status = row["status"]

        if status != "ok":
            print(f"  {probe}: {status}")
            continue

        print(f"  {probe} neighbours:")
        for neighbour in row["neighbours"]:
            print(f"    {neighbour['similarity']:.4f}  {neighbour['node_id']}")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  overall status: {report['overall_status']}")
    if report["errors"]:
        print("  errors:")
        for error in report["errors"]:
            print(f"    - {error}")

    print()
    if report["overall_status"] == "ok":
        print("Done. Step 2 embeddings passed the sanity checks.")
    else:
        print("Done. One or more sanity checks failed; inspect the report above.")


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON report: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report(
    *,
    embeddings_path: Path,
    graph_path: Path | None,
    expected_counts: dict[str, int],
    use_graph_counts: bool,
    required_types: list[str],
    probes: list[str],
    top_k: int,
    min_norm: float,
    max_norm: float,
) -> dict[str, Any]:
    data = load_embeddings(embeddings_path)
    type_keys = validate_schema(data, required_types=required_types or None)

    arrays = [
        {
            "name": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(data.items())
    ]

    errors: list[str] = []

    combined_expected = dict(expected_counts)

    if use_graph_counts and graph_path is not None:
        graph_counts = load_graph_counts(graph_path)
        combined_expected.update(graph_counts)

    count_rows, counts_ok = count_validation(data, type_keys, combined_expected)
    health_rows, health_ok = vector_health(
        data,
        type_keys,
        min_norm=min_norm,
        max_norm=max_norm,
    )
    probe_rows = semantic_probe_checks(data, probes, k=top_k)

    if not counts_ok:
        errors.append("One or more embedding counts do not match expected counts.")
    if not health_ok:
        errors.append("One or more vector-health checks failed.")

    return {
        "embeddings_path": str(embeddings_path),
        "graph_path": str(graph_path) if graph_path is not None else None,
        "arrays": arrays,
        "available_types": type_keys,
        "expected_counts": combined_expected,
        "count_validation": count_rows,
        "vector_health": health_rows,
        "semantic_probes": probe_rows,
        "errors": errors,
        "overall_status": "ok" if not errors else "failed",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help=f"Path to layer_2_embeddings.npz. Default: {DEFAULT_EMBEDDINGS}",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help=f"Optional Step 1 graph for count comparison. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--no-graph-counts",
        action="store_true",
        help="Do not compare embedding counts against node counts from --graph.",
    )
    parser.add_argument(
        "--no-expected-counts",
        action="store_true",
        help="Disable built-in expected counts.",
    )
    parser.add_argument(
        "--expected-count",
        action="append",
        default=[],
        help="Expected count as type=count, e.g. --expected-count ingredients=5537. Can be repeated.",
    )
    parser.add_argument(
        "--require-type",
        action="append",
        default=[],
        help="Require an embedding type such as recipes, ingredients, actions. Can be repeated.",
    )
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        help="Node id to use as nearest-neighbour probe. Can be repeated.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of nearest neighbours to print per probe.",
    )
    parser.add_argument(
        "--min-norm",
        type=float,
        default=0.1,
        help="Minimum acceptable vector norm.",
    )
    parser.add_argument(
        "--max-norm",
        type=float,
        default=100.0,
        help="Maximum acceptable vector norm.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Optional path to write machine-readable sanity report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print human-readable report; useful with --out-json.",
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

    if args.top_k < 1:
        log.error("--top-k must be >= 1")
        return 2
    if args.min_norm < 0:
        log.error("--min-norm must be >= 0")
        return 2
    if args.max_norm <= args.min_norm:
        log.error("--max-norm must be greater than --min-norm")
        return 2

    try:
        expected_counts = {} if args.no_expected_counts else dict(DEFAULT_EXPECTED_COUNTS)
        expected_counts.update(parse_expected_counts(args.expected_count))

        probes = args.probe if args.probe else DEFAULT_PROBES

        graph_path = None if args.no_graph_counts else args.graph
        use_graph_counts = graph_path is not None and graph_path.exists()

        if graph_path is not None and not graph_path.exists():
            log.warning("Graph file not found for graph-count comparison: %s", graph_path)

        report = build_report(
            embeddings_path=args.embeddings,
            graph_path=graph_path if use_graph_counts else None,
            expected_counts=expected_counts,
            use_graph_counts=use_graph_counts,
            required_types=args.require_type,
            probes=probes,
            top_k=args.top_k,
            min_norm=args.min_norm,
            max_norm=args.max_norm,
        )

        if args.out_json:
            write_json(args.out_json, report)

        if not args.quiet:
            print_report(report)

        return 0 if report["overall_status"] == "ok" else 2

    except (FileNotFoundError, ValueError, TypeError, pickle.PickleError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
