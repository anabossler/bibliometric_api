"""
csc_calibration_openalex.py
===========================

Build a small cross-domain calibration set for the CSC score using OpenAlex
citation corpora.

The script scores several transparent subfield-pair corpora with the same
high-level CSC arithmetic:

    CSC = 1 - (S_cross * D_bar)

where:
  - S_cross is the fraction of within-corpus citation edges crossing
    subfield labels;
  - D_bar is the cross-subfield Jensen-Shannon distance between abstract-term
    distributions, weighted by cross-subfield citation flow.

This script is intended as a calibration/stress-test helper, not as the main
pipeline. It helps place the manuscript corpora on an ordered scale by comparing
them to additional OpenAlex corpora selected for expected fragmentation or
integration.

Inputs:
  - OpenAlex Works API access via either:
      OPENALEX_API_KEY, or
      OPENALEX_MAILTO / OPENALEX_EMAIL for the polite pool.

Outputs:
  - JSON file with corpus-level CSC values and diagnostic controls.

Usage:
  # Recommended: set credentials/contact in environment or .env
  export OPENALEX_MAILTO="you@example.edu"
  python csc_calibration_openalex.py --per-subfield 400 \
      --out-json csc_calibration_openalex.json

  # With an API key
  export OPENALEX_API_KEY="..."
  python csc_calibration_openalex.py --api-key "$OPENALEX_API_KEY"

  # Use a custom .env file
  python csc_calibration_openalex.py --dotenv /path/to/.env

Dependencies:
  requests
  numpy
  scipy

"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from scipy.spatial.distance import jensenshannon


OPENALEX_WORKS_URL = "https://api.openalex.org/works"

DEFAULT_MODEL_REFERENCE_POINTS = {
    "_reference_plastics": {
        "CSC": 0.402,
        "S_cross": 0.617,
        "source": "manuscript",
    },
    "_reference_recipes": {
        "CSC": 0.766,
        "S_cross": 0.394,
        "D_bar": 0.592,
        "source": "recipe CSC calibration",
    },
}

# ---------------------------------------------------------------------------
# Calibration set
# ---------------------------------------------------------------------------
# Each corpus is a tuple:
#   (corpus_name, [(subfield_label, OpenAlex title/abstract search query), ...],
#    qualitative_expectation)
#
# Edit this list to match the strongest calibration argument for your revision.
CORPORA: list[tuple[str, list[tuple[str, str]], str]] = [
    (
        "plastics_vs_circular",
        [
            ("plastic_recycling", "plastic recycling"),
            ("circular_economy", "circular economy"),
        ],
        "Expected fragmented: related vocabularies, weak cross-citation.",
    ),
    (
        "crispr_vs_geneediting",
        [
            ("crispr_cas9", "CRISPR Cas9"),
            ("genome_editing", "genome editing"),
        ],
        "Expected integrated: shared method vocabulary, dense cross-citation.",
    ),
    (
        "microbiome_gut_vs_soil",
        [
            ("gut_microbiome", "gut microbiome"),
            ("soil_microbiome", "soil microbiome"),
        ],
        "Expected fragmented: same construct, distinct communities.",
    ),
    (
        "ml_transformers_vs_gnn",
        [
            ("transformers_nlp", "transformer language model"),
            ("graph_neural_networks", "graph neural network"),
        ],
        "Expected mid/integrated: shared deep-learning vocabulary.",
    ),
]

STOP_WORDS = set(
    """
    the a an and or of to in for on with without by from at as is are was were
    be been being this that these those it its their his her our your my we you
    they he she them us into over under between within across per via using used
    use based study studies results result method methods approach approaches
    paper show present propose new novel data model models analysis effect
    effects role can may will also than then thus which who whom whose when
    where while about above after again against all am any because before below
    both did do does doing down during each few further had has have having how
    if more most no nor not only other out own same so some such too very should
    now
    """.split()
)

TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("csc_calibration")


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------
def load_env_file(path: Path | None = None) -> Path | None:
    """Load KEY=VALUE pairs from a .env file.

    If path is None, search from the current working directory upward.
    Existing environment variables are never overwritten.
    """
    candidate: Path | None = path

    if candidate is None:
        cwd = Path.cwd().resolve()
        for parent in (cwd, *cwd.parents):
            maybe = parent / ".env"
            if maybe.exists():
                candidate = maybe
                break

    if candidate is None:
        log.info("No .env file found; using environment variables only.")
        return None

    if not candidate.exists():
        log.warning(".env file not found: %s", candidate)
        return None

    loaded = 0
    with candidate.open(encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key.isidentifier() and key not in os.environ:
                os.environ[key] = value
                loaded += 1

    log.info("Loaded %d value(s) from %s", loaded, candidate)
    return candidate


def get_contact_email(cli_mailto: str | None = None) -> str | None:
    return (
        cli_mailto
        or os.environ.get("OPENALEX_MAILTO")
        or os.environ.get("OPENALEX_EMAIL")
    )


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------
def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct an OpenAlex abstract from abstract_inverted_index."""
    if not inverted_index:
        return ""

    positions: dict[int, str] = {}
    for word, indices in inverted_index.items():
        if not isinstance(indices, list):
            continue
        for idx in indices:
            try:
                positions[int(idx)] = word
            except (TypeError, ValueError):
                continue

    return " ".join(positions[i] for i in sorted(positions))


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [tok for tok in tokens if tok not in STOP_WORDS]


# ---------------------------------------------------------------------------
# OpenAlex fetching
# ---------------------------------------------------------------------------
def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def fetch_subfield(
    session: requests.Session,
    query: str,
    *,
    api_key: str | None,
    mailto: str | None,
    per_subfield: int,
    from_year: int,
    timeout_s: int,
    sleep_s: float,
    max_retries: int,
) -> list[dict[str, Any]]:
    """Pull up to per_subfield works matching a title/abstract query."""
    records: list[dict[str, Any]] = []
    cursor = "*"

    params: dict[str, Any] = {
        "filter": (
            f"title_and_abstract.search:{query},"
            f"from_publication_date:{from_year}-01-01,type:article"
        ),
        "select": (
            "id,referenced_works,abstract_inverted_index,"
            "publication_year,relevance_score"
        ),
        "per-page": 200,
    }
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto

    while len(records) < per_subfield:
        params["cursor"] = cursor

        response: requests.Response | None = None
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                response = session.get(
                    OPENALEX_WORKS_URL,
                    params=params,
                    timeout=timeout_s,
                )
                # Respect common rate-limit responses with backoff.
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        response=response,
                    )
                response.raise_for_status()
                break
            except Exception as exc:  # requests exceptions + HTTPError
                last_error = exc
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"OpenAlex request failed after {max_retries} attempts "
                        f"for query={query!r}"
                    ) from last_error
                wait = sleep_s * (2 ** attempt)
                log.warning(
                    "OpenAlex request failed for query=%r attempt %d/%d; retrying in %.1fs",
                    query,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)

        if response is None:
            raise RuntimeError(f"No OpenAlex response for query={query!r}")

        payload = response.json()
        batch = payload.get("results") or []
        if not batch:
            break

        records.extend(batch)
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor:
            break

        time.sleep(sleep_s)

    return records[:per_subfield]


def build_corpus(
    session: requests.Session,
    subfields: list[tuple[str, str]],
    *,
    api_key: str | None,
    mailto: str | None,
    per_subfield: int,
    from_year: int,
    timeout_s: int,
    sleep_s: float,
    max_retries: int,
) -> tuple[dict[str, str], dict[str, Counter[str]], list[tuple[str, str]]]:
    """Build node labels, abstract-term counters, and intra-corpus citation edges.

    If a work is retrieved by multiple subfield queries, it is assigned to the
    subfield where OpenAlex reports the highest relevance score.
    """
    raw: dict[str, tuple[str, float, set[str], Counter[str]]] = {}

    for label, query in subfields:
        log.info("Fetching subfield=%s query=%r", label, query)
        works = fetch_subfield(
            session,
            query,
            api_key=api_key,
            mailto=mailto,
            per_subfield=per_subfield,
            from_year=from_year,
            timeout_s=timeout_s,
            sleep_s=sleep_s,
            max_retries=max_retries,
        )
        log.info("Fetched %d work(s) for %s", len(works), label)

        for work in works:
            work_id = work.get("id")
            if not work_id:
                continue

            relevance = float(work.get("relevance_score") or 0.0)
            if work_id in raw and raw[work_id][1] >= relevance:
                continue

            refs = set(work.get("referenced_works") or [])
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            terms = Counter(tokenize(abstract))
            raw[work_id] = (label, relevance, refs, terms)

    label_of = {work_id: value[0] for work_id, value in raw.items()}
    terms_of = {work_id: value[3] for work_id, value in raw.items()}

    node_set = set(raw)
    edges: set[tuple[str, str]] = set()
    for work_id, (_, _, refs, _) in raw.items():
        for ref in refs:
            if ref in node_set and ref != work_id:
                edges.add(tuple(sorted((work_id, ref))))

    return label_of, terms_of, sorted(edges)


# ---------------------------------------------------------------------------
# CSC computation and controls
# ---------------------------------------------------------------------------
def term_distribution(
    counters: Iterable[Counter[str]],
    vocab_index: dict[str, int],
) -> np.ndarray:
    vector = np.zeros(len(vocab_index), dtype=np.float64)
    for counter in counters:
        for term, count in counter.items():
            idx = vocab_index.get(term)
            if idx is not None:
                vector[idx] += count
    total = vector.sum()
    return vector / total if total else vector


def safe_js_distance(p: np.ndarray, q: np.ndarray) -> float:
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    return float(jensenshannon(p, q, base=2))


def csc_from_corpus(
    label_of: dict[str, str],
    terms_of: dict[str, Counter[str]],
    edges: list[tuple[str, str]],
    *,
    n_null: int,
    seed: int,
) -> dict[str, Any]:
    labels = sorted(set(label_of.values()))
    if len(labels) < 2:
        raise ValueError("CSC requires at least two subfield labels.")
    if not label_of:
        raise ValueError("Empty corpus.")

    n_cross = sum(1 for a, b in edges if label_of[a] != label_of[b])
    s_cross = n_cross / len(edges) if edges else float("nan")

    subfield_terms: dict[str, Counter[str]] = defaultdict(Counter)
    for work_id, counter in terms_of.items():
        subfield_terms[label_of[work_id]].update(counter)

    vocab = sorted({term for counter in subfield_terms.values() for term in counter})
    vocab_index = {term: idx for idx, term in enumerate(vocab)}

    distributions = {
        label: term_distribution([subfield_terms[label]], vocab_index)
        for label in labels
    }

    pair_weights: Counter[tuple[str, str]] = Counter()
    for a, b in edges:
        if label_of[a] != label_of[b]:
            pair_weights[tuple(sorted((label_of[a], label_of[b])))] += 1

    pairwise: list[tuple[tuple[str, str], float, int]] = []
    weighted_num = 0.0
    weighted_den = 0

    for pair, weight in pair_weights.items():
        x, y = pair
        dist = safe_js_distance(distributions[x], distributions[y])
        if np.isnan(dist):
            continue
        pairwise.append((pair, dist, weight))
        weighted_num += dist * weight
        weighted_den += weight

    if weighted_den:
        d_bar = weighted_num / weighted_den
    else:
        distances = [
            safe_js_distance(distributions[x], distributions[y])
            for x, y in itertools.combinations(labels, 2)
        ]
        distances = [d for d in distances if not np.isnan(d)]
        d_bar = float(np.mean(distances)) if distances else float("nan")

    csc = 1 - (s_cross * d_bar) if not (np.isnan(s_cross) or np.isnan(d_bar)) else float("nan")

    # Control 1: label-shuffle null for S_cross
    nodes = list(label_of)
    node_pos = {node: idx for idx, node in enumerate(nodes)}
    label_array = np.array([label_of[node] for node in nodes])
    edge_indices = [(node_pos[a], node_pos[b]) for a, b in edges]

    rng = np.random.default_rng(seed)
    null = np.empty(n_null, dtype=np.float64)

    for idx in range(n_null):
        if not edge_indices:
            null[idx] = np.nan
            continue
        permuted = rng.permutation(label_array)
        null[idx] = sum(
            1 for i, j in edge_indices if permuted[i] != permuted[j]
        ) / len(edge_indices)

    null_mean = float(np.nanmean(null)) if np.isfinite(null).any() else float("nan")
    null_sd = float(np.nanstd(null)) if np.isfinite(null).any() else float("nan")
    z_score = (
        (s_cross - null_mean) / null_sd
        if null_sd and not np.isnan(s_cross)
        else float("nan")
    )

    # Control 2: homogeneous split of the largest subfield
    largest_label = Counter(label_of.values()).most_common(1)[0][0]
    largest_nodes = [node for node in label_of if label_of[node] == largest_label]
    rng_py = random.Random(seed)
    rng_py.shuffle(largest_nodes)
    half = len(largest_nodes) // 2
    h1, h2 = largest_nodes[:half], largest_nodes[half:]

    d_homog = safe_js_distance(
        term_distribution((terms_of[node] for node in h1), vocab_index),
        term_distribution((terms_of[node] for node in h2), vocab_index),
    )

    separation_ratio = (
        d_bar / d_homog
        if d_homog and not np.isnan(d_bar) and not np.isnan(d_homog)
        else None
    )

    return {
        "labels": labels,
        "n_nodes": len(label_of),
        "n_edges": len(edges),
        "n_cross": n_cross,
        "S_cross": round(float(s_cross), 4) if not np.isnan(s_cross) else None,
        "D_bar": round(float(d_bar), 4) if not np.isnan(d_bar) else None,
        "CSC": round(float(csc), 4) if not np.isnan(csc) else None,
        "null_mean": round(null_mean, 4) if not np.isnan(null_mean) else None,
        "null_sd": round(null_sd, 4) if not np.isnan(null_sd) else None,
        "z": round(float(z_score), 1) if not np.isnan(z_score) else None,
        "D_homog": round(float(d_homog), 4) if not np.isnan(d_homog) else None,
        "separation_ratio": round(float(separation_ratio), 2)
        if separation_ratio is not None
        else None,
        "pairwise_JS": {
            f"{x}-{y}": round(float(dist), 4)
            for (x, y), dist, _ in pairwise
        },
        "pairwise_cross_edge_counts": {
            f"{x}-{y}": int(weight)
            for (x, y), _, weight in pairwise
        },
        "sub_sizes": {
            label: sum(1 for work_label in label_of.values() if work_label == label)
            for label in labels
        },
        "vocab_size": len(vocab),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAlex API key. Defaults to OPENALEX_API_KEY from the environment.",
    )
    parser.add_argument(
        "--mailto",
        default=None,
        help=(
            "Contact email for OpenAlex polite pool. Defaults to "
            "OPENALEX_MAILTO or OPENALEX_EMAIL from the environment."
        ),
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=None,
        help="Optional path to a .env file. If omitted, search upward from cwd.",
    )
    parser.add_argument(
        "--per-subfield",
        type=int,
        default=400,
        help="Maximum number of OpenAlex works to fetch per subfield.",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=2015,
        help="Earliest publication year included in OpenAlex searches.",
    )
    parser.add_argument(
        "--n-null",
        type=int,
        default=200,
        help="Number of label-shuffle permutations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for nulls and homogeneous split.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("csc_calibration_openalex.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Base sleep time between OpenAlex requests and retry backoff.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum request attempts per OpenAlex page.",
    )
    parser.add_argument(
        "--user-agent",
        default="csc-calibration-openalex/1.0",
        help="HTTP User-Agent for OpenAlex requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.dotenv)

    api_key = args.api_key or os.environ.get("OPENALEX_API_KEY")
    mailto = get_contact_email(args.mailto)

    if not api_key and not mailto:
        log.error(
            "Set either OPENALEX_API_KEY/--api-key or "
            "OPENALEX_MAILTO/OPENALEX_EMAIL/--mailto."
        )
        return 1

    if args.per_subfield <= 0:
        log.error("--per-subfield must be positive.")
        return 1
    if args.n_null <= 0:
        log.error("--n-null must be positive.")
        return 1

    session = make_session(args.user_agent)
    results: dict[str, Any] = {}

    for name, subfields, expectation in CORPORA:
        log.info("=== %s :: %s", name, expectation)
        label_of, terms_of, edges = build_corpus(
            session,
            subfields,
            api_key=api_key,
            mailto=mailto,
            per_subfield=args.per_subfield,
            from_year=args.from_year,
            timeout_s=args.timeout,
            sleep_s=args.sleep,
            max_retries=args.max_retries,
        )

        log.info("Corpus %s: nodes=%d citation_edges=%d", name, len(label_of), len(edges))
        if len(edges) < 30:
            log.warning(
                "Corpus %s has very few internal citation edges (%d); "
                "consider increasing --per-subfield.",
                name,
                len(edges),
            )

        try:
            score = csc_from_corpus(
                label_of,
                terms_of,
                edges,
                n_null=args.n_null,
                seed=args.seed,
            )
        except Exception as exc:
            log.exception("Failed to score corpus %s: %s", name, exc)
            results[name] = {
                "error": str(exc),
                "expectation": expectation,
                "subfields": [label for label, _ in subfields],
            }
            continue

        score["expectation"] = expectation
        score["subfields"] = [label for label, _ in subfields]
        results[name] = score

        log.info(
            "%s: CSC=%s S_cross=%s D_bar=%s z=%s sep=%s",
            name,
            score.get("CSC"),
            score.get("S_cross"),
            score.get("D_bar"),
            score.get("z"),
            score.get("separation_ratio"),
        )

    results.update(DEFAULT_MODEL_REFERENCE_POINTS)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    log.info("Wrote %s", args.out_json)

    scored = [
        (name, data.get("CSC"))
        for name, data in results.items()
        if isinstance(data, dict) and data.get("CSC") is not None
    ]

    log.info("CSC calibration order: low = fragmented; high = aligned")
    for name, score in sorted(scored, key=lambda item: item[1]):
        log.info("  %.3f  %s", score, name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
