"""
================================

Check FoodOn coverage for frequent base Ingredient nodes using the EBI OLS
Ontology Lookup Service.

This script does NOT modify the graph. It queries OLS for each frequent
ingredient and produces a human-review CSV of candidate FoodOn mappings ranked
by match quality.

Match classes, best to worst:

  exact_label
  exact_synonym
  token_set_label
  token_set_synonym
  partial_label
  partial_synonym
  none

The output CSV includes a `validated_keep` column. Reviewers should mark TRUE
or FALSE manually. Empty values should be treated as FALSE by downstream steps.

Default input
-------------

  data/graph_step3_flagged.gpickle

Default outputs
---------------

  data/foodon_coverage.csv
  data/foodon_query_cache.jsonl
  data/foodon_coverage_summary.json

Usage
-----

  python step4_0_check_foodon_coverage.py

  python step4_0_check_foodon_coverage.py --top-n 200

  python step4_0_check_foodon_coverage.py --rows 10 --sleep 0.5

  python step4_0_check_foodon_coverage.py --no-cache

  python step4_0_check_foodon_coverage.py --refresh-cache

API
---

  EBI OLS4 search endpoint:
  https://www.ebi.ac.uk/ols4/api/search

"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_GRAPH = Path("data/graph_step3_flagged.gpickle")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_OLS_URL = "https://www.ebi.ac.uk/ols4/api/search"
DEFAULT_ONTOLOGY = "foodon"
DEFAULT_TOP_N = 500
DEFAULT_ROWS = 5
DEFAULT_SLEEP_SECONDS = 0.2
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_USER_AGENT = "relish-recipes-research/1.0 (academic; contact: project-local)"

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step4_0_check_foodon_coverage")


MATCH_ORDER = {
    "exact_label": 0,
    "exact_synonym": 1,
    "token_set_label": 2,
    "token_set_synonym": 3,
    "partial_label": 4,
    "partial_synonym": 5,
    "none": 6,
}


# ---------------------------------------------------------------------------
# Graph and ingredient extraction
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


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def ingredient_label(graph: Any, ingredient_id: str) -> str:
    attrs = graph.nodes.get(ingredient_id, {})
    return safe_str(
        attrs.get("canonical_name")
        or attrs.get("name")
        or attrs.get("label")
        or ingredient_id.replace("ing::", "")
    )


def ingredient_name_to_query(name: str) -> str:
    """Convert an Ingredient node id or label to an OLS-friendly query."""
    text = str(name).replace("ing::", "")
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ingredient_frequencies(graph: Any) -> Counter[str]:
    """Count Ingredient mentions through Recipe --contains--> Ingredient edges."""
    counts: Counter[str] = Counter()

    for source, target, edge_attrs in graph.edges(data=True):
        if edge_attrs.get("edge_type") != "contains":
            continue
        if graph.nodes.get(source, {}).get("node_type") != "Recipe":
            continue
        if graph.nodes.get(target, {}).get("node_type") != "Ingredient":
            continue

        counts[str(target)] += 1

    return counts


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_key(*, query: str, ontology: str, rows: int) -> str:
    return json.dumps(
        {"query": query, "ontology": ontology, "rows": rows},
        sort_keys=True,
        ensure_ascii=False,
    )


def load_cache(path: Path, *, enabled: bool) -> dict[str, dict[str, Any]]:
    if not enabled or not path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                key = row.get("key")
                response = row.get("response")
                if key and isinstance(response, dict):
                    cache[str(key)] = response
            except (json.JSONDecodeError, TypeError):
                log.warning("Skipping malformed cache line %d in %s", line_no, path)

    log.info("Loaded %d cached OLS responses", len(cache))
    return cache


def append_cache(path: Path, *, key: str, query: str, ontology: str, rows: int, response: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "key": key,
        "query": query,
        "ontology": ontology,
        "rows": rows,
        "response": response,
    }

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# OLS querying
# ---------------------------------------------------------------------------

def query_ols(
    *,
    term: str,
    ontology: str,
    rows: int,
    url: str,
    timeout_s: int,
    user_agent: str,
    max_retries: int,
    retry_backoff_s: float,
) -> dict[str, Any]:
    """Query EBI OLS for a single term."""
    params = {
        "q": term,
        "ontology": ontology,
        "rows": rows,
        "queryFields": "label,synonym",
    }

    request_url = f"{url}?{urlencode(params)}"
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            request = Request(request_url, headers=headers)
            with urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            last_error = exc
            # Retry rate limits and transient server errors. Do not hammer.
            if exc.code not in {429, 500, 502, 503, 504}:
                log.warning("Non-retryable HTTP error for %r: %s", term, exc.code)
                return {}

            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "HTTP error for %r attempt %d/%d: %s. Sleeping %.1fs",
                term,
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
                "OLS error for %r attempt %d/%d: %s. Sleeping %.1fs",
                term,
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    log.warning("OLS query failed for %r after %d retries: %s", term, max_retries, last_error)
    return {}


# ---------------------------------------------------------------------------
# Match scoring
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(str(text)).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def synonyms_of(doc: dict[str, Any]) -> list[str]:
    raw = doc.get("synonym") or doc.get("synonyms") or []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def classify_text_match(query: str, candidate: str, *, label_or_synonym: str) -> tuple[str, float]:
    q = normalize_text(query)
    c = normalize_text(candidate)

    if not q or not c:
        return "none", 0.0

    if q == c:
        if label_or_synonym == "label":
            return "exact_label", 1.0
        return "exact_synonym", 0.9

    q_tokens = tokens(q)
    c_tokens = tokens(c)
    token_jaccard = jaccard(q_tokens, c_tokens)

    if q_tokens and q_tokens == c_tokens:
        if label_or_synonym == "label":
            return "token_set_label", 0.84
        return "token_set_synonym", 0.78

    if q in c or c in q or token_jaccard >= 0.50:
        if label_or_synonym == "label":
            return "partial_label", round(0.45 + 0.30 * token_jaccard, 4)
        return "partial_synonym", round(0.35 + 0.25 * token_jaccard, 4)

    return "none", 0.0


def classify_match(query: str, doc: dict[str, Any]) -> tuple[str, float, str]:
    """Return best match_type, confidence, and matched_text for one OLS doc."""
    label = safe_str(doc.get("label"))
    best_type, best_conf, best_text = classify_text_match(query, label, label_or_synonym="label") + (label,)

    for synonym in synonyms_of(doc):
        match_type, confidence = classify_text_match(query, synonym, label_or_synonym="synonym")
        if confidence > best_conf:
            best_type, best_conf, best_text = match_type, confidence, synonym

    return best_type, float(best_conf), best_text


def doc_foodon_id(doc: dict[str, Any]) -> str:
    return safe_str(doc.get("obo_id") or doc.get("short_form") or doc.get("ontology_name") or "")


def doc_description(doc: dict[str, Any], *, max_chars: int) -> str:
    raw = doc.get("description") or []
    if isinstance(raw, str):
        description = raw
    elif isinstance(raw, list):
        description = "; ".join(str(item) for item in raw if item)
    else:
        description = ""
    return description[:max_chars]


def best_doc_for_query(query: str, docs: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, float, str]:
    best_doc: dict[str, Any] | None = None
    best_type = "none"
    best_conf = 0.0
    best_text = ""

    for doc in docs:
        match_type, confidence, matched_text = classify_match(query, doc)

        current_rank = (confidence, -MATCH_ORDER.get(match_type, 99))
        best_rank = (best_conf, -MATCH_ORDER.get(best_type, 99))

        if current_rank > best_rank:
            best_doc = doc
            best_type = match_type
            best_conf = confidence
            best_text = matched_text

    return best_doc, best_type, best_conf, best_text


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def build_rows(
    *,
    graph: Any,
    top_ingredients: list[tuple[str, int]],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    use_cache: bool,
    write_cache: bool,
    refresh_cache: bool,
    ontology: str,
    rows_per_query: int,
    ols_url: str,
    sleep_s: float,
    timeout_s: int,
    user_agent: str,
    max_retries: int,
    retry_backoff_s: float,
    description_chars: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    new_queries = 0

    for rank, (ingredient_id, frequency) in enumerate(top_ingredients, start=1):
        label = ingredient_label(graph, ingredient_id)
        query = ingredient_name_to_query(label)

        if not query:
            continue

        key = cache_key(query=query, ontology=ontology, rows=rows_per_query)

        if use_cache and not refresh_cache and key in cache:
            response = cache[key]
        else:
            response = query_ols(
                term=query,
                ontology=ontology,
                rows=rows_per_query,
                url=ols_url,
                timeout_s=timeout_s,
                user_agent=user_agent,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
            )
            cache[key] = response
            new_queries += 1

            if write_cache:
                append_cache(
                    cache_path,
                    key=key,
                    query=query,
                    ontology=ontology,
                    rows=rows_per_query,
                    response=response,
                )

            if sleep_s > 0:
                time.sleep(sleep_s)

        docs = (response.get("response") or {}).get("docs") or []

        if not docs:
            rows.append({
                "rank": int(rank),
                "ingredient_id": ingredient_id,
                "ingredient_label": label,
                "ingredient_query": query,
                "frequency": int(frequency),
                "match_type": "none",
                "match_order": MATCH_ORDER["none"],
                "confidence": 0.0,
                "matched_text": "",
                "foodon_id": "",
                "foodon_label": "",
                "foodon_description": "",
                "foodon_url": "",
                "validated_keep": "",
            })
        else:
            best_doc, match_type, confidence, matched_text = best_doc_for_query(query, docs)

            if best_doc is None:
                best_doc = docs[0]

            rows.append({
                "rank": int(rank),
                "ingredient_id": ingredient_id,
                "ingredient_label": label,
                "ingredient_query": query,
                "frequency": int(frequency),
                "match_type": match_type,
                "match_order": MATCH_ORDER.get(match_type, 99),
                "confidence": round(float(confidence), 6),
                "matched_text": matched_text,
                "foodon_id": safe_str(best_doc.get("obo_id") or best_doc.get("short_form")),
                "foodon_label": safe_str(best_doc.get("label")),
                "foodon_description": doc_description(best_doc, max_chars=description_chars),
                "foodon_url": safe_str(best_doc.get("iri")),
                "validated_keep": "",
            })

        if rank % 25 == 0:
            log.info("%d/%d ingredients processed (%d new queries this run)", rank, len(top_ingredients), new_queries)

    return rows, new_queries


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote CSV: %s (%d rows)", path, len(df))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON: %s", path)


def build_summary(
    *,
    df: pd.DataFrame,
    args: argparse.Namespace,
    total_mentions_top_n: int,
    total_mentions_all: int,
    n_ingredients_total: int,
    new_queries: int,
) -> dict[str, Any]:
    type_counts = {
        str(key): int(value)
        for key, value in df["match_type"].value_counts().items()
    }

    matched = df[df["match_type"] != "none"]

    return {
        "inputs": {
            "graph": str(args.graph),
        },
        "outputs": {
            "coverage_csv": str(args.out_csv),
            "cache_jsonl": str(args.cache),
            "summary_json": str(args.out_summary),
        },
        "parameters": {
            "top_n": args.top_n,
            "ontology": args.ontology,
            "rows_per_query": args.rows,
            "sleep_seconds": args.sleep,
            "ols_url": args.ols_url,
            "use_cache": not args.no_cache,
            "refresh_cache": args.refresh_cache,
            "write_cache": not args.no_write_cache,
        },
        "n_ingredients_total_in_graph": int(n_ingredients_total),
        "total_ingredient_mentions_in_graph": int(total_mentions_all),
        "total_queried": int(len(df)),
        "new_queries_this_run": int(new_queries),
        "match_type_counts": type_counts,
        "n_with_any_match": int(len(matched)),
        "pct_ingredients_with_match": round(100 * len(matched) / max(len(df), 1), 2),
        "cumulative_freq_top_n": int(total_mentions_top_n),
        "cumulative_freq_with_match": int(matched["frequency"].sum()) if not matched.empty else 0,
        "pct_mentions_top_n_over_all_mentions": round(
            100 * total_mentions_top_n / max(total_mentions_all, 1),
            2,
        ),
        "pct_top_n_mentions_covered_by_matches": round(
            100 * (matched["frequency"].sum() if not matched.empty else 0) / max(df["frequency"].sum(), 1),
            2,
        ),
        "review_instruction": (
            "Open foodon_coverage.csv, inspect candidates, and fill validated_keep "
            "with TRUE/FALSE before running the enrichment step."
        ),
    }


def print_summary(summary: dict[str, Any]) -> None:
    log.info("=" * 72)
    log.info("FOODON COVERAGE SUMMARY")
    log.info("=" * 72)
    log.info("Ingredients queried:       %d", summary["total_queried"])
    log.info(
        "With any FoodOn match:     %d (%.2f%%)",
        summary["n_with_any_match"],
        summary["pct_ingredients_with_match"],
    )
    for match_type in [
        "exact_label",
        "exact_synonym",
        "token_set_label",
        "token_set_synonym",
        "partial_label",
        "partial_synonym",
        "none",
    ]:
        log.info("  %-18s %d", match_type, summary["match_type_counts"].get(match_type, 0))

    log.info("")
    log.info(
        "Top-N mention coverage:    %d mentions = %.2f%% of all ingredient mentions",
        summary["cumulative_freq_top_n"],
        summary["pct_mentions_top_n_over_all_mentions"],
    )
    log.info(
        "Matched top-N mentions:    %d = %.2f%% of queried mentions",
        summary["cumulative_freq_with_match"],
        summary["pct_top_n_mentions_covered_by_matches"],
    )
    log.info("")
    log.info("Next: review foodon_coverage.csv and fill validated_keep TRUE/FALSE.")


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
        help=f"Input graph. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of most frequent ingredients to query.",
    )
    parser.add_argument(
        "--ontology",
        default=DEFAULT_ONTOLOGY,
        help=f"Target ontology in OLS. Default: {DEFAULT_ONTOLOGY}",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="Number of OLS documents to retrieve per ingredient query.",
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
        help="Maximum retries per OLS query.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Initial exponential backoff in seconds.",
    )
    parser.add_argument(
        "--ols-url",
        default=DEFAULT_OLS_URL,
        help="OLS search endpoint.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Cache JSONL path. Default: data-dir/foodon_query_cache.jsonl.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Coverage CSV path. Default: data-dir/foodon_coverage.csv.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        help="Summary JSON path. Default: data-dir/foodon_coverage_summary.json.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read existing cache.",
    )
    parser.add_argument(
        "--no-write-cache",
        action="store_true",
        help="Do not append new query responses to cache.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached hits and re-query all terms, while still writing responses to cache unless --no-write-cache.",
    )
    parser.add_argument(
        "--description-chars",
        type=int,
        default=250,
        help="Maximum characters of FoodOn description to include in CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    args.cache = args.cache or args.data_dir / "foodon_query_cache.jsonl"
    args.out_csv = args.out_csv or args.data_dir / "foodon_coverage.csv"
    args.out_summary = args.out_summary or args.data_dir / "foodon_coverage_summary.json"

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    if args.top_n < 1:
        log.error("--top-n must be >= 1")
        return 2
    if args.rows < 1:
        log.error("--rows must be >= 1")
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
    if args.description_chars < 0:
        log.error("--description-chars must be >= 0")
        return 2

    try:
        args.data_dir.mkdir(parents=True, exist_ok=True)

        cache = load_cache(args.cache, enabled=not args.no_cache and not args.refresh_cache)

        graph = load_graph(args.graph)
        frequencies = ingredient_frequencies(graph)

        if not frequencies:
            raise ValueError("No Recipe --contains--> Ingredient edges found in graph")

        top_ingredients = frequencies.most_common(args.top_n)
        total_mentions_top_n = sum(count for _, count in top_ingredients)
        total_mentions_all = sum(frequencies.values())

        log.info(
            "Selected top %d ingredients: %d/%d mentions = %.2f%% of corpus ingredient mentions",
            len(top_ingredients),
            total_mentions_top_n,
            total_mentions_all,
            100 * total_mentions_top_n / max(total_mentions_all, 1),
        )

        rows, new_queries = build_rows(
            graph=graph,
            top_ingredients=top_ingredients,
            cache=cache,
            cache_path=args.cache,
            use_cache=not args.no_cache,
            write_cache=not args.no_write_cache,
            refresh_cache=args.refresh_cache,
            ontology=args.ontology,
            rows_per_query=args.rows,
            ols_url=args.ols_url,
            sleep_s=args.sleep,
            timeout_s=args.timeout,
            user_agent=args.user_agent,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff,
            description_chars=args.description_chars,
        )

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values(
                by=["match_order", "frequency", "confidence", "rank"],
                ascending=[True, False, False, True],
                kind="mergesort",
            ).drop(columns=["match_order"]).reset_index(drop=True)

        write_csv(df, args.out_csv)

        summary = build_summary(
            df=df,
            args=args,
            total_mentions_top_n=total_mentions_top_n,
            total_mentions_all=total_mentions_all,
            n_ingredients_total=len(frequencies),
            new_queries=new_queries,
        )
        write_json(args.out_summary, summary)
        print_summary(summary)

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
