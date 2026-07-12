"""

===============================

Diachronic translation layer for FoodOn coverage.

For each Ingredient still unmapped after Step 4.0 / 4.2, this script asks an
LLM for a modern English culinary term using source context, then re-queries
OLS/FoodOn with that translated term.

Design principles
-----------------

1. Full provenance
   Every attempt is written to an append-only JSONL audit file with:
     - prompt version
     - model
     - source context
     - LLM translation and certainty
     - OLS query and result
     - acceptance decision

2. Strict injection policy
   A translation only becomes validated_keep=TRUE if:
     - LLM certainty is in --acceptable-certainty
     - OLS confidence is >= --ols-threshold
     - OLS returns a FoodOn ID

   Weak matches are recorded but not accepted.

3. Resumable
   Ingredients already present in the audit JSONL are skipped unless --rerun is
   used.

4. No hardcoded local secrets
   Set OPENROUTER_API_KEY in the environment or pass --dotenv .env.

Default inputs
--------------

  data/foodon_coverage.csv
  data/graph_step3_flagged.gpickle

Default outputs
---------------

  data/foodon_translation_audit.jsonl
  data/foodon_translation_summary.json
  data/foodon_coverage.csv
  data/foodon_query_cache.jsonl

Usage
-----

  python step4_3_translate_historical.py --dry-run --max-items 20

  python step4_3_translate_historical.py --max-items 100 --dotenv .env

  python step4_3_translate_historical.py

  python step4_3_translate_historical.py --model anthropic/claude-haiku-4-5

Notes
-----

This script does not modify the graph. It updates the coverage CSV only.
Run step4_1_enrich_with_foodon.py after reviewing accepted mappings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_COVERAGE_CSV = Path("data/foodon_coverage.csv")
DEFAULT_GRAPH = Path("data/graph_step3_flagged.gpickle")
DEFAULT_AUDIT_JSONL = Path("data/foodon_translation_audit.jsonl")
DEFAULT_SUMMARY_JSON = Path("data/foodon_translation_summary.json")
DEFAULT_OLS_CACHE = Path("data/foodon_query_cache.jsonl")

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OLS_URL = "https://www.ebi.ac.uk/ols4/api/search"
DEFAULT_ONTOLOGY = "foodon"
DEFAULT_ROWS = 5

DEFAULT_OLS_THRESHOLD = 0.30
DEFAULT_ACCEPTABLE_CERTAINTIES = ["certain", "probable"]

DEFAULT_REQUEST_TIMEOUT_S = 60
DEFAULT_OLS_TIMEOUT_S = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_S = 2.0
DEFAULT_OLS_SLEEP_S = 0.15
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 800

DEFAULT_USER_AGENT = "relish-recipes-research/1.0 (academic; contact: project-local)"
DEFAULT_PROMPT_VERSION = "v2"

CERTAINTY_TO_CONFIDENCE = {
    "certain": 0.95,
    "probable": 0.70,
    "uncertain": 0.40,
}

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step4_3_translate_historical")


SYSTEM_PROMPT = """You are a culinary historian. Given an ingredient term from a historical European recipe, produce the modern English culinary term that would appear in the FoodOn ontology, so the term can be looked up successfully.

Follow these steps internally, then emit the JSON result:

Step 1. Detect the source language of the ingredient term. It may be English, Middle English, or a European language such as German, French, Italian, Spanish, Catalan, Portuguese, Old French, Latin, etc. If in doubt, weigh the "Dominant source language" provided in the user prompt as a strong prior.

Step 2. If the term is not in English, translate it literally to English.

Step 3. Normalise the translated term to the form FoodOn is most likely to hold:
- convert plurals to singular when appropriate
- drop cooking-verb prefixes accidentally attached during NER
- keep common English culinary phrasing
- keep multi-word terms when they are canonical ingredients, such as olive oil or rose water

Step 4. Determine certainty:
- certain: unambiguous translation, well-known ingredient
- probable: reasonable inference with minor ambiguity
- uncertain: significant doubt, or not a food ingredient

Rules:
- Output only valid JSON.
- Do not use markdown fences.
- If the term is clearly not a food ingredient, set modern_term to UNKNOWN and certainty to uncertain.
- Do not translate proper nouns; if it is a place-based dish rather than an ingredient, set modern_term to UNKNOWN.
- Prefer the closest FoodOn-compatible generic term over a specific cultivar or subvariety.
"""

USER_PROMPT_TEMPLATE = """Ingredient term: "{term}"
Dominant source language of recipes using this ingredient: {language}
Source region(s): {region}
Historical period(s): {period}
Number of recipes using it: {n_recipes}

Respond with JSON:
{{
  "source_language_detected": "<language identified, or English>",
  "literal_translation": "<literal English translation; if already English, repeat unchanged>",
  "modern_term": "<normalised English culinary term for FoodOn, or UNKNOWN>",
  "certainty": "<certain|probable|uncertain>",
  "notes": "<one short sentence explaining the reasoning>"
}}
"""


MATCH_ORDER = {
    "exact_label_translated": 0,
    "exact_synonym_translated": 1,
    "token_set_label_translated": 2,
    "token_set_synonym_translated": 3,
    "partial_label_translated": 4,
    "partial_synonym_translated": 5,
    "none": 99,
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_dotenv_file(path: Path | None) -> None:
    """Load KEY=VALUE pairs from a .env file without requiring python-dotenv."""
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

            if key and key not in os.environ:
                os.environ[key] = value

    log.info("Loaded environment variables from %s", path)


def get_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Use environment or --dotenv .env.")
    return api_key


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return default
    return text if text else default


def is_true(value: Any) -> bool:
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass

    return str(value).strip().lower() in {
        "true", "1", "yes", "y", "si", "sí", "keep", "accepted", "accept"
    }


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(str(text)).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Coverage CSV
# ---------------------------------------------------------------------------

def load_coverage(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Coverage CSV not found: {path}")

    log.info("Loading coverage CSV: %s", path)
    df = pd.read_csv(path)

    required = {"ingredient_id", "ingredient_query", "frequency", "match_type", "validated_keep"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coverage CSV is missing required columns: {sorted(missing)}")

    if "ner_residual" not in df.columns:
        df["ner_residual"] = ""

    for column in [
        "foodon_id",
        "foodon_label",
        "foodon_description",
        "foodon_url",
        "confidence",
        "translated_query",
        "translation_certainty",
        "translation_prompt_version",
        "translation_model",
    ]:
        if column not in df.columns:
            df[column] = ""

    df["ingredient_id"] = df["ingredient_id"].astype(str)
    df["ingredient_query"] = df["ingredient_query"].fillna("").astype(str)
    df["match_type"] = df["match_type"].fillna("none").astype(str)
    df["ner_residual"] = df["ner_residual"].fillna("").astype(str)
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0).astype(int)

    log.info("Loaded %d coverage rows", len(df))
    return df


def candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    validated = df["validated_keep"].apply(is_true)
    ner_residual = df["ner_residual"].astype(str).str.upper() == "TRUE"

    candidates = df[
        (df["match_type"].astype(str) == "none")
        & (~ner_residual)
        & (~validated)
    ].copy()

    return candidates.sort_values("frequency", ascending=False, kind="mergesort")


# ---------------------------------------------------------------------------
# Graph context
# ---------------------------------------------------------------------------

def load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    log.info("Loading graph for ingredient context: %s", path)
    with path.open("rb") as fh:
        graph = pickle.load(fh)

    if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise TypeError(f"Object loaded from {path} does not look like a NetworkX graph")

    return graph


def build_ingredient_context(graph_path: Path) -> dict[str, dict[str, Any]]:
    graph = load_graph(graph_path)

    recipe_metadata = {
        str(node_id): attrs
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "Recipe"
    }

    context: dict[str, dict[str, Any]] = {}

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "Ingredient":
            continue

        ingredient_id = str(node_id)
        languages: Counter[str] = Counter()
        regions: Counter[str] = Counter()
        periods: Counter[str] = Counter()
        n_recipes = 0

        for source, _, edge_attrs in graph.in_edges(node_id, data=True):
            if edge_attrs.get("edge_type") != "contains":
                continue

            recipe_attrs = recipe_metadata.get(str(source))
            if recipe_attrs is None:
                continue

            n_recipes += 1

            if recipe_attrs.get("source_language"):
                languages[str(recipe_attrs["source_language"])] += 1
            if recipe_attrs.get("source_place"):
                regions[str(recipe_attrs["source_place"])] += 1
            if recipe_attrs.get("period_derived"):
                periods[str(recipe_attrs["period_derived"])] += 1

        context[ingredient_id] = {
            "dominant_language": languages.most_common(1)[0][0] if languages else "Unknown",
            "region": ", ".join(region for region, _ in regions.most_common(3)) or "Unknown",
            "period": ", ".join(period for period, _ in periods.most_common(3)) or "Unknown",
            "n_recipes": int(n_recipes),
        }

    log.info("Built context for %d ingredients", len(context))
    return context


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def strip_json_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    return text.strip()


def parse_llm_response(content: str) -> dict[str, Any]:
    cleaned = strip_json_fences(content)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    modern_term = safe_str(parsed.get("modern_term"), "UNKNOWN")
    certainty = safe_str(parsed.get("certainty"), "uncertain").lower()
    if certainty not in CERTAINTY_TO_CONFIDENCE:
        certainty = "uncertain"

    return {
        "source_language_detected": safe_str(parsed.get("source_language_detected"))[:100],
        "literal_translation": safe_str(parsed.get("literal_translation"))[:250],
        "modern_term": modern_term or "UNKNOWN",
        "certainty": certainty,
        "notes": safe_str(parsed.get("notes"))[:500],
    }


def call_openrouter(
    *,
    api_key: str,
    model: str,
    url: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    request_timeout_s: int,
    max_retries: int,
    retry_backoff_s: float,
    http_referer: str | None,
    x_title: str | None,
) -> str:
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    data = json.dumps(body).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            request = Request(url, data=data, headers=headers)
            with urlopen(request, timeout=request_timeout_s) as response:
                raw = response.read().decode("utf-8")

            payload = json.loads(raw)
            return safe_str(payload["choices"][0]["message"]["content"])

        except HTTPError as exc:
            last_error = exc
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass

            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "OpenRouter HTTP error attempt %d/%d: %s %s. Sleeping %.1fs",
                attempt + 1,
                max_retries,
                exc.code,
                error_body,
                wait,
            )
            time.sleep(wait)

        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError, IndexError) as exc:
            last_error = exc
            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "OpenRouter error attempt %d/%d: %s. Sleeping %.1fs",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"OpenRouter call failed after {max_retries} retries: {last_error}")


# ---------------------------------------------------------------------------
# OLS cache and query
# ---------------------------------------------------------------------------

def ols_cache_key(*, query: str, ontology: str, rows: int) -> str:
    return json.dumps(
        {"query": normalize_text(query), "ontology": ontology, "rows": rows},
        sort_keys=True,
        ensure_ascii=False,
    )


def classify_text_match(query: str, candidate: str, *, label_or_synonym: str) -> tuple[str, float]:
    q = normalize_text(query)
    c = normalize_text(candidate)

    if not q or not c:
        return "none", 0.0

    if q == c:
        if label_or_synonym == "label":
            return "exact_label_translated", 1.0
        return "exact_synonym_translated", 0.9

    q_tokens = token_set(q)
    c_tokens = token_set(c)
    score_jaccard = jaccard(q_tokens, c_tokens)

    if q_tokens and q_tokens == c_tokens:
        if label_or_synonym == "label":
            return "token_set_label_translated", 0.84
        return "token_set_synonym_translated", 0.78

    if q in c or c in q or score_jaccard >= 0.50:
        if label_or_synonym == "label":
            return "partial_label_translated", round(0.45 + 0.30 * score_jaccard, 6)
        return "partial_synonym_translated", round(0.35 + 0.25 * score_jaccard, 6)

    return "none", 0.0


def synonyms_of(doc: dict[str, Any]) -> list[str]:
    raw = doc.get("synonym") or doc.get("synonyms") or []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def classify_doc_match(query: str, doc: dict[str, Any]) -> tuple[str, float, str]:
    label = safe_str(doc.get("label"))
    best_type, best_conf = classify_text_match(query, label, label_or_synonym="label")
    best_text = label

    for synonym in synonyms_of(doc):
        match_type, confidence = classify_text_match(query, synonym, label_or_synonym="synonym")
        if confidence > best_conf:
            best_type, best_conf, best_text = match_type, confidence, synonym

    return best_type, float(best_conf), best_text


def doc_description(doc: dict[str, Any]) -> str:
    raw = doc.get("description") or ""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    return safe_str(raw)


def best_doc_for_query(query: str, response: dict[str, Any]) -> dict[str, Any] | None:
    docs = (response.get("response") or {}).get("docs") or []
    if not docs:
        return None

    best_doc: dict[str, Any] | None = None
    best_type = "none"
    best_conf = 0.0
    best_text = ""

    for doc in docs:
        match_type, confidence, matched_text = classify_doc_match(query, doc)
        current_rank = (confidence, -MATCH_ORDER.get(match_type, 99))
        best_rank = (best_conf, -MATCH_ORDER.get(best_type, 99))

        if current_rank > best_rank:
            best_doc = doc
            best_type = match_type
            best_conf = confidence
            best_text = matched_text

    if best_doc is None:
        return None

    return {
        "foodon_id": safe_str(best_doc.get("obo_id") or best_doc.get("short_form")),
        "foodon_label": safe_str(best_doc.get("label")),
        "foodon_description": doc_description(best_doc),
        "foodon_url": safe_str(best_doc.get("iri")),
        "match_type": best_type,
        "confidence": round(float(best_conf), 6),
        "matched_text": best_text,
    }


def load_ols_cache(path: Path, *, enabled: bool) -> dict[str, dict[str, Any]]:
    """Load OLS cache supporting current and older shapes."""
    if not enabled or not path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}
    n_current = 0
    n_legacy_result = 0
    n_legacy_response = 0

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed cache line %d in %s", line_no, path)
                continue

            if "key" in record and isinstance(record.get("response"), dict):
                cache[str(record["key"])] = {
                    "kind": "response",
                    "query": safe_str(record.get("query")),
                    "value": record["response"],
                }
                n_current += 1
                continue

            # Legacy shape from older Step 4.2 / 4.3.
            query = safe_str(record.get("query"))
            if not query:
                continue

            legacy_key = json.dumps({"query": normalize_text(query)}, sort_keys=True, ensure_ascii=False)

            if "result" in record:
                cache[legacy_key] = {
                    "kind": "best_match",
                    "query": query,
                    "value": record.get("result"),
                }
                n_legacy_result += 1
            elif "response" in record and isinstance(record.get("response"), dict):
                cache[legacy_key] = {
                    "kind": "response",
                    "query": query,
                    "value": record["response"],
                }
                n_legacy_response += 1

    log.info(
        "OLS cache loaded: %d current responses, %d legacy results, %d legacy responses",
        n_current,
        n_legacy_result,
        n_legacy_response,
    )

    return cache


def append_ols_cache(
    path: Path,
    *,
    key: str,
    query: str,
    ontology: str,
    rows: int,
    response: dict[str, Any],
) -> None:
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


def query_ols(
    *,
    term: str,
    ontology: str,
    rows: int,
    ols_url: str,
    timeout_s: int,
    user_agent: str,
    max_retries: int,
    retry_backoff_s: float,
) -> dict[str, Any]:
    params = {
        "q": term,
        "ontology": ontology,
        "type": "class",
        "rows": rows,
        "queryFields": "label,synonym",
    }
    request_url = f"{ols_url}?{urlencode(params)}"

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
            if exc.code not in {429, 500, 502, 503, 504}:
                log.warning("Non-retryable OLS HTTP error for %r: %s", term, exc.code)
                return {}

            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "OLS HTTP error for %r attempt %d/%d: %s. Sleeping %.1fs",
                term,
                attempt + 1,
                max_retries,
                exc.code,
                wait,
            )
            time.sleep(wait)

        except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
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


def get_ols_best_match(
    *,
    query: str,
    ontology: str,
    rows: int,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    use_cache: bool,
    write_cache: bool,
    refresh_cache: bool,
    ols_url: str,
    timeout_s: int,
    user_agent: str,
    max_retries: int,
    retry_backoff_s: float,
    sleep_s: float,
) -> tuple[dict[str, Any] | None, bool]:
    key = ols_cache_key(query=query, ontology=ontology, rows=rows)
    legacy_key = json.dumps({"query": normalize_text(query)}, sort_keys=True, ensure_ascii=False)

    if use_cache and not refresh_cache:
        for candidate_key in [key, legacy_key]:
            cached = cache.get(candidate_key)
            if not cached:
                continue

            if cached.get("kind") == "best_match":
                value = cached.get("value")
                return value if isinstance(value, dict) else None, False

            if cached.get("kind") == "response":
                cached_query = safe_str(cached.get("query")) or query
                return best_doc_for_query(cached_query, cached["value"]), False

    response = query_ols(
        term=query,
        ontology=ontology,
        rows=rows,
        ols_url=ols_url,
        timeout_s=timeout_s,
        user_agent=user_agent,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )

    cache[key] = {
        "kind": "response",
        "query": query,
        "value": response,
    }

    if write_cache:
        append_ols_cache(
            cache_path,
            key=key,
            query=query,
            ontology=ontology,
            rows=rows,
            response=response,
        )

    if sleep_s > 0:
        time.sleep(sleep_s)

    return best_doc_for_query(query, response), True


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def load_audit_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                ingredient_id = safe_str(record.get("ingredient_id"))
                if ingredient_id:
                    records[ingredient_id] = record
            except json.JSONDecodeError:
                log.warning("Skipping malformed audit line %d in %s", line_no, path)

    return records


def append_audit(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def acceptance_decision(
    *,
    modern_term: str,
    certainty: str,
    ols_result: dict[str, Any] | None,
    ols_threshold: float,
    acceptable_certainties: set[str],
) -> tuple[bool, str]:
    if modern_term.upper() == "UNKNOWN":
        return False, "llm_unknown"

    if ols_result is None or not safe_str(ols_result.get("foodon_id")):
        return False, "ols_no_match"

    if certainty not in acceptable_certainties:
        return False, f"llm_certainty_not_accepted:{certainty}"

    ols_conf = float(ols_result.get("confidence", 0.0) or 0.0)
    if ols_conf < ols_threshold:
        return False, f"ols_below_threshold:{ols_conf}<{ols_threshold}"

    return True, f"accepted:llm={certainty};ols_conf={ols_conf}"


def update_coverage_row(
    df: pd.DataFrame,
    *,
    ingredient_id: str,
    llm_result: dict[str, Any],
    ols_result: dict[str, Any],
    model: str,
    prompt_version: str,
) -> None:
    matches = df.index[df["ingredient_id"].astype(str) == ingredient_id].tolist()
    if not matches:
        return

    index = matches[0]

    df.at[index, "match_type"] = ols_result["match_type"]
    df.at[index, "confidence"] = ols_result["confidence"]
    df.at[index, "foodon_id"] = ols_result["foodon_id"]
    df.at[index, "foodon_label"] = ols_result["foodon_label"]
    df.at[index, "foodon_description"] = ols_result.get("foodon_description", "")
    df.at[index, "foodon_url"] = ols_result["foodon_url"]
    df.at[index, "validated_keep"] = "TRUE"
    df.at[index, "translated_query"] = llm_result["modern_term"]
    df.at[index, "translation_certainty"] = llm_result["certainty"]
    df.at[index, "translation_prompt_version"] = prompt_version
    df.at[index, "translation_model"] = model


def process_candidates(
    *,
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    context: dict[str, dict[str, Any]],
    audit_records: dict[str, dict[str, Any]],
    audit_path: Path,
    ols_cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, int]:
    stats = Counter()

    acceptable_certainties = {item.strip().lower() for item in args.acceptable_certainty}

    for _, row in candidates.iterrows():
        ingredient_id = safe_str(row["ingredient_id"])
        term = safe_str(row["ingredient_query"])

        if not ingredient_id or not term:
            continue

        if not args.rerun and ingredient_id in audit_records:
            stats["skipped_existing_audit"] += 1
            continue

        info = context.get(ingredient_id, {
            "dominant_language": "Unknown",
            "region": "Unknown",
            "period": "Unknown",
            "n_recipes": 0,
        })

        user_prompt = USER_PROMPT_TEMPLATE.format(
            term=term,
            language=info.get("dominant_language", "Unknown"),
            region=info.get("region", "Unknown"),
            period=info.get("period", "Unknown"),
            n_recipes=info.get("n_recipes", 0),
        )

        try:
            raw_llm_content = call_openrouter(
                api_key=api_key,
                model=args.model,
                url=args.openrouter_url,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                request_timeout_s=args.request_timeout,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff,
                http_referer=args.http_referer,
                x_title=args.x_title,
            )
            llm_result = parse_llm_response(raw_llm_content)
        except RuntimeError as exc:
            log.error("Skipping %s (%s): LLM failure: %s", ingredient_id, term, exc)
            stats["llm_failure"] += 1
            continue

        modern_term = llm_result["modern_term"]
        certainty = llm_result["certainty"]
        llm_confidence = CERTAINTY_TO_CONFIDENCE.get(certainty, 0.40)

        ols_result: dict[str, Any] | None = None
        ols_query_made = False

        if modern_term and modern_term.upper() != "UNKNOWN":
            ols_result, ols_query_made = get_ols_best_match(
                query=modern_term,
                ontology=args.ontology,
                rows=args.rows,
                cache=ols_cache,
                cache_path=args.ols_cache,
                use_cache=not args.no_cache,
                write_cache=not args.no_write_cache,
                refresh_cache=args.refresh_cache,
                ols_url=args.ols_url,
                timeout_s=args.ols_timeout,
                user_agent=args.user_agent,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff,
                sleep_s=args.ols_sleep,
            )
            if ols_query_made:
                stats["ols_queries_made"] += 1
        else:
            stats["llm_unknown"] += 1

        accepted, reason = acceptance_decision(
            modern_term=modern_term,
            certainty=certainty,
            ols_result=ols_result,
            ols_threshold=args.ols_threshold,
            acceptable_certainties=acceptable_certainties,
        )

        if accepted and ols_result is not None:
            update_coverage_row(
                df,
                ingredient_id=ingredient_id,
                llm_result=llm_result,
                ols_result=ols_result,
                model=args.model,
                prompt_version=args.prompt_version,
            )
            stats["accepted_into_graph"] += 1
        else:
            if reason == "ols_no_match":
                stats["ols_no_match"] += 1
            elif reason.startswith("ols_below_threshold"):
                stats["ols_below_threshold"] += 1
            elif reason.startswith("llm_certainty_not_accepted"):
                stats["llm_certainty_not_accepted"] += 1

        audit_entry = {
            "ingredient_id": ingredient_id,
            "ingredient_query": term,
            "frequency": int(row["frequency"]) if pd.notna(row["frequency"]) else 0,
            "context": info,
            "llm_model": args.model,
            "llm_prompt_version": args.prompt_version,
            "llm_source_language_detected": llm_result["source_language_detected"],
            "llm_literal_translation": llm_result["literal_translation"],
            "llm_translation": modern_term,
            "llm_certainty": certainty,
            "llm_confidence": llm_confidence,
            "llm_notes": llm_result["notes"],
            "ols_query": modern_term if modern_term.upper() != "UNKNOWN" else "",
            "ols_match_type": safe_str((ols_result or {}).get("match_type")),
            "ols_confidence": float((ols_result or {}).get("confidence", 0.0) or 0.0),
            "ols_foodon_id": safe_str((ols_result or {}).get("foodon_id")),
            "ols_foodon_label": safe_str((ols_result or {}).get("foodon_label")),
            "ols_threshold_used": args.ols_threshold,
            "accepted_into_graph": bool(accepted),
            "acceptance_reason": reason,
            "timestamp": timestamp_utc(),
        }
        append_audit(audit_path, audit_entry)
        stats["processed"] += 1

        if stats["processed"] % args.flush_every == 0:
            df.to_csv(args.output_csv, index=False, encoding="utf-8")
            log.info(
                "%d processed; accepted=%d; llm_unknown=%d; ols_no_match=%d",
                stats["processed"],
                stats["accepted_into_graph"],
                stats["llm_unknown"],
                stats["ols_no_match"],
            )

    return {str(key): int(value) for key, value in stats.items()}


def write_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    stats: dict[str, int],
    n_candidates_initial: int,
    n_candidates_after_resume: int,
) -> None:
    summary = {
        "inputs": {
            "coverage_csv": str(args.coverage_csv),
            "graph": str(args.graph),
        },
        "outputs": {
            "output_csv": str(args.output_csv),
            "audit_jsonl": str(args.audit_jsonl),
            "summary_json": str(path),
            "ols_cache": str(args.ols_cache),
        },
        "parameters": {
            "model": args.model,
            "prompt_version": args.prompt_version,
            "ols_threshold": args.ols_threshold,
            "acceptable_certainty": args.acceptable_certainty,
            "max_items": args.max_items,
            "ontology": args.ontology,
            "rows": args.rows,
            "rerun": args.rerun,
            "dry_run": args.dry_run,
        },
        "candidate_counts": {
            "initial_candidates": int(n_candidates_initial),
            "after_resume_filter": int(n_candidates_after_resume),
        },
        "stats": stats,
        "notes": [
            "LLM-mediated mappings are accepted only under strict certainty and OLS-confidence thresholds.",
            "Every attempted translation is recorded in the audit JSONL.",
            "Review the coverage CSV before FoodOn enrichment if these mappings are analytically important.",
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log.info("Wrote summary JSON: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=DEFAULT_COVERAGE_CSV,
        help=f"Input coverage CSV. Default: {DEFAULT_COVERAGE_CSV}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_COVERAGE_CSV,
        help="Output coverage CSV. Default overwrites input in place.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH,
        help=f"Graph for ingredient context. Default: {DEFAULT_GRAPH}",
    )
    parser.add_argument(
        "--audit-jsonl",
        type=Path,
        default=DEFAULT_AUDIT_JSONL,
        help=f"Append-only audit JSONL. Default: {DEFAULT_AUDIT_JSONL}",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help=f"Summary JSON. Default: {DEFAULT_SUMMARY_JSON}",
    )
    parser.add_argument(
        "--ols-cache",
        type=Path,
        default=DEFAULT_OLS_CACHE,
        help=f"Shared OLS cache JSONL. Default: {DEFAULT_OLS_CACHE}",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        help="Optional .env file containing OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--openrouter-url",
        default=DEFAULT_OPENROUTER_URL,
        help="OpenRouter chat completions endpoint.",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt version string written to audit.",
    )
    parser.add_argument(
        "--ols-threshold",
        type=float,
        default=DEFAULT_OLS_THRESHOLD,
        help="Minimum OLS confidence for accepted mapping.",
    )
    parser.add_argument(
        "--acceptable-certainty",
        action="append",
        default=list(DEFAULT_ACCEPTABLE_CERTAINTIES),
        help="LLM certainty accepted into graph. Can be repeated.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Process at most N new ingredients.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned items without calling LLM or OLS.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Do not skip ingredients already present in the audit JSONL.",
    )
    parser.add_argument(
        "--ontology",
        default=DEFAULT_ONTOLOGY,
        help=f"OLS ontology. Default: {DEFAULT_ONTOLOGY}",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="OLS rows per query.",
    )
    parser.add_argument(
        "--ols-url",
        default=DEFAULT_OLS_URL,
        help="OLS search endpoint.",
    )
    parser.add_argument(
        "--ols-timeout",
        type=int,
        default=DEFAULT_OLS_TIMEOUT_S,
        help="OLS timeout in seconds.",
    )
    parser.add_argument(
        "--ols-sleep",
        type=float,
        default=DEFAULT_OLS_SLEEP_S,
        help="Seconds to sleep between uncached OLS calls.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read existing OLS cache.",
    )
    parser.add_argument(
        "--no-write-cache",
        action="store_true",
        help="Do not append new OLS responses to cache.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached hits and re-query.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="OpenRouter request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for OpenRouter and OLS calls.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_S,
        help="Initial exponential backoff in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens for LLM response.",
    )
    parser.add_argument(
        "--http-referer",
        default=None,
        help="Optional HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--x-title",
        default="Relish-FoodOn-Translation",
        help="Optional X-Title header for OpenRouter.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent for OLS.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Write coverage CSV every N processed accepted/rejected attempts.",
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

    if not (0 <= args.ols_threshold <= 1):
        log.error("--ols-threshold must be in [0, 1]")
        return 2
    if args.max_items is not None and args.max_items < 1:
        log.error("--max-items must be >= 1")
        return 2
    if args.rows < 1:
        log.error("--rows must be >= 1")
        return 2
    if args.ols_timeout < 1:
        log.error("--ols-timeout must be >= 1")
        return 2
    if args.ols_sleep < 0:
        log.error("--ols-sleep must be >= 0")
        return 2
    if args.request_timeout < 1:
        log.error("--request-timeout must be >= 1")
        return 2
    if args.max_retries < 1:
        log.error("--max-retries must be >= 1")
        return 2
    if args.retry_backoff < 0:
        log.error("--retry-backoff must be >= 0")
        return 2
    if args.max_output_tokens < 1:
        log.error("--max-output-tokens must be >= 1")
        return 2
    if args.flush_every < 1:
        log.error("--flush-every must be >= 1")
        return 2

    try:
        df = load_coverage(args.coverage_csv)
        candidates = candidate_rows(df)
        n_candidates_initial = len(candidates)

        audit_records = load_audit_records(args.audit_jsonl)
        if not args.rerun and audit_records:
            candidates = candidates[~candidates["ingredient_id"].astype(str).isin(set(audit_records))]
        n_candidates_after_resume = len(candidates)

        if args.max_items is not None:
            candidates = candidates.head(args.max_items).copy()

        log.info("Initial translation candidates: %d", n_candidates_initial)
        log.info("After resume filter: %d", n_candidates_after_resume)
        log.info("Planned for this run: %d", len(candidates))

        if len(candidates) == 0:
            stats = {"processed": 0}
            write_summary(
                path=args.summary_json,
                args=args,
                stats=stats,
                n_candidates_initial=n_candidates_initial,
                n_candidates_after_resume=n_candidates_after_resume,
            )
            log.info("Nothing to do.")
            return 0

        context = build_ingredient_context(args.graph)

        if args.dry_run:
            for _, row in candidates.head(10).iterrows():
                ingredient_id = safe_str(row["ingredient_id"])
                info = context.get(ingredient_id, {})
                log.info(
                    "[dry-run] %s | %s | language=%s, region=%s, period=%s, n=%s",
                    ingredient_id,
                    row["ingredient_query"],
                    info.get("dominant_language", "Unknown"),
                    info.get("region", "Unknown"),
                    info.get("period", "Unknown"),
                    info.get("n_recipes", 0),
                )
            if len(candidates) > 10:
                log.info("[dry-run] ... %d more", len(candidates) - 10)
            log.info("[dry-run] No API calls and no outputs written except optional summary.")
            return 0

        load_dotenv_file(args.dotenv)
        api_key = get_api_key()

        ols_cache = load_ols_cache(
            args.ols_cache,
            enabled=not args.no_cache and not args.refresh_cache,
        )

        log.info("Starting translation layer with model=%s, ols_threshold=%.3f", args.model, args.ols_threshold)

        stats = process_candidates(
            df=df,
            candidates=candidates,
            context=context,
            audit_records=audit_records,
            audit_path=args.audit_jsonl,
            ols_cache=ols_cache,
            args=args,
            api_key=api_key,
        )

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False, encoding="utf-8")
        log.info("Wrote updated coverage CSV: %s", args.output_csv)

        write_summary(
            path=args.summary_json,
            args=args,
            stats=stats,
            n_candidates_initial=n_candidates_initial,
            n_candidates_after_resume=n_candidates_after_resume,
        )

        log.info("=" * 72)
        log.info("TRANSLATION LAYER SUMMARY")
        log.info("Processed:             %d", stats.get("processed", 0))
        log.info("Accepted into graph:   %d", stats.get("accepted_into_graph", 0))
        log.info("LLM UNKNOWN:           %d", stats.get("llm_unknown", 0))
        log.info("OLS no match:          %d", stats.get("ols_no_match", 0))
        log.info("OLS below threshold:   %d", stats.get("ols_below_threshold", 0))
        log.info("LLM certainty rejected:%d", stats.get("llm_certainty_not_accepted", 0))
        log.info("Audit JSONL:           %s", args.audit_jsonl)
        log.info("Next: review coverage CSV, then run step4_1_enrich_with_foodon.py")
        return 0

    except (
        FileNotFoundError,
        RuntimeError,
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
