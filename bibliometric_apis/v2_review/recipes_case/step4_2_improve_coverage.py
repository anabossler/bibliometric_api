"""
===========================

Improve validated FoodOn coverage after the first Step 4.0 pass.

This script reads foodon_coverage.csv and applies three conservative strategies:

1. Lower threshold for partial_label / partial_synonym matches
   Some legitimate OLS matches receive low confidence because FoodOn labels often
   append long descriptors. The threshold is configurable.

2. Rescue simple compounds
   For unmatched compound queries such as "pine nuts" or "chicken breasts",
   try simplified forms such as:
     pine nut, nut, pine
     chicken breast, breast, chicken

3. Flag NER residuals for deletion
   Entries that are clearly not ingredients are written to a deletion CSV and
   are not marked validated_keep.

This script does NOT modify the graph. It updates or writes a coverage CSV and
creates an audit JSON.

Default inputs
--------------

  data/foodon_coverage.csv

Default outputs
---------------

  data/foodon_coverage.csv                         updated in place by default
  data/foodon_ner_residual_deletion.csv
  data/foodon_coverage_improvements.json

Usage
-----

  python step4_2_improve_coverage.py

  python step4_2_improve_coverage.py --dry-run

  python step4_2_improve_coverage.py --skip-ols

  python step4_2_improve_coverage.py --output-csv data/foodon_coverage_improved.csv

  python step4_2_improve_coverage.py --partial-threshold 0.45 --max-ols-queries 200

Notes
-----

Downstream Step 4.1 should still use only rows with validated_keep=TRUE.
Review the updated CSV before enrichment if these mappings matter for analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_DATA_DIR = Path("data")
DEFAULT_COVERAGE_CSV = DEFAULT_DATA_DIR / "foodon_coverage.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_COVERAGE_CSV
DEFAULT_CACHE = DEFAULT_DATA_DIR / "foodon_query_cache.jsonl"
DEFAULT_DELETION_CSV = DEFAULT_DATA_DIR / "foodon_ner_residual_deletion.csv"
DEFAULT_AUDIT_JSON = DEFAULT_DATA_DIR / "foodon_coverage_improvements.json"

DEFAULT_OLS_URL = "https://www.ebi.ac.uk/ols4/api/search"
DEFAULT_ONTOLOGY = "foodon"
DEFAULT_ROWS = 5
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_SLEEP_SECONDS = 0.15
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_USER_AGENT = "relish-recipes-research/1.0 (academic; contact: project-local)"

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step4_2_improve_coverage")


MATCH_ORDER = {
    "exact_label": 0,
    "exact_synonym": 1,
    "token_set_label": 2,
    "token_set_synonym": 3,
    "partial_label": 4,
    "partial_synonym": 5,
    "exact_label_rescued": 6,
    "exact_synonym_rescued": 7,
    "token_set_label_rescued": 8,
    "token_set_synonym_rescued": 9,
    "partial_label_rescued": 10,
    "partial_synonym_rescued": 11,
    "none": 99,
}


JUNK_STANDALONE = {
    # Units and measurements
    "kg", "g", "gr", "mg", "ml", "l", "dl", "cl",
    "cup", "cups", "tablespoon", "tablespoons", "tbsp",
    "teaspoon", "teaspoons", "tsp", "pound", "pounds",
    "ounce", "ounces", "oz", "lb", "pinch", "handful",
    # Tools leaked as ingredients
    "spit", "mortar", "cheesecloth", "skewer", "grater", "sieve",
    "bowl", "pot", "pan", "knife", "spoon", "fork", "oven",
    # Abstract / meaningless
    "thing", "kind", "type", "amount", "quantity", "portion",
    "piece", "pieces", "part", "parts", "bit", "bits",
    # Cooking verbs leaked from instructions
    "chop", "grate", "boil", "cook", "fry", "roast", "bake",
    "mix", "stir", "add", "remove", "cover", "heat", "cool",
    "slice", "dice", "mince", "peel",
    # Vague intermediate preparations; not always junk as compounds, but
    # suspicious alone.
    "paste", "mixture", "batter", "dough", "filling", "topping",
    "serving", "recipe", "dish",
}


GERMAN_RESIDUALS = {
    "zucker", "salz", "wasser", "mehl", "milch", "butter", "ei", "eier",
    "zimt", "rosinen", "tl", "el", "prise", "gramm", "puderzucker",
    "eigelb", "schmand", "speck", "vanilleschote", "apfel", "birne",
    "kirschen", "pflaumen", "strudelteig", "knodel", "knoedel",
    "spatzle", "spaetzle",
    "el zimt", "tl zimt", "el salz", "tl salz", "el zucker", "tl zucker",
    "el butter", "tl butter", "el mehl", "el milch", "tl vanille",
    "el honig", "tl kardamom", "gr zucker", "gr mehl", "gr butter",
}


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


def truthy_series(series: pd.Series) -> pd.Series:
    return series.apply(is_true)


# ---------------------------------------------------------------------------
# Loading coverage
# ---------------------------------------------------------------------------

def load_coverage(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Coverage CSV not found: {path}. Run Step 4.0 first.")

    log.info("Loading coverage CSV: %s", path)
    df = pd.read_csv(path)

    required = {"ingredient_id", "ingredient_query", "frequency", "match_type", "confidence", "validated_keep"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Coverage CSV is missing required columns: {sorted(missing)}")

    df["ingredient_id"] = df["ingredient_id"].astype(str)
    df["ingredient_query"] = df["ingredient_query"].fillna("").astype(str)
    df["match_type"] = df["match_type"].fillna("none").astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0).astype(int)

    if "foodon_id" not in df.columns:
        df["foodon_id"] = ""
    if "foodon_label" not in df.columns:
        df["foodon_label"] = ""
    if "foodon_description" not in df.columns:
        df["foodon_description"] = ""
    if "foodon_url" not in df.columns:
        df["foodon_url"] = ""

    log.info("Loaded %d coverage rows", len(df))
    return df


# ---------------------------------------------------------------------------
# NER residual detection
# ---------------------------------------------------------------------------

def is_ner_residual(
    ingredient_query: str,
    *,
    extra_junk_tokens: set[str],
    extra_german_residuals: set[str],
) -> tuple[bool, str]:
    """Return (is_residual, reason)."""
    query = normalize_text(ingredient_query)

    if not query:
        return True, "empty"

    all_german = GERMAN_RESIDUALS | extra_german_residuals
    all_junk = JUNK_STANDALONE | extra_junk_tokens

    if query in all_german:
        return True, "untranslated_german"

    if re.fullmatch(r"\d+(\.\d+)?\s*\w{0,4}", query):
        return True, "numeric_unit"

    tokens = query.split()

    if len(tokens) == 1 and tokens[0] in all_junk:
        return True, f"junk_word:{tokens[0]}"

    if len(tokens) >= 2 and tokens[0] in {
        "chop", "chopped", "grate", "grated", "sliced", "slice",
        "diced", "dice", "mince", "minced", "peel", "peeled",
        "boil", "boiled", "fry", "fried", "roast", "roasted",
        "cook", "cooked", "mix", "mixed", "stir", "stirred",
    }:
        return True, f"verb_prefix:{tokens[0]}"

    if len(tokens) >= 2 and tokens[0] in {"el", "tl", "gr", "kg", "g", "ml", "l"}:
        return True, "unit_prefix"

    if len(tokens) >= 5:
        return True, "long_phrase_fragment"

    return False, ""


def apply_ner_residual_flags(
    df: pd.DataFrame,
    *,
    extra_junk_tokens: set[str],
    extra_german_residuals: set[str],
) -> tuple[pd.DataFrame, int]:
    df = df.copy()

    df["ner_residual"] = ""
    df["ner_residual_reason"] = ""

    for index, row in df.iterrows():
        is_residual, reason = is_ner_residual(
            str(row["ingredient_query"]),
            extra_junk_tokens=extra_junk_tokens,
            extra_german_residuals=extra_german_residuals,
        )
        if is_residual:
            df.at[index, "ner_residual"] = "TRUE"
            df.at[index, "ner_residual_reason"] = reason

    n_residuals = int((df["ner_residual"] == "TRUE").sum())
    log.info("Flagged %d NER residuals", n_residuals)

    return df, n_residuals


# ---------------------------------------------------------------------------
# Cache and OLS
# ---------------------------------------------------------------------------

def cache_key(*, query: str, ontology: str, rows: int) -> str:
    return json.dumps(
        {"query": normalize_text(query), "ontology": ontology, "rows": rows},
        sort_keys=True,
        ensure_ascii=False,
    )


def load_cache(path: Path, *, enabled: bool) -> dict[str, dict[str, Any]]:
    """Load OLS response cache.

    Supports both:
      - Step 4.0 cache: {"key": ..., "response": raw_response}
      - old Step 4.2 cache: {"query": ..., "result": best_match}
    """
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

                if "key" in row and isinstance(row.get("response"), dict):
                    cache[str(row["key"])] = {"kind": "response", "value": row["response"]}
                elif "query" in row and "result" in row:
                    # Legacy cache lacks ontology/rows. Store by normalized query only.
                    legacy_key = json.dumps(
                        {"query": normalize_text(row["query"])},
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    cache[legacy_key] = {"kind": "best_match", "value": row.get("result")}
            except (json.JSONDecodeError, TypeError):
                log.warning("Skipping malformed cache line %d in %s", line_no, path)

    log.info("Loaded %d cached OLS entries", len(cache))
    return cache


def append_cache(
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

    q_tokens = token_set(q)
    c_tokens = token_set(c)
    score_jaccard = jaccard(q_tokens, c_tokens)

    if q_tokens and q_tokens == c_tokens:
        if label_or_synonym == "label":
            return "token_set_label", 0.84
        return "token_set_synonym", 0.78

    if q in c or c in q or score_jaccard >= 0.50:
        if label_or_synonym == "label":
            return "partial_label", round(0.45 + 0.30 * score_jaccard, 4)
        return "partial_synonym", round(0.35 + 0.25 * score_jaccard, 4)

    return "none", 0.0


def classify_doc_match(query: str, doc: dict[str, Any]) -> tuple[str, float, str]:
    label = safe_str(doc.get("label"))
    best_type, best_conf = classify_text_match(query, label, label_or_synonym="label")
    best_text = label

    for synonym in synonyms_of(doc):
        match_type, confidence = classify_text_match(query, synonym, label_or_synonym="synonym")
        if confidence > best_conf:
            best_type, best_conf, best_text = match_type, confidence, synonym

    return best_type, float(best_conf), best_text


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
        rank = (confidence, -MATCH_ORDER.get(match_type, 99))
        best_rank = (best_conf, -MATCH_ORDER.get(best_type, 99))

        if rank > best_rank:
            best_doc = doc
            best_type = match_type
            best_conf = confidence
            best_text = matched_text

    if best_doc is None:
        return None

    description = best_doc.get("description") or ""
    if isinstance(description, list):
        description = description[0] if description else ""

    return {
        "foodon_id": safe_str(best_doc.get("obo_id") or best_doc.get("short_form")),
        "foodon_label": safe_str(best_doc.get("label")),
        "foodon_description": safe_str(description),
        "foodon_url": safe_str(best_doc.get("iri")),
        "match_type": best_type,
        "confidence": round(float(best_conf), 6),
        "matched_text": best_text,
    }


def get_best_match_cached(
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
    """Return best match and whether a new OLS query was made."""
    key = cache_key(query=query, ontology=ontology, rows=rows)
    legacy_key = json.dumps({"query": normalize_text(query)}, sort_keys=True, ensure_ascii=False)

    if use_cache and not refresh_cache:
        if key in cache:
            cached = cache[key]
            if cached.get("kind") == "response":
                return best_doc_for_query(query, cached["value"]), False
            if cached.get("kind") == "best_match":
                return cached.get("value"), False

        if legacy_key in cache:
            cached = cache[legacy_key]
            if cached.get("kind") == "best_match":
                return cached.get("value"), False

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

    cache[key] = {"kind": "response", "value": response}

    if write_cache:
        append_cache(
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
# Compound simplification
# ---------------------------------------------------------------------------

IRREGULAR_SINGULARS = {
    "leaves": "leaf",
    "loaves": "loaf",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "berries": "berry",
    "cherries": "cherry",
    "knives": "knife",
}


def singularize_token(token: str) -> str:
    token = token.lower().strip()
    if token in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        return token[:-3] + "f"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def simplify_compound(query: str) -> list[str]:
    """Generate candidate simplifications for OLS rescue."""
    query_norm = normalize_text(query)
    tokens = query_norm.split()

    if not tokens:
        return []

    candidates: list[str] = []

    # Singularize all words.
    singular_all = " ".join(singularize_token(token) for token in tokens)
    if singular_all and singular_all != query_norm:
        candidates.append(singular_all)

    # Singularize last word only.
    last_singular = " ".join(tokens[:-1] + [singularize_token(tokens[-1])])
    if last_singular not in {query_norm, singular_all}:
        candidates.append(last_singular)

    if len(tokens) >= 2:
        # English head noun: last token.
        head = singularize_token(tokens[-1])
        if head:
            candidates.append(head)

        # Modifier fallback.
        first = singularize_token(tokens[0])
        if first:
            candidates.append(first)

    # Remove duplicates while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)

    return unique


def rescue_compounds(
    df: pd.DataFrame,
    *,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    use_cache: bool,
    write_cache: bool,
    refresh_cache: bool,
    ontology: str,
    rows: int,
    ols_url: str,
    timeout_s: int,
    user_agent: str,
    max_retries: int,
    retry_backoff_s: float,
    sleep_s: float,
    max_ols_queries: int,
    min_rescue_confidence: float,
) -> tuple[pd.DataFrame, int, int]:
    df = df.copy()
    rescued_count = 0
    ols_queries_made = 0

    candidates_df = df[
        (df["match_type"] == "none")
        & (df["ner_residual"] != "TRUE")
        & (df["ingredient_query"].str.contains(" ", na=False))
    ].sort_values("frequency", ascending=False)

    log.info("%d unmatched compound candidates for OLS rescue", len(candidates_df))

    for index, row in candidates_df.iterrows():
        if ols_queries_made >= max_ols_queries:
            log.info("Hit max OLS query cap: %d", max_ols_queries)
            break

        original_query = str(row["ingredient_query"])
        simplifications = simplify_compound(original_query)
        best_match: dict[str, Any] | None = None
        rescued_from = ""

        for candidate_query in simplifications:
            match, was_new = get_best_match_cached(
                query=candidate_query,
                ontology=ontology,
                rows=rows,
                cache=cache,
                cache_path=cache_path,
                use_cache=use_cache,
                write_cache=write_cache,
                refresh_cache=refresh_cache,
                ols_url=ols_url,
                timeout_s=timeout_s,
                user_agent=user_agent,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                sleep_s=sleep_s,
            )

            if was_new:
                ols_queries_made += 1

            if match and safe_str(match.get("foodon_id")) and float(match.get("confidence", 0.0)) >= min_rescue_confidence:
                best_match = match
                rescued_from = candidate_query
                break

            if ols_queries_made >= max_ols_queries:
                break

        if best_match:
            df.at[index, "match_type"] = f"{best_match['match_type']}_rescued"
            df.at[index, "confidence"] = float(best_match["confidence"])
            df.at[index, "foodon_id"] = safe_str(best_match.get("foodon_id"))
            df.at[index, "foodon_label"] = safe_str(best_match.get("foodon_label"))
            df.at[index, "foodon_description"] = safe_str(best_match.get("foodon_description"))
            df.at[index, "foodon_url"] = safe_str(best_match.get("foodon_url"))
            df.at[index, "rescued_from_query"] = rescued_from
            rescued_count += 1

            if rescued_count % 25 == 0:
                log.info("Rescued %d compounds so far (%d new OLS queries)", rescued_count, ols_queries_made)

    log.info("Rescued %d compounds via %d new OLS queries", rescued_count, ols_queries_made)
    return df, rescued_count, ols_queries_made


# ---------------------------------------------------------------------------
# Validation rule
# ---------------------------------------------------------------------------

def should_accept(
    row: pd.Series,
    *,
    partial_threshold: float,
    rescued_partial_threshold: float,
    accept_token_set: bool,
) -> str:
    if row.get("ner_residual") == "TRUE":
        return ""

    match_type = str(row.get("match_type", "none"))
    confidence = float(row.get("confidence", 0.0) or 0.0)

    if match_type in {"exact_label", "exact_synonym", "exact_label_rescued", "exact_synonym_rescued"}:
        return "TRUE"

    if accept_token_set and match_type in {
        "token_set_label",
        "token_set_synonym",
        "token_set_label_rescued",
        "token_set_synonym_rescued",
    }:
        return "TRUE"

    if match_type in {"partial_label", "partial_synonym"} and confidence >= partial_threshold:
        return "TRUE"

    if match_type in {"partial_label_rescued", "partial_synonym_rescued"} and confidence >= rescued_partial_threshold:
        return "TRUE"

    return ""


def apply_validation_rules(
    df: pd.DataFrame,
    *,
    partial_threshold: float,
    rescued_partial_threshold: float,
    accept_token_set: bool,
    preserve_existing_false: bool,
) -> pd.DataFrame:
    df = df.copy()

    if preserve_existing_false:
        existing_false = df["validated_keep"].astype(str).str.strip().str.lower().isin({
            "false", "0", "no", "n", "reject", "rejected"
        })
    else:
        existing_false = pd.Series(False, index=df.index)

    new_values = []
    for index, row in df.iterrows():
        if existing_false.loc[index]:
            new_values.append("FALSE")
        else:
            new_values.append(
                should_accept(
                    row,
                    partial_threshold=partial_threshold,
                    rescued_partial_threshold=rescued_partial_threshold,
                    accept_token_set=accept_token_set,
                )
            )

    df["validated_keep"] = new_values
    return df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def load_extra_tokens(items: list[str], path: Path | None) -> set[str]:
    tokens = {normalize_text(item) for item in items or [] if normalize_text(item)}

    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Token file not found: {path}")

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                tokens.add(normalize_text(line))

    return {token for token in tokens if token}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Wrote CSV: %s (%d rows)", path, len(df))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    log.info("Wrote JSON: %s", path)


def build_deletion_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ingredient_id",
        "ingredient_query",
        "frequency",
        "ner_residual_reason",
        "validated_keep",
    ]

    deletion_df = df[df["ner_residual"] == "TRUE"].copy()
    for column in columns:
        if column not in deletion_df.columns:
            deletion_df[column] = ""

    return deletion_df[columns].sort_values(
        by=["frequency", "ingredient_query"],
        ascending=[False, True],
        kind="mergesort",
    )


def build_audit(
    *,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    args: argparse.Namespace,
    n_ner_residuals: int,
    rescued_count: int,
    ols_queries_made: int,
) -> dict[str, Any]:
    before_validated = int(truthy_series(df_before["validated_keep"]).sum())
    after_validated = int(truthy_series(df_after["validated_keep"]).sum())

    total_mentions = int(df_after["frequency"].sum())
    matched_mentions = int(df_after[truthy_series(df_after["validated_keep"])]["frequency"].sum())

    return {
        "inputs": {
            "coverage_csv": str(args.coverage_csv),
        },
        "outputs": {
            "output_csv": str(args.output_csv),
            "deletion_csv": str(args.deletion_csv),
            "audit_json": str(args.audit_json),
            "cache_jsonl": str(args.cache),
        },
        "config": {
            "partial_threshold": args.partial_threshold,
            "rescued_partial_threshold": args.rescued_partial_threshold,
            "accept_token_set": args.accept_token_set,
            "preserve_existing_false": args.preserve_existing_false,
            "skip_ols": args.skip_ols,
            "max_ols_queries": args.max_ols_queries,
            "min_rescue_confidence": args.min_rescue_confidence,
            "ontology": args.ontology,
            "rows": args.rows,
        },
        "counts": {
            "total_rows": int(len(df_after)),
            "validated_before": before_validated,
            "validated_after": after_validated,
            "validated_delta": after_validated - before_validated,
            "ner_residuals_flagged": int(n_ner_residuals),
            "compounds_rescued": int(rescued_count),
            "ols_queries_made": int(ols_queries_made),
        },
        "coverage": {
            "matched_mentions": matched_mentions,
            "total_mentions": total_mentions,
            "coverage_pct": round(100 * matched_mentions / max(total_mentions, 1), 2),
        },
        "match_type_counts_after": {
            str(key): int(value)
            for key, value in df_after["match_type"].value_counts().items()
        },
    }


def print_audit(audit: dict[str, Any]) -> None:
    log.info("=" * 72)
    log.info("FOODON COVERAGE IMPROVEMENT")
    log.info("=" * 72)
    counts = audit["counts"]
    coverage = audit["coverage"]
    log.info("Validated before:          %d", counts["validated_before"])
    log.info(
        "Validated after:           %d (%+d)",
        counts["validated_after"],
        counts["validated_delta"],
    )
    log.info("NER residuals flagged:     %d", counts["ner_residuals_flagged"])
    log.info("Compounds rescued:         %d", counts["compounds_rescued"])
    log.info("New OLS queries made:      %d", counts["ols_queries_made"])
    log.info(
        "Mention coverage:          %d/%d = %.2f%%",
        coverage["matched_mentions"],
        coverage["total_mentions"],
        coverage["coverage_pct"],
    )


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
        default=DEFAULT_OUTPUT_CSV,
        help="Output coverage CSV. Default overwrites data/foodon_coverage.csv.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"OLS cache JSONL. Default: {DEFAULT_CACHE}",
    )
    parser.add_argument(
        "--deletion-csv",
        type=Path,
        default=DEFAULT_DELETION_CSV,
        help=f"NER residual deletion CSV. Default: {DEFAULT_DELETION_CSV}",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=DEFAULT_AUDIT_JSON,
        help=f"Audit JSON. Default: {DEFAULT_AUDIT_JSON}",
    )
    parser.add_argument(
        "--partial-threshold",
        type=float,
        default=0.4,
        help="Minimum confidence to accept partial_label / partial_synonym.",
    )
    parser.add_argument(
        "--rescued-partial-threshold",
        type=float,
        default=0.4,
        help="Minimum confidence to accept rescued partial matches.",
    )
    parser.add_argument(
        "--accept-token-set",
        action="store_true",
        default=True,
        help="Accept token_set_label / token_set_synonym matches. Default: true.",
    )
    parser.add_argument(
        "--no-accept-token-set",
        dest="accept_token_set",
        action="store_false",
        help="Do not automatically accept token_set matches.",
    )
    parser.add_argument(
        "--preserve-existing-false",
        action="store_true",
        help="Do not override rows explicitly marked FALSE/NO/REJECT in validated_keep.",
    )
    parser.add_argument(
        "--skip-ols",
        action="store_true",
        help="Do not query OLS for compound rescues.",
    )
    parser.add_argument(
        "--max-ols-queries",
        type=int,
        default=500,
        help="Maximum new OLS queries for compound rescue.",
    )
    parser.add_argument(
        "--min-rescue-confidence",
        type=float,
        default=0.4,
        help="Minimum confidence for a rescue match to be copied into the CSV.",
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
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="OLS timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Seconds between new OLS queries.",
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
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header.",
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
        "--junk-token",
        action="append",
        default=[],
        help="Additional standalone junk token. Can be repeated.",
    )
    parser.add_argument(
        "--junk-token-file",
        type=Path,
        help="Optional file with one additional standalone junk token per line.",
    )
    parser.add_argument(
        "--german-residual",
        action="append",
        default=[],
        help="Additional untranslated residual token/query. Can be repeated.",
    )
    parser.add_argument(
        "--german-residual-file",
        type=Path,
        help="Optional file with one additional untranslated residual per line.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute changes but do not write outputs unless --write-dry-run-outputs is set.",
    )
    parser.add_argument(
        "--write-dry-run-outputs",
        action="store_true",
        help="With --dry-run, still write output CSV/deletion CSV/audit JSON.",
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

    if not (0 <= args.partial_threshold <= 1):
        log.error("--partial-threshold must be in [0, 1]")
        return 2
    if not (0 <= args.rescued_partial_threshold <= 1):
        log.error("--rescued-partial-threshold must be in [0, 1]")
        return 2
    if args.max_ols_queries < 0:
        log.error("--max-ols-queries must be >= 0")
        return 2
    if not (0 <= args.min_rescue_confidence <= 1):
        log.error("--min-rescue-confidence must be in [0, 1]")
        return 2
    if args.rows < 1:
        log.error("--rows must be >= 1")
        return 2
    if args.timeout < 1:
        log.error("--timeout must be >= 1")
        return 2
    if args.sleep < 0:
        log.error("--sleep must be >= 0")
        return 2
    if args.max_retries < 1:
        log.error("--max-retries must be >= 1")
        return 2
    if args.retry_backoff < 0:
        log.error("--retry-backoff must be >= 0")
        return 2

    try:
        df_before = load_coverage(args.coverage_csv)
        df = df_before.copy()

        original_validated = int(truthy_series(df_before["validated_keep"]).sum())
        log.info("Validated before: %d", original_validated)

        extra_junk_tokens = load_extra_tokens(args.junk_token, args.junk_token_file)
        extra_german_residuals = load_extra_tokens(args.german_residual, args.german_residual_file)

        log.info("Strategy 3: detect NER residuals")
        df, n_ner_residuals = apply_ner_residual_flags(
            df,
            extra_junk_tokens=extra_junk_tokens,
            extra_german_residuals=extra_german_residuals,
        )

        rescued_count = 0
        ols_queries_made = 0

        if not args.skip_ols and args.max_ols_queries > 0:
            log.info("Strategy 2: rescue compounds via OLS")
            cache = load_cache(args.cache, enabled=not args.no_cache and not args.refresh_cache)
            df, rescued_count, ols_queries_made = rescue_compounds(
                df,
                cache=cache,
                cache_path=args.cache,
                use_cache=not args.no_cache,
                write_cache=not args.no_write_cache,
                refresh_cache=args.refresh_cache,
                ontology=args.ontology,
                rows=args.rows,
                ols_url=args.ols_url,
                timeout_s=args.timeout,
                user_agent=args.user_agent,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff,
                sleep_s=args.sleep,
                max_ols_queries=args.max_ols_queries,
                min_rescue_confidence=args.min_rescue_confidence,
            )
        else:
            log.info("Strategy 2 skipped")

        log.info("Strategy 1: apply validation rules")
        df = apply_validation_rules(
            df,
            partial_threshold=args.partial_threshold,
            rescued_partial_threshold=args.rescued_partial_threshold,
            accept_token_set=args.accept_token_set,
            preserve_existing_false=args.preserve_existing_false,
        )

        deletion_df = build_deletion_dataframe(df)
        audit = build_audit(
            df_before=df_before,
            df_after=df,
            args=args,
            n_ner_residuals=n_ner_residuals,
            rescued_count=rescued_count,
            ols_queries_made=ols_queries_made,
        )

        print_audit(audit)

        if args.dry_run and not args.write_dry_run_outputs:
            log.info("[dry-run] No outputs written.")
            print(json.dumps({
                "counts": audit["counts"],
                "coverage": audit["coverage"],
            }, indent=2, ensure_ascii=False))
            return 0

        write_csv(df, args.output_csv)
        write_csv(deletion_df, args.deletion_csv)
        write_json(args.audit_json, audit)

        log.info("Next: review %s, then run step4_1_enrich_with_foodon.py", args.output_csv)
        return 0

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
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
