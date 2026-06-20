"""
extract_paper_tmo.py
====================

Paper-level TMO extraction for the AWS revision.

For each paper in the plastic recycling corpus, extracts three structured
fields via Claude Haiku 4.5 (OpenRouter):
  - techniques : specific technical processes / methods used
  - methods    : analytical / experimental procedures
  - objectives : high-level research goals or intended outcomes

INPUTS:
  backup_recycled_a/full_corpus/abstracts_full.csv  (doi, abstract)

OUTPUTS (in runs/paper_tmo/):
  paper_tmo.csv         clean table (doi, techniques, methods, objectives,
                                     n_terms, status, model, timestamp)
  paper_tmo_raw.jsonl   one JSON per paper with the full LLM response
                        (audit trail for reviewers)
  paper_tmo_log.txt     summary log

RESUME BEHAVIOUR:
  If interrupted, rerun the script. It reads paper_tmo.csv and skips any
  doi already processed successfully. Failed papers (status='error') are
  retried on the next run. Safe to Ctrl+C at any time.

ENV (set in .env):
  OPENROUTER_API_KEY

USAGE:
  python extract_paper_tmo.py
  python extract_paper_tmo.py --limit 50         # quick smoke test
  python extract_paper_tmo.py --retry-errors     # rerun only failed papers
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path.home() / "Desktop/openalex/.env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "google/gemini-2.5-flash-lite"

BASE_DIR  = Path("backup_recycled_a/full_corpus")
ABSTRACTS = BASE_DIR / "abstracts_full.csv"

OUT_DIR   = Path("runs/paper_tmo")
CSV_PATH  = OUT_DIR / "paper_tmo.csv"
JSONL_PATH = OUT_DIR / "paper_tmo_raw.jsonl"
LOG_PATH  = OUT_DIR / "paper_tmo_log.txt"

MAX_RETRIES   = 3
RETRY_BASE_S  = 2.0
INTER_CALL_S  = 0.4    # gentle rate limit; OpenRouter handles bursts
MAX_TOKENS    = 500
TEMPERATURE   = 0.0    # deterministic for reproducibility

CSV_FIELDS = [
    "doi", "techniques", "methods", "objectives",
    "n_terms", "status", "model", "timestamp",
]

EXTRACTION_PROMPT = """\
You are a scientific knowledge extractor working on plastic recycling literature.

Given a single paper abstract, extract three structured lists describing what the paper does:
- techniques: specific technical processes the paper uses or studies (e.g., pyrolysis, mechanical recycling, glycolysis, enzymatic depolymerization)
- methods: analytical or experimental procedures used to evaluate or characterize (e.g., thermogravimetric analysis, GC-MS, life cycle assessment, kinetic modeling)
- objectives: high-level research goals or intended outcomes (e.g., monomer recovery, depolymerization, waste valorization, plastic-to-fuel conversion)

Rules:
- Each item is a short noun phrase (2-5 words max).
- 3-10 items per list. If a list does not apply, return an empty array [].
- Be specific to THIS paper. Do not include generic terms like "research", "study", "analysis".
- Prefer technical phrasing as used in the abstract.
- Return ONLY a valid JSON object with keys "techniques", "methods", "objectives". No markdown, no preamble, no commentary.

Abstract:
{abstract}
"""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tmo")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_haiku(abstract: str) -> tuple[dict | None, str | None]:
    """Call Claude Haiku via OpenRouter. Returns (parsed_dict, raw_text)."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set in environment")

    prompt = EXTRACTION_PROMPT.format(abstract=abstract.strip())
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(OPENROUTER_URL, json=payload,
                              headers=headers, timeout=60)
            if r.status_code == 429:
                wait = RETRY_BASE_S ** attempt + random.uniform(0, 1)
                log.warning("429 rate-limited, sleeping %.1fs", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            parsed = _parse_json(text)
            return parsed, text
        except Exception as ex:
            last_err = ex
            wait = RETRY_BASE_S ** attempt + random.uniform(0, 1)
            log.warning("Attempt %d failed (%s), sleeping %.1fs",
                        attempt, type(ex).__name__, wait)
            time.sleep(wait)

    log.error("All %d attempts failed: %s", MAX_RETRIES, last_err)
    return None, None


def _parse_json(text: str) -> dict | None:
    """Strip markdown fences and find the JSON object."""
    t = text.strip()
    if t.startswith("```"):
        # remove opening fence (with optional 'json' tag)
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        obj = json.loads(t[s:e + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    for k in ("techniques", "methods", "objectives"):
        v = obj.get(k, [])
        if not isinstance(v, list):
            obj[k] = []
        else:
            obj[k] = [str(x).strip() for x in v if str(x).strip()]
    return obj


# ---------------------------------------------------------------------------
# Resume / IO helpers
# ---------------------------------------------------------------------------

def load_existing() -> dict[str, str]:
    """Return {doi: status} for already-processed papers."""
    if not CSV_PATH.exists():
        return {}
    df = pd.read_csv(CSV_PATH, usecols=["doi", "status"])
    return dict(zip(df["doi"].astype(str), df["status"].astype(str)))


def append_csv_row(row: dict) -> None:
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def append_jsonl(record: dict) -> None:
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_csv_dropping_errors() -> int:
    """For --retry-errors: keep only successful rows, return number dropped."""
    if not CSV_PATH.exists():
        return 0
    df = pd.read_csv(CSV_PATH)
    keep = df[df["status"] == "ok"]
    dropped = len(df) - len(keep)
    keep.to_csv(CSV_PATH, index=False)
    return dropped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper-level TMO extraction")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N unprocessed papers (smoke test)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Drop existing 'error' rows and retry them")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ABSTRACTS.exists():
        log.error("Abstracts file not found: %s", ABSTRACTS)
        sys.exit(1)

    if args.retry_errors:
        dropped = rewrite_csv_dropping_errors()
        log.info("Dropped %d error rows; will retry them", dropped)

    log.info("Loading corpus from %s", ABSTRACTS)
    df = pd.read_csv(ABSTRACTS)
    if "doi" not in df.columns or "abstract" not in df.columns:
        log.error("CSV must contain 'doi' and 'abstract' columns")
        sys.exit(1)
    df = df[df["abstract"].notna() & (df["abstract"].astype(str).str.len() > 50)]
    df = df.drop_duplicates(subset=["doi"]).reset_index(drop=True)
    log.info("Corpus loaded: %d papers with usable abstracts", len(df))

    existing = load_existing()
    n_done   = sum(1 for s in existing.values() if s == "ok")
    n_err    = sum(1 for s in existing.values() if s == "error")
    log.info("Resume state: %d ok, %d error, %d total in CSV",
             n_done, n_err, len(existing))

    to_process = [
        (row["doi"], row["abstract"])
        for _, row in df.iterrows()
        if existing.get(row["doi"]) != "ok"
    ]
    if args.limit:
        to_process = to_process[:args.limit]
    log.info("Will process %d papers", len(to_process))

    if not to_process:
        log.info("Nothing to do.")
        return

    t0 = time.time()
    n_ok = n_fail = 0

    try:
        for i, (doi, abstract) in enumerate(to_process, 1):
            parsed, raw = call_haiku(str(abstract))
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")

            if parsed is None:
                n_fail += 1
                append_csv_row({
                    "doi": doi, "techniques": "", "methods": "", "objectives": "",
                    "n_terms": 0, "status": "error",
                    "model": MODEL, "timestamp": now,
                })
                append_jsonl({"doi": doi, "status": "error",
                              "raw": raw, "timestamp": now})
            else:
                techs = parsed.get("techniques", [])
                meths = parsed.get("methods", [])
                objs  = parsed.get("objectives", [])
                n_terms = len(techs) + len(meths) + len(objs)
                n_ok += 1
                append_csv_row({
                    "doi": doi,
                    "techniques": " | ".join(techs),
                    "methods":    " | ".join(meths),
                    "objectives": " | ".join(objs),
                    "n_terms":    n_terms,
                    "status":     "ok",
                    "model":      MODEL,
                    "timestamp":  now,
                })
                append_jsonl({"doi": doi, "status": "ok",
                              "parsed": parsed, "raw": raw, "timestamp": now})

            if i % 25 == 0 or i == len(to_process):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(to_process) - i) / rate if rate > 0 else 0
                log.info("[%d/%d] ok=%d fail=%d | %.1f/s | ETA %.1f min",
                         i, len(to_process), n_ok, n_fail,
                         rate, remaining / 60)

            time.sleep(INTER_CALL_S)

    except KeyboardInterrupt:
        log.warning("Interrupted by user. Progress saved. Rerun to resume.")
    finally:
        elapsed = time.time() - t0
        summary = (
            f"\n{'=' * 60}\n"
            f"PAPER TMO EXTRACTION SUMMARY\n"
            f"{'=' * 60}\n"
            f"Corpus           : {ABSTRACTS}\n"
            f"Model            : {MODEL}\n"
            f"Run finished     : {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"Processed (run)  : {n_ok + n_fail}\n"
            f"  ok             : {n_ok}\n"
            f"  failed         : {n_fail}\n"
            f"Elapsed          : {elapsed:.0f} s ({elapsed / 60:.1f} min)\n"
            f"Outputs          : {CSV_PATH}\n"
            f"                   {JSONL_PATH}\n"
            f"{'=' * 60}\n"
        )
        log.info(summary)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(summary)


if __name__ == "__main__":
    main()
