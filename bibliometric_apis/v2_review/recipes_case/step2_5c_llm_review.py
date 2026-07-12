"""
======================

LLM-assisted classification of ingredient-canonicalization review hubs.

For each hub in canonicalization_needs_review.csv, the script sends the hub and
its candidate ingredient variants to an LLM with a culinary-domain prompt. The
LLM marks each candidate as:

  - MERGE
  - KEEP_SEPARATE
  - DELETE

This step does NOT modify the graph and should NOT be treated as the final
decision. It creates a review file that a human expert can validate.

Reads
-----

  data/canonicalization_needs_review.csv

Writes
------

  data/checkpoints/llm_hub_decisions.jsonl
  data/canonicalization_needs_review_llm.csv
  data/canonicalization_needs_review_llm_summary.json

Usage
-----

  python step2_5c_llm_review.py --dry-run

  python step2_5c_llm_review.py \
    --input data/canonicalization_needs_review.csv \
    --output data/canonicalization_needs_review_llm.csv \
    --dotenv .env

  python step2_5c_llm_review.py --limit 50 --batch-max-hubs 20

  python step2_5c_llm_review.py --build-final-only

API key
-------

Set OPENROUTER_API_KEY in your environment, or pass --dotenv .env.

Safety
------

By default, `accepted_ids` is left empty. Use --prefill-accepted only if you
explicitly want the LLM's MERGE suggestions copied into accepted_ids for later
manual editing. This avoids accidentally committing LLM suggestions as final
canonicalization decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/canonicalization_needs_review.csv")
DEFAULT_OUTPUT = Path("data/canonicalization_needs_review_llm.csv")
DEFAULT_SUMMARY = Path("data/canonicalization_needs_review_llm_summary.json")
DEFAULT_CHECKPOINT_DIR = Path("data/checkpoints")
DEFAULT_CHECKPOINT_NAME = "llm_hub_decisions.jsonl"

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_BATCH_MAX_HUBS = 25
DEFAULT_BATCH_MAX_CANDIDATES = 40
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_S = 2.0
DEFAULT_REQUEST_TIMEOUT_S = 180
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 8000

VALID_ACTIONS = {"MERGE", "KEEP_SEPARATE", "DELETE", "UNKNOWN"}

LOG_FORMAT = "%(levelname)s | %(message)s"
log = logging.getLogger("step2_5c_llm_review")


SYSTEM_PROMPT = """You are a culinary expert helping canonicalize ingredient names extracted by NER from a corpus of European historical recipes translated to English.

The corpus has NER-induced redundancy: the same concept may appear under multiple surface forms. Your job is to decide, for each hub-candidate pair, whether they refer to the same culinary ingredient.

For each candidate, output one of three actions:
- MERGE: candidate is the same ingredient as the hub. Examples: claw/claws; wine_broth/broth_wine; eggplant/aubergine; Wein/wine when both clearly mean wine.
- KEEP_SEPARATE: candidate is functionally distinct in cooking even if linguistically similar. Examples: almond_seed vs almond_shell; chicken_broth vs vegetable_broth; red_wine vs white_wine; wheat_flour vs bread_flour.
- DELETE: candidate is NER noise, not a real culinary ingredient. Examples: phrase fragments, function-word leftovers, generic placeholders such as type_X or kind_X when the modifier carries no culinary information.

When in doubt, prefer KEEP_SEPARATE. Preserve functional distinctions that matter for cooking.

Respond only with valid JSON. Do not use markdown fences. Do not add prose outside JSON."""


USER_TEMPLATE = """Classify these candidates against their hubs. Apply culinary domain knowledge.

{hubs_block}

Respond with this exact JSON schema:
{{
  "results": [
    {{
      "hub": "<hub_id>",
      "decisions": [
        {{"candidate": "<candidate_id>", "action": "MERGE|KEEP_SEPARATE|DELETE", "reason": "<brief>"}}
      ]
    }}
  ]
}}"""


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
# Input parsing
# ---------------------------------------------------------------------------

def load_review_hubs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Needs-review CSV not found: {path}")

    log.info("Loading needs-review hubs: %s", path)
    df = pd.read_csv(path)

    required = {"hub_id", "candidates_summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df["hub_id"] = df["hub_id"].astype(str)
    df["candidates_summary"] = df["candidates_summary"].fillna("").astype(str)

    log.info("Loaded %d hubs", len(df))
    return df


def parse_candidates_summary(summary: str) -> list[tuple[str, float]]:
    """Parse grouped candidate summaries into [(candidate_id, similarity), ...].

    Supports both common formats:

      ing::orange_juice(0.951)
      ing::orange_juice [orange juice] (0.9510)
      ing::orange_juice (0.9510)

    Items may be separated by semicolons.
    """
    if not summary or pd.isna(summary):
        return []

    pairs: list[tuple[str, float]] = []

    for chunk in str(summary).split(";"):
        item = chunk.strip()
        if not item:
            continue

        # Candidate id is normally the first ing::... token.
        id_match = re.search(r"\b(ing::[^\s;\[\]()]+)", item)
        sim_match = re.search(r"\(([01](?:\.\d+)?)\)\s*$", item)

        if not id_match or not sim_match:
            continue

        try:
            pairs.append((id_match.group(1), float(sim_match.group(1))))
        except ValueError:
            continue

    return pairs


def build_hubs(review_df: pd.DataFrame) -> list[dict[str, Any]]:
    hubs: list[dict[str, Any]] = []

    for _, row in review_df.iterrows():
        candidates = parse_candidates_summary(row["candidates_summary"])
        if not candidates:
            continue

        hubs.append({
            "hub_id": str(row["hub_id"]),
            "hub_label": str(row.get("hub_label", "")),
            "candidates": candidates,
        })

    return hubs


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint_records(path: Path) -> dict[str, list[dict[str, str]]]:
    """Return {hub_id: decisions} from checkpoint JSONL."""
    if not path.exists():
        return {}

    records: dict[str, list[dict[str, str]]] = {}

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                hub = str(rec["hub"])
                decisions = rec.get("decisions", [])
                if isinstance(decisions, list):
                    records[hub] = decisions
            except (json.JSONDecodeError, KeyError, TypeError):
                log.warning("Skipping malformed checkpoint line %d in %s", line_no, path)

    log.info("Checkpoint loaded: %d hubs already done", len(records))
    return records


def append_to_checkpoint(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# Prompt batching
# ---------------------------------------------------------------------------

def build_hubs_block(hubs: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    for index, hub in enumerate(hubs, start=1):
        hub_label = hub.get("hub_label") or ""
        if hub_label:
            lines.append(f"Hub {index}: {hub['hub_id']} [{hub_label}]")
        else:
            lines.append(f"Hub {index}: {hub['hub_id']}")

        for candidate_id, sim in hub["candidates"]:
            lines.append(f"  candidate: {candidate_id}  (cosine={sim:.4f})")

        lines.append("")

    return "\n".join(lines)


def make_dynamic_batches(
    hubs: list[dict[str, Any]],
    *,
    batch_max_hubs: int,
    batch_max_candidates: int,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_candidates = 0

    for hub in hubs:
        n_candidates = len(hub["candidates"])

        if current and (
            len(current) >= batch_max_hubs
            or current_candidates + n_candidates > batch_max_candidates
        ):
            batches.append(current)
            current = []
            current_candidates = 0

        current.append(hub)
        current_candidates += n_candidates

    if current:
        batches.append(current)

    return batches


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def strip_json_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, with a defensive fallback for surrounding chatter."""
    cleaned = strip_json_fences(text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")

    if "results" not in parsed:
        raise ValueError("LLM response is missing 'results' key")

    return parsed


def call_openrouter(
    *,
    api_key: str,
    batch: list[dict[str, Any]],
    model: str,
    url: str,
    temperature: float,
    max_output_tokens: int,
    request_timeout_s: int,
    max_retries: int,
    retry_backoff_s: float,
    http_referer: str | None,
    x_title: str | None,
) -> dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(hubs_block=build_hubs_block(batch))

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }).encode("utf-8")

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
            request = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(request, timeout=request_timeout_s) as response:
                raw = response.read().decode("utf-8")

            payload = json.loads(raw)
            content = payload["choices"][0]["message"]["content"]
            return extract_json_object(content)

        except urllib.error.HTTPError as exc:
            last_error = exc
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "HTTP error attempt %d/%d: %s %s. Sleeping %.1fs",
                attempt + 1,
                max_retries,
                exc,
                error_body,
                wait,
            )
            time.sleep(wait)

        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            last_error = exc
            wait = retry_backoff_s * (2 ** attempt)
            log.warning(
                "LLM error attempt %d/%d: %s. Sleeping %.1fs",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


# ---------------------------------------------------------------------------
# Reconciliation and validation
# ---------------------------------------------------------------------------

def normalize_action(action: Any) -> str:
    text = str(action or "UNKNOWN").strip().upper()
    return text if text in VALID_ACTIONS else "UNKNOWN"


def reconcile_batch_response(
    batch: list[dict[str, Any]],
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    """Match LLM results back to batch entries by hub_id.

    Missing hubs or candidates are filled as UNKNOWN.
    Extra candidates returned by the LLM are ignored.
    """
    results = response.get("results", [])
    if not isinstance(results, list):
        results = []

    by_hub: dict[str, dict[str, Any]] = {}
    for item in results:
        if isinstance(item, dict) and "hub" in item:
            by_hub[str(item["hub"])] = item

    out: list[dict[str, Any]] = []

    for hub in batch:
        hub_id = hub["hub_id"]
        llm_item = by_hub.get(hub_id)

        if not llm_item:
            decisions = [
                {
                    "candidate": candidate_id,
                    "action": "UNKNOWN",
                    "reason": "LLM did not return this hub",
                }
                for candidate_id, _ in hub["candidates"]
            ]
        else:
            raw_decisions = llm_item.get("decisions", [])
            if not isinstance(raw_decisions, list):
                raw_decisions = []

            by_candidate: dict[str, dict[str, Any]] = {}
            for decision in raw_decisions:
                if isinstance(decision, dict) and "candidate" in decision:
                    by_candidate[str(decision["candidate"])] = decision

            decisions = []
            for candidate_id, _ in hub["candidates"]:
                decision = by_candidate.get(candidate_id)
                if not decision:
                    decisions.append({
                        "candidate": candidate_id,
                        "action": "UNKNOWN",
                        "reason": "LLM did not return this candidate",
                    })
                    continue

                decisions.append({
                    "candidate": candidate_id,
                    "action": normalize_action(decision.get("action")),
                    "reason": str(decision.get("reason", ""))[:300],
                })

        out.append({
            "hub": hub_id,
            "decisions": decisions,
        })

    return out


# ---------------------------------------------------------------------------
# Final CSV assembly
# ---------------------------------------------------------------------------

def decisions_to_summary(decisions: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for decision in decisions:
        candidate = decision.get("candidate", "")
        action = normalize_action(decision.get("action"))
        reason = str(decision.get("reason", "")).replace(";", ",")
        parts.append(f"{candidate}={action}({reason})")

    return "; ".join(parts)


def ids_with_action(decisions: list[dict[str, Any]], action: str) -> str:
    action = action.upper()
    return ",".join(
        str(decision.get("candidate"))
        for decision in decisions
        if normalize_action(decision.get("action")) == action
    )


def build_final_dataframe(
    review_df: pd.DataFrame,
    checkpoint_records: dict[str, list[dict[str, Any]]],
    *,
    prefill_accepted: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in review_df.iterrows():
        hub = str(row["hub_id"])
        decisions = checkpoint_records.get(hub, [])

        merge_ids = ids_with_action(decisions, "MERGE")
        keep_ids = ids_with_action(decisions, "KEEP_SEPARATE")
        delete_ids = ids_with_action(decisions, "DELETE")
        unknown_ids = ids_with_action(decisions, "UNKNOWN")

        out_row = dict(row)
        out_row.update({
            "llm_status": "done" if decisions else "missing",
            "llm_suggestion": decisions_to_summary(decisions),
            "llm_merge_ids": merge_ids,
            "llm_keep_separate_ids": keep_ids,
            "llm_delete_ids": delete_ids,
            "llm_unknown_ids": unknown_ids,
            "accepted_ids": merge_ids if prefill_accepted else "",
        })
        rows.append(out_row)

    return pd.DataFrame(rows)


def write_final_outputs(
    *,
    review_df: pd.DataFrame,
    checkpoint_path: Path,
    output_path: Path,
    summary_path: Path,
    prefill_accepted: bool,
    parameters: dict[str, Any],
) -> None:
    records = load_checkpoint_records(checkpoint_path)
    out_df = build_final_dataframe(
        review_df,
        records,
        prefill_accepted=prefill_accepted,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Wrote final review CSV: %s", output_path)

    action_counts = {"MERGE": 0, "KEEP_SEPARATE": 0, "DELETE": 0, "UNKNOWN": 0}
    for decisions in records.values():
        for decision in decisions:
            action_counts[normalize_action(decision.get("action"))] += 1

    summary = {
        "input_hubs": int(len(review_df)),
        "checkpoint_hubs_done": int(len(records)),
        "final_rows": int(len(out_df)),
        "action_counts": action_counts,
        "prefill_accepted": prefill_accepted,
        "parameters": parameters,
        "outputs": {
            "checkpoint_jsonl": str(checkpoint_path),
            "final_csv": str(output_path),
            "summary_json": str(summary_path),
        },
        "notes": [
            "LLM suggestions are not final decisions.",
            "accepted_ids is only prefilled when --prefill-accepted is used.",
            "Review accepted_ids before applying canonicalization merges.",
        ],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("Wrote summary JSON: %s", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input needs-review CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Final LLM-assisted review CSV. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Summary JSON output. Default: {DEFAULT_SUMMARY}",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint directory. Default: {DEFAULT_CHECKPOINT_DIR}",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help=f"Checkpoint filename. Default: {DEFAULT_CHECKPOINT_NAME}",
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
        "--url",
        default=DEFAULT_OPENROUTER_URL,
        help="OpenRouter chat completions endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show batches without API calls. Does not require API key.",
    )
    parser.add_argument(
        "--build-final-only",
        action="store_true",
        help="Build final CSV from existing checkpoint without API calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N pending hubs.",
    )
    parser.add_argument(
        "--batch-max-hubs",
        type=int,
        default=DEFAULT_BATCH_MAX_HUBS,
        help="Maximum hubs per API call.",
    )
    parser.add_argument(
        "--batch-max-candidates",
        type=int,
        default=DEFAULT_BATCH_MAX_CANDIDATES,
        help="Maximum total candidates per API call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries per API call.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_S,
        help="Initial exponential backoff in seconds.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens for model response.",
    )
    parser.add_argument(
        "--http-referer",
        default=None,
        help="Optional HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--x-title",
        default="Relish-AWS",
        help="Optional X-Title header for OpenRouter.",
    )
    parser.add_argument(
        "--prefill-accepted",
        action="store_true",
        help="Copy LLM MERGE suggestions into accepted_ids. Default leaves accepted_ids empty.",
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

    if args.limit is not None and args.limit < 1:
        log.error("--limit must be >= 1")
        return 2
    if args.batch_max_hubs < 1:
        log.error("--batch-max-hubs must be >= 1")
        return 2
    if args.batch_max_candidates < 1:
        log.error("--batch-max-candidates must be >= 1")
        return 2
    if args.max_retries < 1:
        log.error("--max-retries must be >= 1")
        return 2
    if args.request_timeout_s < 1:
        log.error("--request-timeout-s must be >= 1")
        return 2
    if args.max_output_tokens < 1:
        log.error("--max-output-tokens must be >= 1")
        return 2

    try:
        review_df = load_review_hubs(args.input)
        hubs = build_hubs(review_df)

        log.info("Hubs with parseable candidates: %d", len(hubs))

        checkpoint_path = args.checkpoint_dir / args.checkpoint_name
        parameters = {
            "model": args.model,
            "batch_max_hubs": args.batch_max_hubs,
            "batch_max_candidates": args.batch_max_candidates,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "limit": args.limit,
        }

        if args.build_final_only:
            write_final_outputs(
                review_df=review_df,
                checkpoint_path=checkpoint_path,
                output_path=args.output,
                summary_path=args.summary,
                prefill_accepted=args.prefill_accepted,
                parameters=parameters,
            )
            return 0

        done_records = load_checkpoint_records(checkpoint_path)
        pending = [hub for hub in hubs if hub["hub_id"] not in done_records]

        if args.limit is not None:
            pending = pending[:args.limit]

        batches = make_dynamic_batches(
            pending,
            batch_max_hubs=args.batch_max_hubs,
            batch_max_candidates=args.batch_max_candidates,
        )

        if args.dry_run:
            total_candidates = sum(len(hub["candidates"]) for hub in pending)
            log.info(
                "[dry-run] Would process %d hubs (%d candidates) in %d batches",
                len(pending),
                total_candidates,
                len(batches),
            )
            for index, batch in enumerate(batches[:10], start=1):
                log.info(
                    "[dry-run] batch %d: %d hubs, %d candidates",
                    index,
                    len(batch),
                    sum(len(hub["candidates"]) for hub in batch),
                )
            if len(batches) > 10:
                log.info("[dry-run] ... %d more batches", len(batches) - 10)
            return 0

        load_dotenv_file(args.dotenv)
        api_key = get_api_key()

        log.info(
            "Processing %d pending hubs in %d batches",
            len(pending),
            len(batches),
        )

        start_time = time.time()

        for batch_index, batch in enumerate(batches, start=1):
            n_candidates = sum(len(hub["candidates"]) for hub in batch)
            log.info(
                "Batch %d/%d: %d hubs, %d candidates",
                batch_index,
                len(batches),
                len(batch),
                n_candidates,
            )

            response = call_openrouter(
                api_key=api_key,
                batch=batch,
                model=args.model,
                url=args.url,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                request_timeout_s=args.request_timeout_s,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff_s,
                http_referer=args.http_referer,
                x_title=args.x_title,
            )

            records = reconcile_batch_response(batch, response)

            unknowns = sum(
                1
                for record in records
                for decision in record["decisions"]
                if decision["action"] == "UNKNOWN"
            )
            if unknowns:
                log.warning("Batch %d had %d UNKNOWN decisions", batch_index, unknowns)

            append_to_checkpoint(checkpoint_path, records)
            log.info(
                "Flushed %d hubs to checkpoint; elapsed %.1fs",
                len(records),
                time.time() - start_time,
            )

        write_final_outputs(
            review_df=review_df,
            checkpoint_path=checkpoint_path,
            output_path=args.output,
            summary_path=args.summary,
            prefill_accepted=args.prefill_accepted,
            parameters=parameters,
        )

        log.info("Done. Review %s before applying any merges.", args.output)
        return 0

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        TypeError,
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
