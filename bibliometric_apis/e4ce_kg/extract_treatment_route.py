"""Extract the end-of-life treatment route of each material a paper targets.

Input: papers of the E4CE-KG corpus (supramolecular corpus excluded) with an
abstract and at least one material in a MaterialFamily linked to Eurostat
WasteFlow nodes. The input is frozen once to route_input.jsonl so that every
model sees exactly the same papers and materials.

For each paper the model returns, per material, one or more routes from a
closed set aligned with the Eurostat env_wastrt operations, plus
prevention_reuse and not_stated, each with a verbatim quote from the abstract.
Raw responses are appended to route_raw__<model>.jsonl. The run is resumable:
papers already answered by that model are skipped.

Final labels, fixed before any output is inspected: a route is assigned to a
paper-material pair when at least two of the three models assign it with a
quote found verbatim in the title or abstract; otherwise the pair is
not_stated.

Usage:
    python extract_treatment_route.py MODEL [--limit N] [--workers 8] [--skip-invalid]

MODEL is a key of MODELS. Requires OPENROUTER_API_KEY, and NEO4J_PW the first
time, to freeze the input.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
PROMPT = HERE / "prompt_treatment_route.txt"
INPUT = HERE / "route_input.jsonl"
URL = "https://openrouter.ai/api/v1/chat/completions"
ROUTES = ["recycling", "backfilling", "energy_recovery", "incineration", "landfill",
          "other_disposal", "prevention_reuse", "not_stated"]
QWEN_PROVIDER = os.environ.get("QWEN_PROVIDER", "deepinfra")
MODELS = {
    "gemini": {"id": "google/gemini-2.5-flash-lite", "extra": {}},
    # GPT-5.4 endpoints reject temperature and seed; the model runs at its default.
    "gpt": {"id": "openai/gpt-5.4-nano-20260317", "extra": {"reasoning": {"effort": "none"}},
            "drop": ["temperature", "seed"]},
    "qwen": {"id": "qwen/qwen3-235b-a22b-2507",
             "extra": {"provider": {"order": [QWEN_PROVIDER], "allow_fallbacks": False}}},
}
SCHEMA = {
    "type": "object",
    "properties": {"materials": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "material": {"type": "string"},
            "routes": {"type": "array", "items": {
                "type": "object",
                "properties": {"route": {"type": "string", "enum": ROUTES},
                               "evidence": {"type": "string"}},
                "required": ["route", "evidence"], "additionalProperties": False}},
        },
        "required": ["material", "routes"], "additionalProperties": False}}},
    "required": ["materials"], "additionalProperties": False,
}
INPUT_QUERY = """
MATCH (p:Paper)-[:TARGETS_MATERIAL]->(m:Material)-[:IN_FAMILY]->(f:MaterialFamily)
WHERE EXISTS { (f)<-[:CORRESPONDS_TO_FAMILY]-(:WasteFlow) }
WITH p, collect(DISTINCT m.name) AS materials,
     toLower(coalesce(p.title, '') + ' ' + coalesce(p.abstract, '')) AS t
WHERE p.abstract IS NOT NULL AND p.abstract <> ''
  AND (NOT (t CONTAINS 'supramolecular'
            OR EXISTS { (p)-[:HAS_CONCEPT]->(c:Concept)
                        WHERE toLower(c.display_name) CONTAINS 'supramolecular' })
       OR t CONTAINS 'circular econom')
RETURN coalesce(p.openalex_id, p.doi) AS paper_id, p.title AS title,
       p.abstract AS abstract, materials
ORDER BY paper_id
"""
FATAL = {400, 401, 402, 403, 404}


class Fatal(Exception):
    """Configuration error (bad model, key, credits): stop the whole run."""


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def freeze_input():
    """Write the input snapshot once; later runs and other models reuse it."""
    if INPUT.exists():
        return
    from neo4j import GraphDatabase
    pw = os.environ.get("NEO4J_PW")
    if not pw:
        sys.exit("route_input.jsonl does not exist yet: set NEO4J_PW to create it")
    with GraphDatabase.driver("bolt://localhost:7691", auth=("neo4j", pw)) as drv, \
            drv.session(database="circulareconomy") as s:
        rows = [dict(r) for r in s.run(INPUT_QUERY)]
    ids = [r["paper_id"] for r in rows]
    if None in ids or len(ids) != len(set(ids)):
        sys.exit("paper_id is missing or duplicated in the input query")
    tmp = INPUT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            r["materials"] = sorted(r["materials"])
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(INPUT)
    print(f"froze {len(rows)} papers to {INPUT.name}")


def build_prompt(template, paper):
    return (template.replace("{{TITLE}}", paper["title"] or "")
            .replace("{{ABSTRACT}}", paper["abstract"])
            .replace("{{MATERIALS}}", "; ".join(paper["materials"])))


def norm(text):
    return " ".join(text.lower().split())


def check(parsed, paper):
    names = [norm(m["material"]) for m in parsed["materials"]]
    if sorted(names) != sorted(norm(m) for m in paper["materials"]):
        return f"materials returned {sorted(names)} differ from input"
    for m in parsed["materials"]:
        if not m["routes"] or any(r["route"] not in ROUTES for r in m["routes"]):
            return f"invalid routes for {m['material']}"
    return None


def call(session, key, model, body, retries=5):
    for attempt in range(retries):
        try:
            resp = session.post(URL, headers={"Authorization": f"Bearer {key}"},
                                json=body, timeout=180)
        except requests.RequestException as exc:
            err = str(exc)
        else:
            if resp.ok:
                data = resp.json()
                if "choices" in data:
                    return data
                err = json.dumps(data)[:300]
            elif resp.status_code in FATAL:
                raise Fatal(f"{model}: HTTP {resp.status_code} {resp.text[:1500]}")
            else:
                err = f"HTTP {resp.status_code}"
        time.sleep(2 ** attempt)
    raise RuntimeError(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-invalid", action="store_true",
                    help="do not resend papers whose saved answer failed the material or "
                         "route check; the saved answer is kept for validation")
    args = ap.parse_args()

    load_dotenv(Path.home() / "Desktop/openalex/.env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")
    freeze_input()

    template = PROMPT.read_text(encoding="utf-8")
    prompt_sha, input_sha = sha(template.encode()), sha(INPUT.read_bytes())
    papers = [json.loads(line) for line in INPUT.open(encoding="utf-8")]
    cfg = MODELS[args.model]
    out = HERE / f"route_raw__{args.model}.jsonl"
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            rec = json.loads(line)
            if rec["prompt_sha"] != prompt_sha:
                continue
            invalid = (rec.get("error") or "").startswith(("materials returned",
                                                           "invalid routes"))
            if rec["ok"] or (args.skip_invalid and invalid):
                done.add(rec["paper_id"])
    todo = [p for p in papers if p["paper_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.model}: {len(papers)} papers, {len(done)} done, {len(todo)} to run "
          f"| prompt {prompt_sha} | input {input_sha}")

    provider = {"require_parameters": True, **cfg["extra"].get("provider", {})}
    extra = {k: v for k, v in cfg["extra"].items() if k != "provider"}
    lock, stats = threading.Lock(), Counter()
    providers, cost = Counter(), 0.0
    http = requests.Session()

    def work(paper):
        body = {"model": cfg["id"], "temperature": 0, "seed": 42,
                "messages": [{"role": "user", "content": build_prompt(template, paper)}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "routes", "strict": True, "schema": SCHEMA}},
                "usage": {"include": True}, "provider": provider, **extra}
        for k in cfg.get("drop", []):
            body.pop(k, None)
        rec = {"paper_id": paper["paper_id"], "model_key": args.model,
               "model_requested": cfg["id"], "prompt_sha": prompt_sha,
               "input_sha": input_sha,
               "params": {k: body[k] for k in ("temperature", "seed", "reasoning") if k in body},
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            data = call(http, key, args.model, body)
            content = data["choices"][0]["message"]["content"]
            rec.update(model_returned=data.get("model"), provider=data.get("provider"),
                       usage=data.get("usage"), content=content)
            error = check(json.loads(content), paper)
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            error = str(exc)[:300]
        rec.update(ok=error is None, error=error)
        return rec

    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, p) for p in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Fatal as exc:
                for f in futures:
                    f.cancel()
                sys.exit(str(exc))
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                stats["ok" if rec["ok"] else "failed"] += 1
                providers[rec.get("provider")] += 1
                cost += (rec.get("usage") or {}).get("cost") or 0.0
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} | {dict(stats)} | cost {cost:.4f} USD", flush=True)

    print(f"{args.model}: {dict(stats)} | providers {dict(providers)} | cost {cost:.4f} USD")
    if stats["failed"]:
        print("Run the same command again to retry the failed papers.")


if __name__ == "__main__":
    main()
