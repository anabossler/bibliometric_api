"""Adjudicate the paper-material pairs on which the three extraction models disagree.

Secondary analysis. The primary labels follow the rule fixed before the
outputs were inspected (two of three models with a verified quote, see
validate_routes.py). This script sends every pair answered by all three
models whose verified route sets differ to a fourth model of another family.
The adjudicator sees the same definitions and rules as the extraction prompt,
the title and abstract, and the verified routes and quotes of the three
annotators, anonymised as A, B and C in a random order per paper. Pairs on
which the three models agree, and pairs not answered by all three, keep their
primary label.

Outputs:
    route_adjud_raw__<adjudicator>.jsonl   raw answers, resumable
    route_final_adjudicated__<adjudicator>.csv

Usage:
    python adjudicate_routes.py ADJUDICATOR [--limit N] [--workers 4]
    python adjudicate_routes.py ADJUDICATOR --build   # write the CSV only
"""
import argparse
import csv
import json
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from extract_treatment_route import SCHEMA, Fatal, call, check, sha
from validate_routes import MIN_VOTES, MODELS, ROUTES, load, norm

HERE = Path(__file__).resolve().parent
ADJUDICATORS = {
    "deepseek": {"id": "deepseek/deepseek-v3.2",
                 "extra": {"reasoning": {"enabled": False}}},
    "sonnet": {"id": "anthropic/claude-sonnet-5.5",
               "extra": {"temperature": 0, "reasoning": {"enabled": False}}},
}
SEED = 42


def definitions():
    text = (HERE / "prompt_treatment_route.txt").read_text(encoding="utf-8")
    return text.split("Title: {{TITLE}}")[0].rstrip() + "\n"


def collect(papers):
    """Return primary labels, disputed pairs grouped by paper, and verified quotes."""
    answers, quotes = {}, defaultdict(list)
    for m in MODELS:
        answers[m], rows, _, _ = load(m, papers)
        for model, pid, term, _, route, evidence, found in rows:
            if found:
                quotes[(model, pid, term)].append((route, evidence))
    primary, disputed = {}, defaultdict(list)
    for pid, p in papers.items():
        for term in p["materials"]:
            sets = [answers[m].get((pid, term)) for m in MODELS]
            votes = defaultdict(int)
            for s in sets:
                for r in s or ():
                    votes[r] += 1
            primary[(pid, term)] = sorted(r for r, n in votes.items() if n >= MIN_VOTES)
            if all(s is not None for s in sets) and len({frozenset(s) for s in sets}) > 1:
                disputed[pid].append(term)
    return primary, disputed, quotes


def proposals_text(pid, terms, quotes, rng):
    order = MODELS[:]
    rng.shuffle(order)
    lines = []
    for term in terms:
        lines.append(f"- {term}")
        for label, model in zip("ABC", order):
            got = quotes.get((model, pid, term))
            shown = "; ".join(f'{r} ("{e}")' for r, e in got) if got else "not_stated"
            lines.append(f"    annotator {label}: {shown}")
    return "\n".join(lines), order


def build(papers, primary, disputed, adjudicator):
    raw = HERE / f"route_adjud_raw__{adjudicator}.jsonl"
    decided = {}
    if raw.exists():
        for line in raw.open(encoding="utf-8"):
            rec = json.loads(line)
            if rec["ok"]:
                decided[rec["paper_id"]] = json.loads(rec["content"])["materials"]
    out = HERE / f"route_final_adjudicated__{adjudicator}.csv"
    counts = defaultdict(int)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["paper_id", "material", "routes", "source"])
        for pid, p in sorted(papers.items()):
            text = norm(f"{p['title'] or ''} {p['abstract']}")
            given = {norm(m["material"]): m["routes"] for m in decided.get(pid, [])}
            for term in p["materials"]:
                if term in disputed.get(pid, []) and norm(term) in given:
                    routes = sorted({r["route"] for r in given[norm(term)]
                                     if r["route"] in ROUTES and norm(r["evidence"])
                                     and norm(r["evidence"]) in text})
                    source = "adjudicated"
                elif term in disputed.get(pid, []):
                    routes, source = primary[(pid, term)], "disputed_not_adjudicated"
                else:
                    routes, source = primary[(pid, term)], "primary"
                counts[source] += 1
                w.writerow([pid, term, ";".join(routes) or "not_stated", source])
    print(f"wrote {out.name}: {dict(counts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adjudicator", choices=ADJUDICATORS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    papers = {}
    for line in (HERE / "route_input.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        papers[p["paper_id"]] = p
    primary, disputed, quotes = collect(papers)
    n_pairs = sum(len(v) for v in disputed.values())
    if args.build:
        build(papers, primary, disputed, args.adjudicator)
        return

    load_dotenv(Path.home() / "Desktop/openalex/.env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")
    template = definitions() + (HERE / "prompt_adjudicate_route.txt").read_text(
        encoding="utf-8").split("{{DEFINITIONS}}", 1)[1]
    prompt_sha = sha(template.encode())
    cfg = ADJUDICATORS[args.adjudicator]
    out = HERE / f"route_adjud_raw__{args.adjudicator}.jsonl"
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            rec = json.loads(line)
            if rec["prompt_sha"] == prompt_sha and (
                    rec["ok"] or (rec.get("error") or "").startswith(
                        ("materials returned", "invalid routes"))):
                done.add(rec["paper_id"])
    todo = sorted(pid for pid in disputed if pid not in done)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.adjudicator}: {n_pairs} disputed pairs in {len(disputed)} papers | "
          f"{len(done)} papers done | {len(todo)} to run | prompt {prompt_sha}")

    extra = {k: v for k, v in cfg["extra"].items()}
    lock, cost, stats = threading.Lock(), [0.0], defaultdict(int)
    http = requests.Session()

    def work(pid):
        rng = random.Random(f"{SEED}-{pid}")
        terms = sorted(disputed[pid])
        text, order = proposals_text(pid, terms, quotes, rng)
        p = papers[pid]
        prompt = (template.replace("{{TITLE}}", p["title"] or "")
                  .replace("{{ABSTRACT}}", p["abstract"])
                  .replace("{{PROPOSALS}}", text)
                  .replace("{{MATERIALS}}", "; ".join(terms)))
        body = {"model": cfg["id"], "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "routes", "strict": True, "schema": SCHEMA}},
                "usage": {"include": True}, "provider": {"require_parameters": True},
                **extra}
        rec = {"paper_id": pid, "adjudicator": args.adjudicator, "model_requested": cfg["id"],
               "prompt_sha": prompt_sha, "annotator_order": order,
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            data = call(http, key, args.adjudicator, body)
            content = data["choices"][0]["message"]["content"]
            rec.update(model_returned=data.get("model"), provider=data.get("provider"),
                       usage=data.get("usage"), content=content)
            error = check(json.loads(content), {"materials": terms})
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            error = str(exc)[:300]
        rec.update(ok=error is None, error=error)
        return rec

    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, pid) for pid in todo]
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
                cost[0] += (rec.get("usage") or {}).get("cost") or 0.0
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} | {dict(stats)} | cost {cost[0]:.4f} USD", flush=True)
    print(f"{args.adjudicator}: {dict(stats)} | cost {cost[0]:.4f} USD")
    build(papers, primary, disputed, args.adjudicator)


if __name__ == "__main__":
    main()
