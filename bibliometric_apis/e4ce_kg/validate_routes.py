"""Validate the treatment-route extraction of the three models.

Steps:
  1. Read every saved answer (valid and invalid) and map the returned
     materials onto the input materials with fixed rules:
       exact      same name after lower-casing and collapsing whitespace
       split      the input term is the concatenation of several returned
                  names (an input term that merges two materials); the
                  routes of the parts are pooled
       spelling   one-to-one match with difflib ratio >= 0.9
     Returned materials that match no input term are dropped; input terms
     with no match get no answer from that model.
  2. Check that every quote occurs verbatim in the title or abstract after
     Unicode NFKC normalisation, lower-casing and whitespace collapsing.
  3. Apply the decision rule fixed before inspecting outputs: a route is
     kept for a paper-material pair when at least two models assign it with
     a verified quote; otherwise the pair is not_stated.
  4. Report agreement on pairs answered by all three models: Fleiss' kappa
     and pairwise Cohen's kappa per route (binary, verified quotes only),
     exact agreement of route sets, and quote verification per model.
  5. Draw a blind stratified sample (PER_FAMILY pairs per material family)
     for human annotation, plus the key with the model answers.

Usage:
    python validate_routes.py
"""
import csv
import json
import random
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["gemini", "gpt", "qwen"]
ROUTES = ["recycling", "backfilling", "energy_recovery", "incineration", "landfill",
          "other_disposal", "prevention_reuse"]
SEED, PER_FAMILY, MIN_VOTES = 42, 8, 2


def norm(text):
    text = unicodedata.normalize("NFKC", text or "")
    for a, b in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("−", "-")):
        text = text.replace(a, b)
    return " ".join(text.lower().split())


def match_materials(inputs, returned):
    """Map input terms to lists of returned entries; return (mapping, rule per term)."""
    mapping, rules = {}, {}
    pool = {i: r for i, r in enumerate(returned)}
    by_norm = defaultdict(list)
    for i, r in pool.items():
        by_norm[norm(r["material"])].append(i)
    for term in inputs:
        hits = by_norm.get(norm(term))
        if hits:
            mapping[term], rules[term] = [pool[hits[0]]], "exact"
            for i in hits:
                pool.pop(i, None)
    for term in inputs:
        if term in mapping:
            continue
        target = norm(term)
        parts = [i for i, r in pool.items() if norm(r["material"]) in target]
        parts.sort(key=lambda i: target.find(norm(pool[i]["material"])))
        if len(parts) > 1 and " ".join(norm(pool[i]["material"]) for i in parts) == target:
            mapping[term], rules[term] = [pool[i] for i in parts], "split"
            for i in parts:
                pool.pop(i)
    for term in inputs:
        if term in mapping:
            continue
        scored = [(SequenceMatcher(None, norm(term), norm(r["material"])).ratio(), i)
                  for i, r in pool.items()]
        if scored:
            best, i = max(scored)
            if best >= 0.9:
                mapping[term], rules[term] = [pool.pop(i)], "spelling"
    return mapping, rules


def load(model, papers):
    """Return {(paper_id, term): set of verified routes} plus long rows and counters."""
    latest = {}
    for line in (HERE / f"route_raw__{model}.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("content") and (rec["ok"] or rec["paper_id"] not in latest
                                   or not latest[rec["paper_id"]]["ok"]):
            latest[rec["paper_id"]] = rec
    answers, rows, rules_count, quotes = {}, [], Counter(), Counter()
    for pid, rec in latest.items():
        try:
            returned = json.loads(rec["content"])["materials"]
        except (ValueError, KeyError, TypeError):
            rules_count["unparsable"] += 1
            continue
        paper = papers[pid]
        text = norm(f"{paper['title'] or ''} {paper['abstract']}")
        mapping, rules = match_materials(paper["materials"], returned)
        for term in paper["materials"]:
            if term not in mapping:
                rules_count["unmatched"] += 1
                continue
            rules_count[rules[term]] += 1
            verified = set()
            for entry in mapping[term]:
                for r in entry.get("routes", []):
                    if r["route"] not in ROUTES:
                        continue
                    found = bool(norm(r["evidence"])) and norm(r["evidence"]) in text
                    quotes["verified" if found else "not_found"] += 1
                    if found:
                        verified.add(r["route"])
                    rows.append([model, pid, term, rules[term], r["route"], r["evidence"], found])
            answers[(pid, term)] = verified
    return answers, rows, rules_count, quotes


def fleiss_binary(counts, raters=3):
    n = len(counts)
    if n == 0:
        return float("nan")
    p1 = sum(counts) / (n * raters)
    pbar = sum((c * (c - 1) + (raters - c) * (raters - c - 1)) / (raters * (raters - 1))
               for c in counts) / n
    pe = p1 ** 2 + (1 - p1) ** 2
    return float("nan") if pe == 1 else (pbar - pe) / (1 - pe)


def cohen_binary(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main():
    papers = {}
    for line in (HERE / "route_input.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        papers[p["paper_id"]] = p
    with (HERE / "tmo_material_classified.csv").open(newline="", encoding="utf-8") as fh:
        family = {r["term"]: r["category"] for r in csv.DictReader(fh)}

    answers, long_rows = {}, []
    for m in MODELS:
        answers[m], rows, rules_count, quotes = load(m, papers)
        long_rows += rows
        total = sum(quotes.values())
        print(f"{m}: material mapping {dict(rules_count)} | quotes verified "
              f"{quotes['verified']} of {total} ({100 * quotes['verified'] / total:.2f}%)")

    pairs = sorted((pid, term) for pid, p in papers.items() for term in p["materials"])
    common = [k for k in pairs if all(k in answers[m] for m in MODELS)]
    print(f"paper-material pairs: {len(pairs)} | "
          f"answered by all three models: {len(common)}")

    print("route | assigned by gemini gpt qwen | Fleiss kappa | Cohen kappa by pair")
    for route in ROUTES:
        votes = {m: [int(route in answers[m][k]) for k in common] for m in MODELS}
        counts = [sum(votes[m][i] for m in MODELS) for i in range(len(common))]
        cohen = " ".join(f"{a}-{b} {cohen_binary(votes[a], votes[b]) + 0.0:.4f}"
                         for a, b in combinations(MODELS, 2))
        print(f"{route} | {' '.join(str(sum(votes[m])) for m in MODELS)} | "
              f"{fleiss_binary(counts) + 0.0:.4f} | {cohen}")
    exact = sum(len({frozenset(answers[m][k]) for m in MODELS}) == 1 for k in common)
    print(f"identical route sets across the three models: {exact} of {len(common)} "
          f"({100 * exact / len(common):.2f}%)")
    for m in MODELS:
        ns = sum(not answers[m][k] for k in common)
        print(f"{m}: no verified route (not_stated) {ns} of {len(common)} "
              f"({100 * ns / len(common):.2f}%)")

    final = []
    for pid, term in pairs:
        votes = Counter(r for m in MODELS for r in answers[m].get((pid, term), set()))
        kept = sorted(r for r, n in votes.items() if n >= MIN_VOTES)
        final.append([pid, term, family.get(term, ""), ";".join(kept) or "not_stated",
                      sum((pid, term) in answers[m] for m in MODELS),
                      ";".join(f"{r}:{n}" for r, n in sorted(votes.items()))])
    with (HERE / "route_final.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["paper_id", "material", "family", "routes", "models_answering", "votes"])
        w.writerows(final)
    with (HERE / "route_labels_long.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "paper_id", "material", "mapping", "route", "evidence",
                    "evidence_verified"])
        w.writerows(long_rows)
    print(f"final labels: {Counter(r[3] for r in final).most_common(10)}")

    rng = random.Random(SEED)
    by_family = defaultdict(list)
    for k in common:
        by_family[family.get(k[1], "")].append(k)
    sample = []
    for fam in sorted(by_family):
        items = sorted(by_family[fam])
        rng.shuffle(items)
        sample += items[:PER_FAMILY]
    rng.shuffle(sample)
    with (HERE / "audit_route_blind.csv").open("w", newline="", encoding="utf-8") as fb, \
            (HERE / "audit_route_key.csv").open("w", newline="", encoding="utf-8") as fk:
        wb, wk = csv.writer(fb), csv.writer(fk)
        wb.writerow(["item", "paper_id", "title", "abstract", "material",
                     "routes_annotator", "notes"])
        wk.writerow(["item", "paper_id", "material", "family", *MODELS, "final"])
        final_by = {(r[0], r[1]): r[3] for r in final}
        for i, (pid, term) in enumerate(sample, 1):
            p = papers[pid]
            wb.writerow([i, pid, p["title"], p["abstract"], term, "", ""])
            wk.writerow([i, pid, term, family.get(term, ""),
                         *(";".join(sorted(answers[m][(pid, term)])) or "not_stated"
                           for m in MODELS), final_by[(pid, term)]])
    print(f"annotation sample: {len(sample)} pairs from {len(by_family)} families "
          f"-> audit_route_blind.csv / audit_route_key.csv")


if __name__ == "__main__":
    main()
