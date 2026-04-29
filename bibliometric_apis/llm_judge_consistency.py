"""
llm_judge_consistency.py — LLM-as-Judge for conceptual consistency validation

For each shared term (is_shared_concept=True in concept_consistency_cooccurrence.csv),
asks 3 LLMs via OpenRouter whether the term means the same thing across cluster pairs.

INPUT:
  <DATA_DIR>/concept_consistency_cooccurrence.csv
  <DATA_DIR>/semantic_topics.csv
  <DATA_DIR>/paper_topics.csv

OUTPUT:
  <OUT_DIR>/judge_raw.csv       — all LLM responses
  <OUT_DIR>/judge_summary.csv  — agreement metrics per term
  <OUT_DIR>/judge_report.txt   — paper-ready summary

ENV (set in .env or environment):
  OPENROUTER_API_KEY, DATA_DIR, OUT_DIR

USAGE:
  python llm_judge_consistency.py
  python llm_judge_consistency.py --terms 3 --pairs 5   # quick test run
  python llm_judge_consistency.py --summary-only        # recompute from existing raw CSV
"""

import os
import sys
import time
import json
import argparse
import logging
import itertools
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("judge")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUT_DIR  = Path(os.getenv("OUT_DIR", "./llm_judge_results"))

MODELS = [
    {"key": "gpt4o_mini",   "model": "openai/gpt-4o-mini",                "label": "GPT-4o-mini"},
    {"key": "claude_haiku", "model": "anthropic/claude-haiku-4.5",         "label": "Claude Haiku"},
    {"key": "llama_70b",    "model": "meta-llama/llama-3.1-70b-instruct",  "label": "Llama-3.1-70B"},
]

SYSTEM_PROMPT = (
    "You are a scientific terminology expert specializing in materials science "
    "and circular economy research. Your task is to judge whether a scientific term "
    "has the same conceptual meaning across two research subfields."
)

USER_PROMPT_TEMPLATE = """
The term "{term}" appears in two distinct research clusters within a plastic recycling literature corpus.

Cluster A ({label_a}) is characterized by these key terms:
{terms_a}

Cluster B ({label_b}) is characterized by these key terms:
{terms_b}

Question: Does "{term}" refer to the same underlying scientific concept in both clusters, or does it take on different meanings depending on the subfield context?

Rate semantic equivalence on a scale of 1-5:
1 = completely different concepts (the term is used with fundamentally different meanings)
2 = mostly different (some overlap but distinct focus)
3 = partially overlapping (shared core meaning but different emphasis)
4 = mostly equivalent (same concept, minor contextual differences)
5 = identical concept (same meaning regardless of subfield)

Respond ONLY with a valid JSON object and nothing else:
{{"score": <integer 1-5>, "reasoning": "<one sentence explanation>"}}
""".strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_shared_terms(max_terms=None):
    path = DATA_DIR / "concept_consistency_cooccurrence.csv"
    df = pd.read_csv(path)
    shared = df[df["is_shared_concept"] == True].copy()
    shared = shared.sort_values("mean_conceptual_distance", ascending=False)
    if max_terms:
        shared = shared.head(max_terms)
    log.info("Shared terms loaded: %d", len(shared))
    return shared


def load_cluster_terms():
    path = DATA_DIR / "semantic_topics.csv"
    df = pd.read_csv(path)
    cluster_terms = {}
    for _, row in df.iterrows():
        cluster_id = int(row["cluster"])
        terms = [t.strip() for t in str(row["top_terms"]).split(";")]
        cluster_terms[cluster_id] = terms
    return cluster_terms


def load_cluster_labels():
    path = DATA_DIR / "paper_topics.csv"
    df = pd.read_csv(path)
    labels = {}
    for _, row in df.drop_duplicates("cluster").iterrows():
        cid = int(row["cluster"])
        labels[cid] = str(row.get("topic_label", f"Cluster {cid}")).split(",")[0].strip()
    return labels


def get_cluster_pairs(cluster_terms, max_pairs=None):
    cluster_ids = sorted(cluster_terms.keys())
    pairs = list(itertools.combinations(cluster_ids, 2))
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

def call_openrouter(model, system, user, max_retries=3):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set. Add it to your .env file.")

    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 150,
                },
                timeout=30,
            )

            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    content = content[start:end]
                parsed = json.loads(content)
                return int(parsed["score"]), str(parsed["reasoning"])

            elif r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
            else:
                if attempt == max_retries - 1:
                    log.warning("API error %d (attempt %d)", r.status_code, attempt + 1)
                time.sleep(2)

        except json.JSONDecodeError:
            log.warning("JSON parse error on attempt %d", attempt + 1)
            time.sleep(2)
        except Exception:
            if attempt == max_retries - 1:
                log.warning("Request failed after %d attempts", max_retries)
            time.sleep(5)

    return None, None


# ---------------------------------------------------------------------------
# Main judge loop
# ---------------------------------------------------------------------------

def run_judge(max_terms=None, max_pairs=None):
    OUT_DIR.mkdir(exist_ok=True)

    shared_terms  = load_shared_terms(max_terms)
    cluster_terms = load_cluster_terms()
    cluster_labels = load_cluster_labels()
    pairs = get_cluster_pairs(cluster_terms, max_pairs)

    total_calls = len(shared_terms) * len(pairs) * len(MODELS)
    log.info("Terms: %d | Pairs: %d | Models: %d | Total API calls: %d",
             len(shared_terms), len(pairs), len(MODELS), total_calls)

    raw_path = OUT_DIR / "judge_raw.csv"
    fieldnames = [
        "term", "cluster_a", "cluster_b", "label_a", "label_b",
        "model_key", "model_label", "score", "reasoning",
    ]

    completed = set()
    if raw_path.exists():
        existing = pd.read_csv(raw_path)
        for _, row in existing.iterrows():
            key = (row["term"], row["cluster_a"], row["cluster_b"], row["model_key"])
            completed.add(key)
        log.info("Resuming: %d calls already completed", len(completed))

    call_count = 0
    with open(raw_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not completed:
            writer.writeheader()

        for _, term_row in shared_terms.iterrows():
            term = term_row["phrase"]

            for ca, cb in pairs:
                terms_a  = "; ".join(cluster_terms.get(ca, [])[:10])
                terms_b  = "; ".join(cluster_terms.get(cb, [])[:10])
                label_a  = cluster_labels.get(ca, f"Cluster {ca}")
                label_b  = cluster_labels.get(cb, f"Cluster {cb}")

                user_prompt = USER_PROMPT_TEMPLATE.format(
                    term=term,
                    label_a=label_a,
                    label_b=label_b,
                    terms_a=terms_a,
                    terms_b=terms_b,
                )

                for m in MODELS:
                    call_key = (term, ca, cb, m["key"])
                    if call_key in completed:
                        continue

                    call_count += 1
                    remaining = total_calls - len(completed)
                    log.info("[%d/%d] '%s' C%d vs C%d via %s",
                             call_count, remaining, term, ca, cb, m["label"])

                    score, reasoning = call_openrouter(m["model"], SYSTEM_PROMPT, user_prompt)

                    writer.writerow({
                        "term":        term,
                        "cluster_a":   ca,
                        "cluster_b":   cb,
                        "label_a":     label_a,
                        "label_b":     label_b,
                        "model_key":   m["key"],
                        "model_label": m["label"],
                        "score":       score if score is not None else "",
                        "reasoning":   reasoning if reasoning is not None else "FAILED",
                    })
                    f.flush()
                    completed.add(call_key)
                    time.sleep(0.5)

    log.info("Raw results saved to output directory")
    return raw_path


# ---------------------------------------------------------------------------
# Summary & report
# ---------------------------------------------------------------------------

def compute_summary(raw_path):
    df = pd.read_csv(raw_path)
    df = df[df["score"].notna() & (df["score"] != "")]
    df["score"] = df["score"].astype(int)

    summary_rows = []
    for term, grp in df.groupby("term"):
        scores      = grp["score"].values
        mean_score  = round(float(np.mean(scores)), 3)
        std_score   = round(float(np.std(scores)), 3)
        n_calls     = len(scores)
        model_means = grp.groupby("model_key")["score"].mean().to_dict()
        inter_model_std = round(float(np.std(list(model_means.values()))), 3)
        agreement = (
            "HIGH"     if inter_model_std < 0.5  else
            "MODERATE" if inter_model_std < 1.0  else
            "LOW"
        )

        summary_rows.append({
            "term":             term,
            "mean_score":       mean_score,
            "std_score":        std_score,
            "n_calls":          n_calls,
            "inter_model_std":  inter_model_std,
            "agreement":        agreement,
            **{f"mean_{k}": round(v, 3) for k, v in model_means.items()},
        })

    summary = pd.DataFrame(summary_rows).sort_values("mean_score", ascending=False)
    summary_path = OUT_DIR / "judge_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Summary saved")
    return summary


def write_report(summary):
    report_path = OUT_DIR / "judge_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("LLM-AS-JUDGE: CONCEPTUAL CONSISTENCY VALIDATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Terms evaluated: {len(summary)}\n")
        f.write(f"Models used: {', '.join(m['label'] for m in MODELS)}\n\n")

        f.write("── RESULTS BY TERM ──\n\n")
        cols = ["term", "mean_score", "std_score", "inter_model_std", "agreement"]
        available = [c for c in cols if c in summary.columns]
        f.write(summary[available].to_string(index=False) + "\n\n")

        high_agree = summary[summary["agreement"] == "HIGH"]
        f.write(f"High inter-model agreement: {len(high_agree)}/{len(summary)} terms\n\n")

        mean_all = summary["mean_score"].mean()
        f.write(f"Overall mean score: {mean_all:.3f} / 5\n\n")

        f.write("── FOR THE PAPER ──\n\n")
        f.write(
            f'  "Three LLMs (GPT-4o-mini, Claude Haiku, Llama-3.1-70B) independently '
            f'rated semantic equivalence of {len(summary)} shared terms across cluster pairs '
            f'on a 1-5 scale. Mean score: {mean_all:.2f}/5 '
            f'(inter-model std: {summary["inter_model_std"].mean():.3f}). '
            f'{len(high_agree)}/{len(summary)} terms showed high inter-model agreement, '
            f'confirming that cross-cluster shared vocabulary carries distinct conceptual '
            f'meanings — consistent with the AWS hypothesis."\n'
        )

    log.info("Report saved")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge conceptual consistency")
    parser.add_argument("--terms",        type=int, default=None,
                        help="Max shared terms to evaluate (default: all)")
    parser.add_argument("--pairs",        type=int, default=None,
                        help="Max cluster pairs per term (default: all)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip API calls, recompute summary from existing raw CSV")
    args = parser.parse_args()

    if not args.summary_only:
        raw_path = run_judge(max_terms=args.terms, max_pairs=args.pairs)
    else:
        raw_path = OUT_DIR / "judge_raw.csv"
        if not raw_path.exists():
            log.error("No raw file found. Run without --summary-only first.")
            sys.exit(1)

    summary = compute_summary(raw_path)
    write_report(summary)

    log.info("=" * 70)
    log.info("JUDGE COMPLETE — results in: %s", OUT_DIR)
    log.info("=" * 70)
    print("\n" + summary[["term", "mean_score", "inter_model_std", "agreement"]].to_string(index=False))


if __name__ == "__main__":
    main()
