"""

===================

Multi-judge evaluation of cross-subdomain TMO bridge candidates.

Pipeline
--------
1. Load TMO nodes (Technique, Material, Objective) per cluster from Neo4j.
2. For each AWS-isolated cluster (C10, C11, C14), generate candidate bridge
   pairs with each non-isolated cluster (C2-C9), per category.
3. Pre-filter pairs by SBERT cosine similarity to keep the top-N most
   semantically plausible candidates per (cluster_a, cluster_b, category).
4. Submit each candidate to three LLM judges concurrently:
       openai/gpt-4o-mini
       anthropic/claude-haiku-4.5
       meta-llama/llama-3.1-70b-instruct
   Each judge scores 1-5 on:
       Semantic Equivalence (SE)
       Cross-subdomain Bridge Validity (BV)
       Ontological Groundedness (OG)
5. Aggregate: a pair is a CONSENSUS BRIDGE if at least 2 of 3 judges
   assign score >= 3 on all three dimensions.

Output
------
    runs/judge/candidates.json   all generated candidates with SBERT scores
    runs/judge/judgements.json   parsed per-judge scores for every candidate
    runs/judge/consensus.json    consensus bridges (2/3 agreement, score>=3)
    runs/judge/summary.csv       aggregate consensus rate per category/pair

Resumable: existing judgement files are reused on rerun.

Requirements
------------
    pip install neo4j openai sentence-transformers numpy tenacity

Environment
-----------
    OPENROUTER_API_KEY     must be set
    NEO4J_URI              default bolt://localhost:7687
    NEO4J_USER             default neo4j
    NEO4J_PASSWORD         must be set
    NEO4J_DATABASE         default circulareconomy

Usage
-----
    python tmo_bridge_judge.py --top-k 10 --prefilter-n 20 \
        --output-dir runs/judge
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "circulareconomy")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ISOLATED_CLUSTERS = ["C10", "C11", "C14"]
NON_ISOLATED_CLUSTERS = ["C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

CLUSTER_LABELS = {
    "C2": "Energy, LCA & Decarbonization",
    "C3": "CE Governance & Urban Metabolism",
    "C4": "CE Business Models & Green Finance",
    "C5": "CE Policy & Sustainability Transitions",
    "C6": "CE Frameworks & Digital Passports",
    "C7": "Fashion, Textiles & Consumer Circularity",
    "C8": "Industry 4.0 & Circular Manufacturing",
    "C9": "Construction Waste & Built Environment",
    "C10": "Advanced Plastics & Biocomposites",
    "C11": "Recycled Construction Materials & Geotechnical",
    "C14": "Industrial Waste & Material Recovery",
}

CATEGORIES = {
    "Technique": "USES_TECHNIQUE",
    "Material": "TARGETS_MATERIAL",
    "Objective": "PURSUES_OBJECTIVE",
}

JUDGES: list[str] = [
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4.5",
    "meta-llama/llama-3.1-70b-instruct",
]

CONSENSUS_MIN_JUDGES = 2  # at least 2 of 3 must agree
CONSENSUS_MIN_SCORE = 3  # on all three dimensions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("judge")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A candidate cross-subdomain TMO bridge pair."""
    cluster_a: str
    cluster_b: str
    category: str
    term_a: str
    term_b: str
    sbert_cosine: float


@dataclass
class Judgement:
    """One judge's evaluation of one candidate."""
    judge: str
    semantic_equivalence: int
    bridge_validity: int
    ontological_groundedness: int
    parse_ok: bool = True


# ---------------------------------------------------------------------------
# Neo4j: load top-K TMO nodes per cluster
# ---------------------------------------------------------------------------

def load_tmo_terms(top_k: int) -> dict[str, dict[str, list[str]]]:
    """
    Return {cluster_id: {category: [term, ...]}}.

    Ranks by paper-degree within the cluster, pre-2021 papers only.
    """
    if not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD environment variable is not set.")

    out: dict[str, dict[str, list[str]]] = {}
    all_clusters = ISOLATED_CLUSTERS + NON_ISOLATED_CLUSTERS

    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    ) as driver:
        driver.verify_connectivity()
        with driver.session(database=NEO4J_DATABASE) as session:
            for cid in all_clusters:
                out[cid] = {}
                for category, relationship in CATEGORIES.items():
                    query = f"""
                    MATCH (p:Paper)
                    WHERE p.cluster_id = $cid
                      AND (p.year IS NULL OR p.year <= 2020)
                    MATCH (p)-[:{relationship}]->(n:{category})
                    RETURN n.name AS name, count(DISTINCT p) AS degree
                    ORDER BY degree DESC
                    LIMIT $top_k
                    """
                    result = session.run(query, cid=cid, top_k=top_k)
                    out[cid][category] = [record["name"] for record in result]
                log.info(
                    "Cluster %s | T=%d M=%d O=%d",
                    cid,
                    len(out[cid]["Technique"]),
                    len(out[cid]["Material"]),
                    len(out[cid]["Objective"]),
                )

    return out


# ---------------------------------------------------------------------------
# Candidate generation + SBERT prefilter
# ---------------------------------------------------------------------------

def generate_candidates(
    tmo: dict[str, dict[str, list[str]]],
    prefilter_n: int,
    embedder: SentenceTransformer,
) -> list[Candidate]:
    """
    For each (isolated_cluster, non_isolated_cluster, category) triple,
    compute pairwise SBERT cosine similarities between top-K terms and
    keep the top-N most similar candidate pairs.

    This prefilter reduces the candidate set without prejudging semantic
    equivalence: the judges then make the final call. Pairs with low
    cosine similarity are unlikely to be true bridges and would only
    waste judge calls.
    """
    candidates: list[Candidate] = []

    for cid_a, cid_b in product(ISOLATED_CLUSTERS, NON_ISOLATED_CLUSTERS):
        for cat in CATEGORIES:
            terms_a = tmo[cid_a][cat]
            terms_b = tmo[cid_b][cat]
            if not terms_a or not terms_b:
                continue

            emb_a = embedder.encode(terms_a, normalize_embeddings=True)
            emb_b = embedder.encode(terms_b, normalize_embeddings=True)
            sims = emb_a @ emb_b.T  # cosine because embeddings are normalized

            # Flatten and keep top-N
            pairs: list[tuple[int, int, float]] = [
                (i, j, float(sims[i, j]))
                for i in range(len(terms_a))
                for j in range(len(terms_b))
            ]
            pairs.sort(key=lambda x: x[2], reverse=True)
            for i, j, sim in pairs[:prefilter_n]:
                candidates.append(
                    Candidate(
                        cluster_a=cid_a,
                        cluster_b=cid_b,
                        category=cat,
                        term_a=terms_a[i],
                        term_b=terms_b[j],
                        sbert_cosine=sim,
                    )
                )

    log.info(
        "Generated %d candidate pairs after SBERT prefilter (top-%d per group)",
        len(candidates),
        prefilter_n,
    )
    return candidates


# ---------------------------------------------------------------------------
# LLM judge (lazy init)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set.")
        _client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _client


JUDGE_SYSTEM = """You are an expert evaluator of cross-subdomain vocabulary alignments in scholarly knowledge graphs.

A bridge candidate is a pair of terms drawn from two different research subdomains in a circular economy corpus. Your task is to assess whether the pair represents a genuine cross-subdomain semantic alignment.

Rate each pair on three independent dimensions (integer 1-5):

1. Semantic Equivalence (SE)
   1 = terms refer to entirely different phenomena
   3 = terms partially overlap but are not equivalent
   5 = terms refer to the same underlying phenomenon from different disciplinary angles

2. Cross-subdomain Bridge Validity (BV)
   1 = generic shared vocabulary, not a meaningful bridge
   3 = plausible but weak link
   5 = pair reveals a substantive cross-subdomain dependency between the two research communities

3. Ontological Groundedness (OG)
   1 = at least one term is hallucinated or implausible for its subdomain
   3 = both terms plausible but not strongly characteristic of their subdomains
   5 = both terms are well-attested technical vocabulary of their respective subdomains

Return ONLY this XML structure with integer values 1-5. No prose, no markdown.

<evaluation>
  <semantic_equivalence>N</semantic_equivalence>
  <bridge_validity>N</bridge_validity>
  <ontological_groundedness>N</ontological_groundedness>
</evaluation>
"""

JUDGE_USER_TEMPLATE = """Subdomain A: {cluster_a} ({label_a})
Subdomain B: {cluster_b} ({label_b})
Category: {category}

Candidate bridge: "{term_a}" (in A)  <->  "{term_b}" (in B)

Provide your evaluation now."""


SCORE_PATTERNS = {
    "semantic_equivalence": re.compile(
        r"<semantic_equivalence>\s*(\d)\s*</semantic_equivalence>", re.I
    ),
    "bridge_validity": re.compile(
        r"<bridge_validity>\s*(\d)\s*</bridge_validity>", re.I
    ),
    "ontological_groundedness": re.compile(
        r"<ontological_groundedness>\s*(\d)\s*</ontological_groundedness>", re.I
    ),
}


def parse_judge_response(raw: str) -> Judgement | None:
    """Extract three 1-5 scores from XML. Returns None on failure."""
    scores: dict[str, int] = {}
    for key, pattern in SCORE_PATTERNS.items():
        match = pattern.search(raw)
        if not match:
            return None
        try:
            value = int(match.group(1))
            if not 1 <= value <= 5:
                return None
            scores[key] = value
        except ValueError:
            return None
    return Judgement(judge="", **scores)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
def call_judge(judge_model: str, user_prompt: str) -> str:
    response = get_client().chat.completions.create(
        model=judge_model,
        temperature=0.0,
        max_tokens=200,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        extra_headers={"X-Title": "AWS-TMO-Bridge-Judge"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"Judge {judge_model} returned an empty response.")
    return content


def failed_judgement(judge_model: str) -> Judgement:
    """Return a sentinel judgement for a failed API call or parse."""
    return Judgement(
        judge=judge_model,
        semantic_equivalence=0,
        bridge_validity=0,
        ontological_groundedness=0,
        parse_ok=False,
    )


def evaluate_candidate(candidate: Candidate) -> list[Judgement]:
    user_prompt = JUDGE_USER_TEMPLATE.format(
        cluster_a=candidate.cluster_a,
        cluster_b=candidate.cluster_b,
        label_a=CLUSTER_LABELS[candidate.cluster_a],
        label_b=CLUSTER_LABELS[candidate.cluster_b],
        category=candidate.category,
        term_a=candidate.term_a,
        term_b=candidate.term_b,
    )

    # Initialize before starting worker threads to avoid a client-creation race.
    get_client()
    results: dict[str, Judgement] = {}

    with ThreadPoolExecutor(max_workers=len(JUDGES)) as executor:
        futures = {
            executor.submit(call_judge, judge_model, user_prompt): judge_model
            for judge_model in JUDGES
        }
        for future in as_completed(futures):
            judge_model = futures[future]
            try:
                raw = future.result()
            except Exception as exc:
                log.warning("Judge call failed | %s | %s", judge_model, exc)
                results[judge_model] = failed_judgement(judge_model)
                continue

            parsed = parse_judge_response(raw)
            if parsed is None:
                log.warning(
                    "Parse failed | %s | %.120s",
                    judge_model,
                    raw.replace("\n", " "),
                )
                results[judge_model] = failed_judgement(judge_model)
                continue

            parsed.judge = judge_model
            results[judge_model] = parsed

    # Preserve the configured judge order in serialized output.
    return [results[judge_model] for judge_model in JUDGES]


# ---------------------------------------------------------------------------
# Consensus aggregation
# ---------------------------------------------------------------------------

def is_consensus_bridge(judgements: list[Judgement]) -> bool:
    valid = [j for j in judgements if j.parse_ok]
    if len(valid) < CONSENSUS_MIN_JUDGES:
        return False
    agree = sum(
        1
        for j in valid
        if j.semantic_equivalence >= CONSENSUS_MIN_SCORE
        and j.bridge_validity >= CONSENSUS_MIN_SCORE
        and j.ontological_groundedness >= CONSENSUS_MIN_SCORE
    )
    return agree >= CONSENSUS_MIN_JUDGES


# ---------------------------------------------------------------------------
# File helpers and CLI
# ---------------------------------------------------------------------------


def candidate_key(candidate: Candidate) -> str:
    """Build the stable cache key used for one candidate pair."""
    return "|".join(
        (
            candidate.cluster_a,
            candidate.cluster_b,
            candidate.category,
            candidate.term_a,
            candidate.term_b,
        )
    )


def write_json(path: Path, payload: Any) -> None:
    """Atomically write UTF-8 JSON so an interrupted run cannot corrupt it."""
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def positive_int(value: str) -> int:
    """Argparse type for strictly positive integer options."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=10,
        help="Top-K TMO nodes per category per cluster (default: 10).",
    )
    parser.add_argument(
        "--prefilter-n",
        type=positive_int,
        default=20,
        help=(
            "Top-N pairs per cluster pair and category after the SBERT "
            "prefilter (default: 20)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/judge"),
        help="Output directory (default: runs/judge).",
    )
    parser.add_argument(
        "--refresh-candidates",
        action="store_true",
        help="Regenerate candidates instead of reusing candidates.json.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cand_file = args.output_dir / "candidates.json"
    judgements_file = args.output_dir / "judgements.json"
    consensus_file = args.output_dir / "consensus.json"
    summary_file = args.output_dir / "summary.csv"

    # 1. Candidates (cache)
    if cand_file.exists() and not args.refresh_candidates:
        log.info("Loading cached candidates: %s", cand_file)
        candidate_data = json.loads(cand_file.read_text(encoding="utf-8"))
        candidates = [Candidate(**item) for item in candidate_data]
    else:
        log.info("Loading TMO terms from Neo4j...")
        tmo = load_tmo_terms(args.top_k)
        log.info("Loading SBERT model: %s", EMBEDDING_MODEL)
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        candidates = generate_candidates(tmo, args.prefilter_n, embedder)
        write_json(cand_file, [asdict(candidate) for candidate in candidates])
        log.info("Candidates cached: %s", cand_file)

    # 2. Judge each candidate (resumable)
    existing_judgements: dict[str, list[Judgement]] = {}
    if judgements_file.exists():
        raw = json.loads(judgements_file.read_text(encoding="utf-8"))
        existing_judgements = {
            key: [Judgement(**item) for item in items]
            for key, items in raw.items()
        }
        log.info("Resuming with %d cached judgements", len(existing_judgements))

    candidates_since_save = 0
    for idx, candidate in enumerate(candidates, 1):
        key = candidate_key(candidate)
        if key in existing_judgements:
            continue

        log.info(
            "[%d/%d] %s/%s %s: '%s' <-> '%s'",
            idx,
            len(candidates),
            candidate.cluster_a,
            candidate.cluster_b,
            candidate.category,
            candidate.term_a,
            candidate.term_b,
        )
        judgements = evaluate_candidate(candidate)
        existing_judgements[key] = judgements
        candidates_since_save += 1

        # Persist after every 10 candidates so a crash doesn't lose progress
        if candidates_since_save >= 10:
            write_json(
                judgements_file,
                {
                    cache_key: [asdict(judgement) for judgement in values]
                    for cache_key, values in existing_judgements.items()
                },
            )
            candidates_since_save = 0

    write_json(
        judgements_file,
        {
            cache_key: [asdict(judgement) for judgement in values]
            for cache_key, values in existing_judgements.items()
        },
    )
    log.info("All judgements saved: %s", judgements_file)

    # 3. Consensus
    consensus: list[dict[str, Any]] = []
    for candidate in candidates:
        key = candidate_key(candidate)
        judgements = existing_judgements.get(key, [])
        if is_consensus_bridge(judgements):
            consensus.append({
                **asdict(candidate),
                "judgements": [asdict(j) for j in judgements],
            })

    write_json(consensus_file, consensus)
    log.info(
        "Consensus bridges: %d / %d candidates (%.1f%%)",
        len(consensus),
        len(candidates),
        100.0 * len(consensus) / max(len(candidates), 1),
    )

    # 4. Per-category, per-cluster-pair summary
    totals: dict[tuple[str, str, str], int] = defaultdict(int)
    accepts: dict[tuple[str, str, str], int] = defaultdict(int)
    for candidate in candidates:
        triple = (candidate.cluster_a, candidate.cluster_b, candidate.category)
        totals[triple] += 1
        key = candidate_key(candidate)
        if is_consensus_bridge(existing_judgements.get(key, [])):
            accepts[triple] += 1

    with summary_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "cluster_a",
                "cluster_b",
                "category",
                "n_candidates",
                "n_consensus",
                "consensus_rate",
            ]
        )
        for (ca, cb, cat), n in sorted(totals.items()):
            k = accepts[(ca, cb, cat)]
            writer.writerow([ca, cb, cat, n, k, f"{k / n:.3f}"])
    log.info("Per-pair summary: %s", summary_file)

    log.info("Done.")


if __name__ == "__main__":
    main()
