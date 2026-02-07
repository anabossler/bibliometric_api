#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
temporal_concept_consistency.py

EVOLUCIÓN TEMPORAL de la consistencia conceptual por co-ocurrencias.

Pregunta:
¿Los conceptos centrales (e.g. "plastic recycling") convergen o divergen
semánticamente entre clusters A LO LARGO DEL TIEMPO?

Salida:
- temporal_concept_consistency.csv
- temporal_concept_consistency.png
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

# ========= CONFIG =========
WINDOW_YEARS = 3          # ventana temporal (rolling)
MIN_PAPERS = 80           # mínimo por ventana
N_CLUSTERS = 6
TARGET_PHRASES = [
    "plastic recycling",
    "recycled plastic",
    "chemical recycling",
    "mechanical recycling",
    "circular economy",
    "waste management"
]

# ========= LOG =========
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("temporal_concept_consistency")

# ========= NEO4J =========
class Neo4jClient:
    def __init__(self):
        load_dotenv()
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "neo4j")
            )
        )
        self.db = os.getenv("NEO4J_DATABASE", "neo4j")

    def fetch(self):
        query = """
        MATCH (p:Paper)
        WHERE p.abstract IS NOT NULL
          AND p.abstract <> ''
          AND p.publication_year IS NOT NULL
        RETURN
            p.abstract AS abstract,
            toInteger(p.publication_year) AS year
        """
        with self.driver.session(database=self.db) as s:
            rows = [r.data() for r in s.run(query)]
        return pd.DataFrame(rows)

    def close(self):
        self.driver.close()

# ========= CO-OCCURRENCES =========
def extract_cooccurrences(texts: List[str], phrase: str, window: int = 10):
    words_phrase = phrase.split()
    L = len(words_phrase)
    out = []

    for txt in texts:
        words = txt.lower().split()
        for i in range(len(words) - L + 1):
            if words[i:i+L] == words_phrase:
                w = words[max(0, i-window):i] + words[i+L:i+L+window]
                out.extend([x for x in w if x.isalpha() and len(x) > 2])
    return out

# ========= MAIN ANALYSIS =========
def temporal_consistency(df: pd.DataFrame, outdir: str):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    years = sorted(df.year.unique())
    results = []

    for i in range(len(years) - WINDOW_YEARS + 1):
        y0 = years[i]
        y1 = years[i + WINDOW_YEARS - 1]
        df_w = df[(df.year >= y0) & (df.year <= y1)]

        if len(df_w) < MIN_PAPERS:
            continue

        texts = df_w.abstract.tolist()

        for phrase in TARGET_PHRASES:
            co = extract_cooccurrences(texts, phrase)
            if len(co) < 50:
                continue

            top = [w for w, _ in Counter(co).most_common(50)]
            emb = embedder.encode(top, convert_to_numpy=True)
            centroid = emb.mean(axis=0)

            results.append({
                "year_start": y0,
                "year_end": y1,
                "phrase": phrase,
                "n_papers": len(df_w),
                "semantic_spread": np.std(cosine_similarity(emb, centroid.reshape(1,-1)))
            })

    df_out = pd.DataFrame(results)
    csv_path = os.path.join(outdir, "temporal_concept_consistency.csv")
    df_out.to_csv(csv_path, index=False)
    log.info(f"✓ Saved {csv_path}")

    plot_temporal(df_out, outdir)
    return df_out

# ========= PLOT =========
def plot_temporal(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(10,6))

    for phrase in df.phrase.unique():
        d = df[df.phrase == phrase]
        x = d.year_start + (WINDOW_YEARS / 2)
        plt.plot(x, d.semantic_spread, marker='o', label=phrase)

    plt.xlabel("Year")
    plt.ylabel("Semantic dispersion (↓ = convergence)")
    plt.title("Temporal evolution of conceptual consistency")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    path = os.path.join(outdir, "temporal_concept_consistency.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    log.info(f"✓ Plot saved {path}")

# ========= RUN =========
def main():
    outdir = "./new_results_plastic/full_corpus"
    os.makedirs(outdir, exist_ok=True)

    neo = Neo4jClient()
    df = neo.fetch()
    neo.close()

    if df.empty:
        log.error("No data loaded")
        sys.exit(1)

    temporal_consistency(df, outdir)

if __name__ == "__main__":
    main()
