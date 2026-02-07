"""
temporal_concept_consistency_nulls_FIXED.py

NULL MODEL CORRECTO:
- Observed: Co-ocurrencias con "plastic recycling"
- NULL: Palabras RANDOM del vocabulario (no co-ocurrentes)

Si observed < NULL → Genuino consenso conceptual
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
import logging

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("temporal_nulls_fixed")

# ============================================================================
# CONFIG
# ============================================================================
OUTPUT_DIR = "./new_results_plastic/full_corpus"

N_NULLS = 200
CONTEXT_WINDOW = 10
TOP_N_WORDS = 50

TARGET_PHRASES = [
    "plastic recycling",
    "recycled plastic",
    "chemical recycling",
    "mechanical recycling",
    "circular economy",
    "waste management"
]

# ============================================================================
# NEO4J
# ============================================================================
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

    def fetch_abstracts(self):
        query = """
        MATCH (p:Paper)
        WHERE p.abstract IS NOT NULL
          AND p.abstract <> ''
        RETURN p.abstract AS abstract
        LIMIT 10000
        """
        with self.driver.session(database=self.db) as s:
            rows = [r.data() for r in s.run(query)]
        return pd.DataFrame(rows)

    def close(self):
        self.driver.close()


# ============================================================================
# BUILD VOCABULARY
# ============================================================================
def build_vocabulary(texts: List[str], min_freq: int = 50) -> List[str]:
    """Construye vocabulario de palabras frecuentes del corpus"""
    log.info("Building vocabulary...")
    
    word_counts = Counter()
    
    for txt in texts:
        words = [w.lower() for w in txt.split() if w.isalpha() and len(w) > 3]
        word_counts.update(words)
    
    # Palabras con frecuencia >= min_freq
    vocab = [w for w, count in word_counts.items() if count >= min_freq]
    
    log.info(f"✓ Vocabulary: {len(vocab):,} words (freq >= {min_freq})")
    
    return vocab


# ============================================================================
# CO-OCCURRENCES
# ============================================================================
def extract_cooccurrences(texts: List[str], phrase: str, window: int = 10):
    """Extrae palabras co-ocurrentes"""
    words_phrase = phrase.lower().split()
    L = len(words_phrase)
    out = []

    for txt in texts:
        words = txt.lower().split()
        for i in range(len(words) - L + 1):
            if words[i:i+L] == words_phrase:
                context = words[max(0, i-window):i] + words[i+L:i+L+window]
                valid = [w for w in context if w.isalpha() and len(w) > 3]
                out.extend(valid)
    
    return out


# ============================================================================
# OBSERVED
# ============================================================================
def compute_observed_spread(texts: List[str], 
                           phrase: str, 
                           embedder: SentenceTransformer,
                           top_n: int = 50) -> tuple:
    """
    Calcula semantic_spread observado.
    
    Returns:
        (spread, top_words)
    """
    # Co-ocurrencias
    co = extract_cooccurrences(texts, phrase, window=CONTEXT_WINDOW)
    
    if len(co) < 100:
        return None, None
    
    # Top-N palabras
    top_words = [w for w, _ in Counter(co).most_common(top_n)]
    
    # Embeddings
    emb = embedder.encode(top_words, convert_to_numpy=True, show_progress_bar=False)
    
    # Centroid
    centroid = emb.mean(axis=0)
    
    # Spread = std de similitudes
    sims = cosine_similarity(emb, centroid.reshape(1, -1)).ravel()
    spread = float(np.std(sims))
    
    return spread, top_words


# ============================================================================
# NULL MODEL (RANDOM WORDS)
# ============================================================================
def compute_null_spread(vocab: List[str],
                       embedder: SentenceTransformer,
                       n_words: int = 50,
                       seed: int = 42) -> float:
    """
    NULL model: Palabras RANDOM del vocabulario (no co-ocurrentes).
    
    Esto representa lo que esperaríamos si NO hubiera asociación
    semántica específica.
    """
    rng = np.random.default_rng(seed)
    
    # Sample random words
    random_words = rng.choice(vocab, size=min(n_words, len(vocab)), replace=False)
    
    # Embed
    emb = embedder.encode(random_words.tolist(), convert_to_numpy=True, show_progress_bar=False)
    
    # Centroid
    centroid = emb.mean(axis=0)
    
    # Spread
    sims = cosine_similarity(emb, centroid.reshape(1, -1)).ravel()
    spread = float(np.std(sims))
    
    return spread


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def analyze_phrase(texts: List[str],
                  vocab: List[str],
                  phrase: str, 
                  embedder: SentenceTransformer,
                  n_nulls: int = 200) -> Dict:
    """Análisis completo para un phrase"""
    
    log.info(f"\n→ Analyzing: '{phrase}'")
    
    # 1. Observed
    obs_spread, top_words = compute_observed_spread(
        texts, phrase, embedder, top_n=TOP_N_WORDS
    )
    
    if obs_spread is None:
        log.warning(f"  Insufficient co-occurrences")
        return None
    
    log.info(f"  Observed spread: {obs_spread:.4f}")
    log.info(f"  Co-occurrence words: {len(top_words)}")
    
    # 2. NULL distribution (random words)
    log.info(f"  Generating {n_nulls} NULL models (random vocab)...")
    
    null_spreads = []
    
    for i in tqdm(range(n_nulls), desc=f"  NULL {phrase[:20]}", ncols=70):
        null_spread = compute_null_spread(vocab, embedder, n_words=len(top_words), seed=42+i)
        null_spreads.append(null_spread)
    
    null_spreads = np.array(null_spreads)
    
    # 3. Statistics
    mean_null = float(np.mean(null_spreads))
    std_null = float(np.std(null_spreads))
    
    z_score = (obs_spread - mean_null) / std_null if std_null > 0 else np.nan
    p_value = float((null_spreads <= obs_spread).sum() / len(null_spreads))
    
    ci_lower = float(np.percentile(null_spreads, 2.5))
    ci_upper = float(np.percentile(null_spreads, 97.5))
    
    log.info(f"  NULL: {mean_null:.4f} ± {std_null:.4f}")
    log.info(f"  Z-score: {z_score:.2f}, p={p_value:.4f}")
    
    interpretation = "✅ CONSENSUS" if obs_spread < ci_lower else ("⚠️ DIVERGENCE" if obs_spread > ci_upper else "➖ NO DIFF")
    log.info(f"  {interpretation}")
    
    return {
        'phrase': phrase,
        'observed_spread': obs_spread,
        'null_mean': mean_null,
        'null_std': std_null,
        'z_score': z_score,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_nulls': len(null_spreads),
        'null_spreads': null_spreads,
        'n_cooccurrence_words': len(top_words)
    }


# ============================================================================
# PLOT
# ============================================================================
def plot_distributions(results: List[Dict], outdir: str):
    log.info("\nGenerating plots...")
    
    n_phrases = len(results)
    ncols = min(3, n_phrases)
    nrows = (n_phrases + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        phrase = result['phrase']
        obs = result['observed_spread']
        nulls = result['null_spreads']
        
        ax.hist(nulls, bins=30, alpha=0.7, color='steelblue', edgecolor='black',
                label='NULL (random vocab)')
        ax.axvline(obs, color='red', linewidth=2.5, linestyle='--', 
                  label=f'Observed={obs:.4f}')
        ax.axvline(result['null_mean'], color='gray', linewidth=1.5, 
                  linestyle='-', alpha=0.7)
        
        ax.axvline(result['ci_lower'], color='green', linewidth=1, 
                  linestyle=':', alpha=0.5)
        ax.axvline(result['ci_upper'], color='green', linewidth=1, 
                  linestyle=':', alpha=0.5, label='95% CI')
        
        # Color de fondo según resultado
        if obs < result['ci_lower']:
            ax.set_facecolor('#e8f5e9')  # Verde claro
        elif obs > result['ci_upper']:
            ax.set_facecolor('#fff3e0')  # Naranja claro
        
        ax.set_xlabel('Semantic Spread (std)')
        ax.set_ylabel('Frequency')
        ax.set_title(phrase, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_phrases, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    plot_path = os.path.join(outdir, "conceptual_consistency_nulls_FINAL.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"✓ Plot: {plot_path}")


# ============================================================================
# SAVE
# ============================================================================
def save_results(results: List[Dict], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    
    df_results = pd.DataFrame([{
        'phrase': r['phrase'],
        'observed_spread': r['observed_spread'],
        'null_mean': r['null_mean'],
        'null_std': r['null_std'],
        'z_score': r['z_score'],
        'p_value': r['p_value'],
        'ci_lower': r['ci_lower'],
        'ci_upper': r['ci_upper'],
        'n_nulls': r['n_nulls']
    } for r in results])
    
    csv_path = os.path.join(outdir, "conceptual_consistency_nulls_FINAL.csv")
    df_results.to_csv(csv_path, index=False)
    
    log.info(f"\n✓ CSV: {csv_path}")
    
    # Report
    report_path = os.path.join(outdir, "conceptual_consistency_nulls_FINAL_report.txt")
    with open(report_path, "w") as f:
        f.write("# CONCEPTUAL CONSISTENCY NULL MODEL (FINAL)\n")
        f.write("="*70 + "\n\n")
        
        f.write("## Methodology\n\n")
        f.write("**Observed:**\n")
        f.write(f"  1. Extract co-occurrences (±{CONTEXT_WINDOW} words from phrase)\n")
        f.write(f"  2. Take top-{TOP_N_WORDS} most frequent co-occurring words\n")
        f.write("  3. Embed words → compute centroid\n")
        f.write("  4. semantic_spread = std(cosine similarities to centroid)\n\n")
        
        f.write("**NULL model:**\n")
        f.write("  - Sample random words from corpus vocabulary\n")
        f.write("  - Same number of words as observed\n")
        f.write("  - Embed → centroid → spread\n")
        f.write(f"  - Repeat {N_NULLS} times\n\n")
        
        f.write("**Interpretation:**\n")
        f.write("  - obs < NULL → Co-occurrences more semantically cohesive than random\n")
        f.write("  - obs > NULL → Co-occurrences more semantically dispersed\n")
        f.write("  - obs ≈ NULL → No special semantic structure\n\n")
        
        f.write("## Results\n\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Summary\n\n")
        
        n_consensus = (df_results['observed_spread'] < df_results['ci_lower']).sum()
        n_divergence = (df_results['observed_spread'] > df_results['ci_upper']).sum()
        n_total = len(df_results)
        
        f.write(f"Total phrases: {n_total}\n")
        f.write(f"Significant consensus (obs < NULL): {n_consensus}\n")
        f.write(f"Significant divergence (obs > NULL): {n_divergence}\n")
        f.write(f"No difference: {n_total - n_consensus - n_divergence}\n\n")
        
        if n_consensus >= n_total * 0.7:
            f.write("**FINDING: STRONG CONCEPTUAL CONSENSUS**\n\n")
            f.write("Low semantic spread (observed < NULL baseline) demonstrates that\n")
            f.write("co-occurring words are MORE semantically cohesive than random.\n\n")
            f.write("This validates that the observed magnitudes reflect genuine\n")
            f.write("semantic structure, NOT embedding artifacts.\n")
        elif n_divergence >= n_total * 0.7:
            f.write("**FINDING: UNEXPECTED DIVERGENCE**\n\n")
            f.write("Co-occurrences show HIGHER spread than random vocabulary.\n")
            f.write("Suggests heterogeneous usage contexts.\n")
        else:
            f.write("**FINDING: MIXED OR NO CLEAR PATTERN**\n\n")
    
    log.info(f"✓ Report: {report_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    log.info("\n" + "="*70)
    log.info("CONCEPTUAL CONSISTENCY NULL MODEL (FIXED)")
    log.info("Response to Reviewer: Random vocabulary baseline")
    log.info("="*70 + "\n")
    
    # 1. Fetch abstracts (sample)
    neo = Neo4jClient()
    try:
        df = neo.fetch_abstracts()
    finally:
        neo.close()
    
    if df.empty:
        log.error("ERROR: No data")
        return
    
    texts = df['abstract'].tolist()
    log.info(f"✓ Abstracts: {len(texts):,}\n")
    
    # 2. Build vocabulary
    vocab = build_vocabulary(texts, min_freq=50)
    
    # 3. Load embedder
    log.info("\nLoading SBERT...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    log.info("✓ Model loaded\n")
    
    # 4. Analyze each phrase
    results = []
    
    for phrase in TARGET_PHRASES:
        result = analyze_phrase(texts, vocab, phrase, embedder, n_nulls=N_NULLS)
        
        if result is not None:
            results.append(result)
    
    if not results:
        log.error("ERROR: No valid results")
        return
    
    # 5. Plot
    plot_distributions(results, OUTPUT_DIR)
    
    # 6. Save
    save_results(results, OUTPUT_DIR)
    
    log.info("\n" + "="*70)
    log.info("✅ ANALYSIS COMPLETE")
    log.info("="*70)


if __name__ == "__main__":
    main()