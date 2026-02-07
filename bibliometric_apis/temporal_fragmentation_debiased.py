#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_consistency_common_terms.py

Analiza si términos comunes ("plastics", "recycling") significan
lo MISMO en diferentes clusters usando análisis de CO-OCURRENCIAS.

PREGUNTA CLAVE:
¿"Reciclado de plástico" para químico = "reciclado de plástico" para sociólogo?

MÉTODO:
1. Extraer términos que CO-OCURREN con la frase target en cada cluster
2. Embed co-ocurrencias con SBERT (representan "aspectos" del concepto)
3. Medir distancia semántica entre clusters
4. Si distancia BAJA → co-ocurrencias relacionadas → MISMO CONCEPTO
   Si distancia ALTA → co-ocurrencias divergentes → CONCEPTOS DIFERENTES

USO:
    python semantic_consistency_common_terms.py
"""

import os, sys, json, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import Counter
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

try:
    from semantic import (
        Analyzer, clean_abstract, tokenize,
        label_topics_c_tfidf, cluster_kmeans
    )
except ImportError:
    print("ERROR: No se puede importar complex_semantic_full.py")
    print("Asegúrate de que está en el mismo directorio")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: Install sentence-transformers")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("semantic_consistency")

# ========= Neo4j Client =========
class Neo4jClient:
    def __init__(self):
        load_dotenv()
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.pwd = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.db = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
    
    def close(self):
        try: 
            self.driver.close()
        except: 
            pass
    
    def fetch_papers(self):
        cypher = """
        MATCH (p:Paper)
        WHERE p.abstract IS NOT NULL 
          AND p.abstract <> ''
          AND p.has_abstract = true
          AND p.publication_year IS NOT NULL
          AND (
            toLower(p.abstract) CONTAINS 'plastic recycling'
            OR toLower(p.abstract) CONTAINS 'recycled plastic'
          )
          AND NOT toLower(p.abstract) CONTAINS 'pulp'
        RETURN p.doi AS doi, p.title AS title,
               p.abstract AS abstract, toString(p.publication_year) AS year
        """
        with self.driver.session(database=self.db) as s:
            rows = [r.data() for r in s.run(cypher)]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["abstract"] = df["abstract"].fillna("").astype(str).apply(clean_abstract)
        df["abstract_len"] = df["abstract"].str.split().apply(len)
        df = df[df["abstract_len"] >= 20].drop(columns=["abstract_len"]).reset_index(drop=True)
        return df

# ========= Co-occurrence Analysis =========
def extract_cooccurrences(texts: List[str], 
                         target_phrase: str,
                         window_size: int = 10,
                         stopwords: set = None) -> List[str]:
    """
    Extrae términos que CO-OCURREN con la frase target.
    Esto revela QUÉ ASPECTOS del concepto se discuten.
    
    Ejemplo:
      "plastic recycling via pyrolysis" → cooccur: [via, pyrolysis]
      "plastic recycling programs" → cooccur: [programs]
    """
    if stopwords is None:
        stopwords = {
            'the', 'and', 'or', 'for', 'of', 'to', 'in', 'on', 'a', 
            'is', 'are', 'was', 'were', 'be', 'been', 'with', 'by',
            'from', 'that', 'this', 'these', 'those', 'can', 'will',
            'has', 'have', 'had', 'as', 'at', 'which', 'who', 'when',
            'where', 'how', 'what', 'such', 'their', 'our', 'its',
            'an', 'but', 'not', 'they', 'them', 'than', 'into', 'through',
            'about', 'would', 'could', 'should', 'may', 'might', 'must',
            'also', 'more', 'most', 'other', 'some', 'any', 'all', 'both',
            'each', 'few', 'many', 'much', 'several', 'such', 'only', 'own',
            'same', 'so', 'then', 'there', 'very', 'well', 'being', 'using',
            'used', 'made', 'make', 'making', 'showed', 'shown', 'found'
        }
    
    cooccurrences = []
    phrase_words = target_phrase.lower().split()
    phrase_len = len(phrase_words)
    
    for text in texts:
        # Tokenize simple
        words = text.lower().split()
        
        # Buscar frase target
        for i in range(len(words) - phrase_len + 1):
            # Check si coincide la frase
            if words[i:i+phrase_len] == phrase_words:
                # Ventana de contexto
                start = max(0, i - window_size)
                end = min(len(words), i + phrase_len + window_size)
                
                # Extraer palabras en ventana (excluyendo target)
                window = words[start:i] + words[i+phrase_len:end]
                
                # Filtrar y limpiar
                window_clean = []
                for w in window:
                    # Remover puntuación
                    w_clean = w.strip('.,;:()[]{}!?"\'`')
                    if (w_clean not in stopwords 
                        and len(w_clean) > 2
                        and w_clean not in phrase_words
                        and w_clean.isalpha()):  # Solo letras
                        window_clean.append(w_clean)
                
                cooccurrences.extend(window_clean)
    
    return cooccurrences

def concept_consistency_via_cooccurrence(
    df: pd.DataFrame,
    labels: np.ndarray,
    target_phrases: List[str],
    embedder: SentenceTransformer,
    outdir: str
) -> pd.DataFrame:
    """
    Analiza si el CONCEPTO es el mismo entre clusters midiendo
    co-ocurrencias semánticas.
    
    Pregunta: ¿Qué términos aparecen CERCA de "plastic recycling" 
              en diferentes clusters? ¿Están relacionados?
    
    Si químico dice "plastic recycling via pyrolysis" y 
    sociólogo dice "plastic recycling programs collection",
    ¿"pyrolysis" y "collection" son aspectos del MISMO proceso?
    """
    
    log.info("="*60)
    log.info("CONCEPT CONSISTENCY VIA CO-OCCURRENCE ANALYSIS")
    log.info("="*60)
    
    results = []
    cooccur_data = {}  # Para visualización
    
    for phrase in target_phrases[:10]:  # Top 10
        log.info(f"\n{'='*50}")
        log.info(f"Analyzing: '{phrase}'")
        log.info(f"{'='*50}")
        
        # Co-ocurrencias por cluster
        cooccur_by_cluster = {}
        embeddings_by_cluster = {}
        
        for cluster in sorted(np.unique(labels)):
            # Textos de este cluster
            texts = df[labels == cluster]['abstract'].tolist()
            
            # Extraer co-ocurrencias
            cooccur = extract_cooccurrences(texts, phrase, window_size=10)
            
            if len(cooccur) < 20:
                log.info(f"  Cluster {cluster}: {len(cooccur)} cooccurrences (SKIP - too few)")
                continue
            
            # Top co-ocurrencias
            top_cooccur = Counter(cooccur).most_common(50)
            top_words = [word for word, count in top_cooccur]
            
            log.info(f"  Cluster {cluster}: {len(cooccur)} cooccurrences")
            log.info(f"    Top: {', '.join(top_words[:8])}")
            
            # Embed co-ocurrencias (representan "aspectos" del concepto)
            # Usamos palabras individuales, no oraciones
            cooccur_embeddings = embedder.encode(
                top_words, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Promedio = "espacio semántico" del concepto en este cluster
            mean_embedding = cooccur_embeddings.mean(axis=0)
            
            cooccur_by_cluster[cluster] = top_words
            embeddings_by_cluster[cluster] = mean_embedding
        
        if len(embeddings_by_cluster) < 2:
            log.warning(f"  → Not enough clusters, SKIP")
            continue
        
        # Calcular distancias semánticas entre "espacios conceptuales"
        cluster_ids = sorted(embeddings_by_cluster.keys())
        n_clusters = len(cluster_ids)
        
        dist_matrix = np.zeros((n_clusters, n_clusters))
        
        for i, c1 in enumerate(cluster_ids):
            for j, c2 in enumerate(cluster_ids):
                if i == j:
                    dist_matrix[i, j] = 0.0
                else:
                    sim = cosine_similarity(
                        embeddings_by_cluster[c1].reshape(1, -1),
                        embeddings_by_cluster[c2].reshape(1, -1)
                    )[0, 0]
                    dist_matrix[i, j] = 1 - sim
        
        # Stats
        upper_tri = dist_matrix[np.triu_indices(n_clusters, k=1)]
        mean_dist = float(upper_tri.mean())
        max_dist = float(upper_tri.max())
        min_dist = float(upper_tri.min())
        
        # Interpretación basada en co-ocurrencias
        if mean_dist < 0.35:
            interpretation = "SHARED CONCEPT (co-occurrences semantically related)"
        elif mean_dist < 0.50:
            interpretation = "PARTIAL OVERLAP (some shared aspects)"
        else:
            interpretation = "DIVERGENT CONCEPTS (different phenomena)"
        
        log.info(f"\n  Conceptual distance:")
        log.info(f"    Mean: {mean_dist:.3f}")
        log.info(f"    Range: [{min_dist:.3f}, {max_dist:.3f}]")
        log.info(f"    → {interpretation}")
        
        # Comparaciones específicas
        if len(cluster_ids) >= 2:
            log.info(f"\n  Pairwise examples:")
            n_examples = min(3, len(cluster_ids))
            for i in range(min(2, len(cluster_ids))):
                for j in range(i+1, n_examples):
                    c1, c2 = cluster_ids[i], cluster_ids[j]
                    dist = dist_matrix[i, j]
                    
                    log.info(f"    C{c1} vs C{c2} (dist={dist:.3f}):")
                    log.info(f"      C{c1}: {', '.join(cooccur_by_cluster[c1][:5])}")
                    log.info(f"      C{c2}: {', '.join(cooccur_by_cluster[c2][:5])}")
        
        # Guardar
        results.append({
            'phrase': phrase,
            'n_clusters': n_clusters,
            'mean_conceptual_distance': mean_dist,
            'max_distance': max_dist,
            'min_distance': min_dist,
            'interpretation': interpretation,
            'is_shared_concept': mean_dist < 0.40
        })
        
        cooccur_data[phrase] = {
            'cluster_ids': cluster_ids,
            'cooccurrences': cooccur_by_cluster,
            'embeddings': embeddings_by_cluster,
            'distance_matrix': dist_matrix
        }
    
    # DataFrame
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        log.warning("No results generated")
        return df_results
    
    # Guardar
    csv_path = os.path.join(outdir, "concept_consistency_cooccurrence.csv")
    df_results.to_csv(csv_path, index=False)
    log.info(f"\n✓ Results saved: {csv_path}")
    
    # Plots
    plot_concept_consistency(df_results, outdir)
    plot_cooccurrence_heatmaps(cooccur_data, outdir)
    
    return df_results

# ========= Visualizations =========
def plot_concept_consistency(df: pd.DataFrame, outdir: str):
    """
    Plot resumen de consistencia conceptual
    """
    
    if df.empty:
        log.warning("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort por distancia
    df_sorted = df.sort_values('mean_conceptual_distance')
    
    # Plot 1: Bar chart
    colors = ['#2ecc71' if d < 0.40 else '#e74c3c' 
              for d in df_sorted['mean_conceptual_distance']]
    
    ax1.barh(range(len(df_sorted)), df_sorted['mean_conceptual_distance'], 
             color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['phrase'], fontsize=9)
    ax1.set_xlabel('Mean Conceptual Distance', fontsize=11, fontweight='bold')
    ax1.set_title('Concept Consistency via Co-occurrence\n(lower = shared concept)', 
                  fontsize=12, fontweight='bold')
    ax1.axvline(0.40, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Scatter
    ax2.scatter(df['mean_conceptual_distance'], df['n_clusters'],
                s=100, alpha=0.6, c=colors)
    
    for _, row in df.iterrows():
        ax2.annotate(row['phrase'], 
                    (row['mean_conceptual_distance'], row['n_clusters']),
                    fontsize=8, alpha=0.7)
    
    ax2.axvline(0.40, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mean Conceptual Distance', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Clusters', fontsize=11, fontweight='bold')
    ax2.set_title('Consistency vs Coverage', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(outdir, "concept_consistency_summary.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"✓ Plot saved: {plot_path}")

def plot_cooccurrence_heatmaps(cooccur_data: Dict, outdir: str, max_phrases: int = 6):
    """
    Heatmaps de distancia conceptual para top frases
    """
    
    if not cooccur_data:
        log.warning("No data for heatmaps")
        return
    
    # Select top phrases by coverage
    phrases_sorted = sorted(cooccur_data.keys(), 
                           key=lambda p: len(cooccur_data[p]['cluster_ids']),
                           reverse=True)[:max_phrases]
    
    n_phrases = len(phrases_sorted)
    if n_phrases == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, phrase in enumerate(phrases_sorted):
        if idx >= len(axes):
            break
        
        data = cooccur_data[phrase]
        dist_mat = data['distance_matrix']
        cluster_ids = data['cluster_ids']
        
        ax = axes[idx]
        
        im = ax.imshow(dist_mat, cmap='RdYlGn_r', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(cluster_ids)))
        ax.set_yticks(range(len(cluster_ids)))
        ax.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=9)
        ax.set_yticklabels([f"C{c}" for c in cluster_ids], fontsize=9)
        
        mean_val = dist_mat[np.triu_indices(len(cluster_ids), k=1)].mean()
        ax.set_title(f"'{phrase}'\n(mean={mean_val:.2f})",
                    fontsize=10, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Annotations
        for i in range(len(cluster_ids)):
            for j in range(len(cluster_ids)):
                if i != j:
                    text = ax.text(j, i, f'{dist_mat[i, j]:.2f}',
                                 ha="center", va="center", color="black",
                                 fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_phrases, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    plot_path = os.path.join(outdir, "concept_consistency_heatmaps.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"✓ Heatmaps saved: {plot_path}")

def print_summary(df: pd.DataFrame):
    """
    Print executive summary
    """
    
    log.info("\n" + "="*60)
    log.info("CONCEPT CONSISTENCY SUMMARY (CO-OCCURRENCE METHOD)")
    log.info("="*60)
    
    if df.empty:
        log.warning("No results to summarize")
        return
    
    n_shared = (df['mean_conceptual_distance'] < 0.40).sum()
    n_divergent = (df['mean_conceptual_distance'] >= 0.55).sum()
    n_total = len(df)
    
    print(f"\nPhrases analyzed: {n_total}")
    print(f"  Shared concept (< 0.40):   {n_shared} ({100*n_shared/n_total:.0f}%)")
    print(f"  Divergent (≥ 0.55):        {n_divergent} ({100*n_divergent/n_total:.0f}%)")
    
    if n_total > 0:
        print(f"\nMost shared concepts (same phenomenon):")
        for _, row in df.nsmallest(min(5, n_total), 'mean_conceptual_distance').iterrows():
            print(f"  - {row['phrase']:30s} (dist={row['mean_conceptual_distance']:.3f})")
        
        print(f"\nMost divergent concepts (different phenomena):")
        for _, row in df.nlargest(min(5, n_total), 'mean_conceptual_distance').iterrows():
            print(f"  - {row['phrase']:30s} (dist={row['mean_conceptual_distance']:.3f})")
    
    # Overall interpretation
    mean_dist = df['mean_conceptual_distance'].mean()
    
    print(f"\n{'='*50}")
    print(f"OVERALL ASSESSMENT")
    print(f"{'='*50}")
    print(f"Mean conceptual distance: {mean_dist:.3f}")
    
    if mean_dist < 0.35:
        print("→ SHARED CONCEPTS ACROSS DOMAINS")
        print("   Different specializations discuss SAME phenomena")
        print("   Convergence is REAL (semantic bridges exist)")
    elif mean_dist < 0.50:
        print("→ PARTIAL CONCEPTUAL OVERLAP")
        print("   Some shared understanding, some divergence")
    else:
        print("→ DIVERGENT CONCEPTS")
        print("   Same terms refer to DIFFERENT phenomena")
        print("   Convergence is SUPERFICIAL (terminological only)")
    
    log.info("="*60)

# ========= Main =========
def main():
    outroot = os.getenv("DATA_DIR", "./new_results_plastic")
    outdir = os.path.join(outroot, "full_corpus")
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    
    log.info(f"Output directory: {outdir}")
    
    # Target phrases to analyze
    target_phrases = [
        'plastic recycling',
        'recycled plastic',
        'recycling process',
        'circular economy',
        'waste management',
        'environmental impact',
        'sustainability',
        'polymer recycling',
        'chemical recycling',
        'mechanical recycling'
    ]
    
    log.info(f"Analyzing {len(target_phrases)} key phrases")
    
    # Connect Neo4j
    neo = Neo4jClient()
    
    try:
        log.info("Loading corpus from Neo4j...")
        df = neo.fetch_papers()
        
        if df.empty:
            log.error("No papers found")
            sys.exit(1)
        
        log.info(f"Papers loaded: {len(df)}")
        
        # Use recent papers (2021+) for analysis
        df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
        df_recent = df[df["year_num"] >= 2021].reset_index(drop=True)
        
        log.info(f"Recent papers (2021+): {len(df_recent)}")
        
        if len(df_recent) < 100:
            log.error("Not enough recent papers for analysis")
            sys.exit(1)
        
        # Embedding + clustering
        log.info("Initializing analyzer...")
        an = Analyzer(backend="sbert")
        an.set_df(df_recent)
        an.preprocess()
        an.embed()
        
        log.info("Clustering (k=6)...")
        labels = cluster_kmeans(an.X, k=6, random_state=42)
        
        log.info(f"Clusters: {len(np.unique(labels))}")
        
        # Concept consistency analysis via co-occurrence
        embedder = an.model  # Reuse SBERT model
        
        df_results = concept_consistency_via_cooccurrence(
            df_recent, labels, target_phrases, embedder, outdir
        )
        
        print_summary(df_results)
        
        log.info("\n✅ ANALYSIS COMPLETED")
        log.info(f"✅ Results in: {outdir}/")
        log.info("   - concept_consistency_cooccurrence.csv")
        log.info("   - concept_consistency_summary.png")
        log.info("   - concept_consistency_heatmaps.png")
        
    except Exception as e:
        log.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        neo.close()

if __name__ == "__main__":
    main()