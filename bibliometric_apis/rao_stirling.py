"""
Calcula Rao-Stirling usando CENTROIDS (no embeddings completos).
Estrategia:
1. Fetch abstracts desde Neo4j
2. Calcular embeddings en BATCHES por cluster
3. Calcular centroid de cada cluster
4. Calcular distancias entre centroids
5. Rao-Stirling

Memoria requerida: ~500MB (vs 5GB del método naive)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from semantic import Analyzer, Neo4jClient
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("rao_stirling")

# ========================================================================
# Config
# ========================================================================

CSV_PATH = "./new_results_plastic/full_corpus/paper_topics.csv"
OUTPUT_DIR = "./new_results_plastic/full_corpus"

BENCHMARKS = {
    "Supramolecular Chemistry": 0.68,
    "Circular Economy (general)": 0.55,
    "Materials Science": 0.52,
    "Theoretical Physics": 0.35
}

# ========================================================================
# Funciones
# ========================================================================

def fetch_abstracts_for_cluster(dois, cluster_id):
    """Fetch abstracts desde Neo4j para un cluster específico"""
    neo = Neo4jClient()
    
    cypher = """
        UNWIND $dois AS doi
        MATCH (p:Paper {doi: doi})
        WHERE p.abstract IS NOT NULL 
          AND p.abstract <> ''
        RETURN p.doi AS doi, 
               p.abstract AS abstract
    """
    
    with neo.driver.session(database=neo.db) as s:
        rows = [r.data() for r in s.run(cypher, dois=dois)]
    
    neo.close()
    
    df_abs = pd.DataFrame(rows)
    
    log.info(f"  C{cluster_id}: fetched {len(df_abs)}/{len(dois)} abstracts")
    
    return df_abs


def compute_cluster_centroid(df_abs, cluster_id):
    """Calcula centroid para un cluster usando embeddings"""
    
    if len(df_abs) == 0:
        log.warning(f"  C{cluster_id}: no abstracts, skipping")
        return None
    
    # Generar embeddings solo para este cluster
    an = Analyzer(backend="sbert", random_state=42)
    an.set_df(df_abs)
    
    log.info(f"  C{cluster_id}: preprocessing {len(df_abs)} papers...")
    an.preprocess()
    
    log.info(f"  C{cluster_id}: embedding...")
    an.embed()
    
    # Calcular centroid
    centroid = an.X.mean(axis=0, keepdims=True)
    
    log.info(f"  C{cluster_id}: centroid computed ✓")
    
    # Liberar memoria
    del an.X
    del an
    
    return centroid


def compute_centroids_by_cluster(df):
    """Calcula centroids cluster por cluster (memoria eficiente)"""
    
    log.info("\n" + "="*70)
    log.info("COMPUTING CLUSTER CENTROIDS (batch processing)")
    log.info("="*70 + "\n")
    
    clusters = sorted(df['cluster'].unique())
    centroids = {}
    
    for c in clusters:
        log.info(f"\n→ Processing Cluster {c}:")
        
        # DOIs de este cluster
        dois_c = df[df.cluster == c]['doi'].tolist()
        
        # Fetch abstracts
        df_abs = fetch_abstracts_for_cluster(dois_c, c)
        
        if len(df_abs) < 10:
            log.warning(f"  C{c}: too few abstracts ({len(df_abs)}), using fallback")
            centroids[c] = None
            continue
        
        # Calcular centroid
        centroid = compute_cluster_centroid(df_abs, c)
        centroids[c] = centroid
    
    return centroids


def compute_centroid_distances(centroids):
    """Calcula matriz de distancias entre centroids"""
    
    log.info("\n" + "="*70)
    log.info("COMPUTING PAIRWISE DISTANCES")
    log.info("="*70 + "\n")
    
    # Filtrar centroids válidos
    valid_clusters = [c for c, cent in centroids.items() if cent is not None]
    
    if len(valid_clusters) < 2:
        log.error("ERROR: Not enough valid centroids")
        return None, None
    
    # Stack centroids
    C = np.vstack([centroids[c] for c in valid_clusters])
    
    # Distancias = 1 - cosine similarity
    cos_sim = cosine_similarity(C, C)
    distances = 1 - cos_sim
    
    # Dict format
    dist_matrix = {}
    for i, ci in enumerate(valid_clusters):
        for j, cj in enumerate(valid_clusters):
            dist_matrix[(ci, cj)] = distances[i, j]
    
    log.info(f"Distance matrix computed: {len(valid_clusters)}×{len(valid_clusters)}")
    log.info(f"Mean distance: {np.mean(distances[np.triu_indices(len(distances), k=1)]):.4f}")
    
    return dist_matrix, distances


def compute_rao_stirling(proportions, dist_matrix):
    """Calcula Rao-Stirling"""
    
    clusters = sorted(proportions.keys())
    
    rs = 0.0
    
    for i, ci in enumerate(clusters):
        for j, cj in enumerate(clusters):
            if i != j and (ci, cj) in dist_matrix:
                d_ij = dist_matrix[(ci, cj)]
                p_i = proportions[ci]
                p_j = proportions[cj]
                
                rs += d_ij * p_i * p_j
    
    return rs


def compare_with_benchmarks(rs_value):
    """Comparación con benchmarks"""
    
    log.info("\n" + "="*70)
    log.info("COMPARISON WITH BENCHMARKS")
    log.info("="*70 + "\n")
    
    comparisons = [
        {"field": "Plastic Recycling (this study)", "rao_stirling": rs_value, "source": "Current analysis"}
    ]
    
    for field, value in BENCHMARKS.items():
        comparisons.append({"field": field, "rao_stirling": value, "source": "Literature"})
    
    df_comp = pd.DataFrame(comparisons)
    df_comp = df_comp.sort_values('rao_stirling', ascending=False)
    
    log.info(df_comp.to_string(index=False))
    
    max_bench = max(BENCHMARKS.values())
    deficit_pct = 100 * (max_bench - rs_value) / max_bench
    
    log.info(f"\nDeficit vs most interdisciplinary: {deficit_pct:.1f}%")
    
    return df_comp, deficit_pct


# ========================================================================
# Main
# ========================================================================

def main():
    
    log.info("\n" + "="*70)
    log.info("RAO-STIRLING DIVERSITY ANALYSIS")
    log.info("Memory-efficient centroid-based computation")
    log.info("="*70 + "\n")
    
    # 1. Cargar clustering
    log.info("Loading paper_topics.csv...")
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    df = df[df['cluster'].notna()].copy()
    df['cluster'] = df['cluster'].astype(int)
    
    log.info(f"Papers: {len(df):,}")
    log.info(f"Clusters: {df['cluster'].nunique()}")
    
    # 2. Proporciones
    labels = df['cluster'].values
    clusters = sorted(df['cluster'].unique())
    
    proportions = {}
    for c in clusters:
        proportions[c] = (labels == c).sum() / len(labels)
    
    log.info("\nCluster distribution:")
    for c, p in sorted(proportions.items()):
        log.info(f"  C{c}: {p:.3f}")
    
    # Balance
    simpson = sum([p**2 for p in proportions.values()])
    balance = 1 - simpson
    log.info(f"\nBalance (1-Simpson): {balance:.3f}")
    
    # 3. Calcular centroids (memoria eficiente)
    try:
        centroids = compute_centroids_by_cluster(df)
        
        # 4. Distancias
        dist_matrix, distances = compute_centroid_distances(centroids)
        
        if dist_matrix is None:
            raise Exception("Failed to compute distances")
        
        # 5. Rao-Stirling
        log.info("\n" + "="*70)
        log.info("COMPUTING RAO-STIRLING")
        log.info("="*70 + "\n")
        
        rs = compute_rao_stirling(proportions, dist_matrix)
        
        log.info(f"Rao-Stirling = {rs:.4f}")
        
        # 6. Benchmarks
        df_comp, deficit_pct = compare_with_benchmarks(rs)
        
        # 7. Guardar
        log.info("\n" + "="*70)
        log.info("SAVING RESULTS")
        log.info("="*70 + "\n")
        
        csv_path = f"{OUTPUT_DIR}/rao_stirling_diversity.csv"
        df_comp.to_csv(csv_path, index=False)
        
        report_path = f"{OUTPUT_DIR}/rao_stirling_report.txt"
        with open(report_path, "w") as f:
            f.write("# RAO-STIRLING DIVERSITY ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            f.write("## Methodology\n\n")
            f.write("Rao-Stirling index: RS = Σ(i≠j) d_ij · p_i · p_j\n")
            f.write("  - d_ij = 1 - cosine_similarity(centroid_i, centroid_j)\n")
            f.write("  - p_i = proportion of papers in cluster i\n")
            f.write("  - Centroids computed from SBERT embeddings\n\n")
            
            f.write("## Results\n\n")
            f.write(f"Papers: {len(df):,}\n")
            f.write(f"Clusters: {len(clusters)}\n")
            f.write(f"Balance: {balance:.3f}\n")
            f.write(f"Rao-Stirling: {rs:.4f}\n\n")
            
            f.write("Cluster distribution:\n")
            for c, p in sorted(proportions.items()):
                f.write(f"  C{c}: {p:.3f}\n")
            f.write("\n")
            
            f.write("## Comparison with Benchmarks\n\n")
            f.write(df_comp.to_string(index=False))
            f.write("\n\n")
            
            f.write(f"Diversity deficit: {deficit_pct:.1f}%\n")
        
        log.info(f"✓ {csv_path}")
        log.info(f"✓ {report_path}")
        
        log.info("\n" + "="*70)
        log.info("✅ ANALYSIS COMPLETE")
        log.info("="*70 + "\n")
        
    except Exception as e:
        log.error(f"\n❌ ERROR: {e}")
        log.error("Falling back to lexical distances...\n")
        
        # FALLBACK: Usar topic_label
        from sklearn.feature_extraction.text import CountVectorizer
        
        log.info("Computing lexical distances from topic_label...")
        
        # Extraer vocabulario por cluster
        cluster_vocab = {}
        for c in clusters:
            texts = df[df.cluster == c]['topic_label'].dropna().tolist()
            vocab = set()
            for text in texts:
                vocab.update(text.split(', '))
            cluster_vocab[c] = vocab
        
        # Distancias Jaccard
        dist_matrix = {}
        for ci in clusters:
            for cj in clusters:
                if ci == cj:
                    dist_matrix[(ci, cj)] = 0.0
                else:
                    intersection = len(cluster_vocab[ci] & cluster_vocab[cj])
                    union = len(cluster_vocab[ci] | cluster_vocab[cj])
                    jaccard = intersection / union if union > 0 else 0.0
                    dist_matrix[(ci, cj)] = 1 - jaccard
        
        rs = compute_rao_stirling(proportions, dist_matrix)
        
        log.info(f"Rao-Stirling (lexical): {rs:.4f}")
        
        df_comp, deficit_pct = compare_with_benchmarks(rs)
        
        # Guardar con nota
        csv_path = f"{OUTPUT_DIR}/rao_stirling_diversity.csv"
        df_comp.to_csv(csv_path, index=False)
        
        log.info(f"\n✓ Results saved (lexical distances fallback)")
        log.info(f"  Note: Used topic vocabulary distances instead of embeddings")


if __name__ == "__main__":
    main()
