"""
k_selection_metrics.py

OUTPUT:
    - k_selection_metrics.csv (tabla completa)
    - k_selection_plot.png (visualización)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, 
    calinski_harabasz_score, adjusted_rand_score
)
from sklearn.model_selection import StratifiedShuffleSplit
import logging

from semantic import Analyzer, Neo4jClient

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("k_selection")

# =========================================================================
# CONFIG
# =========================================================================

PAPER_TOPICS_CSV = "./new_results_plastic/full_corpus/paper_topics.csv"
OUTPUT_DIR = "./new_results_plastic/full_corpus"
K_MIN = 2
K_MAX = 16
BOOTSTRAP_RUNS = 20
RANDOM_SEED = 42

# =========================================================================
# FUNCTIONS
# =========================================================================

def fetch_abstracts_from_dois(neo: Neo4jClient, dois: list) -> pd.DataFrame:
    """Fetch abstracts desde Neo4j"""
    
    log.info("Fetching abstracts from Neo4j...")
    
    cypher = """
        UNWIND $dois AS doi
        MATCH (p:Paper {doi: doi})
        WHERE p.abstract IS NOT NULL 
          AND p.abstract <> ''
          AND p.has_abstract = true
        RETURN p.doi AS doi, 
               p.abstract AS abstract,
               toString(p.publication_year) AS year
    """
    
    with neo.driver.session(database=neo.db) as s:
        rows = [r.data() for r in s.run(cypher, dois=dois)]
    
    df = pd.DataFrame(rows)
    
    log.info(f"✓ Abstracts fetched: {len(df):,}")
    
    return df


def compute_bootstrap_ari(X, k: int, n_bootstrap: int = 20, seed: int = 42) -> tuple:
    """Bootstrap ARI para estimar estabilidad"""
    
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    
    if n < 100:
        return np.nan, (np.nan, np.nan)
    
    aris = []
    
    for b in range(n_bootstrap):
        # Sample 90%
        sample_size = int(0.9 * n)
        idx1 = rng.choice(n, size=sample_size, replace=False)
        idx2 = rng.choice(n, size=sample_size, replace=False)
        
        # Clustering
        km1 = KMeans(n_clusters=k, n_init=10, random_state=seed+b)
        km2 = KMeans(n_clusters=k, n_init=10, random_state=seed+b+1000)
        
        labels1 = km1.fit_predict(X[idx1])
        labels2 = km2.fit_predict(X[idx2])
        
        # Intersect
        common = np.intersect1d(idx1, idx2, return_indices=True)
        
        if len(common[0]) < 50:
            continue
        
        ari = adjusted_rand_score(labels1[common[1]], labels2[common[2]])
        aris.append(ari)
    
    if len(aris) == 0:
        return np.nan, (np.nan, np.nan)
    
    aris = np.array(aris)
    median_ari = np.median(aris)
    q25 = np.percentile(aris, 25)
    q75 = np.percentile(aris, 75)
    
    return median_ari, (q25, q75)


def evaluate_k(X, k: int, bootstrap_runs: int = 20) -> dict:
    """Evalúa métricas para un valor específico de k"""
    
    log.info(f"  Evaluating k={k}...")
    
    # KMeans
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
    labels_km = km.fit_predict(X)
    
    sil_km = silhouette_score(X, labels_km, metric='cosine')
    db_km = davies_bouldin_score(X, labels_km)
    ch_km = calinski_harabasz_score(X, labels_km)
    
    # Bootstrap ARI
    ari_km, ari_iqr_km = compute_bootstrap_ari(X, k, bootstrap_runs, RANDOM_SEED)
    
    # Agglomerative
    try:
        ag = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    except TypeError:
        ag = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average')
    
    labels_ag = ag.fit_predict(X)
    
    sil_ag = silhouette_score(X, labels_ag, metric='cosine')
    db_ag = davies_bouldin_score(X, labels_ag)
    ch_ag = calinski_harabasz_score(X, labels_ag)
    
    ari_ag, ari_iqr_ag = compute_bootstrap_ari(X, k, bootstrap_runs, RANDOM_SEED)
    
    return {
        'k': k,
        'kmeans_silhouette': sil_km,
        'kmeans_db': db_km,
        'kmeans_ch': ch_km,
        'kmeans_ari': ari_km,
        'kmeans_ari_q25': ari_iqr_km[0],
        'kmeans_ari_q75': ari_iqr_km[1],
        'agglo_silhouette': sil_ag,
        'agglo_db': db_ag,
        'agglo_ch': ch_ag,
        'agglo_ari': ari_ag,
        'agglo_ari_q25': ari_iqr_ag[0],
        'agglo_ari_q75': ari_iqr_ag[1]
    }


def plot_k_selection(df: pd.DataFrame, outdir: str):
    """Plot de métricas vs k"""
    
    log.info("Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Silhouette
    ax = axes[0, 0]
    ax.plot(df['k'], df['kmeans_silhouette'], 'o-', label='KMeans', linewidth=2)
    ax.plot(df['k'], df['agglo_silhouette'], 's-', label='Agglomerative', linewidth=2)
    ax.axvline(6, color='red', linestyle='--', alpha=0.5, label='Selected (k=6)')
    ax.set_xlabel('Number of clusters (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
    ax.set_title('Silhouette (higher is better)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Davies-Bouldin
    ax = axes[0, 1]
    ax.plot(df['k'], df['kmeans_db'], 'o-', label='KMeans', linewidth=2)
    ax.plot(df['k'], df['agglo_db'], 's-', label='Agglomerative', linewidth=2)
    ax.axvline(6, color='red', linestyle='--', alpha=0.5, label='Selected (k=6)')
    ax.set_xlabel('Number of clusters (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Davies-Bouldin Index', fontsize=11, fontweight='bold')
    ax.set_title('Davies-Bouldin (lower is better)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calinski-Harabasz
    ax = axes[1, 0]
    ax.plot(df['k'], df['kmeans_ch'], 'o-', label='KMeans', linewidth=2)
    ax.plot(df['k'], df['agglo_ch'], 's-', label='Agglomerative', linewidth=2)
    ax.axvline(6, color='red', linestyle='--', alpha=0.5, label='Selected (k=6)')
    ax.set_xlabel('Number of clusters (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Calinski-Harabasz Index', fontsize=11, fontweight='bold')
    ax.set_title('Calinski-Harabasz (higher is better)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bootstrap ARI
    ax = axes[1, 1]
    ax.plot(df['k'], df['kmeans_ari'], 'o-', label='KMeans', linewidth=2)
    ax.fill_between(df['k'], df['kmeans_ari_q25'], df['kmeans_ari_q75'], alpha=0.2)
    ax.plot(df['k'], df['agglo_ari'], 's-', label='Agglomerative', linewidth=2)
    ax.fill_between(df['k'], df['agglo_ari_q25'], df['agglo_ari_q75'], alpha=0.2)
    ax.axvline(6, color='red', linestyle='--', alpha=0.5, label='Selected (k=6)')
    ax.set_xlabel('Number of clusters (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Bootstrap ARI (median)', fontsize=11, fontweight='bold')
    ax.set_title('Clustering Stability (higher is better)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(outdir, "k_selection_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"✓ Plot saved: {plot_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    log.info("\n" + "="*70)
    log.info("K-SELECTION METRICS (k=2-16)")
    log.info("="*70 + "\n")
    
    # 1. Load clusters
    log.info(f"Loading: {PAPER_TOPICS_CSV}")
    df = pd.read_csv(PAPER_TOPICS_CSV)
    
    if 'doi' not in df.columns:
        raise ValueError("CSV must have 'doi' column")
    
    log.info(f"✓ Papers: {len(df):,}")
    
    # 2. Fetch abstracts
    neo = Neo4jClient()
    
    try:
        dois = df['doi'].dropna().tolist()
        df_abs = fetch_abstracts_from_dois(neo, dois)
        
        if df_abs.empty:
            raise ValueError("No abstracts fetched")
        
        # 3. Embed
        log.info("\nEmbedding with SBERT...")
        an = Analyzer(backend="sbert", random_state=RANDOM_SEED)
        an.set_df(df_abs)
        an.preprocess()
        an.embed()
        
        X = an.X
        
        log.info(f"✓ Embeddings: {X.shape}")
        
        # 4. Evaluate all k
        log.info(f"\nEvaluating k={K_MIN} to k={K_MAX}...")
        log.info("(This may take 20-30 minutes...)")
        
        results = []
        
        for k in range(K_MIN, K_MAX + 1):
            metrics = evaluate_k(X, k, bootstrap_runs=BOOTSTRAP_RUNS)
            results.append(metrics)
        
        df_results = pd.DataFrame(results)
        
        # 5. Save
        log.info("\nSaving results...")
        
        csv_path = os.path.join(OUTPUT_DIR, "k_selection_metrics.csv")
        df_results.to_csv(csv_path, index=False)
        
        log.info(f"✓ CSV saved: {csv_path}")
        
        # 6. Plot
        plot_k_selection(df_results, OUTPUT_DIR)
        
        # 7. Print summary
        log.info("\n" + "="*70)
        log.info("SUMMARY")
        log.info("="*70 + "\n")
        
        print("\nTop 5 by Calinski-Harabasz (KMeans):")
        print(df_results.nlargest(5, 'kmeans_ch')[['k', 'kmeans_ch', 'kmeans_silhouette', 'kmeans_ari']].to_string(index=False))
        
        print("\nMetrics for selected k=6:")
        k6 = df_results[df_results['k'] == 6].iloc[0]
        print(f"  Silhouette: {k6['kmeans_silhouette']:.4f}")
        print(f"  Davies-Bouldin: {k6['kmeans_db']:.4f}")
        print(f"  Calinski-Harabasz: {k6['kmeans_ch']:.1f}")
        print(f"  Bootstrap ARI: {k6['kmeans_ari']:.4f}")
        
        log.info("\n" + "="*70)
        log.info("✅ ANALYSIS COMPLETE")
        log.info("="*70 + "\n")
        
    finally:
        neo.close()


if __name__ == "__main__":
    main()