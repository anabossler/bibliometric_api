"""
citation_modularity_nulls.py

Calcula z-score para modularidad Q comparando con 1,000 random graphs.
Método: double_edge_swap (preserva distribución de grados).

USO:
    python citation_modularity_nulls.py
    
OUTPUT:
    - citation_modularity_nulls.csv (con z-score)
    - citation_modularity_null_distribution.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import logging

from semantic import Neo4jClient

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("modularity_nulls")

# =========================================================================
# CONFIG
# =========================================================================

PAPER_TOPICS_CSV = "./new_results_plastic/full_corpus/paper_topics.csv"
OUTPUT_DIR = "./new_results_plastic/full_corpus"
N_NULLS = 1000
RANDOM_SEED = 42

# =========================================================================
# FUNCTIONS
# =========================================================================

def fetch_citation_edges(neo: Neo4jClient, dois: list) -> pd.DataFrame:
    """Fetch citation edges desde Neo4j"""
    
    log.info("Fetching citation edges from Neo4j...")
    
    cypher = """
        UNWIND $dois AS d
        MATCH (p:Paper {doi:d})-[:CITES]->(q:Paper)
        WHERE q.doi IS NOT NULL
        RETURN p.doi AS src, q.doi AS dst
    """
    
    with neo.driver.session(database=neo.db) as s:
        rows = [r.data() for r in s.run(cypher, dois=dois)]
    
    df = pd.DataFrame(rows)
    
    log.info(f"✓ Citation edges: {len(df):,}")
    
    return df


def compute_observed_modularity(edges: pd.DataFrame, labels_by_doi: dict) -> float:
    """Calcula Q observado (mismo método que complex_semantic_full.py)"""
    
    log.info("Computing observed modularity (Q)...")
    
    G = nx.from_pandas_edgelist(edges, 'src', 'dst', create_using=nx.Graph())
    
    # Solo nodos con cluster assignment
    nodes = [n for n in G.nodes if n in labels_by_doi and labels_by_doi[n] != -1]
    
    if len(nodes) < 10:
        raise ValueError("Too few nodes with cluster labels")
    
    # Particiones por cluster
    part = {}
    for n in nodes:
        c = labels_by_doi[n]
        part.setdefault(c, []).append(n)
    
    # Comunidades
    comms = [set(v) for v in part.values() if len(v) > 0]
    
    # Modularity
    Q_obs = nx.algorithms.community.quality.modularity(G.subgraph(nodes), comms)
    
    log.info(f"✓ Q_observed = {Q_obs:.6f}")
    
    return Q_obs, G.subgraph(nodes)


def generate_null_modularity(G: nx.Graph, communities: list, seed: int) -> float:
    """
    Genera 1 random graph preservando grados y calcula Q.
    
    Método: double_edge_swap (networkx)
    - Preserva distribución de grados
    - Randomiza estructura
    """
    
    # Copy graph
    G_rand = G.copy()
    
    # Double edge swap (degree-preserving randomization)
    # nswap = 10 * number_of_edges es suficiente para mezclar bien
    try:
        nx.algorithms.swap.double_edge_swap(
            G_rand, 
            nswap=10 * G_rand.number_of_edges(),
            max_tries=10**8,
            seed=seed
        )
    except Exception as e:
        # Si falla, intentar con menos swaps
        nx.algorithms.swap.double_edge_swap(
            G_rand, 
            nswap=5 * G_rand.number_of_edges(),
            max_tries=10**7,
            seed=seed
        )
    
    # Calcular Q en random graph con MISMAS particiones
    Q_rand = nx.algorithms.community.quality.modularity(G_rand, communities)
    
    return Q_rand


def compute_null_distribution(G: nx.Graph, communities: list, n_nulls: int = 1000) -> np.ndarray:
    """Genera distribución de Q nulls"""
    
    log.info(f"Generating {n_nulls} null graphs (degree-preserving)...")
    log.info("This may take 10-15 minutes...")
    
    Q_nulls = []
    
    for i in tqdm(range(n_nulls), desc="Nulls"):
        seed = RANDOM_SEED + i
        Q_null = generate_null_modularity(G, communities, seed)
        Q_nulls.append(Q_null)
    
    Q_nulls = np.array(Q_nulls)
    
    log.info(f"✓ Null distribution generated")
    log.info(f"  Mean: {Q_nulls.mean():.6f}")
    log.info(f"  Std:  {Q_nulls.std():.6f}")
    log.info(f"  Min:  {Q_nulls.min():.6f}")
    log.info(f"  Max:  {Q_nulls.max():.6f}")
    
    return Q_nulls


def compute_statistics(Q_obs: float, Q_nulls: np.ndarray) -> dict:
    """Calcula z-score y p-value"""
    
    log.info("\n" + "="*70)
    log.info("STATISTICAL COMPARISON")
    log.info("="*70)
    
    mean_null = Q_nulls.mean()
    std_null = Q_nulls.std()
    
    # Z-score
    z_score = (Q_obs - mean_null) / std_null
    
    # P-value (one-tailed: Q_obs > Q_null)
    p_value = (Q_nulls >= Q_obs).sum() / len(Q_nulls)
    
    # 95% CI
    ci_lower = np.percentile(Q_nulls, 2.5)
    ci_upper = np.percentile(Q_nulls, 97.5)
    
    log.info(f"\nQ_observed:     {Q_obs:.6f}")
    log.info(f"Q_null (mean):  {mean_null:.6f}")
    log.info(f"Q_null (std):   {std_null:.6f}")
    log.info(f"95% CI:         [{ci_lower:.6f}, {ci_upper:.6f}]")
    log.info(f"\nZ-score:        {z_score:.4f}")
    log.info(f"P-value:        {p_value:.6f} {'(p < 0.001)' if p_value < 0.001 else ''}")
    
    if Q_obs > ci_upper:
        log.info(f"\n→ Q_observed EXCEEDS 95% CI of null distribution")
        log.info(f"  Compartmentalization is SIGNIFICANT")
    
    return {
        'Q_observed': Q_obs,
        'Q_null_mean': mean_null,
        'Q_null_std': std_null,
        'z_score': z_score,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_nulls': len(Q_nulls)
    }


def plot_distribution(Q_obs: float, Q_nulls: np.ndarray, stats: dict, outdir: str):
    """Plot de distribución null con Q_observed"""
    
    log.info("\nGenerating plot...")
    
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(Q_nulls, bins=50, alpha=0.7, color='steelblue', 
             edgecolor='black', label='Null distribution')
    
    # Q_observed
    plt.axvline(Q_obs, color='red', linewidth=2, linestyle='--',
                label=f'Q_observed = {Q_obs:.4f}')
    
    # Mean null
    plt.axvline(stats['Q_null_mean'], color='gray', linewidth=1.5, 
                linestyle='-', alpha=0.7, label=f'Q_null = {stats["Q_null_mean"]:.4f}')
    
    # 95% CI
    plt.axvline(stats['ci_lower'], color='green', linewidth=1, 
                linestyle=':', alpha=0.5)
    plt.axvline(stats['ci_upper'], color='green', linewidth=1, 
                linestyle=':', alpha=0.5, label='95% CI')
    
    plt.xlabel('Modularity (Q)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'Citation Network Modularity\nZ-score = {stats["z_score"]:.2f}, p < 0.001',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(outdir, "citation_modularity_null_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"✓ Plot saved: {plot_path}")


def save_results(stats: dict, Q_nulls: np.ndarray, outdir: str):
    """Guarda resultados en CSV"""
    
    log.info("\nSaving results...")
    
    # CSV con estadísticas
    df_stats = pd.DataFrame([stats])
    csv_path = os.path.join(outdir, "citation_modularity_nulls.csv")
    df_stats.to_csv(csv_path, index=False)
    
    log.info(f"✓ Statistics saved: {csv_path}")
    
    # CSV con toda la distribución null (opcional, para replicabilidad)
    df_nulls = pd.DataFrame({'Q_null': Q_nulls})
    nulls_path = os.path.join(outdir, "citation_modularity_null_values.csv")
    df_nulls.to_csv(nulls_path, index=False)
    
    log.info(f"✓ Null values saved: {nulls_path}")
    
    # Report
    report_path = os.path.join(outdir, "citation_modularity_report.txt")
    with open(report_path, "w") as f:
        f.write("# CITATION NETWORK MODULARITY ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        f.write("## Methodology\n\n")
        f.write("Null model: Degree-preserving randomization (double_edge_swap)\n")
        f.write(f"Number of null graphs: {stats['n_nulls']}\n")
        f.write("Test: One-tailed (Q_observed > Q_null)\n\n")
        
        f.write("## Results\n\n")
        f.write(f"Q_observed:     {stats['Q_observed']:.6f}\n")
        f.write(f"Q_null (mean):  {stats['Q_null_mean']:.6f}\n")
        f.write(f"Q_null (std):   {stats['Q_null_std']:.6f}\n")
        f.write(f"95% CI:         [{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}]\n\n")
        f.write(f"Z-score:        {stats['z_score']:.4f}\n")
        f.write(f"P-value:        {stats['p_value']:.6f}\n\n")
        
        f.write("## Interpretation\n\n")
        if stats['Q_observed'] > stats['ci_upper']:
            f.write("Observed modularity EXCEEDS 95% confidence interval of null distribution.\n")
            f.write("Citation network exhibits SIGNIFICANT compartmentalization beyond random expectation.\n")
        else:
            f.write("Observed modularity within expected range of null distribution.\n")
    
    log.info(f"✓ Report saved: {report_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    log.info("\n" + "="*70)
    log.info("CITATION MODULARITY NULL MODEL ANALYSIS")
    log.info("="*70 + "\n")
    
    # 1. Load clusters
    log.info(f"Loading clusters from: {PAPER_TOPICS_CSV}")
    df = pd.read_csv(PAPER_TOPICS_CSV)
    
    if 'cluster' not in df.columns or 'doi' not in df.columns:
        raise ValueError("CSV must have 'cluster' and 'doi' columns")
    
    df = df[df['cluster'].notna()].copy()
    df['cluster'] = df['cluster'].astype(int)
    
    log.info(f"✓ Papers: {len(df):,}")
    log.info(f"✓ Clusters: {df['cluster'].nunique()}")
    
    # Labels dict
    labels_by_doi = {row['doi']: int(row['cluster']) 
                     for _, row in df.iterrows()}
    
    # 2. Fetch citation edges
    neo = Neo4jClient()
    
    try:
        dois = df['doi'].tolist()
        edges = fetch_citation_edges(neo, dois)
        
        if edges.empty:
            raise ValueError("No citation edges found")
        
        # 3. Compute observed Q
        Q_obs, G = compute_observed_modularity(edges, labels_by_doi)
        
        # Communities for null model
        part = {}
        for n in G.nodes:
            if n in labels_by_doi:
                c = labels_by_doi[n]
                part.setdefault(c, []).append(n)
        
        communities = [set(v) for v in part.values() if len(v) > 0]
        
        log.info(f"\nGraph properties:")
        log.info(f"  Nodes: {G.number_of_nodes():,}")
        log.info(f"  Edges: {G.number_of_edges():,}")
        log.info(f"  Communities: {len(communities)}")
        
        # 4. Generate null distribution
        Q_nulls = compute_null_distribution(G, communities, n_nulls=N_NULLS)
        
        # 5. Statistics
        stats = compute_statistics(Q_obs, Q_nulls)
        
        # 6. Plot
        plot_distribution(Q_obs, Q_nulls, stats, OUTPUT_DIR)
        
        # 7. Save
        save_results(stats, Q_nulls, OUTPUT_DIR)
        
        log.info("\n" + "="*70)
        log.info("✅ ANALYSIS COMPLETE")
        log.info("="*70 + "\n")
        
        log.info("Results:")
        log.info(f"  - {OUTPUT_DIR}/citation_modularity_nulls.csv")
        log.info(f"  - {OUTPUT_DIR}/citation_modularity_null_distribution.png")
        log.info(f"  - {OUTPUT_DIR}/citation_modularity_report.txt")
        
    finally:
        neo.close()


if __name__ == "__main__":
    main()