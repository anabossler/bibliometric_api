"""
embedding_sensitivity_complete.py — Análisis de sensibilidad de embeddings
para comparar SBERT vs SPECTER2 vs ChemBERTa

CONFIGURACIÓN:
  - Edita línea 24-25 con tu ruta a results_plastic_rbo/full_corpus
  - Ejecuta: python embedding_sensitivity_complete.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from semantic import Analyzer, centroid_cosine_matrix, Neo4jClient

# ============================================================================
# CONFIGURACIÓN - EDITA ESTAS RUTAS
# ============================================================================
FROZEN_CORPUS_DIR = "/Users/annabossler/Desktop/openalex/results_plastic_rbo/full_corpus/"
FROZEN_PAPER_TOPICS = os.path.join(FROZEN_CORPUS_DIR, "paper_topics.csv")
OUTPUT_DIR = "./embedding_sensitivity_results"

# Modelos a comparar
EMBEDDING_MODELS = ["sbert", "specter2", "chemberta"]

# Parámetros
SAMPLE_FRACTION = 0.20  # 20% del corpus
K_CLUSTERS = 6          # Número de clusters forzado para comparación
RANDOM_SEED = 42


# ============================================================================
# FUNCIONES
# ============================================================================

def load_frozen_dois():
    """Carga los DOIs exactos del corpus original"""
    print("="*80)
    print("CARGANDO DOIs DEL CORPUS ORIGINAL")
    print("="*80)
    
    if not os.path.exists(FROZEN_PAPER_TOPICS):
        print(f"\n❌ ERROR: No se encuentra {FROZEN_PAPER_TOPICS}")
        print(f"   Verifica la ruta en línea 24 del script\n")
        raise FileNotFoundError(FROZEN_PAPER_TOPICS)
    
    df = pd.read_csv(FROZEN_PAPER_TOPICS)
    dois = df['doi'].dropna().unique().tolist()
    
    print(f"\n✓ Archivo cargado: {FROZEN_PAPER_TOPICS}")
    print(f"✓ Papers en corpus original: {len(df):,}")
    print(f"✓ DOIs únicos: {len(dois):,}")
    print("="*80 + "\n")
    
    return dois


def fetch_abstracts_from_dois(dois):
    """Obtiene abstracts desde Neo4j para DOIs específicos"""
    print("="*80)
    print("FETCHING ABSTRACTS DESDE NEO4J")
    print("="*80)
    
    neo = Neo4jClient()
    
    cypher = """
        UNWIND $dois AS doi
        MATCH (p:Paper {doi: doi})
        WHERE p.abstract IS NOT NULL 
          AND p.abstract <> ''
          AND p.has_abstract = true
        RETURN p.doi AS doi, 
               p.openalex_id AS eid,
               p.title AS title,
               p.abstract AS abstract, 
               toString(p.publication_year) AS year,
               COALESCE(toInteger(p.cited_by_count), 0) AS citedBy
    """
    
    print(f"\n[1/2] Ejecutando query para {len(dois):,} DOIs...")
    
    with neo.driver.session(database=neo.db) as s:
        rows = [r.data() for r in s.run(cypher, dois=dois)]
    
    neo.close()
    
    df = pd.DataFrame(rows)
    
    print(f"✓ Abstracts recuperados: {len(df):,}")
    
    if len(df) < len(dois):
        missing = len(dois) - len(df)
        pct = 100 * missing / len(dois)
        print(f"⚠️  Papers sin abstract: {missing} ({pct:.1f}%)")
        print(f"   Posibles causas:")
        print(f"   - OpenAlex reindexó y algunos abstracts ya no están disponibles")
        print(f"   - DOIs con formato diferente en Neo4j")
    
    print("="*80 + "\n")
    
    return df


def stratified_sample(df, frac=0.20, seed=42):
    """Muestra estratificada por año"""
    print("="*80)
    print(f"CREANDO MUESTRA ESTRATIFICADA ({frac*100:.0f}%)")
    print("="*80)
    
    if "year" in df.columns:
        y = pd.to_numeric(df["year"], errors="coerce").fillna(-1).astype(int)
    else:
        y = np.zeros(len(df), dtype=int)

    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Separar clases válidas (>=2 elementos) e inválidas (<2 elementos)
    valid_classes = [c for c in np.unique(y) if sum(y == c) >= 2]
    invalid_classes = [c for c in np.unique(y) if sum(y == c) < 2]

    df_valid = df[y.isin(valid_classes)]
    y_valid = y[y.isin(valid_classes)]
    df_invalid = df[y.isin(invalid_classes)]

    # Estratificación solo para clases válidas
    n_invalid = len(df_invalid)
    
    if len(df_valid) > 0:
        split_frac_valid = (len(df) * frac - n_invalid) / len(df_valid)
        split_frac_valid = max(0.01, min(split_frac_valid, 0.99))

        splitter = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=split_frac_valid, 
            random_state=seed
        )
        _, test_idx = next(splitter.split(df_valid, y_valid))
        df_strat = df_valid.iloc[test_idx]
    else:
        df_strat = pd.DataFrame()

    # Añadir clases no estratificables
    df_sample = pd.concat([df_strat, df_invalid], ignore_index=True)
    
    print(f"\n✓ Corpus completo: {len(df):,} papers")
    print(f"✓ Muestra generada: {len(df_sample):,} papers")
    print(f"✓ Fracción real: {100*len(df_sample)/len(df):.1f}%")
    print("="*80 + "\n")
    
    return df_sample.reset_index(drop=True)


def run_embedding_model(df, backend, k=6, seed=42):
    """Ejecuta embedding y clustering para un modelo específico"""
    print(f"→ Procesando modelo: {backend.upper()}")
    
    an = Analyzer(backend=backend, random_state=seed)
    an.set_df(df)
    
    print(f"  [1/3] Preprocessing...")
    an.preprocess()
    
    print(f"  [2/3] Embedding...")
    an.embed()
    
    print(f"  [3/3] Clustering (k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(an.X)
    
    centroids = centroid_cosine_matrix(an.X, labels)
    
    print(f"  ✓ Completado\n")
    
    return {
        "labels": labels,
        "k": k,
        "centroids": centroids,
        "X": an.X
    }


def compare_models(results):
    """Compara modelos calculando ARI y centroid drift"""
    print("="*80)
    print("COMPARANDO MODELOS")
    print("="*80 + "\n")
    
    comparisons = []
    keys = list(results.keys())
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m1, m2 = keys[i], keys[j]
            
            # ARI entre labels
            ari = adjusted_rand_score(
                results[m1]["labels"], 
                results[m2]["labels"]
            )
            
            # Centroid drift
            C1 = results[m1]["centroids"].values
            C2 = results[m2]["centroids"].values
            drift = np.mean(1 - cosine_similarity(C1, C2))
            
            comparisons.append({
                "model_a": m1,
                "model_b": m2,
                "ARI": float(ari),
                "centroid_drift": float(drift),
                "k": results[m1]["k"]
            })
            
            print(f"{m1.upper()} ↔ {m2.upper()}:")
            print(f"  ARI (label agreement):     {ari:.4f}")
            print(f"  Centroid drift (cosine):   {drift:.4f}")
            print()
    
    return pd.DataFrame(comparisons)


def save_results(df_comparison, corpus_info, outdir):
    """Guarda resultados del análisis"""
    os.makedirs(outdir, exist_ok=True)
    
    # CSV de comparaciones
    csv_path = os.path.join(outdir, "embedding_sensitivity_comparison.csv")
    df_comparison.to_csv(csv_path, index=False)
    
    # Info del corpus
    info_path = os.path.join(outdir, "corpus_info.txt")
    with open(info_path, "w") as f:
        for key, value in corpus_info.items():
            f.write(f"{key}: {value}\n")
    
    print("="*80)
    print("RESULTADOS GUARDADOS")
    print("="*80)
    print(f"\n✓ Comparaciones: {csv_path}")
    print(f"✓ Info corpus:   {info_path}")
    print(f"✓ Directorio:    {outdir}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("EMBEDDING SENSITIVITY ANALYSIS")
    print("Corpus: results_plastic_rbo (n=3,265)")
    print("="*80 + "\n")
    
    # 1. Cargar DOIs del corpus original
    dois_original = load_frozen_dois()
    
    # 2. Fetch abstracts desde Neo4j
    df_full = fetch_abstracts_from_dois(dois_original)
    
    if len(df_full) == 0:
        print("❌ ERROR: No se recuperaron abstracts")
        return
    
    # 3. Crear muestra estratificada
    df_sample = stratified_sample(df_full, frac=SAMPLE_FRACTION, seed=RANDOM_SEED)
    
    # 4. Ejecutar embeddings para cada modelo
    print("="*80)
    print("EJECUTANDO EMBEDDINGS")
    print("="*80 + "\n")
    
    results = {}
    for backend in EMBEDDING_MODELS:
        try:
            results[backend] = run_embedding_model(
                df_sample, 
                backend, 
                k=K_CLUSTERS, 
                seed=RANDOM_SEED
            )
        except Exception as e:
            print(f"⚠️  ERROR en {backend}: {e}\n")
            continue
    
    if len(results) < 2:
        print("❌ ERROR: Se necesitan al menos 2 modelos para comparar")
        return
    
    # 5. Comparar modelos
    df_comparison = compare_models(results)
    
    # 6. Guardar resultados
    corpus_info = {
        "source": FROZEN_PAPER_TOPICS,
        "corpus_size": len(df_full),
        "sample_size": len(df_sample),
        "sample_fraction": SAMPLE_FRACTION,
        "k_clusters": K_CLUSTERS,
        "random_seed": RANDOM_SEED,
        "models_compared": ", ".join(EMBEDDING_MODELS),
        "models_successful": ", ".join(results.keys())
    }
    
    save_results(df_comparison, corpus_info, OUTPUT_DIR)
    
    # 7. Mostrar resumen
    print("="*80)
    print("RESUMEN FINAL")
    print("="*80 + "\n")
    print(df_comparison.to_string(index=False))
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()