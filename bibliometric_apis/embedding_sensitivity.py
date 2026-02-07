"""
embedding_sensitivity.py — Sensibilidad a modelo de embeddings
Ejecuta el pipeline en un subconjunto estratificado para comparar SBERT vs SPECTER2 vs CHEMBERTa.
Produce:
- ARI entre soluciones
- matriz de drift de centroides
- número de clusters detectados
- tabla de estabilidad
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans  # ← AÑADIR ESTA LÍNEA
from semantic import (
    Analyzer, select_model_auto, centroid_cosine_matrix,
    clean_abstract, tokenize, Neo4jClient, cypher_for_full_corpus
)

EMBEDDING_MODELS = ["sbert", "specter2", "chemberta"]

def stratified_sample(df, labels, frac=0.20, seed=42):
    """Toma 20% estratificado por año, pero sin fallar si alguna clase tiene <2 elementos."""
    if "year" in df.columns:
        y = pd.to_numeric(df["year"], errors="coerce").fillna(-1).astype(int)
    else:
        y = np.zeros(len(df), dtype=int)

    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # identificar clases con >= 2
    valid_classes = [c for c in np.unique(y) if sum(y == c) >= 2]
    invalid_classes = [c for c in np.unique(y) if sum(y == c) < 2]

    df_valid = df[y.isin(valid_classes)]
    y_valid = y[y.isin(valid_classes)]

    df_invalid = df[y.isin(invalid_classes)]

    # estratificación solo para las clases válidas
    n_invalid = len(df_invalid)
    split_frac_valid = (len(df) * frac - n_invalid) / len(df_valid)
    split_frac_valid = max(0.01, min(split_frac_valid, 0.99))

    splitter = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=split_frac_valid, 
        random_state=seed
    )
    _, test_idx = next(splitter.split(df_valid, y_valid))

    df_strat = df_valid.iloc[test_idx]

    # añadir clases no estratificables (1 paper)
    df_final = pd.concat([df_strat, df_invalid], ignore_index=True)

    return df_final.reset_index(drop=True)



def run_embedding_sensitivity(outdir="./embedding_sensitivity", kmin=2, kmax=10, frac=0.20):
    os.makedirs(outdir, exist_ok=True)

    # === 1) cargar corpus completo ===
    neo = Neo4jClient()
    df_full = neo.fetch(cypher_for_full_corpus())
    print(f"[INFO] Corpus completo: {len(df_full)} papers")
    neo.close()

    # === 2) sample estratificado ===
    df = stratified_sample(df_full, df_full["year"], frac=frac)
    print(f"[INFO] Submuestra: {len(df)} papers (k forzado = 6)")

    # === 3) ejecutar embeddings alternativos ===
    results = {}
    for backend in EMBEDDING_MODELS:
        print(f"\n[RUN] Backend = {backend}")
        an = Analyzer(backend=backend)
        an.set_df(df)
        an.preprocess()
        an.embed()

        # FORZAR k=6 para comparabilidad entre embeddings
        k_fixed = 6
        kmeans = KMeans(n_clusters=k_fixed, random_state=42, n_init=10)
        labels = kmeans.fit_predict(an.X)
        centroids = centroid_cosine_matrix(an.X, labels)

        results[backend] = {
            "labels": labels,
            "k": k_fixed,
            "centroids": centroids
        }

    # === 4) comparar ARI entre modelos ===
    out_rows = []
    keys = list(results.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m1, m2 = keys[i], keys[j]
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(results[m1]["labels"], results[m2]["labels"])

            # centroid drift
            C1 = results[m1]["centroids"].values
            C2 = results[m2]["centroids"].values
            drift = np.mean(1 - cosine_similarity(C1, C2))

            out_rows.append({
                "model_a": m1,
                "model_b": m2,
                "ARI": float(ari),
                "centroid_drift_mean": float(drift),
                "k_a": results[m1]["k"],
                "k_b": results[m2]["k"]
            })

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(os.path.join(outdir, "embedding_sensitivity_results.csv"), index=False)
    print(f"\n[✓] Guardado: {outdir}/embedding_sensitivity_results.csv")

    print("\n=== Sensitivity Summary (k=6) ===")
    print(df_out)

if __name__ == "__main__":
    run_embedding_sensitivity()