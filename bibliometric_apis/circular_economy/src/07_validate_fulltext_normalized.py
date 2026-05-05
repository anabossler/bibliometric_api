import pandas as pd

# === Cargar los dos CSV ===
cross = pd.read_csv("results_plastic/full_corpus/cross_domain_papers.csv")
topics = pd.read_csv("results_plastic/full_corpus/paper_topics.csv")


# === Añadir columna paper_idx si no está (para merge) ===
# Si paper_topics no tiene paper_idx, le asignamos índices en orden
topics = topics.reset_index().rename(columns={"index": "paper_idx"})

# === Unir ===
merged = pd.merge(
    cross,
    topics,
    on="paper_idx",
    how="left"
)

# === Ordenar por similitud y mostrar ===
merged = merged.sort_values("max_sim_to_other", ascending=False)

# Mostrar solo columnas relevantes
cols = ["paper_idx", "doi", "title", "year", "cluster", 
        "assigned_cluster", "n_close_clusters", "max_sim_to_other"]
print(merged[cols].head(60).to_string(index=False))

# === Guardar resultado completo ===
merged[cols].to_csv("cross_domain_papers_with_titles.csv", index=False)
print("\n✅ Archivo guardado: cross_domain_papers_with_titles.csv")
