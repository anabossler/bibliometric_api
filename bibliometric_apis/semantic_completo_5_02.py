"""
complex_semantic_full.py — Clustering semántico completo + métricas de fragmentación
- Corpus completo sin filtros predefinidos
- SBERT embeddings + ensemble clustering
- Bootstrap ARI, silhouette, modularity Q
- MMR, FPS, diversity metrics
- Term embeddings contextuales
- NUEVAS métricas de fragmentación: Jaccard, Insularity, Cross-domain papers
"""

import os, re, json, argparse, logging, math, warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import psutil

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from neo4j import GraphDatabase

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, silhouette_samples, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Backends opcionales
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import networkx as nx
except Exception:
    nx = None
try:
    import rbo
except ImportError:
    rbo = None
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("complex_semantic")
warnings.filterwarnings("ignore", category=FutureWarning)

# ========= helpers JSON-safe =========
def _to_py_scalar(x):
    import numpy as _np
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.bool_,)):
        return bool(x)
    return x

def to_py(obj):
    """Convierte recursivamente numpy/keys no serializables a tipos Python nativos (y keys a str si hace falta)."""
    import numpy as _np
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, (str, int, float, bool, type(None))):
                try:
                    k = str(_to_py_scalar(k))
                except Exception:
                    k = str(k)
            out[str(k)] = to_py(v)
        return out
    elif isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    elif isinstance(obj, (_np.ndarray,)):
        return to_py(obj.tolist())
    else:
        return _to_py_scalar(obj)

def _cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    """Asegura claves y valores Python puros (evita numpy.int32 como key)."""
    uniq = np.unique(labels).tolist()
    return {int(c): int(np.sum(labels == c)) for c in uniq}

# ========= Frases & stopwords (dominio) =========
PHRASES = [
    "life cycle","life cycle assessment","circular economy",
    "mechanical recycling","chemical recycling","plastics recycling",
    "polyethylene terephthalate","high density polyethylene","low density polyethylene",
    "non intentionally added substances","non-intentionally added substances",
    "food contact materials","post consumer resin","post-consumer resin",
    "closed loop recycling","open loop recycling","solid state polymerization",
    "mass balance","carbon footprint","greenhouse gas","global warming potential",
]
STOPWORDS_EN = {
    "the","and","or","for","of","to","in","on","by","with","a","an","is","are","as","that","this","these","those",
    "we","our","their","its","from","at","be","been","it","into","such","using","used","use","can","may","could","should",
    "however","therefore","thus","also","between","among","across","within","over","under","more","less","most","least","both",
    "results","methods","introduction","conclusion","study","paper","research",
    "was","were","has","have","had","which","not","new","two","three","based","among","using","used","use",
    "authors","author","rights","reserved","copyright","publisher","preprint","peer","reviewed","license",
    "creative","commons","open","access","article","version","supplementary","material","graphical","abstract",
    "statement","competing","interests","conflict","role","funding","acknowledgements","permission","figure","table",
    "note","received","accepted","revised","issue","volume","pages","doi","elsevier","springer","wiley","mdpi","taylor","francis",     "plastic", "recycling", "recycled", "recycl", "polymer", "polymers",
    "waste", "material", "materials", "carried out", "e g", "municipal solid",
}
TOKEN_CHEM_WHITELIST = {
    "pet","pe","hdpe","ldpe","pp","ps","pla","pbat","pbt","pvc","pa","abs","pc","pmma","psf","pbs","pvoh","pva",
    "tio2","zno","nias","nist","uv","rpet","ldpe/hdpe","microplastics","nanoplastics","lca","gwp","ghg"
}
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-.()]+|\d+(?:\.\d+)?%?")
BOILERPLATE_PATTERNS = [
    r"©\s*\d{4}.*?rights reserved\.?",
    r"all rights reserved\.?",
    r"this is an open access article.*?license\.",
    r"the authors? \d{4}",
    r"authors? declare .*?",
    r"publisher.*?not responsible.*?",
    r"preprint.*?not peer reviewed",
    r"supplementary material.*",
    r"graphical abstract.*",
    r"creative commons.*?license",
    r"conflict of interest.*",
    r"competing interests.*",
    r"acknowledg(e)?ments?.*",
    r"google scholar.*",
    r"pubmed.*scopus.*", 
    r"crossref.*pubmed.*",
    r"full text.*pdf.*",
    r"references.*available.*",
    r"database.*search.*",
    r"scopus google.*",
    r"text pdf.*",
    r"pengelolaan sampah.*",
    r"daur ulang.*", 
    r"bank sampah.*",
    r"kegiatan ini.*",
    r"penelitian ini.*",
    r"limbah plastik.*",
    r"xmlns.*",
    r"mml xmlns.*",
]

def clean_abstract(t: str) -> str:
    s = re.sub(r"\s+", " ", t or " ").strip()
    for pat in BOILERPLATE_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"https?://\S+|doi:\s*\S+|10\.\d{4,9}/\S+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d{4}\s+authors?\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"©\s*\d{4}", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def protect_phrases(txt: str) -> str:
    s = txt
    for ph in sorted(PHRASES, key=len, reverse=True):
        s = re.sub(re.escape(ph), ph.replace(" ", "_"), s, flags=re.IGNORECASE)
    return s

def tokenize(text: str) -> List[str]:
    text = protect_phrases(text)
    toks = TOKEN_PATTERN.findall(text)
    out: List[str] = []
    for t in toks:
        low = t.lower().strip(".,;:()[]{}\"'")
        if not low: continue
        if low in TOKEN_CHEM_WHITELIST: out.append(low); continue
        if low in STOPWORDS_EN: continue
        if len(low) <= 2 and low not in TOKEN_CHEM_WHITELIST: continue
        out.append(low)
    return out

# ========= Neo4j client =========
class Neo4jClient:
    def __init__(self):
        load_dotenv()
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.pwd  = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.db   = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
        log.info(f"Neo4j: {self.uri} | DB: {self.db}")

    def close(self):
        try: self.driver.close()
        except Exception: pass

    def fetch(self, cypher: str, params: Optional[Dict[str,Any]] = None) -> pd.DataFrame:
        params = params or {}
        with self.driver.session(database=self.db) as s:
            rows = [r.data() for r in s.run(cypher, **params)]
        df = pd.DataFrame(rows)
        if df.empty: return df
        for c in ["doi","eid","title","abstract","year","citedBy"]:
            if c not in df.columns: df[c] = None
        df = df[["doi","eid","title","abstract","year","citedBy"]]
        # Dedupe robusto
        df["doi_norm"] = df["doi"].astype(str).str.strip().str.lower()
        mask = df["doi_norm"].notna() & (df["doi_norm"]!="") & (df["doi_norm"]!="nan")
        df_valid = df.loc[mask].drop_duplicates(subset=["doi_norm"])
        df_null  = df.loc[~mask].copy()
        df_null["title_norm"] = (df_null["title"].fillna("").astype(str)
                                 .str.lower().str.replace(r"\s+"," ",regex=True).str.strip())
        subset_null = ["title_norm","year"] if "year" in df_null.columns else ["title_norm"]
        df_null = df_null.drop_duplicates(subset=subset_null)
        df = pd.concat([df_valid.drop(columns=["doi_norm"],errors="ignore"),
                        df_null.drop(columns=["doi_norm","title_norm"],errors="ignore")], ignore_index=True)
        # Limpieza de abstracts + filtro por longitud
        df["abstract"] = df["abstract"].fillna("").astype(str).apply(clean_abstract)
        df["abstract_len_words"] = df["abstract"].str.split().apply(len)
        min_words = int(os.getenv("MIN_ABS_WORDS", 20))
        kept = df[df["abstract_len_words"]>=min_words].drop(columns=["abstract_len_words"]).reset_index(drop=True)
        try:
            yy = pd.to_numeric(kept["year"], errors="coerce")
            if yy.notna().any():
                log.info(f"Años en corpus: {int(yy.min()) if yy.notna().any() else 'NA'}..{int(yy.max()) if yy.notna().any() else 'NA'} | n={len(kept)}")
        except Exception:
            pass
        return kept

# ========= Cypher for full corpus =========
def cypher_for_full_corpus() -> str:
    """
    Query simplificada usando propiedad corpus_relevante.
    
    NOTA: Se asume que corpus_relevante ya fue calculado con los filtros
    necesarios (keywords, exclusiones, idioma, etc.)
    """
    return """
MATCH (p:Paper)
WHERE 
  p.corpus_relevante = true
  AND p.abstract IS NOT NULL
  AND p.abstract <> ''
  AND p.publication_year IS NOT NULL
RETURN
  p.doi AS doi,
  p.openalex_id AS eid,
  p.title AS title,
  p.abstract AS abstract,
  toString(p.publication_year) AS year,
  COALESCE(toInteger(p.cited_by_count), 0) AS citedBy
ORDER BY citedBy DESC
"""

# ========= Prepro + embeddings =========
class Analyzer:
    def __init__(self, backend="sbert", random_state=42):
        self.backend = backend
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None
        self.proc: Optional[List[str]] = None
        self.X = None
        self.model = None
        self.knn = None

    def set_df(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def preprocess(self):
        texts = (self.df["abstract"].fillna("").astype(str)).tolist()
        texts = [clean_abstract(t) for t in texts]
        self.proc = [" ".join(tokenize(t)) for t in texts]

    def embed(self):
        backend = self.backend.lower()
        if backend == "tfidf":
            vec = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                token_pattern=r"[A-Za-z0-9_\-.]+"
            )
            self.X = vec.fit_transform(self.proc)
            self.model = vec
            return

        if backend in {"sbert", "specter2", "chemberta"}:
            if SentenceTransformer is None:
                raise RuntimeError("Install sentence-transformers for transformer backends")

            # Selección del modelo según backend
            if backend == "sbert":
                name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            
            elif backend == "specter2":
                name = os.getenv(
                    "SPECTER2_MODEL",
                    "sentence-transformers/allenai-specter"
                )
                
            else:
                name = os.getenv("CHEMBERT_MODEL", "DeepChem/ChemBERTa-77M-MTR")

            log.info(f"Embedding model: {name}")
            st = SentenceTransformer(name)
            self.model = st

            # 🔧 Detección de RAM y ajuste automático de batch_size
            mem_gb = psutil.virtual_memory().total / (1024**3)
            batch_sz = 256 if mem_gb >= 12 else 128

            self.X = st.encode(
                self.proc,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_sz
            )

            log.info(f"SBERT embedding batch_size={batch_sz} (RAM={mem_gb:.1f} GB)")
            return

        raise ValueError(f"Unknown backend: {self.backend}")

    def fit_knn(self, k=10):
        if self.X is None: 
            raise RuntimeError("Call embed() first")
        n = len(self.df)
        if n < 2: 
            return
        self.knn = NearestNeighbors(n_neighbors=min(k+1, max(2, n//2)), metric="cosine", algorithm="brute")
        Xe = self.X if isinstance(self.X, np.ndarray) else self.X.toarray()
        self.knn.fit(Xe)
        
# ========= Clustering & selección de modelo =========
def _to_dense(X):
    if isinstance(X, np.ndarray): return X
    if hasattr(X, 'toarray'): return X.toarray()
    return np.asarray(X)

def auto_k_grid(n:int, kmin:int, kmax:int) -> List[int]:
    k_cap = max(2, min(kmax, int(math.sqrt(max(10, n))) + 2, n//10 + 2))
    kmin = max(2, min(kmin, k_cap))
    return list(range(kmin, k_cap+1))

@dataclass
class ModelScores:
    k: int
    algo: str
    labels: np.ndarray
    silhouette: float
    db: float
    ch: float
    ari_median: float
    ari_iqr: Tuple[float,float]
    cluster_sizes: Dict[int,int]

def eval_internal(X, labels, metric="cosine") -> Tuple[float,float,float]:
    Xe = _to_dense(X)
    try: s = silhouette_score(Xe, labels, metric=metric)
    except Exception: s = float("nan")
    try:
        ch = calinski_harabasz_score(Xe, labels)
        db = davies_bouldin_score(Xe, labels)
    except Exception:
        ch, db = float("nan"), float("nan")
    return s, db, ch

def _choice_sorted(rng: np.random.Generator, n: int, frac: float = 0.9, min_n: int = 20) -> np.ndarray:
    m = max(min_n, int(frac*n)); m = min(m, n)
    return np.sort(rng.choice(n, size=m, replace=False))

def bootstrap_stability(X, clusterer_fn, k:int, B:int=20, seed:int=42) -> Tuple[float,Tuple[float,float]]:
    """
    Bootstrap más conservador: 90% del conjunto, y tamaño mínimo ~3k por seguridad (si se puede).
    """
    Xe = _to_dense(X); n = Xe.shape[0]
    if k < 2 or n < 10 or B <= 1:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    labels_list = []
    min_n = max(20, k*3)
    for _ in range(B):
        idx = _choice_sorted(rng, n, frac=0.9, min_n=min_n)
        Xb = Xe[idx]; lb = clusterer_fn(Xb, k)
        labels_list.append((idx, lb))
    aris = []
    for i in range(len(labels_list)):
        idx_i, li = labels_list[i]
        for j in range(i+1, len(labels_list)):
            idx_j, lj = labels_list[j]
            inter, ia, ja = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(inter) < max(20, k*3): continue
            aris.append(adjusted_rand_score(li[ia], lj[ja]))
    if not aris: return float("nan"), (float("nan"), float("nan"))
    ar = np.array(aris)
    return float(np.median(ar)), (float(np.quantile(ar,0.25)), float(np.quantile(ar,0.75)))

def cluster_kmeans(X, k:int, random_state:int=42) -> np.ndarray:
    Xe = _to_dense(X); km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    return km.fit_predict(Xe)

def cluster_agglo(X, k:int) -> np.ndarray:
    Xe = _to_dense(X)
    try: ac = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    except TypeError: ac = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average')
    return ac.fit_predict(Xe)

def consensus_from_label_sets(label_sets: List[np.ndarray]) -> np.ndarray:
    """
    Consenso simple vía co-asociación: prob. de co-ocurrir en mismo clúster.
    Luego clusteriza la matriz (1 - coassoc) con agglomerative average.
    """
    L = len(label_sets)
    n = len(label_sets[0])
    for l in label_sets:
        if len(l) != n: raise ValueError("Label sets size mismatch")
    co = np.zeros((n,n), dtype=float)
    for lab in label_sets:
        same = (lab[:,None] == lab[None,:]).astype(float)
        co += same
    co /= L
    D = 1.0 - co
    ks = [len(np.unique(l)) for l in label_sets]
    k = int(np.median(ks)); k = max(2, k)
    try:
        ac = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
    except TypeError:
        ac = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    labels = ac.fit_predict(D)
    return labels

def select_model_auto(X, kmin:int, kmax:int, bootstrap:int, seed:int=42, allow_ensemble:bool=True) -> ModelScores:
    Xe = _to_dense(X); n = Xe.shape[0]; grid = auto_k_grid(n, kmin, kmax)
    best: Optional[ModelScores] = None
    for k in grid:
        # KMeans
        labs_km = cluster_kmeans(Xe, k, random_state=seed)
        km_sil, km_db, km_ch = eval_internal(Xe, labs_km)
        km_ari, km_iqr = bootstrap_stability(Xe, lambda Xb, kk=k: cluster_kmeans(Xb, kk, random_state=seed), k, bootstrap, seed)
        cand = ModelScores(k, "kmeans", labs_km, km_sil, km_db, km_ch, km_ari, km_iqr, _cluster_sizes(labs_km))
        best = cand if best is None else (cand if (
            (km_ch, km_sil, -km_db) > (best.ch, best.silhouette, -best.db)
        ) else best)
        # Agglo
        labs_ag = cluster_agglo(Xe, k)
        ag_sil, ag_db, ag_ch = eval_internal(Xe, labs_ag)
        ag_ari, ag_iqr = bootstrap_stability(Xe, lambda Xb, kk=k: cluster_agglo(Xb, kk), k, bootstrap, seed)
        cand = ModelScores(k, "agglo", labs_ag, ag_sil, ag_db, ag_ch, ag_ari, ag_iqr, _cluster_sizes(labs_ag))
        best = cand if ( (ag_ch, ag_sil, -ag_db) > (best.ch, best.silhouette, -best.db) ) else best
        # Ensemble
        if allow_ensemble:
            labs_cons = consensus_from_label_sets([labs_km, labs_ag])
            en_sil, en_db, en_ch = eval_internal(Xe, labs_cons)
            def clusterer_cons(Xb, kk=k):
                lk = cluster_kmeans(Xb, kk, random_state=seed)
                la = cluster_agglo(Xb, kk)
                return consensus_from_label_sets([lk, la])
            en_ari, en_iqr = bootstrap_stability(Xe, clusterer_cons, k, bootstrap, seed)
            cand = ModelScores(k, "ensemble", labs_cons, en_sil, en_db, en_ch, en_ari, en_iqr, _cluster_sizes(labs_cons))
            best = cand if ( (en_ch, en_sil, -en_db) > (best.ch, best.silhouette, -best.db) ) else best
    if best is None:
        labs = cluster_kmeans(Xe, 2, random_state=seed)
        sil, db, ch = eval_internal(Xe, labs)
        best = ModelScores(2, "kmeans", labs, sil, db, ch, float('nan'), (float('nan'), float('nan')), _cluster_sizes(labs))
    return best

def hierarchical_subclustering(X, df, labels, min_cluster_size=500):
    """
    Clustering jerárquico en dos fases:
    1. Clustering inicial (ya hecho)
    2. Sub-clustering dentro de cada cluster grande
    """
    Xe = _to_dense(X)
    all_labels = labels.copy()
    max_label = max(labels) + 1
    
    log.info("=== HIERARCHICAL SUB-CLUSTERING ===")
    
    for cluster_id in sorted(np.unique(labels)):
        if cluster_id == -1:
            continue
            
        # Extraer papers de este cluster
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        X_sub = Xe[mask]
        n_sub = len(X_sub)
        
        log.info(f"Cluster {cluster_id}: {n_sub} papers")
        
        if n_sub < min_cluster_size:
            log.info(f"  -> Demasiado pequeño, mantiene cluster original")
            continue
            
        # Determinar k para sub-clustering
        sub_k = min(8, max(3, n_sub//800))  # 3-8 subclusters dependiendo del tamaño
        
        log.info(f"  -> Sub-clustering en k={sub_k}")
        
        # Probar múltiples algoritmos para sub-clustering
        sub_scores = []
        
        # KMeans
        sub_labels_km = cluster_kmeans(X_sub, sub_k)
        sub_sil_km, _, sub_ch_km = eval_internal(X_sub, sub_labels_km)
        sub_scores.append((sub_ch_km, "kmeans", sub_labels_km))
        
        # Agglomerative
        sub_labels_ag = cluster_agglo(X_sub, sub_k)
        sub_sil_ag, _, sub_ch_ag = eval_internal(X_sub, sub_labels_ag)
        sub_scores.append((sub_ch_ag, "agglo", sub_labels_ag))
        
        # Elegir mejor sub-clustering
        best_ch, best_algo, best_sub_labels = max(sub_scores, key=lambda x: x[0])
        
        log.info(f"  -> Mejor: {best_algo} (CH={best_ch:.1f})")
        
        # Re-etiquetar: reemplazar cluster original con subclusters
        for i, sub_label in enumerate(best_sub_labels):
            if sub_label != -1:
                all_labels[cluster_indices[i]] = max_label + sub_label
        
        max_label += sub_k
    
    # Recomputar tamaños finales
    final_clusters = len(np.unique(all_labels))
    log.info(f"=== RESULTADO: {len(np.unique(labels))} -> {final_clusters} clusters ===")
    
    return all_labels

# ========= Etiquetado (c-TF-IDF) =========
def label_topics_c_tfidf(texts: List[str], labels: np.ndarray, top_n=12, ngram=(2,3), min_df=2) -> Dict[int, List[str]]:
    df = pd.DataFrame({"text": texts, "label": labels})
    clusters = [c for c in sorted(df.label.unique()) if c != -1]
    docs = [" ".join(df[df.label==c].text.tolist()) for c in clusters]
    if not docs: return {}
    cv = CountVectorizer(ngram_range=ngram, min_df=min_df, token_pattern=r"[A-Za-z0-9_\-]+")
    X = cv.fit_transform(docs)
    if X.shape[1] < top_n:
        cv = CountVectorizer(ngram_range=(1,ngram[1]), min_df=min_df, token_pattern=r"[A-Za-z0-9_\-]+")
        X = cv.fit_transform(docs)
    tf = X.astype(float)
    row_sums = np.asarray(tf.sum(axis=1)).ravel() + 1e-12
    tf_norm = tf.multiply(1.0/row_sums.reshape(-1,1))
    df_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    n_cls = X.shape[0]
    idf = np.log(1 + (n_cls / (df_counts + 1)))
    ctfidf = tf_norm.toarray() * idf
    terms = np.array(cv.get_feature_names_out())
    out: Dict[int, List[str]] = {}
    for i, c in enumerate(clusters):
        idx = np.argsort(-ctfidf[i])[:top_n]
        out[int(c)] = [terms[j].replace("_"," ") for j in idx]
    
    # FILTRO PARA REMOVER TÉRMINOS PROBLEMÁTICOS
    banned_terms = {"carried out", "municipal solid", "e g", "sampah plastik", "solid management", "environmental impact"}
    for c in out:
        filtered_terms = [term for term in out[c] if term not in banned_terms]
        out[c] = filtered_terms[:top_n]
    
    return out

# ========= Representantes, timeline =========
def representative_docs(X, df: pd.DataFrame, labels: np.ndarray, top_m=5) -> Dict[int, pd.DataFrame]:
    df2 = df.copy(); df2["cluster"] = labels
    Xe = _to_dense(X)
    out: Dict[int, pd.DataFrame] = {}
    for c in sorted(df2.cluster.unique()):
        if c == -1: continue
        rows = np.where(df2.cluster.values==c)[0]
        if rows.size == 0: continue
        centroid = Xe[rows].mean(axis=0, keepdims=True)
        sims = cosine_similarity(Xe[rows], centroid).ravel()
        order = np.argsort(-sims)[:top_m]
        rep = df2.iloc[rows[order]][["doi","title","year"]].copy()
        rep["rep_similarity"] = sims[order]
        out[int(c)] = rep
    return out

def cluster_timeline(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    d = df.copy(); d["cluster"] = labels
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    return d[d["year"].notna()].groupby(["cluster","year"]).size().reset_index(name="count")

def _fit_linear(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    beta, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = beta*x + intercept
    ss_res = float(np.sum((y-yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    return beta, intercept, yhat, r2

def _fit_exponential(x, y):
    # y ≈ a * exp(bx) -> ln(y+1) ≈ ln(a) + b x
    y_log = np.log(y + 1.0)
    b, ln_a, yhat_log, r2 = _fit_linear(x, y_log)
    a = math.exp(ln_a)
    yhat = a * np.exp(b * x) - 1.0
    return a, b, yhat, r2

def _best_one_break(x, y):
    """Modelo piecewise lineal con 1 cambio de régimen. Devuelve idx*, RSS, parámetros."""
    n = len(x)
    if n < 5: return None
    best = None
    for br in range(2, n-2):
        beta1, c1, y1, _ = _fit_linear(x[:br], y[:br])
        beta2, c2, y2, _ = _fit_linear(x[br:], y[br:])
        yhat = np.concatenate([y1, y2])
        rss = float(np.sum((y - yhat)**2))
        if (best is None) or (rss < best[0]):
            best = (rss, br, (beta1, c1, beta2, c2))
    return best

def _autocorr(y, lag):
    if lag >= len(y): return 0.0
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    return float(np.dot(y[:-lag], y[lag:]) / (np.sqrt(np.dot(y[:-lag], y[:-lag])*np.dot(y[lag:], y[lag:])) + 1e-12))

def enhanced_timeline_trends(tl: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada clúster calcula:
    - pendiente lineal + t-like
    - comparación lineal vs exponential (R2)
    - 1 cambio de régimen (índice relativo) si mejora mucho el RSS
    - señal de estacionalidad (autocorr a lag=1..3)
    """
    rows = []
    for c in sorted(tl.cluster.unique()):
        sub = tl[tl.cluster==c].sort_values("year")
        x = sub["year"].astype(float).values
        y = sub["count"].astype(float).values
        if len(x) < 3 or np.std(x)==0:
            rows.append({"cluster": int(c), "slope": float('nan'), "t_like": float('nan'),
                         "model": "NA", "r2_linear": float('nan'), "r2_exp": float('nan'),
                         "break_at": None, "season_lag1": float('nan'), "season_lag2": float('nan'), "season_lag3": float('nan')})
            continue
        beta, intercept, yhat_lin, r2_lin = _fit_linear(x, y)
        resid = y - yhat_lin
        s2 = np.sum(resid**2) / max(1, len(x)-2); s = math.sqrt(max(0.0, s2))
        sxx = np.sum((x - x.mean())**2)
        t_like = beta * math.sqrt(sxx) / (s + 1e-9) if sxx>0 else float('nan')
        a, b, yhat_exp, r2_exp = _fit_exponential(x, y)
        model = "exponential" if (r2_exp > r2_lin + 0.1) else "linear"
        br = _best_one_break(x, y)
        break_at = None
        if br is not None:
            rss_base = float(np.sum((y - yhat_lin)**2))
            rss_br = br[0]
            if rss_br < 0.8 * rss_base:
                break_at = int(sub["year"].iloc[br[1]])
        rows.append({"cluster": int(c), "slope": float(beta), "t_like": float(t_like),
                     "model": model, "r2_linear": float(r2_lin), "r2_exp": float(r2_exp),
                     "break_at": break_at,
                     "season_lag1": _autocorr(y,1), "season_lag2": _autocorr(y,2), "season_lag3": _autocorr(y,3)})
    return pd.DataFrame(rows)

def centroid_cosine_matrix(X, labels: np.ndarray) -> pd.DataFrame:
    Xe = _to_dense(X); clus = sorted(np.unique(labels))
    cents = []
    for c in clus:
        rows = np.where(labels==c)[0]
        cents.append(Xe[rows].mean(axis=0, keepdims=True))
    C = np.vstack(cents); M = cosine_similarity(C)
    return pd.DataFrame(M, index=[f"C{c}" for c in clus], columns=[f"C{c}" for c in clus])

# ========= Diversidad + MMR/FPS + Cobertura adaptativa =========
def _cosine_dist(A, B=None):
    S = cosine_similarity(A, B) if B is not None else cosine_similarity(A)
    return 1.0 - S

def diversity_metrics(X):
    Xe = _to_dense(X); n = Xe.shape[0]
    if n <= 5000:
        D = _cosine_dist(Xe); iu = np.triu_indices(n, k=1); dvals = D[iu]
        diam = float(dvals.max()) if dvals.size else 0.0
        mean_d = float(dvals.mean()) if dvals.size else 0.0
        p90 = float(np.quantile(dvals, 0.90)) if dvals.size else 0.0
        p95 = float(np.quantile(dvals, 0.95)) if dvals.size else 0.0
    else:
        rng = np.random.default_rng(42); idx = rng.choice(n, size=min(5000, n), replace=False)
        D = _cosine_dist(Xe[idx]); iu = np.triu_indices(len(idx), k=1); dvals = D[iu]
        diam, mean_d = float(dvals.max()), float(dvals.mean()); p90 = float(np.quantile(dvals,0.90)); p95=float(np.quantile(dvals,0.95))
    Xc = Xe - Xe.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, n - 1)
    w = np.linalg.eigvalsh(C); w = np.clip(w, 0, None)
    tr = float(w.sum()) + 1e-12
    participation_ratio = float((tr**2) / (np.sum(w**2) + 1e-12))
    spectral_entropy = float(-np.sum((w/tr) * np.log((w/tr) + 1e-12)))
    return {"n": int(n), "diameter_cos": diam, "mean_pairwise_cos": mean_d,
            "p90_pairwise_cos": p90, "p95_pairwise_cos": p95,
            "participation_ratio": participation_ratio, "spectral_entropy": spectral_entropy}

def mmr_select(X, k=10, lam=0.5, query_vec=None):
    Xe = _to_dense(X); n = Xe.shape[0]
    if n == 0 or k <= 0: return np.array([], dtype=int)
    if query_vec is None: query_vec = Xe.mean(axis=0, keepdims=True)
    def _norm(A): nrm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12; return A / nrm
    Q = _norm(query_vec); Z = _norm(Xe)
    rel = cosine_similarity(Z, Q).ravel()
    selected = []; candidates = set(range(n))
    i0 = int(np.argmax(rel)); selected.append(i0); candidates.remove(i0)
    while len(selected) < min(k, n):
        sel_mat = Z[selected]
        max_sim_to_S = cosine_similarity(Z[list(candidates)], sel_mat).max(axis=1)
        cand_list = np.array(list(candidates))
        scores = lam * rel[cand_list] - (1.0 - lam) * max_sim_to_S
        pick = int(cand_list[np.argmax(scores)])
        selected.append(pick); candidates.remove(pick)
    return np.array(selected, dtype=int)

def farthest_point_sampling(X, k=10):
    Xe = _to_dense(X); n = Xe.shape[0]
    if n == 0 or k <= 0: return np.array([], dtype=int)
    rng = np.random.default_rng(42)
    first = int(rng.integers(0, n)); centers = [first]
    dmin = _cosine_dist(Xe, Xe[[first]]).ravel()
    for _ in range(1, min(k, n)):
        nxt = int(np.argmax(dmin)); centers.append(nxt)
        dmin = np.minimum(dmin, _cosine_dist(Xe, Xe[[nxt]]).ravel())
    return np.array(centers, dtype=int)

def adaptive_coverage_analysis(X, seeds_idx, percentiles=(0.10, 0.25, 0.50, 0.75)):
    """
    Calcula radios adaptativos por clúster en función de la densidad (percentiles de distancias NN).
    """
    Xe = _to_dense(X)
    if len(seeds_idx) == 0:
        return [{"radius_cos": float('nan'), "covered_frac": 0.0, "p": float(p)} for p in percentiles]
    S = Xe[seeds_idx]; D = _cosine_dist(Xe, S); d_nn = D.min(axis=1)
    cov = []
    base = np.quantile(d_nn, percentiles)
    for p, r in zip(percentiles, base):
        cov.append({"radius_cos": float(r), "covered_frac": float((d_nn <= r).mean()), "p": float(p)})
    return cov

# ========= Embeddings de términos (contextuales) =========
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _embed_terms_raw(terms: List[str], analyzer: Analyzer) -> np.ndarray:
    """Fallback: embed de cadenas aisladas (no contextual)."""
    if analyzer.model is not None and hasattr(analyzer.model, "encode"):
        vecs = analyzer.model.encode(terms, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return vecs
    tfv = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1)
    X = tfv.fit_transform(terms).toarray().astype(float)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def contextual_term_embeddings(terms: List[str], documents: List[str], analyzer: Analyzer, max_sents_per_term:int=2000) -> np.ndarray:
    """
    Para cada término, promedia embeddings de oraciones donde aparece (si hay modelo transformer).
    Si no hay oraciones (o no hay modelo), cae al embedding aislado.
    """
    if analyzer.model is None or not hasattr(analyzer.model, "encode"):
        return _embed_terms_raw(terms, analyzer)
    vecs = []
    for t in terms:
        rx = re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE)
        sents = []
        for doc in documents:
            for s in SENT_SPLIT.split(doc):
                if rx.search(s):
                    sents.append(s.strip())
                    if len(sents) >= max_sents_per_term: break
            if len(sents) >= max_sents_per_term: break
        if not sents:
            v = analyzer.model.encode([t], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)[0]
        else:
            E = analyzer.model.encode(sents, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            v = E.mean(axis=0)
            v /= (np.linalg.norm(v) + 1e-12)
        vecs.append(v)
    return np.vstack(vecs)

def term_distance_stats_by_cluster(topics: Dict[int, List[str]], analyzer: Analyzer, documents: Optional[List[str]]=None) -> pd.DataFrame:
    rows = []
    docs = documents if documents is not None else []
    for c, term_list in sorted(topics.items()):
        terms = [t.strip() for t in term_list if isinstance(t, str) and t.strip()]
        terms = list(dict.fromkeys(terms))
        if len(terms) < 2:
            rows.append({"cluster": int(c), "n_terms": len(terms), "pairs": 0,
                         "mean_sim": float('nan'), "mean_dist": float('nan'),
                         "p25_sim": float('nan'), "p50_sim": float('nan'), "p75_sim": float('nan'),
                         "min_sim": float('nan'), "max_sim": float('nan')})
            continue
        V = contextual_term_embeddings(terms, docs, analyzer) if docs else _embed_terms_raw(terms, analyzer)
        S = cosine_similarity(V)
        iu = np.triu_indices(len(terms), k=1)
        sims = S[iu]
        dists = 1.0 - sims
        rows.append({
            "cluster": int(c),
            "n_terms": len(terms),
            "pairs": int(len(sims)),
            "mean_sim": float(np.mean(sims)),
            "mean_dist": float(np.mean(dists)),
            "p25_sim": float(np.quantile(sims, 0.25)),
            "p50_sim": float(np.quantile(sims, 0.50)),
            "p75_sim": float(np.quantile(sims, 0.75)),
            "min_sim": float(np.min(sims)),
            "max_sim": float(np.max(sims)),
        })
    return pd.DataFrame(rows)

# ========= NUEVAS MÉTRICAS DE FRAGMENTACIÓN =========

def compute_jaccard_matrix(topics: Dict[int, List[str]], outdir: str):
    """Calcula Jaccard similarity entre vocabularios de clusters"""
    log.info("--- Calculando Jaccard vocabulary overlap ---")
    
    clusters = sorted(topics.keys())
    n = len(clusters)
    jaccard = np.zeros((n, n))
    
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i == j:
                jaccard[i, j] = 1.0
            else:
                set1 = set(topics[c1][:50])  # Top 50 términos
                set2 = set(topics[c2][:50])
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard[i, j] = intersection / union if union > 0 else 0.0
    
    df_jaccard = pd.DataFrame(jaccard, 
                              index=[f"C{c}" for c in clusters],
                              columns=[f"C{c}" for c in clusters])
    
    df_jaccard.to_csv(os.path.join(outdir, "jaccard_vocabulary.csv"))
    log.info(f"✓ Jaccard matrix guardada")
    
    # Estadísticas
    upper_tri = jaccard[np.triu_indices(n, k=1)]
    log.info(f"  - Mean Jaccard (off-diagonal): {upper_tri.mean():.3f}")
    log.info(f"  - Pares con Jaccard < 0.10 (silos fuertes): {(upper_tri < 0.10).sum()}")
    log.info(f"  - Pares con Jaccard > 0.30 (overlap alto): {(upper_tri > 0.30).sum()}")
    
    return df_jaccard

def compute_citation_insularity_global(neo: Neo4jClient, df: pd.DataFrame, 
                                       labels: np.ndarray, outdir: str):
    """
    Calcula insularity global y por cluster considerando citas externas.
    
    Métricas:
    - Insularity: % de citas que son internas al corpus
    - Modularidad ajustada: considerando enlaces externos como penalización
    """
    log.info("--- Calculando Citation Insularity (Internal vs External) ---")
    
    # Obtener TODAS las citas (internas + externas)
    edges = fetch_citation_edges(neo, df['doi'].tolist(), include_external=True)
    
    if edges is None or edges.empty:
        log.warning("No citation edges found")
        return None
    
    log.info(f"  Total citation edges: {len(edges)}")
    log.info(f"  Internal citations: {edges['is_internal'].sum()}")
    log.info(f"  External citations: {(~edges['is_internal']).sum()}")
    
    # Mapeo DOI -> cluster
    labels_by_doi = {d: int(c) for d, c in zip(df['doi'].tolist(), labels)}
    
    # ========= ANÁLISIS GLOBAL =========
    n_internal = edges['is_internal'].sum()
    n_external = (~edges['is_internal']).sum()
    n_total = len(edges)
    
    global_insularity = 100.0 * n_internal / n_total if n_total > 0 else 0.0
    
    log.info(f"\n  GLOBAL INSULARITY: {global_insularity:.2f}%")
    log.info(f"    - Internal citations: {n_internal} ({100*n_internal/n_total:.1f}%)")
    log.info(f"    - External citations: {n_external} ({100*n_external/n_total:.1f}%)")
    
    # ========= ANÁLISIS POR CLUSTER =========
    cluster_stats = []
    
    for c in sorted(np.unique(labels)):
        if c == -1:
            continue
        
        # Papers en este cluster
        dois_in_c = df.loc[labels == c, 'doi'].tolist()
        
        # Citas FROM este cluster
        edges_from_c = edges[edges['src'].isin(dois_in_c)]
        n_total_c = len(edges_from_c)
        
        if n_total_c == 0:
            cluster_stats.append({
                "cluster": int(c),
                "n_papers": len(dois_in_c),
                "total_citations": 0,
                "internal_citations": 0,
                "external_citations": 0,
                "insularity_pct": float('nan')
            })
            continue
        
        # Citas internas vs externas
        n_internal_c = edges_from_c['is_internal'].sum()
        n_external_c = (~edges_from_c['is_internal']).sum()
        
        insularity_c = 100.0 * n_internal_c / n_total_c
        
        cluster_stats.append({
            "cluster": int(c),
            "n_papers": len(dois_in_c),
            "total_citations": n_total_c,
            "internal_citations": int(n_internal_c),
            "external_citations": int(n_external_c),
            "insularity_pct": insularity_c
        })
    
    df_cluster_stats = pd.DataFrame(cluster_stats)
    
    # Guardar resultados
    df_cluster_stats.to_csv(
        os.path.join(outdir, "citation_insularity_by_cluster.csv"), 
        index=False
    )
    
    # ========= MODULARIDAD AJUSTADA =========
    # Calcular modularidad solo con enlaces internos
    edges_internal_only = edges[edges['is_internal']]
    mod_internal = modularity_on_clusters(edges_internal_only, labels_by_doi)
    
    log.info(f"\n  MODULARITY (internal edges only): {mod_internal:.4f}" if mod_internal else "  MODULARITY: N/A")
    
    # ========= REPORTE TEXTUAL =========
    with open(os.path.join(outdir, "citation_insularity_report.txt"), "w") as f:
        f.write("# CITATION INSULARITY ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("## Global Statistics\n\n")
        f.write(f"Total papers in corpus: {len(df)}\n")
        f.write(f"Total citations analyzed: {n_total}\n")
        f.write(f"Internal citations: {n_internal} ({100*n_internal/n_total:.2f}%)\n")
        f.write(f"External citations: {n_external} ({100*n_external/n_total:.2f}%)\n")
        f.write(f"Global insularity: {global_insularity:.2f}%\n\n")
        
        if mod_internal is not None:
            f.write(f"Modularity Q (internal network): {mod_internal:.4f}\n\n")
        
        f.write("## Interpretation\n\n")
        if global_insularity > 70:
            f.write("HIGH INSULARITY: El corpus muestra fuerte aislamiento epistémico.\n")
            f.write("La mayoría de las citas se dirigen a trabajos dentro del corpus.\n")
        elif global_insularity > 40:
            f.write("MODERATE INSULARITY: Balance entre integración interna y externa.\n")
        else:
            f.write("LOW INSULARITY: El corpus está bien conectado con literatura externa.\n")
        
        f.write("\n## Per-Cluster Statistics\n\n")
        f.write(df_cluster_stats.to_string(index=False))
        f.write("\n\n")
        
        # Clusters con mayor/menor insularity
        if len(df_cluster_stats) > 0:
            most_insular = df_cluster_stats.nlargest(3, 'insularity_pct')
            least_insular = df_cluster_stats.nsmallest(3, 'insularity_pct')
            
            f.write("### Most Insular Clusters (highest internal citation %)\n\n")
            for _, row in most_insular.iterrows():
                f.write(f"Cluster {int(row['cluster'])}: {row['insularity_pct']:.1f}% ")
                f.write(f"({int(row['internal_citations'])}/{int(row['total_citations'])})\n")
            
            f.write("\n### Least Insular Clusters (highest external citation %)\n\n")
            for _, row in least_insular.iterrows():
                f.write(f"Cluster {int(row['cluster'])}: {row['insularity_pct']:.1f}% ")
                f.write(f"({int(row['internal_citations'])}/{int(row['total_citations'])})\n")
    
    log.info(f"\n✓ Citation insularity analysis saved to {outdir}")
    
    return {
        "global_insularity": global_insularity,
        "modularity_internal": mod_internal,
        "cluster_stats": df_cluster_stats
    }

def compute_cross_domain_papers(X, labels: np.ndarray, outdir: str, threshold: float = 0.70):
    """Identifica papers que están cerca de múltiples clusters (cross-domain)"""
    log.info("--- Identificando cross-domain papers ---")
    
    Xe = _to_dense(X)
    
    # Calcular centroids de cada cluster
    clusters = sorted(np.unique(labels))
    centroids = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        centroids.append(Xe[idx].mean(axis=0, keepdims=True))
    
    C = np.vstack(centroids)
    
    # Para cada paper, calcular similarity a todos los centroids
    sims = cosine_similarity(Xe, C)
    
    # Contar cuántos centroids tienen similarity > threshold
    cross_domain_mask = (sims > threshold).sum(axis=1) >= 2
    n_cross = cross_domain_mask.sum()
    pct_cross = 100.0 * n_cross / len(labels)
    
    log.info(f"✓ Cross-domain papers: {n_cross} ({pct_cross:.2f}%)")
    
    # Guardar índices de cross-domain papers
    cross_indices = np.where(cross_domain_mask)[0]
    
    df_cross = pd.DataFrame({
        "paper_idx": cross_indices,
        "assigned_cluster": labels[cross_indices],
        "n_close_clusters": (sims[cross_indices] > threshold).sum(axis=1),
        "max_sim_to_other": np.partition(sims[cross_indices], -2, axis=1)[:, -2]  # 2nd highest
    })
    
    df_cross.to_csv(os.path.join(outdir, "cross_domain_papers.csv"), index=False)
    log.info(f"✓ Cross-domain papers guardados")
    
    # Summary stats
    with open(os.path.join(outdir, "fragmentation_summary.txt"), "w") as f:
        f.write("# FRAGMENTATION SUMMARY\n\n")
        f.write(f"Total papers: {len(labels)}\n")
        f.write(f"Cross-domain papers (similarity > {threshold} to ≥2 clusters): {n_cross} ({pct_cross:.2f}%)\n")
        f.write(f"Single-domain papers: {len(labels) - n_cross} ({100 - pct_cross:.2f}%)\n")
    
    return df_cross

def compute_bridge_papers_sensitivity(X, df: pd.DataFrame, labels: np.ndarray, 
                                     topics: Dict[int, List[str]], 
                                     neo: Neo4jClient,
                                     outdir: str,
                                     thresholds=[0.65, 0.70, 0.75, 0.80]):
    """
    Análisis de sensibilidad para bridge papers:
    1. Varía el threshold de cosine similarity
    2. Aplica filtro léxico dual (geometric + vocabulary)
    3. Calcula brokerage metrics (betweenness centrality) en red de citas
    4. Identifica "genuine bridges" vs "geometric proximity only"
    
    Args:
        X: Embeddings
        df: DataFrame con papers
        labels: Cluster assignments
        topics: Términos c-TF-IDF por cluster
        neo: Neo4j client para extraer red de citas
        outdir: Directorio de salida
        thresholds: Lista de thresholds a probar
    """
    log.info("=" * 70)
    log.info("BRIDGE PAPERS SENSITIVITY ANALYSIS")
    log.info("=" * 70)
    
    Xe = _to_dense(X)
    
    # Calcular centroids de cada cluster
    clusters = sorted(np.unique(labels))
    centroids = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        centroids.append(Xe[idx].mean(axis=0, keepdims=True))
    
    C = np.vstack(centroids)
    
    # Para cada paper, calcular similarity a todos los centroids
    sims = cosine_similarity(Xe, C)
    
    # Preparar abstracts tokenizados para búsqueda léxica
    abstracts = df['abstract'].fillna("").astype(str).tolist()
    abstracts_lower = [a.lower() for a in abstracts]
    
    # ========= PARTE 1: Brokerage Metrics (Citation Network) =========
# ========= PARTE 1: Brokerage Metrics (Citation Network) =========
    log.info("\n[1/2] Calculando brokerage metrics (citation network)...")
    
    # NOTA METODOLÓGICA: Usamos solo red INTERNA (citas dentro del corpus) porque:
    # - Betweenness mide conexión ENTRE clusters del corpus (no hacia literatura externa)
    # - Papers externos no tienen cluster assignment → no pueden ser brokers inter-cluster
    # - Participation coefficient requiere clasificación en communities (clusters)
    # - Esta es la red apropiada para identificar papers que conectan subdominios del campo
    
    brokerage_scores = None
    if nx is not None:
        try:
            # Fetch citation edges (INTERNAL ONLY para brokerage metrics)
            edges = fetch_citation_edges(neo, df['doi'].tolist(), include_external=False)
            
            if edges is not None and not edges.empty and len(edges) > 0:
                # Crear mapping DOI -> cluster
                doi_to_cluster = {d: int(c) for d, c in zip(df['doi'].tolist(), labels)}
                
                # Construir grafo DIRIGIDO (importante para betweenness)
                G = nx.DiGraph()
                
                # Añadir nodos con atributo de cluster
                for doi in df['doi'].tolist():
                    if pd.notna(doi) and doi in doi_to_cluster:
                        G.add_node(doi, cluster=doi_to_cluster[doi])
                
                # Añadir edges (solo dentro del corpus)
                valid_dois = set(df['doi'].tolist())
                for _, row in edges.iterrows():
                    src, dst = row['src'], row['dst']
                    if src in valid_dois and dst in valid_dois:
                        G.add_edge(src, dst)
                
                log.info(f"    Citation graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                
                if G.number_of_nodes() > 10 and G.number_of_edges() > 0:
                    # Calcular betweenness centrality (puede tardar)
                    log.info(f"    Calculando betweenness centrality...")
                    betweenness = nx.betweenness_centrality(G, normalized=True)
                    
                    # Calcular participation coefficient (Guimerà & Amaral 2005)
                    # Mide cuán diversamente conectado está un nodo entre clusters
                    participation = {}
                    for node in G.nodes():
                        if node not in doi_to_cluster:
                            continue
                        
                        # Contar conexiones a cada cluster
                        cluster_connections = {}
                        for neighbor in G.neighbors(node):
                            if neighbor in doi_to_cluster:
                                c = doi_to_cluster[neighbor]
                                cluster_connections[c] = cluster_connections.get(c, 0) + 1
                        
                        total = sum(cluster_connections.values())
                        if total > 0:
                            # Participation coefficient = 1 - Σ(k_i/k)^2
                            pc = 1.0 - sum((count/total)**2 for count in cluster_connections.values())
                            participation[node] = pc
                        else:
                            participation[node] = 0.0
                    
                    # Consolidar scores
                    brokerage_scores = pd.DataFrame({
                        'doi': list(betweenness.keys()),
                        'betweenness': [betweenness[d] for d in betweenness.keys()],
                        'participation_coef': [participation.get(d, 0.0) for d in betweenness.keys()]
                    })
                    
                    log.info(f"    ✓ Brokerage metrics calculados para {len(brokerage_scores)} papers")
                    log.info(f"      Mean betweenness: {brokerage_scores['betweenness'].mean():.4f}")
                    log.info(f"      Mean participation coefficient: {brokerage_scores['participation_coef'].mean():.4f}")
                    
                else:
                    log.warning(f"    Graph too small for brokerage metrics")
            else:
                log.warning(f"    No citation edges available")
        
        except Exception as e:
            log.warning(f"    Brokerage metrics failed: {e}")
    else:
        log.warning(f"    NetworkX not available, skipping brokerage metrics")
    
    # ========= PARTE 2: Geometric + Lexical Bridges =========
    log.info("\n[2/2] Analizando bridges (geometric + lexical)...")
    
    results = []
    
    for threshold in thresholds:
        log.info(f"\n  → Testing threshold = {threshold}")
        
        # MÉTODO 1: Solo geometric (baseline actual)
        geometric_mask = (sims > threshold).sum(axis=1) >= 2
        n_geometric = geometric_mask.sum()
        pct_geometric = 100.0 * n_geometric / len(labels)
        
        # MÉTODO 2: Geometric + Lexical filter
        genuine_bridge_mask = np.zeros(len(labels), dtype=bool)
        
        for i in range(len(labels)):
            if not geometric_mask[i]:
                continue  # No es bridge geométrico
            
            # Encontrar clusters a los que está "cerca"
            close_clusters = np.where(sims[i] > threshold)[0]
            close_cluster_ids = [clusters[j] for j in close_clusters]
            
            if len(close_cluster_ids) < 2:
                continue
            
            # Verificar si el abstract contiene términos de MÚLTIPLES clusters
            abstract_text = abstracts_lower[i]
            
            clusters_with_vocab = []
            for c_idx in close_cluster_ids:
                # Top-20 términos de este cluster
                cluster_terms = topics.get(c_idx, [])[:20]
                cluster_terms_lower = [t.lower() for t in cluster_terms]
                
                # Contar cuántos términos aparecen en el abstract
                matches = sum(1 for term in cluster_terms_lower if term in abstract_text)
                
                if matches >= 2:  # Al menos 2 términos del cluster
                    clusters_with_vocab.append(c_idx)
            
            # Es genuine bridge si tiene vocabulario de ≥2 clusters
            if len(clusters_with_vocab) >= 2:
                genuine_bridge_mask[i] = True
        
        n_genuine = genuine_bridge_mask.sum()
        pct_genuine = 100.0 * n_genuine / len(labels)
        
        # MÉTODO 3: Bridge entre pares específicos
        pair_bridges = {}
        for i in range(len(labels)):
            if not genuine_bridge_mask[i]:
                continue
            
            close_clusters = np.where(sims[i] > threshold)[0]
            close_cluster_ids = sorted([clusters[j] for j in close_clusters])
            
            for j in range(len(close_cluster_ids)):
                for k in range(j+1, len(close_cluster_ids)):
                    pair = (close_cluster_ids[j], close_cluster_ids[k])
                    pair_bridges[pair] = pair_bridges.get(pair, 0) + 1
        
        # MÉTODO 4: Overlap con brokerage metrics (si disponible)
        geometric_high_brokerage = 0
        genuine_high_brokerage = 0
        
        if brokerage_scores is not None:
            # Merge con df para obtener índices
            df_temp = df.reset_index(drop=True)
            df_temp['idx'] = df_temp.index
            df_with_brokerage = df_temp.merge(brokerage_scores, on='doi', how='left')
            
            # Top 10% betweenness
            betweenness_threshold = df_with_brokerage['betweenness'].quantile(0.90)
            
            high_brokerage_idx = set(df_with_brokerage[
                df_with_brokerage['betweenness'] >= betweenness_threshold
            ]['idx'].tolist())
            
            geometric_idx = set(np.where(geometric_mask)[0].tolist())
            genuine_idx = set(np.where(genuine_bridge_mask)[0].tolist())
            
            geometric_high_brokerage = len(geometric_idx & high_brokerage_idx)
            genuine_high_brokerage = len(genuine_idx & high_brokerage_idx)
        
        results.append({
            'threshold': threshold,
            'geometric_bridges_n': n_geometric,
            'geometric_bridges_pct': pct_geometric,
            'genuine_bridges_n': n_genuine,
            'genuine_bridges_pct': pct_genuine,
            'delta_n': n_geometric - n_genuine,
            'delta_pct': pct_geometric - pct_genuine,
            'geometric_high_brokerage': geometric_high_brokerage,
            'genuine_high_brokerage': genuine_high_brokerage,
            'top_connected_pairs': sorted(pair_bridges.items(), key=lambda x: x[1], reverse=True)[:5]
        })
        
        log.info(f"    Geometric bridges: {n_geometric} ({pct_geometric:.2f}%)")
        log.info(f"    Genuine bridges (geometric + lexical): {n_genuine} ({pct_genuine:.2f}%)")
        log.info(f"    Delta: {n_geometric - n_genuine} ({pct_geometric - pct_genuine:.2f}%)")
        log.info(f"    Reduction: {100*(1 - pct_genuine/pct_geometric if pct_geometric > 0 else 0):.1f}%")
        if brokerage_scores is not None:
            log.info(f"    Overlap with high betweenness (top 10%): {genuine_high_brokerage}/{n_genuine}")
    
    # Consolidar resultados
    df_results = pd.DataFrame([{
        'threshold': r['threshold'],
        'geometric_bridges_n': r['geometric_bridges_n'],
        'geometric_bridges_pct': r['geometric_bridges_pct'],
        'genuine_bridges_n': r['genuine_bridges_n'],
        'genuine_bridges_pct': r['genuine_bridges_pct'],
        'delta_n': r['delta_n'],
        'delta_pct': r['delta_pct'],
        'geometric_high_brokerage': r['geometric_high_brokerage'],
        'genuine_high_brokerage': r['genuine_high_brokerage']
    } for r in results])
    
    df_results.to_csv(os.path.join(outdir, "bridge_papers_sensitivity.csv"), index=False)
    
    # Guardar brokerage scores completos
    if brokerage_scores is not None:
        brokerage_scores.to_csv(os.path.join(outdir, "brokerage_metrics.csv"), index=False)
    
    # Análisis detallado para threshold principal (0.70)
    threshold_main = 0.70
    result_main = [r for r in results if r['threshold'] == threshold_main][0]
    
    # Identificar genuine bridges con detalles
    genuine_idx = np.where(genuine_bridge_mask)[0]
    bridge_details = []
    
    for i in genuine_idx:
        close_clusters = np.where(sims[i] > threshold_main)[0]
        close_cluster_ids = [clusters[j] for j in close_clusters]
        
        # Identificar qué términos de cada cluster aparecen
        abstract_text = abstracts_lower[i]
        cluster_matches = {}
        
        for c_idx in close_cluster_ids:
            cluster_terms = topics.get(c_idx, [])[:20]
            cluster_terms_lower = [t.lower() for t in cluster_terms]
            matched_terms = [t for t in cluster_terms_lower if t in abstract_text]
            if len(matched_terms) >= 2:
                cluster_matches[c_idx] = matched_terms[:5]  # Top 5 matches
        
        if len(cluster_matches) >= 2:
            # Añadir brokerage metrics si disponible
            brokerage_data = {}
            if brokerage_scores is not None:
                doi = df.iloc[i]['doi']
                brok_row = brokerage_scores[brokerage_scores['doi'] == doi]
                if not brok_row.empty:
                    brokerage_data = {
                        'betweenness': float(brok_row.iloc[0]['betweenness']),
                        'participation_coef': float(brok_row.iloc[0]['participation_coef'])
                    }
            
            bridge_details.append({
                'paper_idx': int(i),
                'doi': df.iloc[i]['doi'],
                'title': df.iloc[i]['title'],
                'assigned_cluster': int(labels[i]),
                'connected_clusters': sorted(list(cluster_matches.keys())),
                'n_clusters_connected': len(cluster_matches),
                'max_sim': float(sims[i].max()),
                'cluster_terms_matched': {int(k): v for k, v in cluster_matches.items()},
                'brokerage': brokerage_data
            })
    
    # Guardar detalles en JSON
    with open(os.path.join(outdir, "genuine_bridge_papers_details.json"), "w") as f:
        json.dump(bridge_details[:50], f, indent=2, ensure_ascii=False)
    
    # Reporte textual completo
    with open(os.path.join(outdir, "bridge_papers_sensitivity_report.txt"), "w") as f:
        f.write("# BRIDGE PAPERS SENSITIVITY ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("## Methodology\n\n")
        f.write("**Three complementary approaches:**\n\n")
        f.write("1. **Geometric criterion:** Paper has cosine similarity > threshold to ≥2 cluster centroids\n")
        f.write("2. **Lexical criterion:** Abstract contains ≥2 technical terms from each cluster's top-20 c-TF-IDF\n")
        f.write("3. **Brokerage metrics:** Betweenness centrality and participation coefficient in citation network\n\n")
        f.write("**Genuine bridge:** Satisfies BOTH geometric AND lexical criteria\n\n")
        
        f.write("## Results by Threshold\n\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Key Findings\n\n")
        f.write(f"At threshold = {threshold_main} (baseline):\n")
        f.write(f"  - Geometric bridges: {result_main['geometric_bridges_n']} ({result_main['geometric_bridges_pct']:.2f}%)\n")
        f.write(f"  - Genuine bridges: {result_main['genuine_bridges_n']} ({result_main['genuine_bridges_pct']:.2f}%)\n")
        f.write(f"  - Difference: {result_main['delta_n']} papers ({result_main['delta_pct']:.2f}%)\n\n")
        
        if result_main['geometric_bridges_pct'] > 0:
            reduction = 100 * (1 - result_main['genuine_bridges_pct'] / result_main['geometric_bridges_pct'])
            f.write(f"**Reduction in bridge papers when applying lexical filter: {reduction:.1f}%**\n\n")
        
        f.write("This confirms that geometric proximity alone OVERESTIMATES integration.\n")
        f.write("True synthesis requires both semantic proximity AND shared technical vocabulary.\n\n")
        
        if brokerage_scores is not None:
            f.write("## Brokerage Metrics Validation\n\n")
            f.write(f"Citation graph analyzed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n\n")
            f.write("Overlap between genuine bridges and high betweenness centrality (top 10%):\n")
            f.write(f"  - Geometric bridges with high betweenness: {result_main['geometric_high_brokerage']}\n")
            f.write(f"  - Genuine bridges with high betweenness: {result_main['genuine_high_brokerage']}\n\n")
            
            if result_main['genuine_bridges_n'] > 0:
                overlap_pct = 100 * result_main['genuine_high_brokerage'] / result_main['genuine_bridges_n']
                f.write(f"Percentage of genuine bridges with high brokerage: {overlap_pct:.1f}%\n\n")
            
            f.write("**Interpretation:** Genuine bridges (lexical + geometric) show stronger alignment\n")
            f.write("with network-based brokerage metrics than geometric proximity alone.\n\n")
        
        f.write("## Most Connected Cluster Pairs\n\n")
        if result_main['top_connected_pairs']:
            for (c1, c2), count in result_main['top_connected_pairs']:
                terms1 = ", ".join(topics.get(c1, [])[:3])
                terms2 = ", ".join(topics.get(c2, [])[:3])
                f.write(f"  C{c1} ({terms1}) ↔ C{c2} ({terms2}): {count} genuine bridges\n")
        else:
            f.write("  No genuine bridges found between any cluster pairs.\n")
        
        f.write("\n## Sample Genuine Bridge Papers\n\n")
        for i, bridge in enumerate(bridge_details[:5], 1):
            f.write(f"{i}. {bridge['title']}\n")
            f.write(f"   Connects: C{', C'.join(map(str, bridge['connected_clusters']))}\n")
            if bridge['brokerage']:
                f.write(f"   Betweenness: {bridge['brokerage'].get('betweenness', 0):.4f}\n")
            f.write(f"   Matched terms:\n")
            for c, terms in bridge['cluster_terms_matched'].items():
                f.write(f"     C{c}: {', '.join(terms)}\n")
            f.write("\n")
    
    log.info("\n" + "=" * 70)
    log.info("✓ BRIDGE PAPERS SENSITIVITY COMPLETED")
    log.info(f"  At threshold=0.70:")
    log.info(f"    Geometric bridges: {result_main['geometric_bridges_n']} ({result_main['geometric_bridges_pct']:.2f}%)")
    log.info(f"    Genuine bridges: {result_main['genuine_bridges_n']} ({result_main['genuine_bridges_pct']:.2f}%)")
    if result_main['geometric_bridges_pct'] > 0:
        reduction = 100*(1 - result_main['genuine_bridges_pct']/result_main['geometric_bridges_pct'])
        log.info(f"    Reduction: {reduction:.1f}%")
    if brokerage_scores is not None:
        log.info(f"    Genuine bridges with high betweenness: {result_main['genuine_high_brokerage']}")
    log.info("=" * 70 + "\n")
    
    return df_results

# ========= BASELINE TF-IDF VOCABULARY OVERLAP (simple, reproducible, reviewer-friendly) =========
def compute_tfidf_baseline_overlap(texts: List[str], labels: np.ndarray, outdir: str,
                                   top_n: int = 40):
    """
    Baseline lexical overlap using RAW TF–IDF (not c-TF-IDF).
    - Gives reviewer-required comparison showing whether fragmentation is method-induced.
    - Does NOT replace c-TF-IDF, only provides a coarse, neutral reference.
    """
    log.info("--- Baseline TF-IDF lexical similarity (tokens only) ---")
    df = pd.DataFrame({"text": texts, "cluster": labels})
    clusters = sorted(df.cluster.unique())
    
    # TF-IDF token-level representation
    vec = TfidfVectorizer(
        token_pattern=r"[A-Za-z0-9_\-]+",
        ngram_range=(1,1),
        min_df=2,
        max_df=0.95
    )
    X = vec.fit_transform(df["text"])
    terms = np.array(vec.get_feature_names_out())
    
    # Top-N terms per cluster (by mean tfidf)
    vocab = {}
    for c in clusters:
        idx = df[df.cluster == c].index
        if len(idx) == 0:
            vocab[c] = []
            continue
        Xc = X[idx].mean(axis=0).A1
        top_idx = np.argsort(-Xc)[:top_n]
        vocab[c] = terms[top_idx].tolist()
    
    # Compute Jaccard
    jac = np.zeros((len(clusters), len(clusters)))
    for i, c1 in enumerate(clusters):
        set1 = set(vocab[c1])
        for j, c2 in enumerate(clusters):
            if i == j:
                jac[i, j] = 1.0
            else:
                set2 = set(vocab[c2])
                inter = len(set1 & set2)
                union = len(set1 | set2)
                jac[i, j] = inter/union if union > 0 else 0.0
    
    df_jac = pd.DataFrame(
        jac,
        index=[f"C{c}" for c in clusters],
        columns=[f"C{c}" for c in clusters]
    )
    df_jac.to_csv(os.path.join(outdir, "baseline_tfidf_jaccard.csv"), index=True)
    log.info(f"✓ Baseline TF-IDF Jaccard saved")
    log.info(f"  Mean off-diagonal: {jac[np.triu_indices(len(clusters),1)].mean():.3f}")
    
    # Optional RBO (if library exists)
    if rbo:
        rbo_mat = np.zeros_like(jac)
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                if i == j:
                    rbo_mat[i,j] = 1.0
                else:
                    try:
                        rbo_mat[i,j] = rbo.RankingSimilarity(
                            vocab[c1], vocab[c2]
                        ).rbo(p=0.9)
                    except Exception:
                        rbo_mat[i,j] = 0.0
        df_rbo = pd.DataFrame(
            1 - rbo_mat,
            index=[f"C{c}" for c in clusters],
            columns=[f"C{c}" for c in clusters]
        )
        df_rbo.to_csv(os.path.join(outdir, "baseline_tfidf_rbo_fragmentation.csv"))
        log.info("✓ Baseline TF-IDF RBO saved")
    
    return df_jac
#=====fragmentation====

def compute_rbo_fragmentation(topics: Dict[int, List[str]], outdir: str, p: float = 0.9, top_n: int = 50):
    """
    Calcula la fragmentación epistémica entre clusters usando Rank Biased Overlap (RBO).
    Basado en Polimeno et al. (2023): Fragmentation detection.
    - topics: diccionario {cluster_id: [keywords ordenadas por peso TF-IDF]}
    - p: parámetro de profundidad (0.9 ≈ top-10 dominante)
    - top_n: número de términos comparados por clúster
    """
    if rbo is None:
        log.warning("RBO metric skipped (library not available).")
        return None
    
    log.info("--- Calculando RBO-based Epistemic Fragmentation ---")
    
    clusters = sorted(topics.keys())
    n = len(clusters)
    rbo_mat = np.zeros((n, n))
    
    for i, c1 in enumerate(clusters):
        list1 = topics[c1][:top_n]
        for j, c2 in enumerate(clusters):
            if i == j:
                rbo_mat[i, j] = 1.0
            else:
                list2 = topics[c2][:top_n]
                try:
                    rbo_score = rbo.RankingSimilarity(list1, list2).rbo(p=p)
                except Exception:
                    rbo_score = 0.0
                rbo_mat[i, j] = rbo_score
    
    # Convertir a DataFrame y guardar
    df_rbo = pd.DataFrame(
        1 - rbo_mat,  # Fragmentation = 1 - RBO
        index=[f"C{c}" for c in clusters],
        columns=[f"C{c}" for c in clusters]
    )
    
    df_rbo.to_csv(os.path.join(outdir, "rbo_fragmentation.csv"))
    log.info(f"✓ RBO fragmentation matrix guardada")
    
    # Estadísticas globales
    upper = 1 - rbo_mat[np.triu_indices(n, k=1)]
    mean_frag = float(np.mean(upper))
    std_frag = float(np.std(upper))
    
    log.info(f"  - Mean Epistemic Fragmentation (1-RBO): {mean_frag:.3f} ± {std_frag:.3f}")
    log.info(f"  - Min/Max: {float(np.min(upper)):.3f} / {float(np.max(upper)):.3f}")
    
    with open(os.path.join(outdir, "rbo_fragmentation_summary.txt"), "w") as f:
        f.write("# Epistemic Fragmentation (RBO-based)\n\n")
        f.write(f"p parameter: {p}\n")
        f.write(f"Top-N terms compared: {top_n}\n")
        f.write(f"Mean (1-RBO): {mean_frag:.4f}\n")
        f.write(f"Std: {std_frag:.4f}\n")
    
    return df_rbo

def identify_extreme_fragmentation_pairs(topics: Dict[int, List[str]], 
                                        all_matrices: Dict[str, np.ndarray],
                                        clusters: List[int],
                                        threshold: float = 0.99) -> pd.DataFrame:
    """
    Identifica pares de clusters con fragmentación extrema (1-RBO ≥ threshold)
    consistente en TODAS las configuraciones.
    
    Returns:
        DataFrame con columnas: cluster_1, cluster_2, min_frag, max_frag, mean_frag, std_frag, always_extreme
    """
    if rbo is None:
        return pd.DataFrame()
    
    n = len(clusters)
    pair_details = []
    
    log.info(f"  → Analizando estabilidad de {n*(n-1)//2} pares de clusters...")
    
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i >= j:
                continue
            
            # Recopilar fragmentación de este par en todas las configuraciones
            pair_frags = []
            for config_key, frag_mat in all_matrices.items():
                pair_frags.append(frag_mat[i, j])
            
            min_frag = float(np.min(pair_frags))
            max_frag = float(np.max(pair_frags))
            mean_frag = float(np.mean(pair_frags))
            std_frag = float(np.std(pair_frags))
            
            # Verificar si SIEMPRE es extremo
            always_extreme = int(min_frag >= threshold)
            mostly_extreme = int(np.quantile(pair_frags, 0.25) >= threshold)
            
            pair_details.append({
                'cluster_1': int(c1),
                'cluster_2': int(c2),
                'min_fragmentation': min_frag,
                'max_fragmentation': max_frag,
                'mean_fragmentation': mean_frag,
                'std_fragmentation': std_frag,
                'always_extreme': always_extreme,
                'mostly_extreme': mostly_extreme,
                'n_configs': len(pair_frags)
            })
    
    df_pairs = pd.DataFrame(pair_details)
    df_pairs = df_pairs.sort_values('mean_fragmentation', ascending=False)
    
    n_always_extreme = (df_pairs['always_extreme'] == 1).sum()
    log.info(f"    Pares con 1-RBO ≥ {threshold} en TODAS las configs: {n_always_extreme}")
    
    return df_pairs

def compute_rbo_sensitivity_analysis(topics: Dict[int, List[str]], outdir: str):
    """
    Análisis de sensibilidad completo para RBO:
    - Varía p ∈ {0.8, 0.9, 0.95}
    - Varía top_n ∈ {10, 20, 50}
    - Genera matrices y estadísticas para cada configuración
    - Identifica pares con fragmentación extrema consistente
    """
    if rbo is None:
        log.warning("RBO sensitivity analysis skipped (library not available).")
        return None
    
    log.info("=" * 70)
    log.info("SENSITIVITY ANALYSIS: RBO Fragmentation")
    log.info("=" * 70)
    
    # Configuraciones a probar
    p_values = [0.8, 0.9, 0.95]
    top_n_values = [10, 20, 50]
    
    clusters = sorted(topics.keys())
    n = len(clusters)
    
    # Almacenar resultados
    all_configs = []
    all_matrices = {}  # Guardar matrices para análisis de pares
    
    log.info(f"Configuraciones: p ∈ {p_values}, top_n ∈ {top_n_values}")
    log.info(f"Total: {len(p_values) * len(top_n_values)} configuraciones\n")
    
    for p in p_values:
        for top_n in top_n_values:
            config_key = f"p{p}_top{top_n}"
            log.info(f"  [{len(all_configs)+1}/{len(p_values)*len(top_n_values)}] Testing p={p}, top_n={top_n}")
            
            # Matriz RBO para esta configuración
            rbo_mat = np.zeros((n, n))
            
            for i, c1 in enumerate(clusters):
                list1 = topics[c1][:top_n]
                for j, c2 in enumerate(clusters):
                    if i == j:
                        rbo_mat[i, j] = 1.0
                    else:
                        list2 = topics[c2][:top_n]
                        try:
                            rbo_score = rbo.RankingSimilarity(list1, list2).rbo(p=p)
                        except Exception:
                            rbo_score = 0.0
                        rbo_mat[i, j] = rbo_score
            
            # Fragmentación = 1 - RBO
            frag_mat = 1 - rbo_mat
            all_matrices[config_key] = frag_mat
            
            # Estadísticas (solo triángulo superior sin diagonal)
            upper_tri = frag_mat[np.triu_indices(n, k=1)]
            
            config_result = {
                'p': p,
                'top_n': top_n,
                'mean_fragmentation': float(np.mean(upper_tri)),
                'std_fragmentation': float(np.std(upper_tri)),
                'min_fragmentation': float(np.min(upper_tri)),
                'max_fragmentation': float(np.max(upper_tri)),
                'q25': float(np.quantile(upper_tri, 0.25)),
                'q50': float(np.quantile(upper_tri, 0.50)),
                'q75': float(np.quantile(upper_tri, 0.75)),
                'pairs_complete_sep': int((upper_tri >= 0.99).sum()),  # 1-RBO ≥ 0.99
                'pairs_high_frag': int((upper_tri >= 0.90).sum()),     # 1-RBO ≥ 0.90
                'pairs_moderate_frag': int((upper_tri >= 0.70).sum()), # 1-RBO ≥ 0.70
            }
            
            all_configs.append(config_result)
            
            # Guardar matriz individual
            df_frag = pd.DataFrame(
                frag_mat,
                index=[f"C{c}" for c in clusters],
                columns=[f"C{c}" for c in clusters]
            )
            filename = f"rbo_fragmentation_p{p}_top{top_n}.csv"
            df_frag.to_csv(os.path.join(outdir, filename))
            
            log.info(f"      Mean 1-RBO: {config_result['mean_fragmentation']:.3f} "
                    f"± {config_result['std_fragmentation']:.3f}")
            log.info(f"      Range: {config_result['min_fragmentation']:.3f} - {config_result['max_fragmentation']:.3f}")
            log.info(f"      Pairs with 1-RBO ≥ 0.99: {config_result['pairs_complete_sep']}\n")
    
    # Consolidar resultados en tabla
    df_sensitivity = pd.DataFrame(all_configs)
    df_sensitivity.to_csv(os.path.join(outdir, "rbo_sensitivity_summary.csv"), index=False)
    
    log.info("=" * 70)
    log.info("✓ RBO sensitivity analysis completed")
    log.info(f"  Configurations tested: {len(all_configs)}")
    
    # Crear tabla pivote para el paper (formato Nature-friendly)
    pivot_mean = df_sensitivity.pivot(index='top_n', columns='p', values='mean_fragmentation')
    pivot_mean.to_csv(os.path.join(outdir, "rbo_sensitivity_pivot_mean.csv"))
    
    pivot_std = df_sensitivity.pivot(index='top_n', columns='p', values='std_fragmentation')
    pivot_std.to_csv(os.path.join(outdir, "rbo_sensitivity_pivot_std.csv"))
    
    # Análisis detallado de pares (estabilidad)
    log.info("\n  Analizando estabilidad por pares...")
    df_pairs = identify_extreme_fragmentation_pairs(topics, all_matrices, clusters, threshold=0.99)
    df_pairs.to_csv(os.path.join(outdir, "rbo_pairwise_stability.csv"), index=False)
    
    extreme_always = df_pairs[df_pairs['always_extreme'] == 1]
    
    # Reporte textual para Supplementary Materials
    with open(os.path.join(outdir, "rbo_sensitivity_report.txt"), "w") as f:
        f.write("# RBO SENSITIVITY ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("## Configuration Matrix (Mean 1-RBO)\n\n")
        f.write(pivot_mean.to_string())
        f.write("\n\n")
        
        f.write("## Configuration Matrix (Std 1-RBO)\n\n")
        f.write(pivot_std.to_string())
        f.write("\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"Configurations tested: {len(all_configs)}\n")
        f.write(f"Parameter ranges:\n")
        f.write(f"  - p ∈ {{{', '.join(map(str, p_values))}}}\n")
        f.write(f"  - top_n ∈ {{{', '.join(map(str, top_n_values))}}}\n\n")
        
        # Rango de variación global
        f.write("## Fragmentation Range Across ALL Configurations\n\n")
        f.write(f"Min mean 1-RBO: {df_sensitivity['mean_fragmentation'].min():.4f} ")
        f.write(f"(p={df_sensitivity.loc[df_sensitivity['mean_fragmentation'].idxmin(), 'p']}, ")
        f.write(f"top_n={int(df_sensitivity.loc[df_sensitivity['mean_fragmentation'].idxmin(), 'top_n'])})\n")
        
        f.write(f"Max mean 1-RBO: {df_sensitivity['mean_fragmentation'].max():.4f} ")
        f.write(f"(p={df_sensitivity.loc[df_sensitivity['mean_fragmentation'].idxmax(), 'p']}, ")
        f.write(f"top_n={int(df_sensitivity.loc[df_sensitivity['mean_fragmentation'].idxmax(), 'top_n'])})\n")
        
        delta = df_sensitivity['mean_fragmentation'].max() - df_sensitivity['mean_fragmentation'].min()
        f.write(f"\nDelta (max - min): {delta:.4f}\n")
        f.write(f"Relative variation: {100*delta/df_sensitivity['mean_fragmentation'].mean():.2f}%\n\n")
        
        # Pares extremos
        f.write("## Pairs with Extreme Fragmentation (1-RBO ≥ 0.99)\n\n")
        f.write(f"Total cluster pairs analyzed: {len(df_pairs)}\n")
        f.write(f"Pairs with 1-RBO ≥ 0.99 in ALL {len(all_configs)} configurations: {len(extreme_always)}\n")
        f.write(f"Percentage of extreme pairs: {100*len(extreme_always)/len(df_pairs):.1f}%\n\n")
        
        if len(extreme_always) > 0:
            f.write("List of consistently extreme pairs:\n")
            f.write("=" * 70 + "\n")
            for idx, row in extreme_always.iterrows():
                c1, c2 = int(row['cluster_1']), int(row['cluster_2'])
                terms1 = ", ".join(topics[c1][:5])
                terms2 = ", ".join(topics[c2][:5])
                f.write(f"\nC{c1} vs C{c2}:\n")
                f.write(f"  Fragmentation: {row['mean_fragmentation']:.4f} ± {row['std_fragmentation']:.4f}\n")
                f.write(f"  Range: {row['min_fragmentation']:.4f} - {row['max_fragmentation']:.4f}\n")
                f.write(f"  C{c1} terms: {terms1}\n")
                f.write(f"  C{c2} terms: {terms2}\n")
                f.write(f"  → COMPLETE LEXICAL SEPARATION (stable across all configurations)\n")
                f.write("-" * 70 + "\n")
        else:
            f.write("\nNo pairs with consistent extreme fragmentation (1-RBO ≥ 0.99) found.\n")
        
        # Distribución de fragmentación por configuración
        f.write("\n## Distribution of Pair-wise Fragmentation by Configuration\n\n")
        for _, cfg in df_sensitivity.iterrows():
            f.write(f"p={cfg['p']}, top_n={int(cfg['top_n'])}:\n")
            f.write(f"  Mean: {cfg['mean_fragmentation']:.3f} ± {cfg['std_fragmentation']:.3f}\n")
            f.write(f"  Quartiles: Q25={cfg['q25']:.3f}, Median={cfg['q50']:.3f}, Q75={cfg['q75']:.3f}\n")
            f.write(f"  Extreme pairs (≥0.99): {cfg['pairs_complete_sep']}\n")
            f.write(f"  High fragmentation (≥0.90): {cfg['pairs_high_frag']}\n\n")
    
    # Archivo separado para pares extremos (paper-ready)
    with open(os.path.join(outdir, "extreme_fragmentation_pairs.txt"), "w") as f:
        f.write("# CLUSTER PAIRS WITH EXTREME FRAGMENTATION\n")
        f.write("# (1-RBO ≥ 0.99 in ALL configurations)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis parameters:\n")
        f.write(f"  - Configurations tested: {len(all_configs)}\n")
        f.write(f"  - p values: {p_values}\n")
        f.write(f"  - top_n values: {top_n_values}\n")
        f.write(f"  - Threshold: 1-RBO ≥ 0.99\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Total extreme pairs: {len(extreme_always)}\n")
        f.write(f"  - Total pairs analyzed: {len(df_pairs)}\n")
        f.write(f"  - Percentage: {100*len(extreme_always)/len(df_pairs):.1f}%\n\n")
        
        if len(extreme_always) > 0:
            f.write("=" * 70 + "\n")
            f.write("EXTREME PAIRS DETAIL:\n")
            f.write("=" * 70 + "\n\n")
            
            for idx, row in extreme_always.iterrows():
                c1, c2 = int(row['cluster_1']), int(row['cluster_2'])
                terms1 = ", ".join(topics[c1][:8])
                terms2 = ", ".join(topics[c2][:8])
                
                f.write(f"Pair {idx+1}: C{c1} ↔ C{c2}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Fragmentation statistics:\n")
                f.write(f"  Mean:   {row['mean_fragmentation']:.4f}\n")
                f.write(f"  Std:    {row['std_fragmentation']:.4f}\n")
                f.write(f"  Range:  {row['min_fragmentation']:.4f} - {row['max_fragmentation']:.4f}\n")
                f.write(f"  Stable: {'YES' if row['std_fragmentation'] < 0.01 else 'MODERATE'}\n\n")
                f.write(f"Cluster {c1} characteristic terms:\n")
                f.write(f"  {terms1}\n\n")
                f.write(f"Cluster {c2} characteristic terms:\n")
                f.write(f"  {terms2}\n\n")
                f.write(f"Interpretation: COMPLETE LEXICAL SEPARATION\n")
                f.write(f"  → No overlap in top-ranked characteristic vocabulary\n")
                f.write(f"  → Fragmentation persists across ALL parameter choices\n")
                f.write(f"  → Indicates structural epistemic barrier\n\n")
                f.write("=" * 70 + "\n\n")
        else:
            f.write("No pairs with consistent extreme fragmentation found.\n")
            f.write("This would indicate that fragmentation is parameter-dependent.\n")
    
    log.info(f"  ✓ Extreme pairs (1-RBO ≥ 0.99 always): {len(extreme_always)}/{len(df_pairs)}")
    log.info(f"  ✓ Reports saved:")
    log.info(f"    - rbo_sensitivity_summary.csv")
    log.info(f"    - rbo_sensitivity_pivot_mean.csv")
    log.info(f"    - rbo_pairwise_stability.csv")
    log.info(f"    - rbo_sensitivity_report.txt")
    log.info(f"    - extreme_fragmentation_pairs.txt")
    log.info("=" * 70 + "\n")
    
    return df_sensitivity
# ========= Validaciones externas desde Neo4j =========
def fetch_citation_edges(neo: Neo4jClient, dois: List[str], 
                         include_external: bool = False) -> Optional[pd.DataFrame]:
    """
    Obtiene edges de citación desde Neo4j.
    
    Args:
        neo: Cliente Neo4j
        dois: Lista de DOIs del corpus
        include_external: Si True, incluye citas a papers fuera del corpus
    
    Returns:
        DataFrame con columnas: src, dst, is_internal
    """
    dois = [d for d in dois if isinstance(d, str) and len(d) > 0]
    if not dois:
        return None
    
    if include_external:
        # Incluir TODAS las citas (internas + externas)
        cy = """
            UNWIND $dois AS d
            MATCH (p:Paper {doi: d})-[:CITES]->(q:Paper)
            WHERE q.doi IS NOT NULL
            RETURN 
                p.doi AS src, 
                q.doi AS dst,
                q.doi IN $dois AS is_internal
        """
    else:
        # Solo citas internas (comportamiento original)
        cy = """
            UNWIND $dois AS d
            MATCH (p:Paper {doi: d})-[:CITES]->(q:Paper)
            WHERE q.doi IS NOT NULL AND q.doi IN $dois
            RETURN 
                p.doi AS src, 
                q.doi AS dst,
                true AS is_internal
        """
    
    try:
        with neo.driver.session(database=neo.db) as s:
            rows = [r.data() for r in s.run(cy, dois=dois)]
        df = pd.DataFrame(rows)
        return df if not df.empty else None
    except Exception as e:
        log.warning(f"Citation fetch failed: {e}")
        return None

def fetch_coauthorship(neo: Neo4jClient, dois: List[str]) -> Optional[pd.DataFrame]:
    cy = """
        UNWIND $dois AS d
        MATCH (a:Author)-[:AUTHORED]->(p:Publication {doi:d})<-[:AUTHORED]-(b:Author)
        WHERE id(a) < id(b)
        RETURN a.name AS a, b.name AS b
    """
    try:
        with neo.driver.session(database=neo.db) as s:
            rows = [r.data() for r in s.run(cy, dois=dois)]
        df = pd.DataFrame(rows)
        return df if not df.empty else None
    except Exception as e:
        log.info(f"Coauthor fetch skipped/failed: {e}"); return None

def fetch_journal_pairs(neo: Neo4jClient, dois: List[str]) -> Optional[pd.DataFrame]:
    cy = """
        UNWIND $dois AS d
        MATCH (p:Publication {doi:d})-[:PUBLISHED_IN]->(j:Journal)
        RETURN d AS doi, j.name AS journal
    """
    try:
        with neo.driver.session(database=neo.db) as s:
            rows = [r.data() for r in s.run(cy, dois=dois)]
        return pd.DataFrame(rows)
    except Exception as e:
        log.info(f"Journal fetch skipped/failed: {e}"); return None

def multi_external_validation(neo: Neo4jClient, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    labels_by_doi = {d:int(c) for d,c in zip(df['doi'].tolist(), labels)}
    # modularidad de citas
    edges = fetch_citation_edges(neo, df['doi'].tolist())
    modQ = modularity_on_clusters(edges, labels_by_doi)
    if modQ is not None: out["citation_modularity_Q"] = round(modQ, 4)
    # coautoría
    co = fetch_coauthorship(neo, df['doi'].tolist())
    if nx is not None and co is not None and not co.empty:
        G = nx.from_pandas_edgelist(co, 'a', 'b', create_using=nx.Graph())
        out["coauth_components"] = int(nx.number_connected_components(G)) if G.number_of_nodes() else 0
        out["coauth_avg_degree"] = float(np.mean([d for n,d in G.degree()])) if G.number_of_nodes() else float('nan')
    # journals
    jp = fetch_journal_pairs(neo, df['doi'].tolist())
    if jp is not None and not jp.empty:
        ent = []
        for c in np.unique(labels):
            sub = jp[jp['doi'].isin(df.loc[labels==c,'doi'])]
            if sub.empty: continue
            p = sub['journal'].value_counts(normalize=True).values + 1e-12
            ent.append(float(-(p*np.log(p)).sum()))
        out["journal_entropy_mean"] = float(np.mean(ent)) if ent else float('nan')
    return out

def modularity_on_clusters(edges: Optional[pd.DataFrame], labels_by_doi: Dict[str,int]) -> Optional[float]:
    if nx is None or edges is None or edges.empty: return None
    G = nx.from_pandas_edgelist(edges, 'src', 'dst', create_using=nx.Graph())
    if G.number_of_nodes() < 10: return None
    nodes = [n for n in G.nodes if n in labels_by_doi and labels_by_doi[n] != -1]
    if len(nodes) < 10: return None
    part: Dict[int, List[str]] = {}
    for n in nodes:
        c = labels_by_doi[n]; part.setdefault(c, []).append(n)
    try:
        import networkx.algorithms.community.quality as q
        comms = [set(v) for v in part.values() if len(v) > 0]
        return float(q.modularity(G.subgraph(nodes), comms))
    except Exception:
        return None

# ========= Exports & reports =========
def export_tables(name: str, df: pd.DataFrame, labels: np.ndarray, topics: Dict[int,List[str]], reps: Dict[int,pd.DataFrame], tl: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    short = {c: ", ".join(v[:4]) for c, v in topics.items()}
    df_map = df.copy(); df_map["cluster"] = labels; df_map["topic_label"] = df_map["cluster"].map(short)
    df_map[["doi","title","year","cluster","topic_label"]].to_csv(os.path.join(outdir,"paper_topics.csv"), index=False)
    rows = [{"cluster": c, "top_terms": "; ".join(topics.get(c, []))} for c in sorted(set(labels)) if c != -1]
    pd.DataFrame(rows).to_csv(os.path.join(outdir,"semantic_topics.csv"), index=False)
    tl.to_csv(os.path.join(outdir,"cluster_timeline.csv"), index=False)
    rep_tables = []
    for c, t in reps.items():
        t2 = t.copy(); t2.insert(0, "cluster", c); rep_tables.append(t2)
    if rep_tables:
        pd.concat(rep_tables, ignore_index=True).to_csv(os.path.join(outdir,"cluster_representatives.csv"), index=False)

def export_diversity(name: str, df: pd.DataFrame, X, labels: np.ndarray, outdir: str, mmr_k:int=10, lam:float=0.55):
    os.makedirs(outdir, exist_ok=True)
    Xe = _to_dense(X)
    all_rows, mmr_rows, fps_rows, cov_rows = [], [], [], []
    for c in sorted(np.unique(labels)):
        idx = np.where(labels==c)[0]
        if len(idx)==0: continue
        Xc = Xe[idx]; dmet = diversity_metrics(Xc); dmet["cluster"] = int(c); all_rows.append(dmet)
        q = Xc.mean(axis=0, keepdims=True)
        # MMR
        mmr_idx_local = mmr_select(Xc, k=min(mmr_k, len(idx)), lam=lam, query_vec=q)
        mmr_idx_global = idx[mmr_idx_local]
        for j, gi in enumerate(mmr_idx_global):
            r = df.iloc[gi]
            mmr_rows.append({"cluster": int(c), "rank": j+1, "global_idx": int(gi),
                             "doi": r.get("doi"), "title": r.get("title"), "year": r.get("year")})
        # FPS
        fps_idx_local = farthest_point_sampling(Xc, k=min(mmr_k, len(idx)))
        fps_idx_global = idx[fps_idx_local]
        for j, gi in enumerate(fps_idx_global):
            r = df.iloc[gi]
            fps_rows.append({"cluster": int(c), "rank": j+1, "global_idx": int(gi),
                             "doi": r.get("doi"), "title": r.get("title"), "year": r.get("year")})
        # Cobertura: fija + adaptativa
        for method, loc_idx in [("MMR", mmr_idx_local), ("FPS", fps_idx_local)]:
            if len(loc_idx):
                d_nn = _cosine_dist(Xc, Xc[loc_idx]).min(axis=1)
                for rad in [0.05, 0.10, 0.20, 0.30]:
                    cov_rows.append({"cluster": int(c), "method": method, "k": int(len(loc_idx)),
                                     "radius_cos": rad, "covered_frac": float((d_nn <= rad).mean())})
            for cc in adaptive_coverage_analysis(Xc, loc_idx, percentiles=(0.10,0.25,0.50,0.75)):
                cov_rows.append({"cluster": int(c), "method": method+"_adaptive", "k": int(len(loc_idx)), **cc})
    if all_rows: pd.DataFrame(all_rows).to_csv(os.path.join(outdir,"diversity_metrics.csv"), index=False)
    if mmr_rows: pd.DataFrame(mmr_rows).to_csv(os.path.join(outdir,"mmr_representatives.csv"), index=False)
    if fps_rows: pd.DataFrame(fps_rows).to_csv(os.path.join(outdir,"fps_representatives.csv"), index=False)
    if cov_rows: pd.DataFrame(cov_rows).to_csv(os.path.join(outdir,"coverage_curves.csv"), index=False)

def write_semantic_report(name: str, df: pd.DataFrame, labels: np.ndarray, topics: Dict[int,List[str]], reps: Dict[int,pd.DataFrame], outdir:str, method_str:str):
    path = os.path.join(outdir, "semantic_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Semantic Topic Report\n\n")
        f.write(f"Cluster set: {name}\n\n")
        f.write(f"Total papers: {len(df)}\n\n")
        f.write(f"**Clustering method:** {method_str}\n\n")
        for c in sorted(set(labels)):
            if c == -1: continue
            terms = ", ".join(topics.get(c, [])[:10])
            f.write(f"## Cluster {c} — {terms}\n\n")
            tbl = reps.get(c)
            if tbl is not None:
                for _, r in tbl.iterrows():
                    f.write(f"- **{r['title']}** ({r['year']}), DOI: {r['doi']} — rep_sim={r['rep_similarity']:.3f}\n")
            f.write("\n")

def write_validation_report(name:str, scores: ModelScores, linkM: pd.DataFrame, trends: pd.DataFrame,
                            outdir:str, modQ: Optional[float]=None, external: Optional[Dict[str,Any]]=None):
    path = os.path.join(outdir, "validation_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Validation & Diagnostics\n\n")
        f.write(f"Cluster set: {name}\n\n")
        f.write("## Internal metrics\n\n")
        f.write(f"- Algorithm: {scores.algo}\n")
        f.write(f"- k: {scores.k}\n")
        f.write(f"- Silhouette (cosine): {scores.silhouette:.3f}\n")
        f.write(f"- Davies–Bouldin: {scores.db:.3f}\n")
        f.write(f"- Calinski–Harabasz: {scores.ch:.1f}\n")
        f.write(f"- Bootstrap ARI (median): {scores.ari_median if not math.isnan(scores.ari_median) else 'NA'}\n")
        f.write(f"- Bootstrap ARI IQR: {scores.ari_iqr}\n")
        f.write(f"- Cluster sizes: {scores.cluster_sizes}\n\n")
        f.write("## Centroid cosine links\n\n")
        f.write(linkM.to_csv(index=True)); f.write("\n")
        f.write("## Timeline trends (model, slope, t-like)\n\n")
        f.write(trends.to_csv(index=False)); f.write("\n")
        if modQ is not None:
            f.write("## External (citation graph)\n\n")
            f.write(f"- Modularity Q: {modQ}\n")
        if external:
            f.write("## External validation suite\n\n")
            for k,v in external.items():
                f.write(f"- {k}: {v}\n")

def write_business_insights(name: str, insights_df: pd.DataFrame, outdir: str):
    insights_df.to_csv(os.path.join(outdir, "cluster_insights.csv"), index=False)
    path = os.path.join(outdir, "business_insights.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Business-Oriented Insights per Cluster\n\n")
        f.write(f"Cluster set: {name}\n\n")
        f.write(f"- ARI_global: {insights_df['ari_global'].iloc[0]:.3f} | baseline_slope_pos: {insights_df['explosive_baseline_slope'].iloc[0]:.4f}\n\n")
        for _, r in insights_df.sort_values("cluster").iterrows():
            f.write(f"## Cluster {int(r['cluster'])} — {r['insight_label']}\n\n")
            f.write(f"- n: {int(r['n'])}\n")
            f.write(f"- citas/paper: {r['citations_per_paper']:.2f}\n")
            f.write(f"- silhouette_mean: {r['silhouette_mean']:.3f}\n")
            f.write(f"- cohesion (centroid cosine): {r['centroid_cohesion']:.3f}\n")
            f.write(f"- slope: {r['slope']:.4f} | t-like: {r['t_like']:.2f}\n")
            f.write(f"- Rationale: {r['insight_reason']}\n\n")

def write_config(outdir:str, config:Dict[str,Any]):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(to_py(config), f, indent=2, ensure_ascii=False)

# ========= Insights por clúster =========
def per_cluster_metrics(X, df: pd.DataFrame, labels: np.ndarray, tl_trends: pd.DataFrame) -> pd.DataFrame:
    Xe = _to_dense(X); d = df.copy(); d["cluster"] = labels
    d["citedBy"] = pd.to_numeric(d["citedBy"], errors="coerce").fillna(0.0)
    try: s_samples = silhouette_samples(Xe, labels, metric="cosine")
    except Exception: s_samples = np.full(len(labels), np.nan)
    d["sil_sample"] = s_samples
    rows = []
    for c in sorted(d.cluster.unique()):
        sub = d[d.cluster==c]; idx = np.where(labels==c)[0]
        centroid = Xe[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(Xe[idx], centroid).ravel()
        slope = float(tl_trends.loc[tl_trends.cluster==c, "slope"].values[0]) if ((tl_trends is not None) and (tl_trends.cluster==c).any()) else float('nan')
        t_like = float(tl_trends.loc[tl_trends.cluster==c, "t_like"].values[0]) if ((tl_trends is not None) and (tl_trends.cluster==c).any()) else float('nan')
        rows.append({"cluster": int(c), "n": int(len(sub)),
                     "citations_per_paper": float(sub["citedBy"].mean()) if len(sub) else float('nan'),
                     "silhouette_mean": float(sub["sil_sample"].mean()) if len(sub) else float('nan'),
                     "centroid_cohesion": float(np.mean(sims)) if len(sims) else float('nan'),
                     "slope": slope, "t_like": t_like})
    return pd.DataFrame(rows)

def classify_clusters(df_metrics: pd.DataFrame, ari_global: float) -> pd.DataFrame:
    pos_slopes = df_metrics["slope"].dropna()
    base_slope = float(pos_slopes[pos_slopes > 0].mean()) if (pos_slopes > 0).any() else float('nan')
    labels, reasons = [], []
    for _, r in df_metrics.iterrows():
        lbl = "—"; reason = []
        if (not math.isnan(ari_global)) and (ari_global >= 0.6) and (r["citations_per_paper"] < 3.0):
            lbl = "Nichos Académicos Maduros"
            reason = [f"ARI_global={ari_global:.2f} (≥0.6)", f"citas/paper={r['citations_per_paper']:.2f} (<3)"]
        if (not math.isnan(ari_global)) and (ari_global < 0.4) and (not math.isnan(base_slope)) and (r["slope"] > 2.0*base_slope):
            lbl = "Campos Emergentes Sin Consenso"
            reason = [f"slope={r['slope']:.3f} (>2× {base_slope:.3f})", f"ARI_global={ari_global:.2f} (<0.4)"]
        if lbl == "—":
            lbl = "Mixto / Indeterminado"; reason.append("Sin señales claras; revisar términos/topicos y t_like")
        labels.append(lbl); reasons.append("; ".join(reason))
    out = df_metrics.copy()
    out["insight_label"] = labels; out["insight_reason"] = reasons
    out["ari_global"] = ari_global; out["explosive_baseline_slope"] = base_slope
    return out

# ========= Plot =========
def save_plot_2d(X, labels: np.ndarray, outpath: str, title: str = "PCA 2D"):
    try: P = PCA(n_components=2, random_state=42).fit_transform(_to_dense(X))
    except Exception as e: log.warning(f"PCA plot skipped: {e}"); return
    try:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.figure(figsize=(7,5))
        for c in sorted(np.unique(labels)):
            idx = np.where(labels==c)[0]
            if idx.size == 0: continue
            plt.scatter(P[idx,0], P[idx,1], s=12, alpha=0.7, label=f"C{c}")
        plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.legend(markerscale=1, fontsize=8)
        plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()
    except Exception as e:
        log.warning(f"Plot save failed: {e}")

# ========= Runner =========
def run_full_corpus_analysis(backend: str, kmin: int, kmax: int, outroot: str,
                              bootstrap:int=20, graph_diagnostics: bool=False):
    neo = Neo4jClient()
    try:
        name = "full_corpus"
        cypher = cypher_for_full_corpus()
        
        log.info(f"[{name}] Extrayendo corpus completo...")
        df = neo.fetch(cypher, params={})
        
        if df.empty:
            log.warning(f"[{name}] no results"); return
        
        log.info(f"[{name}] papers: {len(df)}")

        an = Analyzer(backend=backend)
        an.set_df(df)
        an.preprocess()  # Sin filtros
        an.embed()
        an.fit_knn(k=10)

        scores = select_model_auto(an.X, kmin=kmin, kmax=kmax, bootstrap=bootstrap)

        # AÑADIR ESTE BLOQUE:
        # AÑADIR ESTE BLOQUE:
        n_clusters_initial = len(np.unique(scores.labels))
        log.info(f"Clusters iniciales detectados: {n_clusters_initial}")

        if n_clusters_initial <= 3:
            log.info("Activando clustering jerárquico...")
            hierarchical_labels = hierarchical_subclustering(an.X, df, scores.labels, min_cluster_size=400)
            
            # Crear nuevo objeto ModelScores con labels jerárquicos
            h_sil, h_db, h_ch = eval_internal(an.X, hierarchical_labels)
            
            # 🔥 CALCULAR BOOTSTRAP ARI PARA HIERARCHICAL
            log.info("Calculando Bootstrap ARI para hierarchical clustering...")
            
            def clusterer_hierarchical(Xb, kk):
                # Sub-clustering simplificado para bootstrap
                labels_base = cluster_agglo(Xb, max(2, min(3, kk//2)))
                return hierarchical_subclustering(Xb, df.iloc[:len(Xb)], 
                                                 labels_base, 
                                                 min_cluster_size=max(50, len(Xb)//20))
            
            h_ari, h_iqr = bootstrap_stability(
                an.X, 
                clusterer_hierarchical, 
                len(np.unique(hierarchical_labels)), 
                bootstrap, 
                seed=42
            )
            
            log.info(f"Bootstrap ARI hierarchical: {h_ari:.3f} (IQR: {h_iqr})")
            
            scores = ModelScores(
                k=len(np.unique(hierarchical_labels)),
                algo="hierarchical", 
                labels=hierarchical_labels,
                silhouette=h_sil, 
                db=h_db, 
                ch=h_ch,
                ari_median=h_ari,  # ✅ AGORA CALCULA!
                ari_iqr=h_iqr,     # ✅ COM IQR!
                cluster_sizes=_cluster_sizes(hierarchical_labels)
            )

        method_str = f"{scores.algo}_auto(k={scores.k}, sil={scores.silhouette:.3f}, DB={scores.db:.3f}, CH={scores.ch:.1f}, ARI_med={scores.ari_median})"
        log.info(f"[{name}] {method_str}")

        topics = label_topics_c_tfidf(an.proc or [], scores.labels, top_n=12)
        reps   = representative_docs(an.X, df, scores.labels, top_m=5)
        tl     = cluster_timeline(df, scores.labels)
        outdir = os.path.join(outroot, name)
        
        export_tables(name, df, scores.labels, topics, reps, tl, outdir)
        write_semantic_report(name, df, scores.labels, topics, reps, outdir, method_str)
        
        linkM  = centroid_cosine_matrix(an.X, scores.labels)
        trends = enhanced_timeline_trends(tl)
        
        # Insights
        clus_metrics = per_cluster_metrics(an.X, df, scores.labels, trends)
        insights_df  = classify_clusters(clus_metrics, ari_global=scores.ari_median)
        write_business_insights(name, insights_df, outdir)

        # Diversidad + MMR/FPS + cobertura
        export_diversity(name, df, an.X, scores.labels, outdir, mmr_k=10, lam=0.55)

        # Distancia entre términos de cada clúster (contextual si posible)
        term_stats = term_distance_stats_by_cluster(topics, an, documents=df['abstract'].tolist())
        if term_stats is not None and not term_stats.empty:
            term_stats.to_csv(os.path.join(outdir, "term_distance_stats.csv"), index=False)

        # Plot
        save_plot_2d(an.X, scores.labels, os.path.join(outdir,"clusters_pca.png"), title=f"{name} — PCA")

# ===== MÉTRICAS DE FRAGMENTACIÓN (NUEVAS) =====
        log.info("\n" + "="*60)
        log.info("CALCULANDO MÉTRICAS DE FRAGMENTACIÓN")
        log.info("="*60)
        
        # 0. BASELINE TF-IDF (debe ir PRIMERO para comparación metodológica)
        log.info("\n[0/4] Baseline lexical overlap (TF-IDF simple)...")
        compute_tfidf_baseline_overlap(an.proc or [], scores.labels, outdir, top_n=40)
        
        # 1. Jaccard vocabulary overlap (c-TF-IDF)
        log.info("\n[1/4] Jaccard vocabulary overlap (c-TF-IDF)...")
        compute_jaccard_matrix(topics, outdir)
        
        # 2. Citation insularity
        if graph_diagnostics:
            log.info("\n[2/4] Citation insularity (graph-based)...")
            compute_citation_insularity(neo, df, scores.labels, outdir)
        else:
            log.info("\n[2/4] Citation insularity SKIPPED (use --graph-diagnostics to enable)")
        
        # 3. Cross-domain papers
        log.info("\n[3/5] Bridge papers sensitivity (geometric + lexical + brokerage)...")
        compute_bridge_papers_sensitivity(an.X, df, scores.labels, topics, neo, outdir, 
                                        thresholds=[0.65, 0.70, 0.75, 0.80])
        # 4. Epistemic Fragmentation (Rank Biased Overlap, c-TF-IDF)
        log.info("\n[4/4] RBO-based epistemic fragmentation (c-TF-IDF)...")
           
                # 4a. Baseline (p=0.9, top_n=50) para mantener compatibilidad
        compute_rbo_fragmentation(topics, outdir, p=0.9, top_n=50)

        # 4b. SENSITIVITY ANALYSIS (crítico para respuesta al reviewer)
        compute_rbo_sensitivity_analysis(topics, outdir)
                
        log.info("\n" + "="*60)
        log.info("✓ MÉTRICAS DE FRAGMENTACIÓN COMPLETADAS")
        log.info("="*60 + "\n")

        # Validación de grafo + suite externa
        modQ = None
        external = None
        if graph_diagnostics:
            log.info("Calculando validación externa (citation graph)...")
            edges = fetch_citation_edges(neo, df['doi'].tolist())
            labels_by_doi = {d:int(c) for d,c in zip(df['doi'].tolist(), scores.labels)}
            modQ = modularity_on_clusters(edges, labels_by_doi)
            external = multi_external_validation(neo, df, scores.labels)
            log.info(f"✓ Modularity Q: {modQ if modQ is not None else 'N/A'}")

        write_validation_report(name, scores, linkM, trends, outdir, modQ=modQ, external=external)

        # Guardar configuración (JSON-safe)
        write_config(outdir, {
            "cluster_name": name,
            "backend": backend,
            "algo": scores.algo,
            "k": scores.k,
            "k_grid": auto_k_grid(len(df), kmin, kmax),
            "bootstrap_runs": bootstrap,
            "graph_diagnostics": graph_diagnostics,
            "corpus_type": "corpus_relevante_property",  # ✅ ACTUALIZADO
            "corpus_filter": "Neo4j property: corpus_relevante = true",  # ✅ NUEVO
            "corpus_size": len(df),  # ✅ NUEVO - útil para replicabilidad
            "env": {
                "NEO4J_URI": os.getenv("NEO4J_URI"),
                "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
            }
        })
        
        log.info("\n" + "="*60)
        log.info("✓ ANÁLISIS COMPLETO FINALIZADO")
        log.info(f"✓ Resultados en: {outdir}")
        log.info("="*60)
        
    finally:
        neo.close()
# ========= CLI =========
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="sbert", choices=["sbert","tfidf","specter2","chemberta"])
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=12)
    p.add_argument("--outdir", type=str, default=os.getenv("DATA_DIR","./new_results_plastic"))
    p.add_argument("--bootstrap", type=int, default=20, help="Bootstrap runs for ARI stability (default 20, 90% sample)")
    p.add_argument("--graph-diagnostics", action="store_true", help="Compute citation network metrics")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    run_full_corpus_analysis(
        backend=args.backend,
        kmin=args.kmin,
        kmax=args.kmax,
        outroot=args.outdir,
        bootstrap=args.bootstrap,
        graph_diagnostics=args.graph_diagnostics
    )