#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
complex_semantic_full.py — Clustering semántico completo + métricas de fragmentación
- Corpus completo sin filtros predefinidos (pero con query base de plásticos)
- SBERT embeddings + ensemble clustering
- Bootstrap ARI, silhouette, modularity Q
- MMR, FPS, diversity metrics
- Term embeddings contextuales
- NUEVAS métricas de fragmentación: Jaccard, Insularity, Cross-domain papers, RBO
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
    "note","received","accepted","revised","issue","volume","pages","doi","elsevier","springer","wiley","mdpi","taylor","francis",
    "plastic", "recycling", "recycled", "recycl", "polymer", "polymers",
    "waste", "material", "materials", "carried", "out", "e", "g", "municipal", "solid",
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
        if not low:
            continue
        if low in TOKEN_CHEM_WHITELIST:
            out.append(low); continue
        if low in STOPWORDS_EN:
            continue
        if len(low) <= 2 and low not in TOKEN_CHEM_WHITELIST:
            continue
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
        try:
            self.driver.close()
        except Exception:
            pass

    def fetch(self, cypher: str, params: Optional[Dict[str,Any]] = None) -> pd.DataFrame:
        params = params or {}
        with self.driver.session(database=self.db) as s:
            rows = [r.data() for r in s.run(cypher, **params)]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for c in ["doi","eid","title","abstract","year","citedBy"]:
            if c not in df.columns:
                df[c] = None
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

        df = pd.concat(
            [df_valid.drop(columns=["doi_norm"],errors="ignore"),
             df_null.drop(columns=["doi_norm","title_norm"],errors="ignore")],
            ignore_index=True
        )

        # Limpieza abstracts + filtro por longitud
        df["abstract"] = df["abstract"].fillna("").astype(str).apply(clean_abstract)
        df["abstract_len_words"] = df["abstract"].str.split().apply(len)

        min_words = int(os.getenv("MIN_ABS_WORDS", 20))
        kept = df[df["abstract_len_words"]>=min_words].drop(columns=["abstract_len_words"]).reset_index(drop=True)

        try:
            yy = pd.to_numeric(kept["year"], errors="coerce")
            if yy.notna().any():
                log.info(f"Años en corpus: {int(yy.min())}..{int(yy.max())} | n={len(kept)}")
        except Exception:
            pass

        return kept


# ========= Cypher for full corpus =========
def cypher_for_full_corpus() -> str:
    return """
    MATCH (p:Paper)
    WHERE p.abstract IS NOT NULL 
      AND p.abstract <> ''
      AND coalesce(p.has_abstract, true) = true
      AND p.publication_year IS NOT NULL
      
      // Términos específicos de reciclaje de plásticos/polímeros
      AND (
        toLower(p.abstract) CONTAINS 'plastic recycling'
        OR toLower(p.abstract) CONTAINS 'recycled plastic'
        OR toLower(p.abstract) CONTAINS 'plastic recycled'
        OR toLower(p.abstract) CONTAINS 'recycled plastics'
        OR toLower(p.abstract) CONTAINS 'plastics recycling'
        OR toLower(p.abstract) CONTAINS 'recycling plastic'
        OR toLower(p.abstract) CONTAINS 'recycling plastics'
        OR toLower(p.abstract) CONTAINS 'recycling of plastic'
        OR toLower(p.abstract) CONTAINS 'recycling of plastics'
        OR toLower(p.abstract) CONTAINS 'polymer recycling'
        OR toLower(p.abstract) CONTAINS 'recycling of polymers'
        OR toLower(p.abstract) CONTAINS 'recycled polymers'
        OR toLower(p.abstract) CONTAINS 'polymers recycling'
      )
      
      // EXCLUIR falsos positivos
      AND NOT toLower(p.abstract) CONTAINS 'pulp'
      AND NOT toLower(p.abstract) CONTAINS 'paper recycling'
    
    RETURN p.doi AS doi, p.openalex_id AS eid, p.title AS title,
           p.abstract AS abstract, toString(p.publication_year) AS year,
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

            if backend == "sbert":
                name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            elif backend == "specter2":
                name = os.getenv("SPECTER2_MODEL", "allenai/specter2_base")
            else:
                name = os.getenv("CHEMBERT_MODEL", "DeepChem/ChemBERTa-77M-MTR")

            log.info(f"Embedding model: {name}")
            st = SentenceTransformer(name)
            self.model = st

            mem_gb = psutil.virtual_memory().total / (1024**3)
            batch_sz = 256 if mem_gb >= 12 else 128

            self.X = st.encode(
                self.proc,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_sz
            )
            log.info(f"Transformer embedding batch_size={batch_sz} (RAM={mem_gb:.1f} GB)")
            return

        raise ValueError(f"Unknown backend: {self.backend}")

    def fit_knn(self, k=10):
        if self.X is None:
            raise RuntimeError("Call embed() first")
        n = len(self.df)
        if n < 2:
            return
        self.knn = NearestNeighbors(
            n_neighbors=min(k+1, max(2, n//2)),
            metric="cosine",
            algorithm="brute"
        )
        Xe = self.X if isinstance(self.X, np.ndarray) else self.X.toarray()
        self.knn.fit(Xe)


# ========= Clustering & selección de modelo =========
def _to_dense(X):
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, 'toarray'):
        return X.toarray()
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
    try:
        s = silhouette_score(Xe, labels, metric=metric)
    except Exception:
        s = float("nan")
    try:
        ch = calinski_harabasz_score(Xe, labels)
        db = davies_bouldin_score(Xe, labels)
    except Exception:
        ch, db = float("nan"), float("nan")
    return s, db, ch

def _choice_sorted(rng: np.random.Generator, n: int, frac: float = 0.9, min_n: int = 20) -> np.ndarray:
    m = max(min_n, int(frac*n))
    m = min(m, n)
    return np.sort(rng.choice(n, size=m, replace=False))

def bootstrap_stability(X, clusterer_fn, k:int, B:int=20, seed:int=42) -> Tuple[float,Tuple[float,float]]:
    Xe = _to_dense(X)
    n = Xe.shape[0]
    if k < 2 or n < 10 or B <= 1:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    labels_list = []
    min_n = max(20, k*3)
    for _ in range(B):
        idx = _choice_sorted(rng, n, frac=0.9, min_n=min_n)
        Xb = Xe[idx]
        lb = clusterer_fn(Xb, k)
        labels_list.append((idx, lb))
    aris = []
    for i in range(len(labels_list)):
        idx_i, li = labels_list[i]
        for j in range(i+1, len(labels_list)):
            idx_j, lj = labels_list[j]
            inter, ia, ja = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(inter) < max(20, k*3):
                continue
            aris.append(adjusted_rand_score(li[ia], lj[ja]))
    if not aris:
        return float("nan"), (float("nan"), float("nan"))
    ar = np.array(aris)
    return float(np.median(ar)), (float(np.quantile(ar,0.25)), float(np.quantile(ar,0.75)))

def cluster_kmeans(X, k:int, random_state:int=42) -> np.ndarray:
    Xe = _to_dense(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    return km.fit_predict(Xe)

def cluster_agglo(X, k:int) -> np.ndarray:
    Xe = _to_dense(X)
    try:
        ac = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
    except TypeError:
        ac = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average')
    return ac.fit_predict(Xe)

def consensus_from_label_sets(label_sets: List[np.ndarray]) -> np.ndarray:
    L = len(label_sets)
    n = len(label_sets[0])
    for l in label_sets:
        if len(l) != n:
            raise ValueError("Label sets size mismatch")
    co = np.zeros((n,n), dtype=float)
    for lab in label_sets:
        same = (lab[:,None] == lab[None,:]).astype(float)
        co += same
    co /= L
    D = 1.0 - co
    ks = [len(np.unique(l)) for l in label_sets]
    k = int(np.median(ks))
    k = max(2, k)
    try:
        ac = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
    except TypeError:
        ac = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    labels = ac.fit_predict(D)
    return labels

def select_model_auto(X, kmin:int, kmax:int, bootstrap:int, seed:int=42, allow_ensemble:bool=True) -> ModelScores:
    Xe = _to_dense(X)
    n = Xe.shape[0]
    grid = auto_k_grid(n, kmin, kmax)
    best: Optional[ModelScores] = None

    for k in grid:
        # KMeans
        labs_km = cluster_kmeans(Xe, k, random_state=seed)
        km_sil, km_db, km_ch = eval_internal(Xe, labs_km)
        km_ari, km_iqr = bootstrap_stability(
            Xe, lambda Xb, kk=k: cluster_kmeans(Xb, kk, random_state=seed),
            k, bootstrap, seed
        )
        cand = ModelScores(k, "kmeans", labs_km, km_sil, km_db, km_ch, km_ari, km_iqr, _cluster_sizes(labs_km))
        best = cand if best is None else (cand if ((km_ch, km_sil, -km_db) > (best.ch, best.silhouette, -best.db)) else best)

        # Agglo
        labs_ag = cluster_agglo(Xe, k)
        ag_sil, ag_db, ag_ch = eval_internal(Xe, labs_ag)
        ag_ari, ag_iqr = bootstrap_stability(
            Xe, lambda Xb, kk=k: cluster_agglo(Xb, kk),
            k, bootstrap, seed
        )
        cand = ModelScores(k, "agglo", labs_ag, ag_sil, ag_db, ag_ch, ag_ari, ag_iqr, _cluster_sizes(labs_ag))
        best = cand if ((ag_ch, ag_sil, -ag_db) > (best.ch, best.silhouette, -best.db)) else best

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
            best = cand if ((en_ch, en_sil, -en_db) > (best.ch, best.silhouette, -best.db)) else best

    if best is None:
        labs = cluster_kmeans(Xe, 2, random_state=seed)
        sil, db, ch = eval_internal(Xe, labs)
        best = ModelScores(2, "kmeans", labs, sil, db, ch, float('nan'), (float('nan'), float('nan')), _cluster_sizes(labs))

    return best


# ========= Etiquetado (c-TF-IDF) =========
def label_topics_c_tfidf(texts: List[str], labels: np.ndarray, top_n=12, ngram=(2,3), min_df=2) -> Dict[int, List[str]]:
    df = pd.DataFrame({"text": texts, "label": labels})
    clusters = [c for c in sorted(df.label.unique()) if c != -1]
    docs = [" ".join(df[df.label==c].text.tolist()) for c in clusters]
    if not docs:
        return {}
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

    banned_terms = {"carried out", "municipal solid", "e g", "sampah plastik", "solid management", "environmental impact"}
    for c in out:
        filtered_terms = [term for term in out[c] if term not in banned_terms]
        out[c] = filtered_terms[:top_n]

    return out


# ========= Representantes, timeline =========
def representative_docs(X, df: pd.DataFrame, labels: np.ndarray, top_m=5) -> Dict[int, pd.DataFrame]:
    df2 = df.copy()
    df2["cluster"] = labels
    Xe = _to_dense(X)
    out: Dict[int, pd.DataFrame] = {}
    for c in sorted(df2.cluster.unique()):
        if c == -1:
            continue
        rows = np.where(df2.cluster.values==c)[0]
        if rows.size == 0:
            continue
        centroid = Xe[rows].mean(axis=0, keepdims=True)
        sims = cosine_similarity(Xe[rows], centroid).ravel()
        order = np.argsort(-sims)[:top_m]
        rep = df2.iloc[rows[order]][["doi","title","year"]].copy()
        rep["rep_similarity"] = sims[order]
        out[int(c)] = rep
    return out

def cluster_timeline(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    d = df.copy()
    d["cluster"] = labels
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
    y_log = np.log(y + 1.0)
    b, ln_a, yhat_log, r2 = _fit_linear(x, y_log)
    a = math.exp(ln_a)
    yhat = a * np.exp(b * x) - 1.0
    return a, b, yhat, r2

def _best_one_break(x, y):
    n = len(x)
    if n < 5:
        return None
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
    if lag >= len(y):
        return 0.0
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    return float(np.dot(y[:-lag], y[lag:]) / (np.sqrt(np.dot(y[:-lag], y[:-lag])*np.dot(y[lag:], y[lag:])) + 1e-12))

def enhanced_timeline_trends(tl: pd.DataFrame) -> pd.DataFrame:
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
        s2 = np.sum(resid**2) / max(1, len(x)-2)
        s = math.sqrt(max(0.0, s2))
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
    Xe = _to_dense(X)
    clus = sorted(np.unique(labels))
    cents = []
    for c in clus:
        rows = np.where(labels==c)[0]
        cents.append(Xe[rows].mean(axis=0, keepdims=True))
    C = np.vstack(cents)
    M = cosine_similarity(C)
    return pd.DataFrame(M, index=[f"C{c}" for c in clus], columns=[f"C{c}" for c in clus])


# ========= Diversidad + MMR/FPS + Cobertura adaptativa =========
def _cosine_dist(A, B=None):
    S = cosine_similarity(A, B) if B is not None else cosine_similarity(A)
    return 1.0 - S

def diversity_metrics(X):
    Xe = _to_dense(X)
    n = Xe.shape[0]
    if n <= 5000:
        D = _cosine_dist(Xe)
        iu = np.triu_indices(n, k=1)
        dvals = D[iu]
        diam = float(dvals.max()) if dvals.size else 0.0
        mean_d = float(dvals.mean()) if dvals.size else 0.0
        p90 = float(np.quantile(dvals, 0.90)) if dvals.size else 0.0
        p95 = float(np.quantile(dvals, 0.95)) if dvals.size else 0.0
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=min(5000, n), replace=False)
        D = _cosine_dist(Xe[idx])
        iu = np.triu_indices(len(idx), k=1)
        dvals = D[iu]
        diam, mean_d = float(dvals.max()), float(dvals.mean())
        p90 = float(np.quantile(dvals,0.90))
        p95 = float(np.quantile(dvals,0.95))

    Xc = Xe - Xe.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, n - 1)
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 0, None)
    tr = float(w.sum()) + 1e-12
    participation_ratio = float((tr**2) / (np.sum(w**2) + 1e-12))
    spectral_entropy = float(-np.sum((w/tr) * np.log((w/tr) + 1e-12)))
    return {"n": int(n), "diameter_cos": diam, "mean_pairwise_cos": mean_d,
            "p90_pairwise_cos": p90, "p95_pairwise_cos": p95,
            "participation_ratio": participation_ratio, "spectral_entropy": spectral_entropy}

def mmr_select(X, k=10, lam=0.5, query_vec=None):
    Xe = _to_dense(X)
    n = Xe.shape[0]
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    if query_vec is None:
        query_vec = Xe.mean(axis=0, keepdims=True)

    def _norm(A):
        nrm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        return A / nrm

    Q = _norm(query_vec)
    Z = _norm(Xe)
    rel = cosine_similarity(Z, Q).ravel()

    selected = []
    candidates = set(range(n))

    i0 = int(np.argmax(rel))
    selected.append(i0)
    candidates.remove(i0)

    while len(selected) < min(k, n):
        sel_mat = Z[selected]
        cand_list = np.array(list(candidates))
        max_sim_to_S = cosine_similarity(Z[cand_list], sel_mat).max(axis=1)
        scores = lam * rel[cand_list] - (1.0 - lam) * max_sim_to_S
        pick = int(cand_list[np.argmax(scores)])
        selected.append(pick)
        candidates.remove(pick)

    return np.array(selected, dtype=int)

def farthest_point_sampling(X, k=10):
    Xe = _to_dense(X)
    n = Xe.shape[0]
    if n == 0 or k <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(42)
    first = int(rng.integers(0, n))
    centers = [first]
    dmin = _cosine_dist(Xe, Xe[[first]]).ravel()
    for _ in range(1, min(k, n)):
        nxt = int(np.argmax(dmin))
        centers.append(nxt)
        dmin = np.minimum(dmin, _cosine_dist(Xe, Xe[[nxt]]).ravel())
    return np.array(centers, dtype=int)

def adaptive_coverage_analysis(X, seeds_idx, percentiles=(0.10, 0.25, 0.50, 0.75)):
    Xe = _to_dense(X)
    if len(seeds_idx) == 0:
        return [{"radius_cos": float('nan'), "covered_frac": 0.0, "p": float(p)} for p in percentiles]
    S = Xe[seeds_idx]
    D = _cosine_dist(Xe, S)
    d_nn = D.min(axis=1)
    cov = []
    base = np.quantile(d_nn, percentiles)
    for p, r in zip(percentiles, base):
        cov.append({"radius_cos": float(r), "covered_frac": float((d_nn <= r).mean()), "p": float(p)})
    return cov


# ========= Embeddings de términos (contextuales) =========
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _embed_terms_raw(terms: List[str], analyzer: Analyzer) -> np.ndarray:
    if analyzer.model is not None and hasattr(analyzer.model, "encode"):
        vecs = analyzer.model.encode(terms, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return vecs
    tfv = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=1)
    X = tfv.fit_transform(terms).toarray().astype(float)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def contextual_term_embeddings(terms: List[str], documents: List[str], analyzer: Analyzer, max_sents_per_term:int=2000) -> np.ndarray:
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
                    if len(sents) >= max_sents_per_term:
                        break
            if len(sents) >= max_sents_per_term:
                break
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
    log.info("--- Calculando Jaccard vocabulary overlap ---")

    clusters = sorted(topics.keys())
    n = len(clusters)
    jaccard = np.zeros((n, n))

    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i == j:
                jaccard[i, j] = 1.0
            else:
                set1 = set(topics[c1][:50])
                set2 = set(topics[c2][:50])
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard[i, j] = intersection / union if union > 0 else 0.0

    df_jaccard = pd.DataFrame(
        jaccard,
        index=[f"C{c}" for c in clusters],
        columns=[f"C{c}" for c in clusters]
    )
    df_jaccard.to_csv(os.path.join(outdir, "jaccard_vocabulary.csv"))
    log.info("✓ Jaccard matrix guardada")

    upper_tri = jaccard[np.triu_indices(n, k=1)]
    log.info(f"  - Mean Jaccard (off-diagonal): {upper_tri.mean():.3f}")
    log.info(f"  - Pares con Jaccard < 0.10 (silos fuertes): {(upper_tri < 0.10).sum()}")
    log.info(f"  - Pares con Jaccard > 0.30 (overlap alto): {(upper_tri > 0.30).sum()}")

    return df_jaccard

def compute_cross_domain_papers(X, labels: np.ndarray, outdir: str, threshold: float = 0.70):
    log.info("--- Identificando cross-domain papers ---")
    Xe = _to_dense(X)

    clusters = sorted(np.unique(labels))
    if len(clusters) < 2:
        log.warning("Solo hay 1 cluster: cross-domain no aplica.")
        return pd.DataFrame(columns=["paper_idx","assigned_cluster","n_close_clusters","max_sim_to_other"])

    centroids = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        centroids.append(Xe[idx].mean(axis=0, keepdims=True))
    C = np.vstack(centroids)

    sims = cosine_similarity(Xe, C)
    close_counts = (sims > threshold).sum(axis=1)
    cross_domain_mask = close_counts >= 2
    n_cross = int(cross_domain_mask.sum())
    pct_cross = 100.0 * n_cross / len(labels)

    log.info(f"✓ Cross-domain papers: {n_cross} ({pct_cross:.2f}%)")
    cross_indices = np.where(cross_domain_mask)[0]

    # 2nd highest sim (max sim to OTHER centroid)
    sims_sub = sims[cross_indices]
    if sims_sub.size == 0:
        second_best = np.array([], dtype=float)
    else:
        # sort each row desc and take 2nd
        second_best = np.sort(sims_sub, axis=1)[:, -2]

    df_cross = pd.DataFrame({
        "paper_idx": cross_indices,
        "assigned_cluster": labels[cross_indices],
        "n_close_clusters": close_counts[cross_indices],
        "max_sim_to_other": second_best
    })

    df_cross.to_csv(os.path.join(outdir, "cross_domain_papers.csv"), index=False)
    log.info("✓ Cross-domain papers guardados")

    with open(os.path.join(outdir, "fragmentation_summary.txt"), "w") as f:
        f.write("# FRAGMENTATION SUMMARY\n\n")
        f.write(f"Total papers: {len(labels)}\n")
        f.write(f"Cross-domain papers (similarity > {threshold} to ≥2 clusters): {n_cross} ({pct_cross:.2f}%)\n")
        f.write(f"Single-domain papers: {len(labels) - n_cross} ({100 - pct_cross:.2f}%)\n")

    return df_cross

def compute_tfidf_baseline_overlap(texts: List[str], labels: np.ndarray, outdir: str, top_n: int = 40):
    log.info("--- Baseline TF-IDF lexical similarity (tokens only) ---")
    df = pd.DataFrame({"text": texts, "cluster": labels})
    clusters = sorted(df.cluster.unique())

    vec = TfidfVectorizer(
        token_pattern=r"[A-Za-z0-9_\-]+",
        ngram_range=(1,1),
        min_df=2,
        max_df=0.95
    )
    X = vec.fit_transform(df["text"])
    terms = np.array(vec.get_feature_names_out())

    vocab = {}
    for c in clusters:
        idx = df[df.cluster == c].index
        if len(idx) == 0:
            vocab[c] = []
            continue
        Xc = X[idx].mean(axis=0).A1
        top_idx = np.argsort(-Xc)[:top_n]
        vocab[c] = terms[top_idx].tolist()

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
    log.info("✓ Baseline TF-IDF Jaccard saved")
    log.info(f"  Mean off-diagonal: {jac[np.triu_indices(len(clusters),1)].mean():.3f}")

    if rbo:
        rbo_mat = np.zeros_like(jac)
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                if i == j:
                    rbo_mat[i,j] = 1.0
                else:
                    try:
                        rbo_mat[i,j] = rbo.RankingSimilarity(vocab[c1], vocab[c2]).rbo(p=0.9)
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

def compute_rbo_fragmentation(topics: Dict[int, List[str]], outdir: str, p: float = 0.9, top_n: int = 50):
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

    df_rbo = pd.DataFrame(
        1 - rbo_mat,
        index=[f"C{c}" for c in clusters],
        columns=[f"C{c}" for c in clusters]
    )
    df_rbo.to_csv(os.path.join(outdir, "rbo_fragmentation.csv"))
    log.info("✓ RBO fragmentation matrix guardada")

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


# ========= Validaciones externas desde Neo4j =========
def fetch_citation_edges(neo: Neo4jClient, dois: List[str]) -> Optional[pd.DataFrame]:
    dois = [d for d in dois if isinstance(d, str) and len(d)>0]
    if not dois:
        return None
    cy = """
        UNWIND $dois AS d
        MATCH (p:Paper {doi:d})-[:CITES]->(q:Paper)
        WHERE q.doi IS NOT NULL
        RETURN p.doi AS src, q.doi AS dst
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
        MATCH (a:Author)-[:AUTHORED]->(p:Paper {doi:d})<-[:AUTHORED]-(b:Author)
        WHERE id(a) < id(b)
        RETURN a.name AS a, b.name AS b
    """
    try:
        with neo.driver.session(database=neo.db) as s:
            rows = [r.data() for r in s.run(cy, dois=dois)]
        df = pd.DataFrame(rows)
        return df if not df.empty else None
    except Exception as e:
        log.info(f"Coauthor fetch skipped/failed: {e}")
        return None

def fetch_journal_pairs(neo: Neo4jClient, dois: List[str]) -> Optional[pd.DataFrame]:
    cy = """
        UNWIND $dois AS d
        MATCH (p:Paper {doi:d})-[:PUBLISHED_IN]->(j:Journal)
        RETURN d AS doi, j.name AS journal
    """
    try:
        with neo.driver.session(database=neo.db) as s:
            rows = [r.data() for r in s.run(cy, dois=dois)]
        df = pd.DataFrame(rows)
        return df if not df.empty else None
    except Exception as e:
        log.info(f"Journal fetch skipped/failed: {e}")
        return None

def modularity_on_clusters(edges: Optional[pd.DataFrame], labels_by_doi: Dict[str,int]) -> Optional[float]:
    if nx is None or edges is None or edges.empty:
        return None
    G = nx.from_pandas_edgelist(edges, 'src', 'dst', create_using=nx.Graph())
    if G.number_of_nodes() < 10:
        return None
    nodes = [n for n in G.nodes if n in labels_by_doi and labels_by_doi[n] != -1]
    if len(nodes) < 10:
        return None
    part: Dict[int, List[str]] = {}
    for n in nodes:
        c = labels_by_doi[n]
        part.setdefault(c, []).append(n)
    try:
        import networkx.algorithms.community.quality as q
        comms = [set(v) for v in part.values() if len(v) > 0]
        return float(q.modularity(G.subgraph(nodes), comms))
    except Exception:
        return None

def multi_external_validation(neo: Neo4jClient, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    labels_by_doi = {d:int(c) for d,c in zip(df['doi'].tolist(), labels)}

    edges = fetch_citation_edges(neo, df['doi'].tolist())
    modQ = modularity_on_clusters(edges, labels_by_doi)
    if modQ is not None:
        out["citation_modularity_Q"] = round(modQ, 4)

    co = fetch_coauthorship(neo, df['doi'].tolist())
    if nx is not None and co is not None and not co.empty:
        G = nx.from_pandas_edgelist(co, 'a', 'b', create_using=nx.Graph())
        out["coauth_components"] = int(nx.number_connected_components(G)) if G.number_of_nodes() else 0
        out["coauth_avg_degree"] = float(np.mean([d for _,d in G.degree()])) if G.number_of_nodes() else float('nan')

    jp = fetch_journal_pairs(neo, df['doi'].tolist())
    if jp is not None and not jp.empty:
        ent = []
        for c in np.unique(labels):
            sub = jp[jp['doi'].isin(df.loc[labels==c,'doi'])]
            if sub.empty:
                continue
            p = sub['journal'].value_counts(normalize=True).values + 1e-12
            ent.append(float(-(p*np.log(p)).sum()))
        out["journal_entropy_mean"] = float(np.mean(ent)) if ent else float('nan')

    return out


def compute_citation_insularity(neo: Neo4jClient, df: pd.DataFrame, labels: np.ndarray, outdir: str):
    log.info("--- Calculando citation insularity ---")

    edges = fetch_citation_edges(neo, df['doi'].tolist())
    if edges is None or edges.empty:
        log.warning("No citation edges found")
        return None

    rows = []
    for c in sorted(np.unique(labels)):
        dois_in_c = df.loc[labels==c, 'doi'].tolist()
        edges_from_c = edges[edges['src'].isin(dois_in_c)]
        n_total = len(edges_from_c)

        if n_total == 0:
            rows.append({"cluster": int(c), "total_citations": 0,
                         "internal_citations": 0, "external_citations": 0,
                         "insularity_pct": float('nan')})
            continue

        edges_internal = edges_from_c[edges_from_c['dst'].isin(dois_in_c)]
        n_internal = len(edges_internal)
        n_external = n_total - n_internal
        insularity = 100.0 * n_internal / n_total

        rows.append({
            "cluster": int(c),
            "total_citations": n_total,
            "internal_citations": n_internal,
            "external_citations": n_external,
            "insularity_pct": insularity
        })

    df_insular = pd.DataFrame(rows)
    df_insular.to_csv(os.path.join(outdir, "citation_insularity.csv"), index=False)
    log.info("✓ Citation insularity guardada")

    mean_ins = df_insular['insularity_pct'].mean()
    log.info(f"  - Mean insularity: {mean_ins:.1f}%")
    log.info(f"  - Clusters con >70% insularity (silos): {(df_insular['insularity_pct'] > 70).sum()}")

    return df_insular


# ========= Exports & reports =========
def export_tables(name: str, df: pd.DataFrame, labels: np.ndarray, topics: Dict[int,List[str]],
                  reps: Dict[int,pd.DataFrame], tl: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    short = {c: ", ".join(v[:4]) for c, v in topics.items()}
    df_map = df.copy()
    df_map["cluster"] = labels
    df_map["topic_label"] = df_map["cluster"].map(short)
    df_map[["doi","title","year","cluster","topic_label"]].to_csv(os.path.join(outdir,"paper_topics.csv"), index=False)

    rows = [{"cluster": c, "top_terms": "; ".join(topics.get(c, []))} for c in sorted(set(labels)) if c != -1]
    pd.DataFrame(rows).to_csv(os.path.join(outdir,"semantic_topics.csv"), index=False)

    tl.to_csv(os.path.join(outdir,"cluster_timeline.csv"), index=False)

    rep_tables = []
    for c, t in reps.items():
        t2 = t.copy()
        t2.insert(0, "cluster", c)
        rep_tables.append(t2)
    if rep_tables:
        pd.concat(rep_tables, ignore_index=True).to_csv(os.path.join(outdir,"cluster_representatives.csv"), index=False)

def export_diversity(name: str, df: pd.DataFrame, X, labels: np.ndarray, outdir: str, mmr_k:int=10, lam:float=0.55):
    os.makedirs(outdir, exist_ok=True)
    Xe = _to_dense(X)
    all_rows, mmr_rows, fps_rows, cov_rows = [], [], [], []
    for c in sorted(np.unique(labels)):
        idx = np.where(labels==c)[0]
        if len(idx)==0:
            continue
        Xc = Xe[idx]
        dmet = diversity_metrics(Xc)
        dmet["cluster"] = int(c)
        all_rows.append(dmet)

        q = Xc.mean(axis=0, keepdims=True)

        mmr_idx_local = mmr_select(Xc, k=min(mmr_k, len(idx)), lam=lam, query_vec=q)
        mmr_idx_global = idx[mmr_idx_local]
        for j, gi in enumerate(mmr_idx_global):
            r = df.iloc[gi]
            mmr_rows.append({"cluster": int(c), "rank": j+1, "global_idx": int(gi),
                             "doi": r.get("doi"), "title": r.get("title"), "year": r.get("year")})

        fps_idx_local = farthest_point_sampling(Xc, k=min(mmr_k, len(idx)))
        fps_idx_global = idx[fps_idx_local]
        for j, gi in enumerate(fps_idx_global):
            r = df.iloc[gi]
            fps_rows.append({"cluster": int(c), "rank": j+1, "global_idx": int(gi),
                             "doi": r.get("doi"), "title": r.get("title"), "year": r.get("year")})

        for method, loc_idx in [("MMR", mmr_idx_local), ("FPS", fps_idx_local)]:
            if len(loc_idx):
                d_nn = _cosine_dist(Xc, Xc[loc_idx]).min(axis=1)
                for rad in [0.05, 0.10, 0.20, 0.30]:
                    cov_rows.append({"cluster": int(c), "method": method, "k": int(len(loc_idx)),
                                     "radius_cos": rad, "covered_frac": float((d_nn <= rad).mean())})
            for cc in adaptive_coverage_analysis(Xc, loc_idx, percentiles=(0.10,0.25,0.50,0.75)):
                cov_rows.append({"cluster": int(c), "method": method+"_adaptive", "k": int(len(loc_idx)), **cc})

    if all_rows:
        pd.DataFrame(all_rows).to_csv(os.path.join(outdir,"diversity_metrics.csv"), index=False)
    if mmr_rows:
        pd.DataFrame(mmr_rows).to_csv(os.path.join(outdir,"mmr_representatives.csv"), index=False)
    if fps_rows:
        pd.DataFrame(fps_rows).to_csv(os.path.join(outdir,"fps_representatives.csv"), index=False)
    if cov_rows:
        pd.DataFrame(cov_rows).to_csv(os.path.join(outdir,"coverage_curves.csv"), index=False)

def write_semantic_report(name: str, df: pd.DataFrame, labels: np.ndarray, topics: Dict[int,List[str]],
                          reps: Dict[int,pd.DataFrame], outdir:str, method_str:str):
    path = os.path.join(outdir, "semantic_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Semantic Topic Report\n\n")
        f.write(f"Cluster set: {name}\n\n")
        f.write(f"Total papers: {len(df)}\n\n")
        f.write(f"**Clustering method:** {method_str}\n\n")
        for c in sorted(set(labels)):
            if c == -1:
                continue
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

def write_config(outdir:str, config:Dict[str,Any]):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(to_py(config), f, indent=2, ensure_ascii=False)


# ========= Insights por clúster =========
def per_cluster_metrics(X, df: pd.DataFrame, labels: np.ndarray, tl_trends: pd.DataFrame) -> pd.DataFrame:
    Xe = _to_dense(X)
    d = df.copy()
    d["cluster"] = labels
    d["citedBy"] = pd.to_numeric(d["citedBy"], errors="coerce").fillna(0.0)

    try:
        s_samples = silhouette_samples(Xe, labels, metric="cosine")
    except Exception:
        s_samples = np.full(len(labels), np.nan)
    d["sil_sample"] = s_samples

    rows = []
    for c in sorted(d.cluster.unique()):
        sub = d[d.cluster==c]
        idx = np.where(labels==c)[0]
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

    labels_out, reasons = [], []
    for _, r in df_metrics.iterrows():
        lbl = "—"
        reason = []
        if (not math.isnan(ari_global)) and (ari_global >= 0.6) and (r["citations_per_paper"] < 3.0):
            lbl = "Nichos Académicos Maduros"
            reason = [f"ARI_global={ari_global:.2f} (≥0.6)", f"citas/paper={r['citations_per_paper']:.2f} (<3)"]
        if (not math.isnan(ari_global)) and (ari_global < 0.4) and (not math.isnan(base_slope)) and (r["slope"] > 2.0*base_slope):
            lbl = "Campos Emergentes Sin Consenso"
            reason = [f"slope={r['slope']:.3f} (>2× {base_slope:.3f})", f"ARI_global={ari_global:.2f} (<0.4)"]
        if lbl == "—":
            lbl = "Mixto / Indeterminado"
            reason.append("Sin señales claras; revisar términos/tópicos y t_like")
        labels_out.append(lbl)
        reasons.append("; ".join(reason))

    out = df_metrics.copy()
    out["insight_label"] = labels_out
    out["insight_reason"] = reasons
    out["ari_global"] = ari_global
    out["explosive_baseline_slope"] = base_slope
    return out

def write_business_insights(name: str, insights_df: pd.DataFrame, outdir: str):
    insights_df.to_csv(os.path.join(outdir, "cluster_insights.csv"), index=False)
    path = os.path.join(outdir, "business_insights.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Business-Oriented Insights per Cluster\n\n")
        f.write(f"Cluster set: {name}\n\n")
        if len(insights_df):
            f.write(f"- ARI_global: {insights_df['ari_global'].iloc[0]:.3f} | baseline_slope_pos: {insights_df['explosive_baseline_slope'].iloc[0]:.4f}\n\n")
        for _, r in insights_df.sort_values("cluster").iterrows():
            f.write(f"## Cluster {int(r['cluster'])} — {r['insight_label']}\n\n")
            f.write(f"- n: {int(r['n'])}\n")
            f.write(f"- citas/paper: {r['citations_per_paper']:.2f}\n")
            f.write(f"- silhouette_mean: {r['silhouette_mean']:.3f}\n")
            f.write(f"- cohesion (centroid cosine): {r['centroid_cohesion']:.3f}\n")
            f.write(f"- slope: {r['slope']:.4f} | t-like: {r['t_like']:.2f}\n")
            f.write(f"- Rationale: {r['insight_reason']}\n\n")


# ========= Plot =========
def save_plot_2d(X, labels: np.ndarray, outpath: str, title: str = "PCA 2D"):
    try:
        P = PCA(n_components=2, random_state=42).fit_transform(_to_dense(X))
    except Exception as e:
        log.warning(f"PCA plot skipped: {e}")
        return
    try:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.figure(figsize=(7,5))
        for c in sorted(np.unique(labels)):
            idx = np.where(labels==c)[0]
            if idx.size == 0:
                continue
            plt.scatter(P[idx,0], P[idx,1], s=12, alpha=0.7, label=f"C{c}")
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(markerscale=1, fontsize=8)
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
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
            log.warning(f"[{name}] no results")
            return
        log.info(f"[{name}] papers: {len(df)}")

        an = Analyzer(backend=backend)
        an.set_df(df)
        an.preprocess()
        an.embed()
        an.fit_knn(k=10)

        # ✅ YA NO hardcodea k=6: usa args.kmin/kmax como debe
        scores = select_model_auto(an.X, kmin=kmin, kmax=kmax, bootstrap=bootstrap)

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

        log.info("\n[0/4] Baseline lexical overlap (TF-IDF simple)...")
        compute_tfidf_baseline_overlap(an.proc or [], scores.labels, outdir, top_n=40)

        log.info("\n[1/4] Jaccard vocabulary overlap (c-TF-IDF)...")
        compute_jaccard_matrix(topics, outdir)

        if graph_diagnostics:
            log.info("\n[2/4] Citation insularity (graph-based)...")
            compute_citation_insularity(neo, df, scores.labels, outdir)
        else:
            log.info("\n[2/4] Citation insularity SKIPPED (use --graph-diagnostics to enable)")

        log.info("\n[3/4] Cross-domain papers (embedding-based)...")
        compute_cross_domain_papers(an.X, scores.labels, outdir, threshold=0.70)

        log.info("\n[4/4] RBO-based epistemic fragmentation (c-TF-IDF)...")
        compute_rbo_fragmentation(topics, outdir, p=0.9, top_n=50)

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
            "corpus_type": "full_unfiltered",
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
    p.add_argument("--outdir", type=str, default=os.getenv("DATA_DIR","./results_plastic"))
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
