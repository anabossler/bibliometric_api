#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_tmo_convergence.py

Validez convergente del léxico *practice-to-talk* frente a la capa TMO
(Technique-Material-Objective) del grafo de conocimiento.

Niveles de análisis
-------------------
  1. Artículo  : modelos logit de presencia/tipo de nodos TMO ~ léxico.
  2. 11 clusters y 5 mega-áreas : correlación de Spearman entre el ratio
     léxico (practice/talk) y un `struct_score` derivado del grafo.
  3. Test de permutación (null) sobre el nivel de cluster.

Entradas
--------
  --corpus   CSV con al menos las columnas: doi, abstract
  --topics   CSV con al menos las columnas: doi, cluster
  Capa TMO   : desde Neo4j (por defecto) o desde un CSV previamente exportado
               con --tmo-csv (permite reproducir sin acceso a la base).

Salidas (en --outdir)
---------------------
  tmo_graph_export.csv        capa TMO cruda extraída del grafo
  tmo_article_level.csv       tabla de análisis a nivel de artículo
  tmo_level_cl[_tmo].csv      agregados por cluster (todos / solo con TMO)
  tmo_level_mega[_tmo].csv    agregados por mega-área
  tmo_logit_results.csv       coeficientes, odds ratios e IC de los logits
  tmo_convergence_summary.csv rho de Spearman y p de permutación
  tmo_permutation_null.csv    distribución nula completa
  run_metadata.json           versiones, semilla, hashes de entrada, commit

Credenciales
------------
NUNCA se escriben en el código ni se imprimen. La contraseña se toma de la
variable de entorno indicada en --password-env (por defecto NEO4J_PASSWORD);
si no existe y hay terminal interactivo, se pide por `getpass`.

Uso
---
    export NEO4J_PASSWORD='...'
    python validate_tmo_convergence.py \
        --uri bolt://localhost:7687 --database circulareconomy \
        --corpus corpus_circular_economy.csv \
        --topics results_circular_economy/full_corpus/paper_topics.csv \
        --outdir results_tmo_validation

    # reproducción sin base de datos:
    python validate_tmo_convergence.py --tmo-csv results_tmo_validation/tmo_graph_export.csv ...

Dependencias
------------
    pandas>=2.0  numpy>=1.24  scipy>=1.10  statsmodels>=0.14  neo4j>=5.0
    ce_vocab.py  (fuente única de verdad del léxico; se distribuye con el repo)

Notas metodológicas
-------------------
* Un artículo puede enlazar varios nodos TMO. Con --tmo-agg representative
  (por defecto) se usa el primer nombre en orden alfabético, lo que hace la
  selección determinista y reproducible; con --tmo-agg any un artículo cuenta
  como positivo si *alguno* de sus nodos satisface el criterio.
* Las tablas por nivel se calculan dos veces: sobre todos los artículos
  (los que carecen de TMO cuentan como 0, igual que en el análisis principal)
  y solo sobre los que tienen TMO extraído, como comprobación de robustez
  frente a la cobertura desigual del extractor.
* El p de permutación se reporta con la corrección (1+k)/(1+B) recomendada
  para tests de Monte Carlo, junto al valor sin corregir.

Licencia: MIT.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LOG = logging.getLogger("tmo")

# ---------------------------------------------------------------------------
# 0. Configuración fija del dominio (no contiene datos sensibles)
# ---------------------------------------------------------------------------

MEGA = {
    "Materials & Technical Recycling":    [10, 11],
    "Industrial Sectors & Applied Cases": [7, 8, 9],
    "Energy & Resource Systems":          [2, 14],
    "Business, Policy & Governance":      [4, 6],
    "Sustainability Framing & Society":   [3, 5],
}
CLUSTER_TO_MEGA = {c: m for m, cs in MEGA.items() for c in cs}

# --- clasificadores por reglas (editables y auditables) --------------------
RE_RESEARCH_METHOD = re.compile(
    r'review|survey|bibliometric|case stud|interview|questionnaire|'
    r'content analysis|meta-|framework|conceptual|structural equation|delphi|'
    r'literature|scientometric|regression|econometric|qualitative|quantitative|'
    r'mixed.?method|focus group|multi.?criteria|swot|panel data', re.I)

RE_MATERIAL_OPERATION = re.compile(
    r'pyrolys|recycl|extract|digest|compost|gasif|adsorp|life.?cycle|'
    r'\blca\b|remanufactur|recover|valoris|valoriz|treatment|synthesis|leach|'
    r'manufactur|3d print|additive|incinerat|fermentat|electroly|catalys|'
    r'depolymer|hydrolys|carboniz|sorting|shredd|material flow', re.I)

RE_OBJECTIVE_DISCURSIVE = re.compile(
    r'sustainable development|circular econom|identify|explore|'
    r'understand|examine|assess barrier|research trend|challenge|awareness|'
    r'value creation|framework|conceptual|ce implementation|ce adoption|'
    r'transition|policy', re.I)

RE_OBJECTIVE_OPERATIONAL = re.compile(
    r'valoris|valoriz|recover|improve .*(propert|efficien|performance)|'
    r'reduce (waste|emission|carbon|footprint|cost)|reus|recycl|'
    r'resource efficien|energy recover|material recover|treat|remediat|'
    r'measure circularit|extend .*(life|lifespan)', re.I)

CYPHER_TMO = """
MATCH (p:Paper)
WHERE p.doi IS NOT NULL AND trim(p.doi) <> ''
OPTIONAL MATCH (p)-[:USES_TECHNIQUE]->(t:Technique)
OPTIONAL MATCH (p)-[:TARGETS_MATERIAL]->(m:Material)
OPTIONAL MATCH (p)-[:PURSUES_OBJECTIVE]->(o:Objective)
RETURN toLower(trim(p.doi))      AS doi,
       collect(DISTINCT t.name)  AS tech_names,
       collect(DISTINCT m.name)  AS mat_names,
       collect(DISTINCT o.name)  AS obj_names
"""

LIST_SEP = " | "
STRUCT_COLS = ["tech_op", "mat_pres", "obj_op", "obj_disc"]


def classify_technique(name: str) -> str:
    """material_operation > research_method > unclassified (orden deliberado)."""
    n = str(name)
    if RE_MATERIAL_OPERATION.search(n):
        return "material_operation"
    if RE_RESEARCH_METHOD.search(n):
        return "research_method"
    return "unclassified"


def classify_objective(name: str) -> str:
    """operational > discursive > unclassified (orden deliberado)."""
    n = str(name)
    if RE_OBJECTIVE_OPERATIONAL.search(n):
        return "operational"
    if RE_OBJECTIVE_DISCURSIVE.search(n):
        return "discursive"
    return "unclassified"


# ---------------------------------------------------------------------------
# 1. Utilidades
# ---------------------------------------------------------------------------

_DOI_PREFIX = re.compile(r'^(https?://)?(dx\.)?doi\.org/', re.I)


def normalize_doi(value) -> str:
    """Minúsculas, sin prefijo de resolución y sin puntuación final."""
    s = str(value).strip().lower()
    s = _DOI_PREFIX.sub("", s)
    return s.rstrip(" .,;")


def parse_cluster(value):
    """'C10' / '10' / '10.0' -> 10 ; devuelve None si no hay entero."""
    m = re.search(r'-?\d+', str(value))
    return int(m.group(0)) if m else None


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5,
                           cwd=Path(__file__).resolve().parent)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def require_columns(df: pd.DataFrame, cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] {source}: faltan las columnas {missing}. "
                         f"Presentes: {list(df.columns)[:20]}")


def load_vocabulary():
    """Importa el léxico desde ce_vocab (fuente única de verdad del estudio)."""
    try:
        from ce_vocab import CE_TALK_RE, CE_PRACTICE_RE  # noqa: WPS433
    except ImportError as exc:
        raise SystemExit(
            "[error] no se pudo importar ce_vocab (CE_TALK_RE, CE_PRACTICE_RE).\n"
            "        Coloca ce_vocab.py junto a este script o añádelo a PYTHONPATH."
        ) from exc
    return CE_TALK_RE, CE_PRACTICE_RE


# ---------------------------------------------------------------------------
# 2. Capa TMO: grafo o CSV
# ---------------------------------------------------------------------------

def resolve_password(user: str, password_env: str) -> str:
    pw = os.environ.get(password_env)
    if pw:
        return pw
    if sys.stdin.isatty():
        return getpass.getpass(f"Contraseña Neo4j para '{user}': ")
    raise SystemExit(
        f"[error] falta la contraseña: define ${password_env} o ejecuta en un "
        f"terminal interactivo. La contraseña nunca se almacena en el código."
    )


def fetch_tmo_from_graph(uri: str, database: str, user: str, password: str) -> pd.DataFrame:
    try:
        from neo4j import GraphDatabase  # noqa: WPS433 (import perezoso)
    except ImportError as exc:
        raise SystemExit("[error] falta el paquete 'neo4j' (pip install neo4j) "
                         "o usa --tmo-csv para trabajar sin base de datos.") from exc

    LOG.info("consultando el grafo en %s (db=%s)", uri, database)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            records = [dict(r) for r in session.run(CYPHER_TMO)]
    finally:
        driver.close()

    tmo = pd.DataFrame(records, columns=["doi", "tech_names", "mat_names", "obj_names"])
    for col in ("tech_names", "mat_names", "obj_names"):
        tmo[col] = tmo[col].map(lambda xs: sorted({str(x) for x in (xs or []) if x}))
    return tmo


def load_tmo_csv(path: Path) -> pd.DataFrame:
    tmo = pd.read_csv(path)
    require_columns(tmo, ["doi", "tech_names", "mat_names", "obj_names"], str(path))
    for col in ("tech_names", "mat_names", "obj_names"):
        tmo[col] = tmo[col].fillna("").map(
            lambda s: sorted(x.strip() for x in str(s).split(LIST_SEP) if x.strip()))
    return tmo


def export_tmo_csv(tmo: pd.DataFrame, path: Path) -> None:
    out = tmo.copy()
    for col in ("tech_names", "mat_names", "obj_names"):
        out[col] = out[col].map(LIST_SEP.join)
    out.to_csv(path, index=False)


def build_tmo_features(tmo: pd.DataFrame, agg: str) -> pd.DataFrame:
    """Deriva los indicadores binarios TMO por artículo."""
    t = tmo.copy()
    t["doi"] = t["doi"].map(normalize_doi)
    t = t[t["doi"].astype(bool)].drop_duplicates("doi").reset_index(drop=True)

    first = lambda xs: xs[0] if xs else None          # noqa: E731 (ya viene ordenado)
    t["tech_name"] = t["tech_names"].map(first)
    t["mat_name"] = t["mat_names"].map(first)
    t["obj_name"] = t["obj_names"].map(first)

    t["tech_class"] = t["tech_name"].map(lambda x: classify_technique(x) if x else "none")
    t["obj_class"] = t["obj_name"].map(lambda x: classify_objective(x) if x else "none")

    if agg == "representative":
        t["tech_op"] = (t["tech_class"] == "material_operation").astype(int)
        t["obj_op"] = (t["obj_class"] == "operational").astype(int)
        t["obj_disc"] = (t["obj_class"] == "discursive").astype(int)
        t["obj_is_ce"] = t["obj_name"].map(
            lambda x: int(str(x).strip().lower() == "circular economy") if x else 0)
    else:  # "any": positivo si algún nodo del artículo cumple el criterio
        t["tech_op"] = t["tech_names"].map(
            lambda xs: int(any(classify_technique(x) == "material_operation" for x in xs)))
        t["obj_op"] = t["obj_names"].map(
            lambda xs: int(any(classify_objective(x) == "operational" for x in xs)))
        t["obj_disc"] = t["obj_names"].map(
            lambda xs: int(any(classify_objective(x) == "discursive" for x in xs)))
        t["obj_is_ce"] = t["obj_names"].map(
            lambda xs: int(any(str(x).strip().lower() == "circular economy" for x in xs)))

    t["mat_pres"] = t["mat_names"].map(lambda xs: int(bool(xs)))
    t["n_tech"] = t["tech_names"].map(len)
    t["n_mat"] = t["mat_names"].map(len)
    t["n_obj"] = t["obj_names"].map(len)
    t["has_tmo"] = (t[["n_tech", "n_mat", "n_obj"]].sum(axis=1) > 0)

    LOG.info("grafo: %s papers | tech_class=%s", f"{len(t):,}",
             t["tech_class"].value_counts().to_dict())
    LOG.info("grafo: obj_class=%s", t["obj_class"].value_counts().to_dict())
    return t.drop(columns=["tech_names", "mat_names", "obj_names"])


# ---------------------------------------------------------------------------
# 3. Léxico y dataset de análisis
# ---------------------------------------------------------------------------

def build_dataset(corpus_path: Path, topics_path: Path, tmo: pd.DataFrame) -> pd.DataFrame:
    talk_re, practice_re = load_vocabulary()

    corpus = pd.read_csv(corpus_path)
    require_columns(corpus, ["doi", "abstract"], str(corpus_path))
    corpus["doi"] = corpus["doi"].map(normalize_doi)
    corpus = corpus.drop_duplicates("doi")

    abstracts = corpus["abstract"].fillna("").astype(str)
    n_empty = int((abstracts.str.strip() == "").sum())
    if n_empty:
        LOG.warning("%d abstracts vacíos: cuentan como 0 en el léxico", n_empty)
    corpus["talk_count"] = abstracts.map(lambda s: len(talk_re.findall(s)))
    corpus["practice_count"] = abstracts.map(lambda s: len(practice_re.findall(s)))
    corpus["word_count"] = abstracts.str.split().str.len().fillna(0).astype(int)

    topics = pd.read_csv(topics_path)
    require_columns(topics, ["doi", "cluster"], str(topics_path))
    topics["doi"] = topics["doi"].map(normalize_doi)
    topics = topics.drop_duplicates("doi")
    topics["cl"] = topics["cluster"].map(parse_cluster)
    topics["mega"] = topics["cl"].map(CLUSTER_TO_MEGA)
    n_before = len(topics)
    topics = topics.dropna(subset=["mega"]).copy()
    LOG.info("clusters: %s de %s papers dentro de las 5 mega-áreas",
             f"{len(topics):,}", f"{n_before:,}")

    lex_cols = ["doi", "talk_count", "practice_count", "word_count"]
    df = (topics.merge(corpus[lex_cols], on="doi", how="inner")
                .merge(tmo, on="doi", how="left"))   # LEFT: conserva papers sin TMO

    for c in STRUCT_COLS + ["obj_is_ce"]:
        df[c] = df[c].fillna(0).astype(int)
    df["has_tmo"] = df["has_tmo"].fillna(False).astype(bool)
    df["cl"] = df["cl"].astype(int)

    if df.empty:
        raise SystemExit("[error] el cruce corpus × topics quedó vacío: revisa los DOI.")
    LOG.info("cruce final: %s papers | con TMO: %.1f%%", f"{len(df):,}",
             100 * df["has_tmo"].mean())
    return df


# ---------------------------------------------------------------------------
# 4. Agregados por nivel
# ---------------------------------------------------------------------------

def level_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    t = df.groupby(key).agg(
        n=("doi", "size"),
        coverage=("has_tmo", "mean"),
        practice=("practice_count", "mean"),
        talk=("talk_count", "mean"),
        tech_op=("tech_op", "mean"),
        mat=("mat_pres", "mean"),
        obj_op=("obj_op", "mean"),
        obj_disc=("obj_disc", "mean"),
        obj_is_ce=("obj_is_ce", "mean"),
    )
    t["lex_ratio"] = np.where(t["talk"] > 0, t["practice"] / t["talk"], np.nan)
    if t["lex_ratio"].isna().any():
        LOG.warning("lex_ratio indefinido (talk medio = 0) en: %s",
                    list(t.index[t["lex_ratio"].isna()]))
    t["struct_score"] = t["tech_op"] + t["mat"] + t["obj_op"] - t["obj_disc"]
    return t.round(4)


def spearman_safe(x, y) -> tuple[float, float, int]:
    m = pd.notna(x) & pd.notna(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    rho, p = spearmanr(np.asarray(x)[m], np.asarray(y)[m])
    return float(rho), float(p), n


# ---------------------------------------------------------------------------
# 5. Logits a nivel de artículo
# ---------------------------------------------------------------------------

def run_logits(sub: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    import statsmodels.formula.api as smf  # noqa: WPS433 (import perezoso)
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    data = sub.copy()
    data["cl"] = data["cl"].astype(int).astype(str)
    use_cluster_fe = data["cl"].nunique() > 1
    rhs = "practice_count + talk_count + word_count" + (" + C(cl)" if use_cluster_fe else "")
    if not use_cluster_fe:
        LOG.warning("un solo cluster en el subconjunto: se omiten los efectos fijos")

    rows = []
    for y in outcomes:
        if data[y].nunique() < 2:
            LOG.warning("logit %s omitido: la variable dependiente es constante", y)
            continue
        try:
            model = smf.logit(f"{y} ~ {rhs}", data=data).fit(disp=0, maxiter=200)
        except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
            LOG.warning("logit %s no estimable (%s)", y, type(exc).__name__)
            continue
        ci = model.conf_int()
        converged = bool(getattr(model, "mle_retvals", {}).get("converged", True))
        if not converged:
            LOG.warning("logit %s: el optimizador no convergió", y)
        for term in model.params.index:
            rows.append({
                "outcome": y, "term": term,
                "coef": float(model.params[term]),
                "std_err": float(model.bse[term]),
                "z": float(model.tvalues[term]),
                "p_value": float(model.pvalues[term]),
                "odds_ratio": float(np.exp(model.params[term])),
                "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
                "n_obs": int(model.nobs), "pseudo_r2": float(model.prsquared),
                "converged": converged, "cluster_fe": use_cluster_fe,
            })
        print(f"  {y:10s}: practice {model.params['practice_count']:+.4f} "
              f"(p={model.pvalues['practice_count']:.2e}) | "
              f"talk {model.params['talk_count']:+.4f} "
              f"(p={model.pvalues['talk_count']:.2e})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Test de permutación
# ---------------------------------------------------------------------------

def permutation_test(df: pd.DataFrame, key: str, n_perm: int, seed: int):
    """Baraja el bloque TMO entre artículos manteniendo fijo el léxico.

    Conserva la estructura conjunta de los indicadores TMO dentro de cada
    artículo (se permuta el bloque completo con el mismo índice) y destruye
    solo su asociación con el cluster/léxico.
    """
    obs_table = level_table(df, key)
    obs_rho, _, _ = spearman_safe(obs_table["lex_ratio"], obs_table["struct_score"])

    codes, levels = pd.factorize(df[key], sort=True)
    counts = np.bincount(codes, minlength=len(levels)).astype(float)
    block = df[STRUCT_COLS].to_numpy(dtype=float)
    signs = np.array([1.0, 1.0, 1.0, -1.0])            # tech_op + mat + obj_op - obj_disc

    lex_ratio = obs_table["lex_ratio"].reindex(levels).to_numpy(dtype=float)
    valid = np.isfinite(lex_ratio)

    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = block[rng.permutation(block.shape[0])]
        means = np.column_stack([
            np.bincount(codes, weights=perm[:, j], minlength=len(levels)) / counts
            for j in range(perm.shape[1])
        ])
        struct = means @ signs
        nulls[i] = spearmanr(lex_ratio[valid], struct[valid])[0]

    nulls = nulls[np.isfinite(nulls)]
    k = int((np.abs(nulls) >= abs(obs_rho)).sum())
    stats = {
        "level": key, "n_groups": int(valid.sum()),
        "observed_rho": obs_rho,
        "null_mean": float(nulls.mean()), "null_sd": float(nulls.std(ddof=1)),
        "null_ci_low": float(np.percentile(nulls, 2.5)),
        "null_ci_high": float(np.percentile(nulls, 97.5)),
        "n_perm_valid": int(nulls.size),
        "p_perm": (1 + k) / (1 + nulls.size),          # corrección de Monte Carlo
        "p_perm_uncorrected": k / nulls.size if nulls.size else float("nan"),
    }
    return stats, nulls


# ---------------------------------------------------------------------------
# 7. CLI y orquestación
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validez convergente léxico practice-to-talk vs capa TMO del grafo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = p.add_argument_group("fuente de la capa TMO")
    src.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                     help="URI Bolt de Neo4j (o variable NEO4J_URI)")
    src.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"),
                     help="nombre de la base (o variable NEO4J_DATABASE)")
    src.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"),
                     help="usuario de Neo4j (o variable NEO4J_USER)")
    src.add_argument("--password-env", default="NEO4J_PASSWORD",
                     help="variable de entorno que contiene la contraseña")
    src.add_argument("--tmo-csv", type=Path, default=None,
                     help="usar un export previo en lugar de consultar el grafo")

    io = p.add_argument_group("entradas y salidas")
    io.add_argument("--corpus", type=Path, default=Path("corpus_circular_economy.csv"))
    io.add_argument("--topics", type=Path,
                    default=Path("results_circular_economy/full_corpus/paper_topics.csv"))
    io.add_argument("--outdir", type=Path, default=Path("results_tmo_validation"))

    an = p.add_argument_group("análisis")
    an.add_argument("--tmo-agg", choices=["representative", "any"], default="representative",
                    help="cómo resumir múltiples nodos TMO por artículo")
    an.add_argument("--n-perm", type=int, default=1000, help="permutaciones del test null")
    an.add_argument("--seed", type=int, default=42, help="semilla del generador")
    an.add_argument("--quiet", action="store_true", help="solo advertencias y errores")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="[%(levelname)s] %(message)s", stream=sys.stderr)

    for path in (args.corpus, args.topics):
        if not path.exists():
            raise SystemExit(f"[error] no existe el fichero de entrada: {path}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- 1. capa TMO -------------------------------------------------------
    if args.tmo_csv:
        if not args.tmo_csv.exists():
            raise SystemExit(f"[error] no existe --tmo-csv: {args.tmo_csv}")
        tmo_raw = load_tmo_csv(args.tmo_csv)
        LOG.info("capa TMO leída de %s (%s filas)", args.tmo_csv, f"{len(tmo_raw):,}")
    else:
        password = resolve_password(args.user, args.password_env)
        tmo_raw = fetch_tmo_from_graph(args.uri, args.database, args.user, password)
        del password
        export_tmo_csv(tmo_raw, args.outdir / "tmo_graph_export.csv")

    tmo = build_tmo_features(tmo_raw, args.tmo_agg)

    # --- 2. dataset a nivel de artículo -----------------------------------
    df = build_dataset(args.corpus, args.topics, tmo)
    df.to_csv(args.outdir / "tmo_article_level.csv", index=False)

    # --- 3. cobertura y posible sesgo del subconjunto con TMO -------------
    print("\n== cobertura TMO por mega-área ==")
    print(df.groupby("mega")["has_tmo"].mean().round(3).to_string())
    print("\n== con vs sin TMO (control de sesgo) ==")
    print(df.groupby("has_tmo")[["talk_count", "practice_count", "word_count"]]
            .mean().round(2).to_string())

    # --- 4. agregados por nivel -------------------------------------------
    summary_rows = []
    subsets = [("all", df), ("tmo", df[df["has_tmo"]])]
    for subset_name, frame in subsets:
        if frame.empty:
            LOG.warning("subconjunto '%s' vacío: omitido", subset_name)
            continue
        for key, label in [("mega", "5 MEGA-ÁREAS"), ("cl", "11 CLUSTERS")]:
            t = level_table(frame, key)
            rho, pval, n = spearman_safe(t["lex_ratio"], t["struct_score"])
            print(f"\n== {label} [{subset_name}] ==  Spearman lex_ratio~struct_score: "
                  f"rho={rho:.3f} p={pval:.4f} (n={n})")
            print(t.to_string())
            suffix = "" if subset_name == "all" else f"_{subset_name}"
            t.to_csv(args.outdir / f"tmo_level_{key}{suffix}.csv")
            summary_rows.append({"subset": subset_name, "level": key, "n_groups": n,
                                 "spearman_rho": rho, "spearman_p": pval})

    # --- 5. logits ---------------------------------------------------------
    sub = df[df["has_tmo"]].copy()
    print(f"\n== LOGITS (subconjunto con TMO, n={len(sub):,}) ==")
    if len(sub) < 50:
        LOG.warning("muy pocos casos con TMO (%d): los logits pueden ser inestables", len(sub))
    logit_df = run_logits(sub, ["tech_op", "mat_pres", "obj_disc", "obj_is_ce"])
    if not logit_df.empty:
        logit_df.to_csv(args.outdir / "tmo_logit_results.csv", index=False)

    # --- 6. permutación ----------------------------------------------------
    null_frames = []
    for subset_name, frame in subsets:
        if frame.empty:
            continue
        stats, nulls = permutation_test(frame, "cl", args.n_perm, args.seed)
        stats["subset"] = subset_name
        print(f"\n== PERMUTACIÓN clusters [{subset_name}], B={args.n_perm} ==")
        print(f"observado rho={stats['observed_rho']:.3f} | "
              f"null 95% = [{stats['null_ci_low']:.3f}, {stats['null_ci_high']:.3f}] | "
              f"p_perm={stats['p_perm']:.4f} (sin corregir {stats['p_perm_uncorrected']:.4f})")
        summary_rows.append({"subset": subset_name, "level": "cl_permutation",
                             "n_groups": stats["n_groups"],
                             "spearman_rho": stats["observed_rho"],
                             "spearman_p": float("nan"), **{
                                 k: stats[k] for k in
                                 ("null_mean", "null_sd", "null_ci_low", "null_ci_high",
                                  "p_perm", "p_perm_uncorrected", "n_perm_valid")}})
        null_frames.append(pd.DataFrame({"subset": subset_name, "rho_null": nulls}))

    pd.DataFrame(summary_rows).to_csv(
        args.outdir / "tmo_convergence_summary.csv", index=False)
    if null_frames:
        pd.concat(null_frames).to_csv(
            args.outdir / "tmo_permutation_null.csv", index=False)

    # --- 7. metadatos de reproducibilidad ---------------------------------
    meta = {
        "script": Path(__file__).name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"pandas": pd.__version__, "numpy": np.__version__},
        "params": {"tmo_agg": args.tmo_agg, "n_perm": args.n_perm, "seed": args.seed,
                   "database": args.database, "tmo_source":
                       str(args.tmo_csv) if args.tmo_csv else "neo4j"},
        "inputs": {str(p): sha256_of(p) for p in (args.corpus, args.topics)
                   if p.exists()},
        "n_papers_analyzed": int(len(df)),
        "tmo_coverage": float(df["has_tmo"].mean()),
    }
    try:
        import scipy, statsmodels  # noqa: WPS433
        meta["packages"].update({"scipy": scipy.__version__,
                                 "statsmodels": statsmodels.__version__})
    except Exception:  # pragma: no cover
        pass
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    LOG.info("resultados escritos en %s", args.outdir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
