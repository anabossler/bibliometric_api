"""Ingest Eurostat waste statistics into the E4CE-KG Neo4j database.

Datasets and fixed dimensions:
  env_wasgen  waste generation         unit=T, hazard=HAZ_NHAZ, nace_r2=TOTAL_HH
  env_wastrt  waste treatment          unit=T, hazard=HAZ_NHAZ, every wst_oper
  cei_srm030  circular material use rate
  cei_pc032   waste generation per GDP unit

Every dimension that is not geo, time or a declared key dimension must take a
single value in the response; otherwise the run stops before writing anything.
Each node therefore holds exactly one Eurostat observation.

WasteFlow and CircularityIndicator nodes are rebuilt from scratch on each run.
Country codes follow ISO 3166-1 alpha-2 (Eurostat EL is stored as GR).

Usage:
    python ingest_eurostat.py [--start-year 2018]

The Neo4j password is read from the NEO4J_PW environment variable. After this
script, run load_material_families.py to relink WasteFlow to MaterialFamily.
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import requests
from neo4j import GraphDatabase

URI = "bolt://localhost:7691"
DB = "circulareconomy"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{}"
EU27 = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "EL", "HU",
        "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"]
ISO = {"EL": "GR"}
DATASETS = {
    "env_wasgen": {"label": "WasteFlow", "keys": ["waste"],
                   "filters": {"unit": "T", "hazard": "HAZ_NHAZ", "nace_r2": "TOTAL_HH"}},
    "env_wastrt": {"label": "WasteFlow", "keys": ["waste", "wst_oper"],
                   "filters": {"unit": "T", "hazard": "HAZ_NHAZ"}},
    "cei_srm030": {"label": "CircularityIndicator", "keys": ["unit"], "filters": {}},
    "cei_pc032": {"label": "CircularityIndicator", "keys": ["unit"], "filters": {}},
}
BATCH = 5000


def fetch(dataset, geo, start_year, filters, retries=4):
    params = {"format": "JSON", "lang": "EN", "geo": geo,
              "sinceTimePeriod": str(start_year), **filters}
    for attempt in range(retries):
        try:
            resp = requests.get(API.format(dataset), params=params, timeout=120)
        except requests.RequestException as exc:
            err = exc
        else:
            if resp.status_code == 404:
                return None
            if resp.ok:
                return resp.json()
            err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        time.sleep(2 ** attempt)
    sys.exit(f"{dataset} {geo}: request failed after {retries} attempts ({err})")


def as_dict(obj):
    """JSON-stat value/status containers can be a list or a sparse dict."""
    if obj is None:
        return {}
    if isinstance(obj, list):
        return {i: v for i, v in enumerate(obj)}
    return {int(k): v for k, v in obj.items()}


def parse_jsonstat(data):
    """Return one dict per observation: dimension codes, labels, value and flag."""
    ids, sizes = data["id"], data["size"]
    codes, labels = [], []
    for dim in ids:
        cat = data["dimension"][dim]["category"]
        index = cat["index"]
        order = sorted(index, key=index.get) if isinstance(index, dict) else list(index)
        codes.append(order)
        labels.append(cat.get("label", {}))
    values, status = as_dict(data.get("value")), as_dict(data.get("status"))
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    rows = []
    for flat, value in values.items():
        if value is None:
            continue
        row = {"value": float(value), "flag": status.get(flat, "")}
        for d, dim in enumerate(ids):
            code = codes[d][(flat // strides[d]) % sizes[d]]
            row[dim] = code
            row[f"{dim}_label"] = labels[d].get(code, code)
        rows.append(row)
    return rows


def collect(dataset, cfg, start_year):
    rows, no_data = [], []
    for geo in EU27:
        data = fetch(dataset, geo, start_year, cfg["filters"])
        parsed = parse_jsonstat(data) if data and data.get("value") else []
        if not parsed:
            no_data.append(geo)
        rows.extend(parsed)
    if not rows:
        sys.exit(f"{dataset}: no observations returned")

    dims = [k for k in rows[0] if not k.endswith("_label") and k not in ("value", "flag")]
    fixed = [d for d in dims if d not in ("geo", "time", *cfg["keys"])]
    for d in fixed:
        seen = sorted({r[d] for r in rows})
        if len(seen) != 1:
            sys.exit(f"{dataset}: dimension '{d}' has several values {seen[:10]}; "
                     "add a filter for it in DATASETS")
    keys = Counter(tuple(r[d] for d in ("geo", "time", *cfg["keys"])) for r in rows)
    dup = [k for k, n in keys.items() if n > 1]
    if dup:
        sys.exit(f"{dataset}: {len(dup)} duplicated observation keys, e.g. {dup[:3]}")
    print(f"{dataset}: {len(rows)} observations | fixed {[(d, rows[0][d]) for d in fixed]} "
          f"| no data for {no_data or 'none'}")
    return [to_node(dataset, cfg, r) for r in rows]


def to_node(dataset, cfg, r):
    node = {"dataset": dataset, "country_code": ISO.get(r["geo"], r["geo"]),
            "year": int(r["time"]), "value": r["value"], "flag": r["flag"],
            "unit": r["unit"], "unit_label": r["unit_label"]}
    if cfg["label"] == "WasteFlow":
        node.update(ewc_stat_code=r["waste"], waste_category=r["waste_label"],
                    hazard=r["hazard"])
        if "nace_r2" in r:
            node["nace_r2"] = r["nace_r2"]
        if "wst_oper" in r:
            node.update(wst_oper=r["wst_oper"], operation=r["wst_oper_label"])
    return node


def write(session, nodes):
    for label in ("WasteFlow", "CircularityIndicator"):
        for rec in list(session.run(
                "SHOW CONSTRAINTS YIELD name, labelsOrTypes "
                "WHERE $l IN labelsOrTypes RETURN name", l=label)):
            session.run(f"DROP CONSTRAINT `{rec['name']}`").consume()
        session.run(f"MATCH (:{label})-[r]-() "
                    "CALL { WITH r DELETE r } IN TRANSACTIONS OF 50000 ROWS").consume()
        session.run(f"MATCH (n:{label}) "
                    "CALL { WITH n DELETE n } IN TRANSACTIONS OF 50000 ROWS").consume()

    orphans = session.run("""
        MATCH (m:Material) WHERE NOT (m)--()
        WITH collect(m) AS ms, collect(m.name) AS names
        FOREACH (x IN ms | DELETE x)
        RETURN names
    """).single()["names"]
    print(f"Material nodes left without relationships and removed: {orphans}")

    by_label = defaultdict(list)
    for n in nodes:
        by_label[DATASETS[n["dataset"]]["label"]].append(n)
    for label, rows in by_label.items():
        for i in range(0, len(rows), BATCH):
            session.run(f"UNWIND $rows AS r CREATE (n:{label}) SET n = r",
                        rows=rows[i:i + BATCH]).consume()
    session.run("CREATE INDEX wasteflow_category IF NOT EXISTS "
                "FOR (w:WasteFlow) ON (w.waste_category)").consume()
    session.run("CREATE CONSTRAINT circularity_indicator_key IF NOT EXISTS "
                "FOR (c:CircularityIndicator) "
                "REQUIRE (c.dataset, c.country_code, c.year, c.unit) IS UNIQUE").consume()


def report(session):
    for r in session.run("""
        MATCH (n) WHERE n:WasteFlow OR n:CircularityIndicator
        RETURN n.dataset AS dataset, count(*) AS nodes,
               count(DISTINCT n.country_code) AS countries,
               min(n.year) AS first, max(n.year) AS last,
               collect(DISTINCT n.unit) AS units
        ORDER BY dataset
    """):
        print(dict(r))
    for r in session.run("""
        MATCH (w:WasteFlow {country_code: 'DE', year: 2020, ewc_stat_code: 'TOTAL'})
        WHERE w.wst_oper IS NULL OR w.wst_oper = 'TRT'
        RETURN w.dataset AS dataset, w.wst_oper AS wst_oper, w.value AS tonnes
    """):
        print("Germany 2020 total waste:", dict(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start-year", type=int, default=2018)
    args = ap.parse_args()
    pw = os.environ.get("NEO4J_PW")
    if not pw:
        sys.exit("Set NEO4J_PW first")

    nodes = []
    for dataset, cfg in DATASETS.items():
        nodes.extend(collect(dataset, cfg, args.start_year))

    with GraphDatabase.driver(URI, auth=("neo4j", pw)) as drv, drv.session(database=DB) as s:
        write(s, nodes)
        report(s)


if __name__ == "__main__":
    main()
