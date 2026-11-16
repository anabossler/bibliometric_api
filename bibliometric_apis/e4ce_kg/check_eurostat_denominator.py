"""Explain why the Eurostat total treated differs from the sum of operations.

For the reference year, reads from the graph the env_wastrt tonnes of every
primary EWC-Stat category of each family, per EU-27 country, for the total
(TRT), the six operations used in the analysis and the two intermediate
aggregates (RCV_R_B = recycling plus backfilling, DSP_L_OTH = landfill plus
other disposal). For each family it reports the recycling share over TRT and
over the six operations, and classifies every country-category cell:

    no TRT        operations reported but no total
    TRT only      total reported but none of the six operations
    ops > TRT     the six operations exceed the total by more than TOL
    ops < TRT     the total exceeds the six operations by more than TOL; the
                  cell then also reports whether the intermediate aggregates
                  cover the gap
    consistent    within TOL

Reads the password from the environment variable NEO4J_PW.

Usage:
    python check_eurostat_denominator.py [YEAR]
"""
import os
import sys
from collections import Counter, defaultdict

from build_family_tables import EU27, OPS

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
AGG = ["RCV_R_B", "DSP_L_OTH"]
TOL = 0.01
QUERY = """
MATCH (w:WasteFlow {dataset: 'env_wastrt'})-[c:CORRESPONDS_TO_FAMILY]->(f:MaterialFamily)
WHERE c.secondary = false AND toInteger(w.year) = $year AND w.wst_oper IN $ops
RETURN f.family_id AS family, w.country_code AS country, w.waste_category AS category,
       w.wst_oper AS op, sum(w.value) AS tonnes
"""


def main():
    from neo4j import GraphDatabase
    pw = os.environ.get("NEO4J_PW")
    if not pw:
        sys.exit("set NEO4J_PW")
    with GraphDatabase.driver("bolt://localhost:7691", auth=("neo4j", pw)) as drv, \
            drv.session(database="circulareconomy") as s:
        rows = list(s.run(QUERY, year=YEAR, ops=["TRT"] + OPS + AGG))
    cells = defaultdict(dict)
    for r in rows:
        if r["country"] in EU27:
            cells[(r["family"], r["country"], r["category"])][r["op"]] = float(r["tonnes"] or 0)

    fam = defaultdict(lambda: defaultdict(float))
    kinds = defaultdict(Counter)
    tonnes = defaultdict(lambda: defaultdict(float))
    for (f, c, cat), v in cells.items():
        six = sum(v.get(op, 0.0) for op in OPS)
        trt = v.get("TRT")
        fam[f]["trt"] += trt or 0.0
        fam[f]["six"] += six
        fam[f]["rcv_r"] += v.get("RCV_R", 0.0)
        has_ops = any(op in v for op in OPS)
        if trt is None and has_ops:
            kind = "no TRT"
            tonnes[f][kind] += six
        elif trt is not None and not has_ops:
            kind = "TRT only"
            tonnes[f][kind] += trt
        elif trt is not None and six > trt * (1 + TOL):
            kind = "ops > TRT"
            tonnes[f][kind] += six - trt
        elif trt is not None and six < trt * (1 - TOL):
            agg = sum(v.get(op, 0.0) for op in AGG)
            split = sum(v.get(op, 0.0) for op in ("RCV_R", "RCV_B", "DSP_L", "DSP_OTH"))
            kind = "ops < TRT, aggregates cover gap" if agg > split * (1 + TOL) else "ops < TRT"
            tonnes[f][kind] += trt - six
        else:
            kind = "consistent"
        kinds[f][kind] += 1

    print(f"year {YEAR} | EU-27 | tolerance {100 * TOL:.0f}%")
    print("family | recycling % of TRT | recycling % of six ops | TRT / six | cells by kind "
          "(tonnes involved)")
    for f in sorted(fam, key=lambda x: -fam[x]["six"]):
        t = fam[f]
        r_trt = 100 * t["rcv_r"] / t["trt"] if t["trt"] else float("nan")
        r_six = 100 * t["rcv_r"] / t["six"] if t["six"] else float("nan")
        ratio = t["trt"] / t["six"] if t["six"] else float("nan")
        detail = "; ".join(f"{k} {n} ({tonnes[f].get(k, 0):.0f} t)" if k != "consistent"
                           else f"{k} {n}" for k, n in kinds[f].most_common())
        print(f"{f} | {r_trt:.2f} | {r_six:.2f} | {ratio:.4f} | {detail}")
    total = Counter()
    for f in kinds:
        total.update(kinds[f])
    print(f"\nall families: {dict(total.most_common())}")


if __name__ == "__main__":
    main()
