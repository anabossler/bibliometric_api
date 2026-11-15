"""Build the family tables that join research routes with Eurostat waste treatment.

Steps:
  1. Export from Neo4j, once, the env_wastrt treatment of every material family
     (primary EWC-Stat categories only) by country, year and operation to
     eurostat_family_ops.csv. Later runs read the CSV.
  2. EU table (REF_YEAR): for each family, tonnes treated, shares of recycling
     (RCV_R), thermal treatment (RCV_E + DSP_I), disposal (DSP_L + DSP_OTH) and
     backfilling (RCV_B) over the six operations, tonnes not recycled, and the
     research side from the primary route labels: papers and the share of them
     that study recycling, thermal treatment or landfill (an upper bound, see the
     disposal audit), or no route with a Eurostat operation. A paper counts once
     per family, with the union of the routes of its materials in that family.
     Tonnes not recycled include backfilling.
  3. Panel of EU27 countries x families: papers with at least one author in the
     country (whole and fractional counts), and the mean tonnes and shares of the
     PANEL_YEARS.

Outputs:
    eurostat_family_ops.csv
    family_eu_table.csv
    country_family_panel.csv

Usage:
    python build_family_tables.py
NEO4J_PW is needed only for the first export.
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS_CSV = HERE / "eurostat_family_ops.csv"
OPS = ["RCV_R", "RCV_B", "RCV_E", "DSP_I", "DSP_L", "DSP_OTH"]
GROUPS = {"recycling": ["RCV_R"], "thermal": ["RCV_E", "DSP_I"],
          "disposal": ["DSP_L", "DSP_OTH"], "backfilling": ["RCV_B"]}
REF_YEAR, PANEL_YEARS = 2022, (2018, 2020, 2022)
EU27 = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
        "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"}
QUERY = """
MATCH (w:WasteFlow {dataset: 'env_wastrt'})-[c:CORRESPONDS_TO_FAMILY]->(f:MaterialFamily)
WHERE c.secondary = false AND w.wst_oper IN $ops
RETURN f.family_id AS family, w.country_code AS country, w.year AS year,
       w.wst_oper AS op, sum(w.value) AS tonnes
ORDER BY family, country, year, op
"""


def export_ops():
    if OPS_CSV.exists():
        return
    from neo4j import GraphDatabase
    pw = os.environ.get("NEO4J_PW")
    if not pw:
        sys.exit(f"{OPS_CSV.name} does not exist yet: set NEO4J_PW to export it")
    with GraphDatabase.driver("bolt://localhost:7691", auth=("neo4j", pw)) as drv, \
            drv.session(database="circulareconomy") as s:
        rows = [dict(r) for r in s.run(QUERY, ops=OPS)]
    with OPS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "country", "year", "op", "tonnes"])
        w.writeheader()
        w.writerows(rows)
    print(f"exported {len(rows)} rows to {OPS_CSV.name}")


def load_ops():
    ops = defaultdict(float)
    with OPS_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ops[(r["family"], r["country"], int(r["year"]), r["op"])] += float(r["tonnes"])
    return ops


def treatment(ops, family, countries, years):
    """Mean over years of the summed tonnes per operation group."""
    per_year = []
    for y in years:
        by_op = {op: sum(ops.get((family, c, y, op), 0.0) for c in countries) for op in OPS}
        total = sum(by_op.values())
        if total > 0:
            per_year.append((total, {g: sum(by_op[o] for o in g_ops)
                                     for g, g_ops in GROUPS.items()}))
    if not per_year:
        return None
    total = sum(t for t, _ in per_year) / len(per_year)
    groups = {g: sum(v[g] for _, v in per_year) / len(per_year) for g in GROUPS}
    return total, groups


def research():
    """Return {(paper_id, family): set of routes} from the primary labels."""
    routes = defaultdict(set)
    with (HERE / "route_final.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            got = {x for x in r["routes"].split(";") if x and x != "not_stated"}
            routes[(r["paper_id"], r["family"])] |= got
    return routes


def research_flags(labels):
    return {"recycling": "recycling" in labels,
            "thermal": bool(labels & {"energy_recovery", "incineration"}),
            "landfill_upper": "landfill" in labels,
            "no_eurostat_route": not (labels - {"prevention_reuse"})}


def main():
    export_ops()
    ops = load_ops()
    routes = research()
    families = sorted({f for (_, f) in routes if f})
    countries_all = sorted({c for (_, c, _, _) in ops})

    eu_rows = []
    for fam in families:
        papers = [labels for (p, f), labels in routes.items() if f == fam]
        flags = [research_flags(l) for l in papers]
        tr = treatment(ops, fam, countries_all, (REF_YEAR,))
        if tr is None:
            continue
        total, g = tr
        row = {"family": fam, "tonnes_treated": round(total),
               **{f"eu_pct_{k}": round(100 * v / total, 2) for k, v in g.items()},
               "tonnes_not_recycled": round(total - g["recycling"]),
               "papers": len(papers),
               **{f"papers_pct_{k}": round(100 * sum(x[k] for x in flags) / len(flags), 2)
                  for k in ("recycling", "thermal", "landfill_upper", "no_eurostat_route")}}
        eu_rows.append(row)
    eu_rows.sort(key=lambda r: -r["tonnes_not_recycled"])
    with (HERE / "family_eu_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(eu_rows[0]))
        w.writeheader()
        w.writerows(eu_rows)

    print(f"EU27 {REF_YEAR} | sorted by tonnes not recycled")
    print("family | Mt treated | EU % recycled / thermal / disposal | Mt not recycled | "
          "papers | % papers recycling / thermal / landfill (upper) / no Eurostat route")
    for r in eu_rows:
        print(f"{r['family']} | {r['tonnes_treated'] / 1e6:.2f} | {r['eu_pct_recycling']:.2f} / "
              f"{r['eu_pct_thermal']:.2f} / {r['eu_pct_disposal']:.2f} | "
              f"{r['tonnes_not_recycled'] / 1e6:.2f} | {r['papers']} | "
              f"{r['papers_pct_recycling']:.2f} / {r['papers_pct_thermal']:.2f} / "
              f"{r['papers_pct_landfill_upper']:.2f} / {r['papers_pct_no_eurostat_route']:.2f}")

    shares = defaultdict(list)
    with (HERE / "paper_countries.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["country_code"] in EU27:
                frac = int(r["authorships_with_country"]) / int(r["authorships_total"])
                shares[r["paper_id"]].append((r["country_code"], frac))
    panel = defaultdict(lambda: defaultdict(float))
    for (pid, fam), labels in routes.items():
        flags = research_flags(labels)
        for cc, frac in shares.get(pid, []):
            cell = panel[(cc, fam)]
            cell["papers"] += 1
            cell["papers_frac"] += frac
            cell["papers_recycling"] += flags["recycling"]
            cell["papers_thermal"] += flags["thermal"]
    rows = []
    for cc in sorted(EU27):
        for fam in families:
            tr = treatment(ops, fam, [cc], PANEL_YEARS)
            if tr is None:
                continue
            total, g = tr
            cell = panel.get((cc, fam), {})
            rows.append({"country": cc, "family": fam,
                         "papers": int(cell.get("papers", 0)),
                         "papers_frac": round(cell.get("papers_frac", 0.0), 4),
                         "papers_recycling": int(cell.get("papers_recycling", 0)),
                         "papers_thermal": int(cell.get("papers_thermal", 0)),
                         "tonnes_treated": round(total),
                         **{f"pct_{k}": round(100 * v / total, 4) for k, v in g.items()}})
    with (HERE / "country_family_panel.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    zero = sum(r["papers"] == 0 for r in rows)
    print(f"\npanel: {len(rows)} country-family cells with Eurostat data | "
          f"cells without papers {zero} | paper-country incidences {sum(r['papers'] for r in rows)}")


if __name__ == "__main__":
    main()
