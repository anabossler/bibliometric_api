"""Load the material-family layer into the E4CE-KG Neo4j database.

Creates one MaterialFamily node per row of vocab/material_families.csv, links
every classified Material term to its family (IN_FAMILY) and every Eurostat
WasteFlow to the family that covers its EWC-Stat category
(CORRESPONDS_TO_FAMILY). The layer is rebuilt on each run, so the result
depends only on the two input files.

Usage:
    python load_material_families.py [tmo_material_classified.csv]

The Neo4j password is read from the NEO4J_PW environment variable.
"""
import csv
import os
import sys
from pathlib import Path

from neo4j import GraphDatabase

URI = "bolt://localhost:7691"
DB = "circulareconomy"
HERE = Path(__file__).resolve().parent
VOCAB = HERE.parents[1] / "vocab" / "material_families.csv"
CLASSIFIED = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "tmo_material_classified.csv"
BATCH = 2000


def split(cell):
    return [x.strip() for x in cell.split(";") if x.strip()]


def read_inputs():
    with VOCAB.open(newline="", encoding="utf-8") as fh:
        families = [
            {
                "family_id": r["family_id"],
                "family_name": r["family_name"],
                "cats": split(r["ewc_stat_categories"]),
                "sec": split(r["ewc_stat_secondary_categories"]),
                "notes": r["notes"],
            }
            for r in csv.DictReader(fh)
        ]
    allowed = {f["family_id"] for f in families}

    # csv keeps every cell as text, so a term such as "nan" is not read as missing.
    with CLASSIFIED.open(newline="", encoding="utf-8") as fh:
        terms = [
            {"term": r["term"], "category": r["category"],
             "model": r["model"], "prompt_sha": r["prompt_sha"]}
            for r in csv.DictReader(fh)
        ]
    unknown = sorted({t["category"] for t in terms} - allowed)
    if unknown:
        sys.exit(f"Categories not in the vocabulary: {unknown}")

    pairs = [{"cat": c, "fid": f["family_id"], "secondary": False}
             for f in families for c in f["cats"]]
    pairs += [{"cat": c, "fid": f["family_id"], "secondary": True}
              for f in families for c in f["sec"]]
    cats = [p["cat"] for p in pairs]
    if len(cats) != len(set(cats)):
        sys.exit("An EWC-Stat category is assigned to more than one family")
    return families, terms, pairs


def main():
    pw = os.environ.get("NEO4J_PW")
    if not pw:
        sys.exit("Set NEO4J_PW first")
    families, terms, pairs = read_inputs()
    print(f"vocabulary: {len(families)} families | classified terms: {len(terms)}")

    with GraphDatabase.driver(URI, auth=("neo4j", pw)) as drv, drv.session(database=DB) as s:
        s.run("CREATE CONSTRAINT material_family_id IF NOT EXISTS "
              "FOR (f:MaterialFamily) REQUIRE f.family_id IS UNIQUE").consume()
        s.run("MATCH ()-[e:IN_FAMILY|CORRESPONDS_TO_FAMILY]->(:MaterialFamily) DELETE e").consume()
        s.run("MATCH (f:MaterialFamily) WHERE NOT f.family_id IN $ids DETACH DELETE f",
              ids=[f["family_id"] for f in families]).consume()
        s.run("""
            UNWIND $rows AS r
            MERGE (f:MaterialFamily {family_id: r.family_id})
            SET f.name = r.family_name,
                f.ewc_stat_categories = r.cats,
                f.ewc_stat_secondary_categories = r.sec,
                f.notes = r.notes
        """, rows=families).consume()

        linked = 0
        for i in range(0, len(terms), BATCH):
            linked += s.run("""
                UNWIND $rows AS r
                MATCH (m:Material {name: r.term})
                MATCH (f:MaterialFamily {family_id: r.category})
                MERGE (m)-[e:IN_FAMILY]->(f)
                SET e.model = r.model, e.prompt_sha = r.prompt_sha
                RETURN count(e) AS n
            """, rows=terms[i:i + BATCH]).single()["n"]

        flows = s.run("""
            UNWIND $rows AS r
            MATCH (w:WasteFlow {waste_category: r.cat})
            MATCH (f:MaterialFamily {family_id: r.fid})
            MERGE (w)-[e:CORRESPONDS_TO_FAMILY]->(f)
            SET e.secondary = r.secondary
            RETURN count(e) AS n
        """, rows=pairs).single()["n"]

        present = {r["c"] for r in s.run(
            "MATCH (w:WasteFlow) RETURN DISTINCT w.waste_category AS c")}
        uncovered = sorted(present - {p["cat"] for p in pairs})
        missing = sorted({p["cat"] for p in pairs} - present)
        n_fam = s.run("MATCH (f:MaterialFamily) RETURN count(f) AS n").single()["n"]

    print(f"MaterialFamily nodes: {n_fam}")
    print(f"IN_FAMILY: {linked} of {len(terms)} terms")
    print(f"CORRESPONDS_TO_FAMILY: {flows} WasteFlow nodes")
    print(f"Eurostat categories without family: {uncovered}")
    print(f"Vocabulary categories absent from the graph: {missing}")


if __name__ == "__main__":
    main()
