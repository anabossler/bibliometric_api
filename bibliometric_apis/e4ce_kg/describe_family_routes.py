"""Describe what the papers of selected material families propose.

The unit is a paper within a family, with the union of the primary route labels
of its materials in that family, as in build_family_tables.py.

1. Recycling in the mineral families: the verified recycling quotes of the
   three models for each paper are matched against a fixed dictionary of
   applications. A paper can match several applications; papers that match none
   are counted and listed.
2. Healthcare papers without a route that has a Eurostat counterpart: how many
   carry prevention or re-use only and how many no route at all, keyword groups
   in their titles, and a seeded sample of titles.

Usage:
    python describe_family_routes.py
"""
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 42
RECYCLING_FAMILIES = ["other_mineral", "combustion_residues"]
N_EXAMPLES, N_UNMATCHED, N_TITLES = 3, 15, 25


def words(*alternatives):
    return re.compile(r"\b(" + "|".join(alternatives) + r")", re.IGNORECASE)


APPLICATIONS = {
    "construction_materials": words(
        "cement", "concrete", "mortar", "clinker", "brick", "geopolymer", "aggregate",
        "asphalt", "binder", "paving", "road", "construction material", "building material",
        "cementitious"),
    "ceramics_glass": words("ceramic", "glass", "tile", "porcelain", "vitrif"),
    "metal_recovery": re.compile(
        r"\brecover\w*[^.;]{0,60}\b(metals?|iron|zinc|copper|alumin(ium|um|a)|rare earths?|"
        r"rees?|lithium|vanadium|titanium|gold|silver|nickel|cobalt|chromium|scandium|"
        r"gallium)\b|\bleach|\bhydrometallurg|\bpyrometallurg", re.IGNORECASE),
    "adsorbents_catalysts": words("adsorb", "sorbent", "catalys"),
    "soil_amendment": words("fertili", "soil amend", "soil improv", "soil condition",
                            "liming"),
}
TITLE_GROUPS = {
    "covid_pandemic": words("covid", "pandemic", "sars-cov", "coronavirus"),
    "management_policy": words("management", "governance", "policy", "policies", "regulat",
                               "legislat", "practice", "awareness", "knowledge",
                               "attitude", "survey"),
    "treatment_technology": words("incinerat", "pyrolys", "steriliz", "sterilis", "autoclav",
                                  "microwave", "plasma", "gasif", "disinfect", "treatment"),
    "ppe_masks": words("mask", "ppe", "personal protective", "glove"),
}


def main():
    papers = {}
    for line in (HERE / "route_input.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        papers[p["paper_id"]] = p
    routes, terms = defaultdict(set), defaultdict(set)
    with (HERE / "route_final.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["paper_id"], r["family"])
            routes[key] |= {x for x in r["routes"].split(";") if x and x != "not_stated"}
            terms[key].add(r["material"])
    quotes = defaultdict(list)
    wanted = {(pid, term) for (pid, fam), ts in terms.items()
              if fam in RECYCLING_FAMILIES for term in ts}
    with (HERE / "route_labels_long.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["paper_id"], r["material"]) in wanted and r["route"] == "recycling" \
                    and r["evidence_verified"] == "True":
                quotes[r["paper_id"], r["material"]].append(r["evidence"])
    rng = random.Random(SEED)

    print("== recycling applications in the verified quotes")
    for fam in RECYCLING_FAMILIES:
        keys = sorted(k for k, v in routes.items() if k[1] == fam and "recycling" in v)
        hits, unmatched = defaultdict(list), []
        for key in keys:
            text = " ".join(q for t in sorted(terms[key]) for q in quotes[key[0], t])
            found = [a for a, rx in APPLICATIONS.items() if rx.search(text)]
            for a in found:
                hits[a].append((key, text))
            if not found:
                unmatched.append((key, text))
        print(f"\n{fam}: {len(keys)} papers with recycling")
        for a in APPLICATIONS:
            n = len(hits[a])
            print(f"  {a}: {n} ({100 * n / len(keys):.2f}%)")
        print(f"  none of the above: {len(unmatched)} ({100 * len(unmatched) / len(keys):.2f}%)")
        for a in APPLICATIONS:
            for key, text in rng.sample(hits[a], min(N_EXAMPLES, len(hits[a]))):
                print(f"    [{a}] {'; '.join(sorted(terms[key]))}: {text[:300]}")
        for key, text in rng.sample(unmatched, min(N_UNMATCHED, len(unmatched))):
            print(f"    [none] {'; '.join(sorted(terms[key]))}: {text[:300]}")

    print("\n== healthcare papers without a route with a Eurostat counterpart")
    keys = sorted(k for k in routes if k[1] == "healthcare")
    none_ = [k for k in keys if not (routes[k] - {"prevention_reuse"})]
    only_pr = [k for k in none_ if routes[k] == {"prevention_reuse"}]
    print(f"healthcare papers {len(keys)} | without Eurostat route {len(none_)} "
          f"({100 * len(none_) / len(keys):.2f}%) | of which prevention_reuse only "
          f"{len(only_pr)} | no route at all {len(none_) - len(only_pr)}")
    for g, rx in TITLE_GROUPS.items():
        n = sum(bool(rx.search(papers[k[0]]["title"] or "")) for k in none_)
        print(f"  title {g}: {n} ({100 * n / len(none_):.2f}%)")
    n = sum(not any(rx.search(papers[k[0]]["title"] or "") for rx in TITLE_GROUPS.values())
            for k in none_)
    print(f"  title none of the above: {n} ({100 * n / len(none_):.2f}%)")
    for k in rng.sample(none_, min(N_TITLES, len(none_))):
        label = ";".join(sorted(routes[k])) or "not_stated"
        print(f"    [{label}] {papers[k[0]]['title']}")


if __name__ == "__main__":
    main()
