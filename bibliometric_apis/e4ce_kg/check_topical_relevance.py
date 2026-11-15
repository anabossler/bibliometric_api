"""Check how many papers in the route corpus are not about waste or the circular
economy, and whether the family results hold without them.

A paper is topical when its title or abstract matches RELEVANT, a fixed list of
waste and circular-economy terms. The unit is a paper within a family, with the
union of its primary route labels in that family (build_family_tables.py).

Reports:
  - papers per family and the share that is not topical;
  - for the families with Eurostat data (family_eu_table.csv), the research
    shares recomputed on topical papers only, and the Spearman correlations of
    analyze_alignment.py on those shares (same seed, permutations, bootstrap);
  - healthcare: the share without a route with a Eurostat counterpart among
    topical papers, the material terms that bring non-topical papers into the
    family, and seeded samples of titles.

Usage:
    python check_topical_relevance.py
"""
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from analyze_alignment import N_BOOT, N_PERM, PAIRS, SEED, spearman
from build_family_tables import research, research_flags

HERE = Path(__file__).resolve().parent
RELEVANT = re.compile(
    r"waste|recycl|circular econom|circularity|by-?products?|landfill|incinerat|valori[sz]"
    r"|end-of-life|scrap|upcycl|secondary raw|\bre-?us(e|ed|ing)\b|tailings|slag|fly ash"
    r"|bottom ash|sludge|(crop|agricultural|forest|industrial) residues?|compost"
    r"|anaerobic digest|biorefiner|resource recovery|sustainab", re.IGNORECASE)
FLAGS = ("recycling", "thermal", "landfill_upper", "no_eurostat_route")
N_TITLES = 15


def correlation(a, b, rng):
    rho = spearman(a, b)
    perm = sum(abs(spearman(a, rng.sample(b, len(b)))) >= abs(rho) - 1e-12
               for _ in range(N_PERM))
    idx, boots = list(range(len(a))), []
    for _ in range(N_BOOT):
        s = [rng.choice(idx) for _ in idx]
        value = spearman([a[i] for i in s], [b[i] for i in s])
        if not math.isnan(value):
            boots.append(value)
    boots.sort()
    return (rho, (perm + 1) / (N_PERM + 1),
            boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1])


def main():
    papers = {}
    for line in (HERE / "route_input.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        papers[p["paper_id"]] = p
    topical = {pid: bool(RELEVANT.search(f"{p['title'] or ''} {p['abstract'] or ''}"))
               for pid, p in papers.items()}
    off = sum(not v for v in topical.values())
    print(f"papers {len(papers)} | not topical {off} ({100 * off / len(papers):.2f}%)")

    routes = research()
    by_family = defaultdict(list)
    for (pid, fam), labels in routes.items():
        by_family[fam].append((pid, labels))
    with (HERE / "family_eu_table.csv").open(newline="", encoding="utf-8") as fh:
        eu = {r["family"]: r for r in csv.DictReader(fh)}

    print("\nfamily | papers | not topical (%) | topical: papers, % recycling / thermal / "
          "landfill (upper) / no Eurostat route | Eurostat data")
    shares = {}
    for fam in sorted(by_family, key=lambda f: -len(by_family[f])):
        items = by_family[fam]
        keep = [labels for pid, labels in items if topical[pid]]
        flags = [research_flags(l) for l in keep]
        pct = {k: 100 * sum(x[k] for x in flags) / len(flags) if flags else float("nan")
               for k in FLAGS}
        if fam in eu:
            shares[fam] = pct
        n_off = len(items) - len(keep)
        print(f"{fam or '(none)'} | {len(items)} | {n_off} ({100 * n_off / len(items):.2f}) | "
              f"{len(keep)}, " + " / ".join(f"{pct[k]:.2f}" for k in FLAGS) +
              f" | {'yes' if fam in eu else 'no'}")

    rng = random.Random(SEED)
    fams = sorted(shares)
    print(f"\ntopical papers only, {len(fams)} families with Eurostat data")
    for name, eu_key, res_key in PAIRS:
        a = [float(eu[f][eu_key]) for f in fams]
        b = [shares[f][res_key.replace("papers_pct_", "")] for f in fams]
        rho, p, lo, hi = correlation(a, b, rng)
        print(f"{name}: rho {rho:.4f} | permutation p {p:.4f} | bootstrap 95% CI "
              f"{lo:.4f} to {hi:.4f}")

    hc = by_family.get("healthcare", [])
    keep = [(pid, l) for pid, l in hc if topical[pid]]
    none_ = [(pid, l) for pid, l in keep if not (l - {"prevention_reuse"})]
    only_pr = sum(l == {"prevention_reuse"} for _, l in none_)
    print(f"\nhealthcare topical {len(keep)} of {len(hc)} | without Eurostat route {len(none_)} "
          f"({100 * len(none_) / len(keep):.2f}%) | prevention_reuse only {only_pr} | "
          f"no route at all {len(none_) - only_pr}")
    off_ids = {pid for pid, _ in hc if not topical[pid]}
    terms = Counter()
    with (HERE / "route_final.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["family"] == "healthcare" and r["paper_id"] in off_ids:
                terms[r["material"]] += 1
    print(f"material terms of non-topical healthcare papers: {terms.most_common(20)}")
    print("\nsample of non-topical papers, all families:")
    off_items = sorted((pid, fam) for (pid, fam) in routes if not topical[pid])
    for pid, fam in rng.sample(off_items, min(N_TITLES, len(off_items))):
        print(f"    [{fam}] {papers[pid]['title']}")
    print("\nsample of topical healthcare papers without Eurostat route:")
    for pid, labels in rng.sample(none_, min(N_TITLES, len(none_))):
        print(f"    [{';'.join(sorted(labels)) or 'not_stated'}] {papers[pid]['title']}")


if __name__ == "__main__":
    main()
