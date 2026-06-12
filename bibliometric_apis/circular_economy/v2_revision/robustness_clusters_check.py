"""
ROBUSTNESS ANALYSIS OF CLUSTER AGGREGATION STRATEGIES

This script evaluates the sensitivity of the practice-to-talk analysis
to alternative treatments of the five clusters excluded from the
main mega-area framework.

Two aggregation schemes are considered:

Option A:
Reassign the five excluded clusters to the existing five
mega-areas based on thematic similarity.

Option B:
Group the five excluded clusters into a sixth mega-area,
"Bio & Environmental Processes".

For each configuration, the script recalculates:

```
- Practice-to-talk ratios by mega-area
- Mega-area rankings
- Aggregate counts and descriptive statistics
```

Input files:

```
corpus_circular_economy.csv
    DOI, title, abstract, year

results_circular_economy/full_corpus/paper_topics.csv
    DOI-level topic assignments including all clusters
```

Usage:

```
python robustness_16_clusters.py
```

Notes:

```
The main analysis is based on the original five mega-area
framework. This script is intended as a sensitivity analysis
to evaluate whether the observed patterns depend on the
treatment of the five excluded clusters.
```

"""
...
print("\n" + "=" * 70)
print("REFERENCE CONFIGURATION")
print("11 clusters aggregated into 5 mega-areas")
print("=" * 70)

ref = ptt_by_mega(BASE, "Reference configuration")

print("\n" + "=" * 70)
print("OPTION A")
print("16 clusters aggregated into 5 mega-areas")
print("=" * 70)

a = ptt_by_mega(MAP_A, "Alternative aggregation A")

print("\n" + "=" * 70)
print("OPTION B")
print("16 clusters aggregated into 6 mega-areas")
print("=" * 70)

b = ptt_by_mega(MAP_B, "Alternative aggregation B")

# Ranking comparison

print("\n" + "=" * 70)
print("RANKING COMPARISON")
print("=" * 70)

def order(rows):
return [r[0] for r in rows]

oref = order(ref)
oa = order(a)
ob = order(b)

print("Reference ranking:")
print(" > ".join(m.split()[0] for m in oref))

print("\nOption A ranking:")
print(" > ".join(m.split()[0] for m in oa))

print(f"\nIdentical ranking (Reference vs Option A): {oref == oa}")

print("\nOption B ranking:")
print(" > ".join(m.split()[0] for m in ob))

bio_pos = [i for i, m in enumerate(ob)
if "Bio & Environmental Processes" in m]

if bio_pos:
print(
f"\nBio & Environmental Processes ranking position: "
f"{bio_pos[0] + 1}/{len(ob)}"
)

print("\nAnalysis complete.")
print("Results may be compared across aggregation schemes")
print("to assess the stability of observed practice-to-talk patterns.")
...
