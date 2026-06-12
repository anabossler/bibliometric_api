"""
GEOGRAPHIC ROBUSTNESS ANALYSIS USING ALL 16 CLUSTERS

This script evaluates the sensitivity of the geographic attribution
results to alternative treatments of the five clusters excluded from
the primary mega-area framework.

Two aggregation schemes are considered:

Option A:
Reassign the five excluded clusters to the existing five
mega-areas based on thematic similarity.

Option B:
Group the five excluded clusters into an additional
mega-area, "Bio & Environmental Processes".

For each aggregation scheme, the script recalculates:

```
- Global North / Global South distributions
- Country-level attribution counts
- Indonesia-related counts within
  Sustainability Framing & Society
```

Input files:

```
results_circular_economy/full_corpus/paper_topics.csv
ce_openalex_v2/ce_papers_meta.csv
ce_openalex_v2/ce_affiliations.csv
ce_openalex_v2/ce_institutions.csv
```

Requirements:

```
The affiliations file must contain:
    author_id, paper_id, institution_id

This preserves paper-level affiliation attribution and
avoids institution propagation across all papers authored
by the same researcher.
```

Usage:

```
python geographic_robustness_16_clusters.py
```

"""
...

print("\n" + "=" * 64)
print("REFERENCE CONFIGURATION")
print("11 clusters aggregated into 5 mega-areas")
print("=" * 64)

run(BASE, "Reference configuration")

print("\n" + "=" * 64)
print("OPTION A")
print("16 clusters aggregated into 5 mega-areas")
print("=" * 64)

run(MAP_A, "Alternative aggregation A")

print("\n" + "=" * 64)
print("OPTION B")
print("16 clusters aggregated into 6 mega-areas")
print("=" * 64)

run(MAP_B, "Alternative aggregation B")

print("\n" + "=" * 64)
print("COMPARISON ACROSS AGGREGATION SCHEMES")
print("=" * 64)
print("The resulting tables may be compared across")
print("aggregation strategies to evaluate the stability")
print("of geographic attribution patterns.")
print("=" * 64)
