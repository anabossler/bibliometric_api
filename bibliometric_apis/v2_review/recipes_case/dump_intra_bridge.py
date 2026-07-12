"""
Find the best intra-graph bridge: two historical recipes from DIFFERENT
centuries AND different countries sharing 4+ DISTINCTIVE ingredients.

"""
import pickle, sys
from pathlib import Path
from collections import Counter
import itertools

graph_path = Path("data/graph_step2_canonical.gpickle")

print(f"Loading graph...")
with graph_path.open("rb") as fh:
    G = pickle.load(fh)

# Top 50 universal ingredients (filter out)
ing_freq = {}
for n, d in G.nodes(data=True):
    if d.get("node_type") == "Ingredient":
        ing_freq[n] = d.get("n_occurrences", 0)

top50 = set(k for k, v in sorted(ing_freq.items(), key=lambda x: -x[1])[:50])
print(f"Filtering out top 50 universal ingredients (n_occ >= {min(ing_freq[k] for k in top50)})")

# Build recipe → distinctive ingredients map
recipe_ings = {}
recipe_meta = {}
for n, d in G.nodes(data=True):
    if d.get("node_type") != "Recipe":
        continue
    ings = set()
    for _, target, edata in G.out_edges(n, data=True):
        if edata.get("edge_type") == "contains" and target not in top50:
            ings.add(target)
    if len(ings) >= 3:  # skip recipes with too few distinctive ingredients
        recipe_ings[n] = ings
        recipe_meta[n] = {
            "title": d.get("title", "?"),
            "place": d.get("source_place", "?"),
            "year": d.get("source_year"),
            "period": d.get("period_derived", "?"),
            "source": d.get("source_title", "?"),
            "author": d.get("source_author", "?"),
        }

print(f"{len(recipe_ings)} recipes with 3+ distinctive ingredients")

# Build inverted index: distinctive ingredient → recipes
ing_to_recipes = {}
for rid, ings in recipe_ings.items():
    for ing in ings:
        ing_to_recipes.setdefault(ing, set()).add(rid)

# Find pairs from different periods AND different places with most shared distinctive
print("Searching for cross-century, cross-country bridges...")
best_pairs = []

# Sample approach: for each recipe, find best match from different period+place
checked = 0
for rid, ings in recipe_ings.items():
    meta = recipe_meta[rid]
    if meta["year"] is None:
        continue
    
    # Count overlaps via inverted index
    candidates = Counter()
    for ing in ings:
        for other in ing_to_recipes.get(ing, set()):
            if other != rid:
                candidates[other] += 1
    
    for other_rid, overlap_count in candidates.most_common(20):
        if overlap_count < 4:
            break
        other_meta = recipe_meta[other_rid]
        if other_meta["year"] is None:
            continue
        # Different century
        if meta["period"] == other_meta["period"]:
            continue
        # Different country (rough check on place string)
        if meta["place"] == other_meta["place"]:
            continue
        
        shared = ings & recipe_ings[other_rid]
        best_pairs.append((overlap_count, rid, other_rid, shared))
    
    checked += 1
    if checked % 500 == 0:
        print(f"  checked {checked}/{len(recipe_ings)}...")

best_pairs.sort(key=lambda x: -x[0])

print(f"\nTOP 15 CROSS-CENTURY CROSS-COUNTRY BRIDGES:")
seen = set()
shown = 0
for count, r1, r2, shared in best_pairs:
    key = tuple(sorted([r1, r2]))
    if key in seen:
        continue
    seen.add(key)
    
    m1 = recipe_meta[r1]
    m2 = recipe_meta[r2]
    shared_names = sorted(s.replace("ing::", "") for s in shared)
    
    print(f"\n  === {count} distinctive shared ingredients ===")
    print(f"  A: {m1['title'][:50]}")
    print(f"     {m1['source']} ({m1['author']})")
    print(f"     {m1['place']} · {m1['year']} · {m1['period']}")
    print(f"  B: {m2['title'][:50]}")
    print(f"     {m2['source']} ({m2['author']})")
    print(f"     {m2['place']} · {m2['year']} · {m2['period']}")
    print(f"  Shared: {', '.join(shared_names[:10])}")
    if len(shared_names) > 10:
        print(f"          +{len(shared_names)-10} more")
    
    # Show tools and actions for both
    for rid, label in [(r1, "A"), (r2, "B")]:
        tools = [t.replace("tool::","") for _, t, ed in G.out_edges(rid, data=True) if ed.get("edge_type") == "uses_tool"]
        actions = [t.replace("act::","") for _, t, ed in G.out_edges(rid, data=True) if ed.get("edge_type") == "performs"]
        places = [t for _, t, ed in G.out_edges(rid, data=True) if ed.get("edge_type") == "origin"]
        periods = [t for _, t, ed in G.out_edges(rid, data=True) if ed.get("edge_type") == "dated"]
        if tools:
            print(f"  {label} tools: {', '.join(tools[:5])}")
        if actions:
            print(f"  {label} actions: {', '.join(actions[:5])}")
    
    shown += 1
    if shown >= 15:
        break

print("\n\nDone.")
