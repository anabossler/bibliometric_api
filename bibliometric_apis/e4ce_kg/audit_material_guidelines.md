# Annotation guidelines: material families

Task: for each row of the sheet, write in `family_annotator` the `family_id`
from `vocab/material_families.csv` that best describes the term. One family
per term.

## Decision rules

1. If the term names a product or stream that has its own family (batteries,
   weee, vehicles, photovoltaic, construction_demolition), use that family.
2. Otherwise classify by material (plastic, metal, glass, paper_cardboard,
   wood, textile, ...). Packaging with a stated material goes to that
   material; packaging with no material stated goes to packaging.
3. Use critical_raw_materials, fuels_energy_carriers and advanced_materials
   only when the term names the element, fuel or engineered material itself.
4. Escape labels: sector_or_concept for economic sectors, activities and
   abstract concepts; generic_waste_resource for unspecific terms;
   other_material for a specific material that fits no family.

## Procedure

- Judge the term alone; do not look up the papers.
- If unsure, choose the best family and write the alternative in `notes`.
- Do not open `audit_material_key.csv`.
- Annotators work independently and do not discuss cases until both sheets
  are complete.
