# Annotation guidelines: end-of-life treatment routes

## Task

Each row of `audit_route_blind.csv` holds one paper (title and abstract) and one material that the paper targets. Read the title and abstract and write in `routes_annotator` the treatment route or routes that the paper studies or proposes for that material when it is waste.

Use only the codes below, separated by `;` when there is more than one (for example `recycling;landfill`). Write `not_stated` when the abstract does not say which route the paper studies or proposes for the material.

## Codes

| Code | Definition |
|---|---|
| `recycling` | Waste is reprocessed into products, materials or substances: mechanical and chemical recycling, use of waste as raw material or aggregate in new products (for example ash or slag in concrete), composting, anaerobic digestion. |
| `backfilling` | Suitable non-hazardous waste is used to fill excavated areas or for engineering purposes in landscaping. |
| `energy_recovery` | Waste is used principally as a fuel or to generate energy, including conversion into fuels (pyrolysis oil or gas used as fuel, refuse-derived fuel, co-incineration with energy recovery). |
| `incineration` | Thermal treatment without energy recovery. |
| `landfill` | Deposit into or onto land, including landfilling of residues. |
| `other_disposal` | Other disposal operations: land treatment, deep injection, release into water. |
| `prevention_reuse` | Avoiding the waste or keeping the product in use: waste reduction at source, reuse, repair, refurbishment, remanufacturing, product life extension. |
| `not_stated` | The abstract does not say which route the paper studies or proposes for this material. |

Definitions follow Directive 2008/98/EC and the Eurostat waste-treatment operations (env_wastrt).

## Decision rules

1. Assign a route only if the paper studies, tests, compares, models or proposes it for the material. A route mentioned only as background, or as the current practice the paper wants to replace, does not count.
2. Generic terms such as "valorization", "upcycling", "circular use" or "sustainable management" do not identify a route on their own. Use `not_stated` unless the abstract says what is actually done with the material.
3. Assign several routes when the paper compares or combines them.
4. If the material is not waste in the paper (for example a virgin feedstock or a catalyst), use `not_stated` and write `not waste` in `notes`.
5. If the material term merges several materials (for example "steel slag paper sludge"), annotate the routes for any of its parts and write `merged term` in `notes`.

## Procedure

- Annotate alone and without any AI tool. Do not open `audit_route_key.csv` or the model outputs until all rows are annotated.
- Work in the row order of the file. Do not edit the columns `item`, `paper_id`, `title`, `abstract` or `material`.
- Write doubts in `notes`; leave `routes_annotator` with your best decision even when in doubt.
- Save the annotated file as `annotations/audit_route_A.csv` (second annotator: `audit_route_B.csv`).
