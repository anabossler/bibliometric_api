
## Reproducing the analysis

### 1. Requirements

```bash
conda create -n aws python=3.11
conda activate aws
pip install -r requirements.txt
```

### 2. Data

Download the data deposit from Zenodo and place the files in `data/`:

```
data/
  corpus_with_clusters.csv     # 20,066 papers with cluster and mega_area assignments
  ce_authorships.csv
  ce_affiliations.csv
  ce_papers_meta.csv
  paper_topics_clean.csv
  openAlex_query.txt           # exact query and retrieval date
```

### 3. Pipeline

Run scripts in order:

```bash
# Step 1: fetch author affiliations from OpenAlex (only needed if replicating from scratch)
python src/01_fetch_affiliations.py \
    --corpus data/corpus_with_clusters.csv \
    --out data/

# Step 2: geographic distribution by mega-area
python src/02_geography.py \
    --topics data/paper_topics_clean.csv \
    --authorships data/ce_authorships.csv \
    --affiliations data/ce_affiliations.csv \
    --institutions data/ce_institutions.csv \
    --out results/

# Step 3: CE-talk vs CE-practice ratio (main result)
python src/03_review_metrics.py \
    --topics data/paper_topics_clean.csv \
    --corpus data/corpus_with_clusters.csv \
    --out results/

# Step 4: underexplored sectors count
python src/04_underexplored_sectors.py \
    --corpus data/corpus_with_clusters.csv \
    --out results/

# Step 5: representativeness test for geographic subset
python src/05_representativeness_test.py \
    --topics data/paper_topics_clean.csv \
    --authorships data/ce_authorships.csv \
    --out results/

# Step 6: fetch full-text PDFs for validation sample (optional, takes ~2h)
python src/06_fetch_fulltexts.py \
    --corpus data/corpus_with_clusters.csv \
    --out cluster_fulltexts/

# Step 7-8: full-text validation of practice-to-talk ratio
python src/07_validate_fulltext.py \
    --summary cluster_fulltexts/fetch_summary.csv \
    --fulltext cluster_fulltexts/ \
    --corpus data/corpus_with_clusters.csv \
    --out results/fulltext_validation.csv

python src/08_validate_fulltext_norm.py \
    --summary cluster_fulltexts/fetch_summary.csv \
    --fulltext cluster_fulltexts/ \
    --corpus data/corpus_with_clusters.csv \
    --out results/fulltext_validation_normalized.csv

# Step 9: chi-square test for North-South gradient
python src/09_chi2_northsouth.py
# or, if you have the raw affiliation CSV:
python src/09_chi2_northsouth.py --affiliations results/paper_country.csv
```

**Note:** Full-text PDFs are not deposited due to copyright. The fetch pipeline in `src/06_fetch_fulltexts.py` reproduces the sample using Open Access sources (OpenAlex, Unpaywall, Semantic Scholar).

---

- Code (`src/`): MIT License
- Paper (`paper/`): CC-BY 4.0
- Data (Zenodo): CC-BY 4.0
