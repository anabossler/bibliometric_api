"""
semcse_retrieval.py
===================
SemCSE (Brinner & Zarriess, EMNLP 2025) embeddings for the plastic
corpus, aligned with paper_tmo.csv order.

Adds condition S (SemCSE) to phase4_results.csv.
"""

import os, sys, time, logging, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("semcse")

ROOT       = Path("/Users/")
ABSTRACTS  = ROOT / "backup_recycled_a/full_corpus/abstracts_full.csv"
DOI_ORDER  = ROOT / "runs/hygrag_v3/phase0_doi_order.csv"
OUT_DIR    = ROOT / "runs/hygrag_v3/phase0_embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUT_DIR / "semcse.npy"

# Cosine-similarity-compatible version of SemCSE
MODEL_NAME = "CLAUSE-Bielefeld/SemCSE"
BATCH      = 4          
MAX_LEN    = 256        


def load_corpus_in_order():
    log.info("Loading DOI order from phase0_doi_order.csv...")
    order = pd.read_csv(DOI_ORDER)
    log.info("  %d DOIs in canonical order", len(order))

    abs_df = pd.read_csv(ABSTRACTS)
    abs_map = dict(zip(abs_df["doi"], abs_df["abstract"]))

    abstracts = []
    missing = 0
    for d in order["doi"]:
        t = abs_map.get(d, "")
        if not isinstance(t, str) or not t.strip():
            missing += 1
            t = ""
        abstracts.append(t)
    log.info("  Abstracts retrieved: %d (%d missing)",
             len(abstracts) - missing, missing)
    return abstracts


def embed_semcse(abstracts):
    log.info("Loading %s ...", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME)
    mdl.eval()

    embs = []
    n = len(abstracts)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, BATCH):
            batch = abstracts[i:i + BATCH]
            ins = tok(batch, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            out = mdl(**ins)
            # CLS token: out.last_hidden_state[:, 0, :]
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            embs.append(cls)
            if (i // BATCH) % 25 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = (i + BATCH) / elapsed
                eta = (n - i) / max(rate, 0.1) / 60
                log.info("  %d/%d   rate=%.1f/s   ETA %.1f min",
                         i + BATCH, n, rate, eta)

    X = np.vstack(embs)
    # L2-normalize for cosine retrieval consistency
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    return X.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if semcse.npy exists")
    args = ap.parse_args()

    if OUT_PATH.exists() and not args.force:
        log.info("semcse.npy already exists at %s (use --force to redo)", OUT_PATH)
        X = np.load(OUT_PATH)
        log.info("  shape=%s  norm[0]=%.3f", X.shape, np.linalg.norm(X[0]))
        return

    abstracts = load_corpus_in_order()
    X = embed_semcse(abstracts)
    np.save(OUT_PATH, X)
    log.info("Saved %s  shape=%s", OUT_PATH, X.shape)


if __name__ == "__main__":
    main()
