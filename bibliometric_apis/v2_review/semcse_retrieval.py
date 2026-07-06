#!/usr/bin/env python3
"""
semcse_retrieval.py
===================

SemCSE (Brinner & Zarriess, EMNLP 2025) embeddings for the plastic corpus,
aligned with the canonical phase0_doi_order.csv used by hygrag_aws_v3.py.

Writes phase0_embeddings/semcse.npy so that hygrag_aws_v3.py phase 4 can
score condition S (SemCSE) and condition T (RRF including SemCSE) against the
same 656 cross-cluster citation pairs as every other encoder. The output row
order is identical to phase0_doi_order.csv, so the resulting .npy is a drop-in
sibling of sbert.npy / specter2.npy / qwen3.npy.


CHECKPOINTING
-------------
Embeddings are written to a partial file after every checkpoint interval and
the completed-row count is tracked in a sidecar JSON. Re-running resumes from
the last completed row, so an interrupted run never recomputes finished rows.

USAGE
-----
  python semcse_retrieval.py            # compute (resumes if interrupted)
  python semcse_retrieval.py --force    # recompute from scratch
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("semcse")

# ---- Paths (edit ROOT if your checkout lives elsewhere) --------------------
ROOT = Path("/Users//Desktop/openalex")
ABSTRACTS = ROOT / "backup_recycled_a/full_corpus/abstracts_full.csv"
DOI_ORDER = ROOT / "runs/hygrag_v3/phase0_doi_order.csv"
OUT_DIR = ROOT / "runs/hygrag_v3/phase0_embeddings"
OUT_PATH = OUT_DIR / "semcse.npy"
PARTIAL_PATH = OUT_DIR / "semcse_partial.npy"
PROGRESS_PATH = OUT_DIR / "semcse_progress.json"

MODEL_NAME = "CLAUSE-Bielefeld/SemCSE"

# Loading strategy: prefer sentence-transformers (correct pooling automatically).
USE_RAW = False          # set True only if repo is not a sentence-transformers model
POOLING = "mean"         # used only when USE_RAW is True: "mean" or "cls"

BATCH = 8                # M2 8GB: 8 is safe for a small encoder; lower if OOM
MAX_LEN = 256
CHECKPOINT_EVERY = 256   # rows between checkpoints


def load_corpus_in_order() -> list[str]:
    log.info("Loading DOI order from %s", DOI_ORDER.name)
    order = pd.read_csv(DOI_ORDER)
    doi_col = "doi" if "doi" in order.columns else order.columns[0]
    log.info("  %d DOIs in canonical order", len(order))

    abs_df = pd.read_csv(ABSTRACTS)
    abs_map = dict(zip(abs_df["doi"].astype(str), abs_df["abstract"].astype(str)))

    abstracts, missing = [], 0
    for d in order[doi_col].astype(str):
        t = abs_map.get(d, "")
        if not isinstance(t, str) or not t.strip() or t == "nan":
            missing += 1
            t = ""
        abstracts.append(t)
    log.info("  Abstracts retrieved: %d (%d missing, encoded as empty)",
             len(abstracts) - missing, missing)
    return abstracts


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---- Encoder backends ------------------------------------------------------

def embed_sentence_transformers(abstracts: list[str], start: int,
                                emb: np.ndarray) -> np.ndarray:
    """Preferred path: pooling is whatever the model was trained with."""
    from sentence_transformers import SentenceTransformer

    log.info("Loading %s via SentenceTransformer ...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_LEN

    n = len(abstracts)
    t0 = time.time()
    for b0 in range(start, n, BATCH):
        b1 = min(b0 + BATCH, n)
        # empty strings -> single space so the model still returns a vector
        batch = [t if t.strip() else " " for t in abstracts[b0:b1]]
        vecs = model.encode(batch, convert_to_numpy=True,
                            normalize_embeddings=False,
                            show_progress_bar=False)
        emb[b0:b1] = vecs.astype(np.float32)

        if b1 % CHECKPOINT_EVERY < BATCH or b1 == n:
            _checkpoint(emb, b1, n)
            rate = (b1 - start) / max(1e-6, time.time() - t0)
            eta = (n - b1) / max(1e-6, rate) / 60
            log.info("  [%d/%d] %.1f rows/s  ETA %.1f min", b1, n, rate, eta)
    return emb


def embed_raw(abstracts: list[str], start: int, emb: np.ndarray) -> np.ndarray:
    """Fallback: raw transformers with explicit pooling. Confirm POOLING
    against the model card before trusting the result."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    log.info("Loading %s via AutoModel (pooling=%s) ...", MODEL_NAME, POOLING)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME)
    mdl.eval()

    n = len(abstracts)
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(start, n, BATCH):
            b1 = min(b0 + BATCH, n)
            batch = [t if t.strip() else " " for t in abstracts[b0:b1]]
            ins = tok(batch, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            out = mdl(**ins)
            hidden = out.last_hidden_state           # (B, L, H)

            if POOLING == "cls":
                vecs = hidden[:, 0, :]
            else:  # mean pooling over non-padding tokens
                mask = ins["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                vecs = summed / counts

            emb[b0:b1] = vecs.cpu().numpy().astype(np.float32)

            if b1 % CHECKPOINT_EVERY < BATCH or b1 == n:
                _checkpoint(emb, b1, n)
                rate = (b1 - start) / max(1e-6, time.time() - t0)
                eta = (n - b1) / max(1e-6, rate) / 60
                log.info("  [%d/%d] %.1f rows/s  ETA %.1f min", b1, n, rate, eta)
    return emb


def _checkpoint(emb: np.ndarray, rows_done: int, n: int) -> None:
    np.save(PARTIAL_PATH, emb)
    PROGRESS_PATH.write_text(json.dumps(
        {"n": n, "model": MODEL_NAME, "rows_done": int(rows_done)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if semcse.npy exists")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists() and not args.force:
        X = np.load(OUT_PATH)
        log.info("semcse.npy already exists: shape=%s norm[0]=%.3f "
                 "(use --force to redo)", X.shape, np.linalg.norm(X[0]))
        return

    abstracts = load_corpus_in_order()
    n = len(abstracts)

    # Resume from checkpoint if consistent.
    start = 0
    emb = None
    if not args.force and PARTIAL_PATH.exists() and PROGRESS_PATH.exists():
        prog = json.loads(PROGRESS_PATH.read_text())
        if prog.get("n") == n and prog.get("model") == MODEL_NAME:
            partial = np.load(PARTIAL_PATH)
            done = int(prog.get("rows_done", 0))
            if 0 < done <= n:
                emb = partial
                start = done
                log.info("Resuming from checkpoint at row %d/%d", start, n)

    if emb is None:
        # dim is discovered from a single probe encode
        probe_dim = _probe_dim()
        emb = np.zeros((n, probe_dim), dtype=np.float32)

    if USE_RAW:
        emb = embed_raw(abstracts, start, emb)
    else:
        emb = embed_sentence_transformers(abstracts, start, emb)

    emb = l2_normalize(emb)
    np.save(OUT_PATH, emb)
    PARTIAL_PATH.unlink(missing_ok=True)
    PROGRESS_PATH.unlink(missing_ok=True)
    log.info("Saved %s  shape=%s  norm[0]=%.3f",
             OUT_PATH, emb.shape, np.linalg.norm(emb[0]))


def _probe_dim() -> int:
    """Encode one short string to discover the embedding dimension."""
    if USE_RAW:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModel.from_pretrained(MODEL_NAME)
        mdl.eval()
        with torch.no_grad():
            ins = tok(["probe"], return_tensors="pt", truncation=True,
                      max_length=MAX_LEN)
            out = mdl(**ins)
            return int(out.last_hidden_state.shape[-1])
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        v = model.encode(["probe"], convert_to_numpy=True,
                         show_progress_bar=False)
        return int(v.shape[-1])


if __name__ == "__main__":
    main()
