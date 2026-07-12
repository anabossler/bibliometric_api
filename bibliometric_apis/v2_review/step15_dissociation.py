"""
—  Double-dissociation test for the AWS mechanism.

Design:
  Split the usable INTRA-cluster citation edges 80/20 at the EDGE level.
  * Train the encoder (MultipleNegativesRankingLoss) on the 80% train intra pairs.
  * Evaluate base vs adapted encoder, Recall@50, on TWO held-out sets:
      (1) the 20% held-out INTRA edges  — same domain as training signal
      (2) the 656 CROSS-cluster edges    — the target the paper cares about
  Both eval sets are disjoint from the training pairs.

Interpretation:
  intra ↑ AND cross ↓  -> specialisation is REAL and DIRECTIONAL -> AWS mechanism
                          demonstrated (the encoder can learn 'related-by-citation'
                          within a subdomain, but that skill does not cross the
                          semantic boundary).
  both ↓               -> global overfitting on 604 pairs -> report only 'no gain'.

Reuses step14_supervised_finetune as a library (constants + domain_adapt +
tokenize/l2_normalize/bootstrap_ci).  CPU only.  Outputs to $FT_OUT_DIR.
"""
import os
import json
import random
import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import step14_supervised_finetune as ft

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("step15")

OUT_DIR = ft.Path(os.environ.get("FT_OUT_DIR", "runs/ft_dissociation"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def eval_edges(edges_df, doi_to_idx, doc_terms, base_emb, adapt_emb):
    """Recall@50 per-pair hit arrays for base and adapted on the given edges."""
    def dense_top(X, si):
        sims = X @ X[si]
        sims[si] = -np.inf
        return set(np.argsort(-sims)[:ft.TOP_K])

    hb, hz = [], []
    skipped = 0
    for s, t in zip(edges_df["s"], edges_df["t"]):
        if s not in doi_to_idx or t not in doi_to_idx:
            skipped += 1
            continue
        si, ti = doi_to_idx[s], doi_to_idx[t]
        if not doc_terms[si]:          # same skip rule as the paper harness
            skipped += 1
            continue
        hb.append(int(ti in dense_top(base_emb, si)))
        hz.append(int(ti in dense_top(adapt_emb, si)))
    return np.array(hb), np.array(hz), skipped


def wilcox_greater(x, y):
    d = np.asarray(x) - np.asarray(y)
    if np.count_nonzero(d) < 10:
        return None, None
    w, p = wilcoxon(x, y, alternative="greater")
    return float(w), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra-edges", default=str(
        ft.Path("crossref_full_graph/citation_edges_all_crossref.csv")))
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-seq-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    order = pd.read_csv(ft.DOI_ORDER)
    df_abs = pd.read_csv(ft.ABSTRACTS)
    abstr_lc = {str(k).lower().strip(): v for k, v in
                zip(df_abs["doi"], df_abs["abstract"])}
    doi_to_idx = {str(d).lower().strip(): i for i, d in enumerate(order["doi"])}
    doc_terms = [ft.tokenize(abstr_lc.get(d, "") if isinstance(abstr_lc.get(d, ""), str) else "")
                 for d in [str(x).lower().strip() for x in order["doi"]]]

    base_emb = ft.l2_normalize(np.load(ft.EMB_DIR / "sbert.npy"))

    # ---- build usable INTRA edge list (evaluable + trainable) ----
    E = pd.read_csv(args.intra_edges)
    intra = E[E["edge_type"] == "intra"].copy()
    intra["s"] = intra["source_doi"].astype(str).str.lower().str.strip()
    intra["t"] = intra["target_doi"].astype(str).str.lower().str.strip()

    def usable(s, t):
        a, b = abstr_lc.get(s), abstr_lc.get(t)
        return (s in doi_to_idx and t in doi_to_idx
                and isinstance(a, str) and a and isinstance(b, str) and b
                and bool(doc_terms[doi_to_idx[s]]))
    intra = intra[[usable(s, t) for s, t in zip(intra["s"], intra["t"])]].reset_index(drop=True)
    log.info("Usable intra edges: %d", len(intra))

    # ---- 80/20 edge-level split ----
    rng = random.Random(args.seed)
    idx = list(range(len(intra)))
    rng.shuffle(idx)
    n_test = int(round(len(idx) * args.test_frac))
    test_idx = set(idx[:n_test])
    train_edges = intra.iloc[[i for i in idx if i not in test_idx]].reset_index(drop=True)
    test_edges = intra.iloc[[i for i in idx if i in test_idx]].reset_index(drop=True)
    log.info("Split: %d train / %d held-out intra", len(train_edges), len(test_edges))

    # ---- cross edges (the paper's 656 eval set) ----
    cross = pd.read_csv(ft.EDGES)
    cross["s"] = cross["source_doi"].astype(str).str.lower().str.strip()
    cross["t"] = cross["target_doi"].astype(str).str.lower().str.strip()

    # ---- train on 80% intra pairs ----
    train_pairs = [(abstr_lc[s], abstr_lc[t]) for s, t in zip(train_edges["s"], train_edges["t"])]
    log.info("Training on %d intra pairs (%d epochs, batch %d)",
             len(train_pairs), args.epochs, args.batch_size)
    model = ft.domain_adapt([], "supervised", args.epochs, args.batch_size,
                            args.lr, args.device, False,
                            max_seq_length=args.max_seq_length, train_pairs=train_pairs)

    texts_in_order = [abstr_lc.get(str(d).lower().strip(), "") or "" for d in order["doi"]]
    adapt_emb = ft.l2_normalize(model.encode(texts_in_order, batch_size=64,
                                             show_progress_bar=True, convert_to_numpy=True))

    # ---- evaluate on both held-out sets ----
    res = {}
    for name, edf in [("intra_heldout", test_edges), ("cross", cross)]:
        hb, hz, sk = eval_edges(edf, doi_to_idx, doc_terms, base_emb, adapt_emb)
        w, p = wilcox_greater(hz, hb)   # adapted > base ?
        res[name] = {
            "n": int(len(hb)), "skipped": int(sk),
            "base": round(float(hb.mean()), 4),
            "adapt": round(float(hz.mean()), 4),
            "delta_adapt_minus_base": round(float(hz.mean() - hb.mean()), 4),
            "base_ci": [round(x, 4) for x in ft.bootstrap_ci(hb)],
            "adapt_ci": [round(x, 4) for x in ft.bootstrap_ci(hz)],
            "wilcoxon_adapt_gt_base": {"W": w, "p": p},
        }
        log.info("%-14s base=%.4f adapt=%.4f delta=%+.4f p=%s (n=%d)",
                 name, res[name]["base"], res[name]["adapt"],
                 res[name]["delta_adapt_minus_base"], p, len(hb))

    summary = {
        "design": "80/20 edge-level split on usable intra edges; train on 80%, "
                   "eval base vs adapted on 20% held-out intra AND on cross.",
        "n_train_pairs": len(train_pairs),
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "results": res,
    }
    (OUT_DIR / "dissociation_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(OUT_DIR / "adapt_emb_heldout.npy", adapt_emb.astype(np.float32))
    log.info("wrote %s", OUT_DIR / "dissociation_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
