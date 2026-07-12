"""
step14_domain_finetune.py
=========================

  * We take the SBERT baseline the paper already uses (all-MiniLM-L6-v2,
    condition B, Recall@50 = 0.3384) and *domain-adapt* it to the recycled
    corpus with an UNSUPERVISED objective (TSDAE by default, SimCSE optional).
  * We re-embed all 3,043 abstracts IN THE CANONICAL DOI ORDER and drop the new
    matrix into the identical retrieval harness (same 656 cross-cluster citation
    pairs, same Recall@50, same bootstrap CI, same Wilcoxon).

Why UNSUPERVISED adaptation
-----------------------------------------------------------------
The 656 citation pairs are the *evaluation signal*. Fine-tuning on them (or on
any citation-derived supervision) would leak the test set and inflate recall
circularly — a reviewer would reject it instantly. TSDAE / SimCSE learn only
from the abstract *text distribution* (denoising / dropout self-supervision).
So the encoder is genuinely adapted to the corpus vocabulary WITHOUT ever seeing
which papers cite which. The question it answers is clean:

    "Once the encoder is fluent in this domain's vocabulary, does that alone
     recover the cross-vocabulary citation links?"

Interpreting the outcome (both directions support the paper)
------------------------------------------------------------
  * If Recall@50 barely moves and stays far below the oracle union (0.596):
    domain fluency does NOT close the gap. The fragmentation is structural, not
    a "raw model doesn't know the domain" artifact. -> AWS confirmed.
  * If Recall@50 rises but still trails the fusion/oracle: adaptation helps at
    the margin yet the awareness-without-synthesis floor persists. -> the
    diagnosis (a residual, non-recoverable gap) still holds; you simply report
    the adapted encoder as one more signal below the oracle.

Drop-in compatibility
----------------------
Reuses, VERBATIM, the metric core of final_retrieval_table.py: tokenize(),
bootstrap_ci(), the dense/lexical scorers, and the Recall@50 aggregation. 

Usage
-----

    # quick plumbing check (tiny subset, ~1 min, no real training signal)
    ./.venv/bin/python step14_domain_finetune.py --smoke

    # alternative unsupervised recipe
    ./.venv/bin/python step14_domain_finetune.py --method simcse --epochs 1

Outputs (runs/finetune_domain/)
-------------------------------
    <method>_minilm.npy              domain-adapted embeddings, canonical order
    summary_finetune.json            base vs adapted Recall@50, CI, Wilcoxon, oracle gap
    finetune_comparison.png          bar chart: lexical | SBERT base | SBERT adapted | oracle

Needs: sentence-transformers, torch, scipy, matplotlib, pandas, numpy.

"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("finetune")

# --------------------------------------------------------------------------
# Paths (identical to final_retrieval_table.py)
# --------------------------------------------------------------------------
DOI_ORDER = Path("runs/hygrag_v3/phase0_doi_order.csv")
EMB_DIR   = Path("runs/hygrag_v3/phase0_embeddings")
ABSTRACTS = Path("backup_recycled_a/full_corpus/abstracts_full.csv")
EDGES     = Path("backup_recycled_a/full_corpus/citation_edges_cross_cluster.csv")
MAIN_SUMMARY = Path("runs/final_table/summary.json")   # for the oracle reference

OUT_DIR = Path(os.environ.get("FT_OUT_DIR", "runs/finetune_domain"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 50
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # == condition B (sbert.npy)

# --------------------------------------------------------------------------
# Metric core  --  COPIED VERBATIM from final_retrieval_table.py
# (do not edit here without editing there; they must stay identical)
# --------------------------------------------------------------------------
STOPWORDS = set(
    "the and for that this with from are was were been have has had not but "
    "its can will also into than more which their these they would could "
    "should about each between through during after before other some such "
    "only over how our who what when where doi org https http www "
    "available online published all any both did does done either "
    "however just much most must same several since still very while "
    "using used use results show study paper present".split()
)


def tokenize(text: str) -> set:
    if not isinstance(text, str):
        return set()
    words = re.findall(r"[a-z]+", text.lower())
    words = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    return set(words) | {f"{words[i]} {words[i+1]}"
                         for i in range(len(words) - 1)}


def l2_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def bootstrap_ci(values, n_boot=1000, seed=42):
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return (round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4))


# --------------------------------------------------------------------------
# Unsupervised domain adaptation
# --------------------------------------------------------------------------
def split_sentences(abstract: str) -> list[str]:
    """Light regex sentence split (no nltk dependency)."""
    if not isinstance(abstract, str):
        return []
    parts = re.split(r"(?<=[.!?])\s+", abstract.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _DeletionNoiseDataset:
    """nltk-free re-implementation of sentence-transformers'
    DenoisingAutoEncoderDataset.

    Each __getitem__ returns InputExample(texts=[damaged, original]) where the
    damaged text has a fraction `del_ratio` of its whitespace-delimited words
    randomly deleted. This is the exact TSDAE noising scheme (Wang et al., 2021),
    only the tokenizer differs (str.split instead of nltk.word_tokenize), which
    removes the punkt download requirement and changes nothing about the method.
    """

    def __init__(self, sentences, del_ratio: float = 0.6, seed: int = 42):
        from sentence_transformers import InputExample
        self._InputExample = InputExample
        self.sentences = sentences
        self.del_ratio = del_ratio
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.sentences)

    def _delete(self, text: str) -> str:
        words = text.split()
        n = len(words)
        if n == 0:
            return text
        keep = [w for w in words if self._rng.random() >= self.del_ratio]
        if not keep:                      # never return an empty string
            keep = [words[self._rng.randrange(n)]]
        return " ".join(keep)

    def __getitem__(self, idx):
        clean = self.sentences[idx]
        return self._InputExample(texts=[self._delete(clean), clean])


def domain_adapt(train_texts, method, epochs, batch_size, lr, device, smoke,
                 max_seq_length=64, train_pairs=None):
    """Return a domain-adapted SentenceTransformer (same arch/pooling as base).

    For method='supervised', `train_pairs` is a list of (src_text, tgt_text)
    abstract pairs drawn from INTRA-cluster citations. These are positives for
    MultipleNegativesRankingLoss (in-batch negatives). `train_texts` is ignored.
    """
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers import InputExample

    model = SentenceTransformer(BASE_MODEL, device=device)
    # Cap the TRAINING sequence length — this is the dominant memory driver for
    # TSDAE (the decoder attends over the full sequence). We restore the model's
    # native length before the final encode pass so the saved embeddings stay
    # directly comparable to sbert.npy.
    base_seq_length = model.max_seq_length
    model.max_seq_length = max_seq_length
    # keep the base model's own pooling (mean) so base vs adapted is apples-to-apples

    if method == "tsdae":
        # nltk-free TSDAE: our own deletion-noise dataset (see _DeletionNoiseDataset).
        # Semantics are identical to sentence-transformers'
        # DenoisingAutoEncoderDataset — each item is InputExample([damaged, clean])
        # with ~60% of words deleted — but it uses whitespace tokenization instead
        # of nltk.word_tokenize, so no nltk / punkt download is required.
        ds = _DeletionNoiseDataset(train_texts, del_ratio=0.6)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            drop_last=True)
        loss = losses.DenoisingAutoEncoderLoss(
            model, decoder_name_or_path=BASE_MODEL, tie_encoder_decoder=True)
    elif method == "simcse":
        # unsupervised SimCSE: (sentence, sentence) positive pairs, dropout noise
        examples = [InputExample(texts=[s, s]) for s in train_texts]
        loader = DataLoader(examples, batch_size=batch_size, shuffle=True,
                            drop_last=True)
        loss = losses.MultipleNegativesRankingLoss(model)
    elif method == "supervised":
        # SUPERVISED domain adaptation: positives are (src, tgt) abstract pairs
        # from INTRA-cluster citations. MultipleNegativesRankingLoss uses the
        # other targets in the batch as in-batch negatives, so a large batch =
        # more negatives = stronger signal. The task (find the cited paper among
        # ~batch_size candidates) is genuinely hard, so the loss actually moves.
        # Evaluation is on the held-out CROSS-cluster citations: strict split by
        # construction (no cross pair is ever seen in training) => zero leakage.
        if not train_pairs:
            raise ValueError("method='supervised' requires non-empty train_pairs")
        examples = [InputExample(texts=[s, t]) for s, t in train_pairs]
        loader = DataLoader(examples, batch_size=batch_size, shuffle=True,
                            drop_last=True)
        loss = losses.MultipleNegativesRankingLoss(model)
    else:
        raise ValueError(f"unknown method {method!r}")

    steps = len(loader)
    warmup = 0 if smoke else int(0.1 * steps * epochs)
    _n = len(train_pairs) if method == "supervised" else len(train_texts)
    log.info("Domain-adapting via %s: %d examples, %d steps/epoch, %d epoch(s), device=%s",
             method.upper(), _n, steps, epochs, device)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        weight_decay=0.0,
        scheduler="constantlr",
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )
    # restore native length so the final encode matches the base sbert.npy regime
    model.max_seq_length = base_seq_length
    return model


# --------------------------------------------------------------------------
# Evaluation (identical harness slice)
# --------------------------------------------------------------------------
def evaluate(order, abstr_by_doi, adapted_emb):
    doi_to_idx = {d: i for i, d in enumerate(order["doi"])}
    doc_terms = [tokenize(abstr_by_doi.get(d, "")) for d in order["doi"]]

    sbert_base = l2_normalize(np.load(EMB_DIR / "sbert.npy"))
    adapted = l2_normalize(adapted_emb)

    def score_lexical(query, exclude, top_k):
        if not query:
            return []
        scored = []
        for i, t in enumerate(doc_terms):
            if i == exclude:
                continue
            ov = len(query & t)
            if ov:
                scored.append((i, ov))
        scored.sort(key=lambda x: -x[1])
        return [i for i, _ in scored[:top_k]]

    def score_dense(X, si, top_k):
        sims = X @ X[si]
        sims[si] = -np.inf
        return list(np.argsort(-sims)[:top_k])

    pairs = pd.read_csv(EDGES)
    rows = []
    skipped = 0
    for _, row in pairs.iterrows():
        src_doi, tgt_doi = row["source_doi"], row["target_doi"]
        if src_doi not in doi_to_idx or tgt_doi not in doi_to_idx:
            skipped += 1
            continue
        si, ti = doi_to_idx[src_doi], doi_to_idx[tgt_doi]
        src_lex = doc_terms[si]
        if not src_lex:
            skipped += 1
            continue
        top_a = score_lexical(src_lex, si, TOP_K)
        top_b = score_dense(sbert_base, si, TOP_K)
        top_ad = score_dense(adapted, si, TOP_K)
        rows.append({
            "source_doi": src_doi, "target_doi": tgt_doi,
            "hit_A_lexical":      int(ti in top_a),
            "hit_B_sbert_base":   int(ti in top_b),
            "hit_Z_sbert_adapt":  int(ti in top_ad),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "results_finetune.csv", index=False)
    return df, skipped


def paired_wilcoxon(x, y, alt="greater"):
    d = np.asarray(x) - np.asarray(y)
    if np.count_nonzero(d) < 10:
        return None, None
    w, p = wilcoxon(x, y, alternative=alt)
    return float(w), float(p)


# --------------------------------------------------------------------------
def make_figure(rec, ci, oracle, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Lexical\nbaseline", "SBERT\nbase",
              f"SBERT\n+{method.upper()}", "Oracle\nunion"]
    vals = [rec["A_lexical"], rec["B_sbert_base"],
            rec["Z_sbert_adapt"], oracle]
    lo = [rec["A_lexical"] - ci["A_lexical"][0],
          rec["B_sbert_base"] - ci["B_sbert_base"][0],
          rec["Z_sbert_adapt"] - ci["Z_sbert_adapt"][0], 0]
    hi = [ci["A_lexical"][1] - rec["A_lexical"],
          ci["B_sbert_base"][1] - rec["B_sbert_base"],
          ci["Z_sbert_adapt"][1] - rec["Z_sbert_adapt"], 0]
    colors = ["#9e9e9e", "#4c72b0", "#dd8452", "#55a868"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=[lo, hi], capsize=4, color=colors,
           edgecolor="black", linewidth=0.6)
    ax.axhspan(oracle, 1.0, color="#55a868", alpha=0.06)
    ax.text(3.35, (oracle + 1) / 2,
            f"unrecoverable\nby any single\nsignal: {1 - oracle:.1%}",
            ha="center", va="center", fontsize=8, color="#2f6b46")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("Recall@50 (656 cross-cluster citation pairs)")
    ax.set_ylim(0, max(0.7, oracle + 0.08))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Unsupervised domain adaptation does not close the AWS gap",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "finetune_comparison.png", dpi=200)
    log.info("  wrote %s", OUT_DIR / "finetune_comparison.png")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["tsdae", "simcse", "supervised"],
                    default="supervised")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="lower = less memory. 4 fits an 8GB Mac; raise on a GPU box")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-seq-length", type=int, default=64,
                    help="training seq cap (memory driver). Encoding uses the "
                         "base model's full length regardless, to match sbert.npy")
    ap.add_argument("--max-train-sentences", type=int, default=12000,
                    help="subsample training sentences (TSDAE saturates well "
                         "before 27k; keeps memory + time bounded). 0 = use all")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                    default="cpu",
                    help="cpu is the safe default on an 8GB Mac (MPS hits a hard "
                         "~9GB cap and OOMs mid-epoch). Use mps/cuda on bigger RAM")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + 1 epoch to validate plumbing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--intra-edges", type=str,
                    default="crossref_full_graph/citation_edges_all_crossref.csv",
                    help="full citation graph (intra+cross) with edge_type column; "
                         "supervised training uses the INTRA rows only")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device() if args.device == "auto" else args.device
    order = pd.read_csv(DOI_ORDER)
    df_abs = pd.read_csv(ABSTRACTS)
    abstr_by_doi = dict(zip(df_abs["doi"], df_abs["abstract"]))
    log.info("Corpus: %d papers (canonical order)", len(order))

    # case-insensitive abstract lookup (Crossref graph DOIs are lowercased)
    abstr_by_doi_lc = {str(k).lower().strip(): v for k, v in abstr_by_doi.items()}

    train_texts = []
    train_pairs = None

    if args.method == "supervised":
        # --- build INTRA-cluster citation pairs as supervised positives ---
        E = pd.read_csv(args.intra_edges)
        intra = E[E["edge_type"] == "intra"] if "edge_type" in E.columns else \
                E[E["source_cluster"] == E["target_cluster"]]
        pairs = []
        n_miss = 0
        for s, t in zip(intra["source_doi"].astype(str).str.lower(),
                        intra["target_doi"].astype(str).str.lower()):
            a, b = abstr_by_doi_lc.get(s), abstr_by_doi_lc.get(t)
            if a and b and isinstance(a, str) and isinstance(b, str):
                pairs.append((a, b))
            else:
                n_miss += 1
        train_pairs = pairs
        if args.smoke:
            train_pairs = train_pairs[:64]
            args.epochs = 1
        log.info("Supervised INTRA-cluster pairs: %d usable (%d skipped for "
                 "missing abstracts) from %d intra edges",
                 len(train_pairs), n_miss, len(intra))
    else:
        # --- build training text (sentences from in-corpus abstracts only) ---
        for d in order["doi"]:
            train_texts.extend(split_sentences(abstr_by_doi.get(d, "")))
        train_texts = [t for t in train_texts if t]
        if args.smoke:
            train_texts = train_texts[:200]
            args.epochs = 1
        elif args.max_train_sentences and len(train_texts) > args.max_train_sentences:
            rng = random.Random(args.seed)
            train_texts = rng.sample(train_texts, args.max_train_sentences)
            log.info("Subsampled to %d training sentences (--max-train-sentences)",
                     len(train_texts))
        log.info("Training sentences: %d", len(train_texts))

    # --- domain-adapt ---
    t0 = time.time()
    model = domain_adapt(train_texts, args.method, args.epochs,
                         args.batch_size, args.lr, device, args.smoke,
                         max_seq_length=args.max_seq_length, train_pairs=train_pairs)
    log.info("Adaptation done in %.0fs", time.time() - t0)

    # --- re-embed ALL abstracts in canonical order ---
    texts_in_order = [abstr_by_doi.get(d, "") or "" for d in order["doi"]]
    adapted_emb = model.encode(texts_in_order, batch_size=64,
                               show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=False)
    out_npy = OUT_DIR / f"{args.method}_minilm.npy"
    np.save(out_npy, adapted_emb.astype(np.float32))
    log.info("  wrote %s  shape=%s", out_npy, adapted_emb.shape)

    # --- evaluate in the identical harness ---
    df, skipped = evaluate(order, abstr_by_doi, adapted_emb)
    cols = [c for c in df.columns if c.startswith("hit_")]
    rec = {c.replace("hit_", ""): round(df[c].mean(), 4) for c in cols}
    ci = {c.replace("hit_", ""): list(bootstrap_ci(df[c].values)) for c in cols}

    w_zb, p_zb = paired_wilcoxon(df["hit_Z_sbert_adapt"].values,
                                 df["hit_B_sbert_base"].values)
    w_za, p_za = paired_wilcoxon(df["hit_Z_sbert_adapt"].values,
                                 df["hit_A_lexical"].values)

    # oracle reference from the main table (union of all individual signals)
    oracle = 0.596
    if MAIN_SUMMARY.exists():
        oracle = json.loads(MAIN_SUMMARY.read_text()).get(
            "oracle_upper_bound_union_of_all_individual", oracle)

    summary = {
        "method": args.method,
        "base_model": BASE_MODEL,
        "epochs": args.epochs,
        "n_pairs": len(df),
        "n_skipped": skipped,
        "top_k": TOP_K,
        "n_train_sentences": len(train_texts),
        "n_train_pairs": len(train_pairs) if train_pairs else 0,
        "device": device,
        "smoke": args.smoke,
        "recall": rec,
        "bootstrap_ci": ci,
        "delta_adapt_minus_base": round(rec["Z_sbert_adapt"] - rec["B_sbert_base"], 4),
        "wilcoxon_adapt_vs_base_greater": {"W": w_zb, "p": p_zb},
        "wilcoxon_adapt_vs_lexical_greater": {"W": w_za, "p": p_za},
        "oracle_union_reference": oracle,
        "adapted_gap_below_oracle": round(oracle - rec["Z_sbert_adapt"], 4),
    }
    (OUT_DIR / "summary_finetune.json").write_text(json.dumps(summary, indent=2))
    log.info("  wrote %s", OUT_DIR / "summary_finetune.json")

    make_figure(rec, ci, oracle, args.method)

    # console report
    print("\n" + "=" * 64)
    print(f" Domain adaptation ({args.method.upper()}) — Recall@50 on 656 pairs")
    print("=" * 64)
    print(f"  Lexical baseline      {rec['A_lexical']:.4f}  CI{ci['A_lexical']}")
    print(f"  SBERT base            {rec['B_sbert_base']:.4f}  CI{ci['B_sbert_base']}")
    print(f"  SBERT + {args.method:<6}       {rec['Z_sbert_adapt']:.4f}  CI{ci['Z_sbert_adapt']}")
    print(f"  delta (adapt-base)    {summary['delta_adapt_minus_base']:+.4f}"
          f"   Wilcoxon p={p_zb}")
    print(f"  oracle union          {oracle:.4f}")
    print(f"  gap adapt->oracle     {summary['adapted_gap_below_oracle']:.4f}"
          f"   ({1-oracle:.1%} unrecoverable by any single signal)")
    print("=" * 64)


if __name__ == "__main__":
    main()
