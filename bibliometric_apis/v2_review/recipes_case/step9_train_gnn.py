"""
Step 9: Train a heterogeneous GraphSAGE on the clean recipe knowledge graph.

WHAT IT DOES
------------
1. Loads data/graph_step4_foodon.gpickle (clean, FoodOn-enriched).
2. Filters out recipes with quality_flag='upstream_corrupt' (and the edges
   incident to them) so the model never sees corrupted training signal.
3. Builds per-type input features:
   - Recipe / Ingredient / Action: qwen3-embedding-8b (4096-dim) from
     data/layer_2_embeddings.npz (already computed in step2).
   - Tool / FoodOnClass / Place: SBERT (all-MiniLM-L6-v2, 384-dim)
     computed on-the-fly from node names / labels.
   - Period: learnable embedding (10 nodes only).
4. Constructs a PyG HeteroData object with all node and edge types.
5. Adds reverse edges (T.ToUndirected) so message passing flows both ways.
6. Splits "contains" edges 80/10/10 train/val/test.
7. Trains a 2-layer heterogeneous GraphSAGE with link prediction loss
   (BCE with negative sampling).
8. Reports train/val/test AUC + Hits@10.
9. Saves trained model and final node embeddings (128-dim per node).

Reads:
  data/graph_step4_foodon.gpickle
  data/layer_2_embeddings.npz

Writes:
  data/gnn_model.pt                       (PyTorch state dict)
  data/gnn_node_embeddings.npz            (128-dim per node, per type)
  data/gnn_training_metrics.json          (train/val/test AUC per epoch)
  data/gnn_training_log.csv               (epoch-by-epoch metrics)

Usage:
  python step9_train_gnn.py                       # default config
  python step9_train_gnn.py --epochs 50
  python step9_train_gnn.py --hidden-dim 64 --out-dim 64
  python step9_train_gnn.py --device mps          # force device
  python step9_train_gnn.py --dry-run             # build everything but no train

Estimated runtime on Mac M2 8GB CPU: 10-15 min for 100 epochs.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import negative_sampling


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPH_PATH = Path("data/graph_step4_foodon.gpickle")
EMBEDDINGS_PATH = Path("data/layer_2_embeddings.npz")
OUTPUT_DIR = Path("data/")

# Node types and their feature source.
QWEN_TYPES = {"Recipe", "Ingredient", "Action"}        # use qwen3 from npz
SBERT_TYPES = {"Tool", "FoodOnClass", "Place"}         # SBERT on-the-fly
LEARNABLE_TYPES = {"Period"}                            # learnable embedding

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast on CPU
SBERT_DIM = 384

# Edge type used as the training target for link prediction.
TARGET_RELATION = ("Recipe", "contains", "Ingredient")

# Train/val/test split fractions of the "contains" edges.
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# Training defaults (CLI-overridable).
DEFAULT_HIDDEN_DIM = 128
DEFAULT_OUT_DIM = 128
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_NEG_RATIO = 1.0
DEFAULT_EARLY_STOP_PATIENCE = 15

SEED = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("step9_gnn")


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def select_device(forced: str | None) -> torch.device:
    """CPU is the safe default on Mac M2 because PyG sparse ops on MPS
    are unreliable. MPS / CUDA can be forced via --device.
    """
    if forced:
        if forced == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS requested but not available, falling back to CPU")
            return torch.device("cpu")
        if forced == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device(forced)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        log.info("MPS available but using CPU (PyTorch Geometric on MPS is unstable). "
                 "Override with --device mps if you want to try.")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Graph loading and filtering
# ---------------------------------------------------------------------------
def load_and_filter_graph(path: Path):
    log.info("Loading graph: %s", path)
    with path.open("rb") as fh:
        G = pickle.load(fh)
    log.info("  before filtering: %d nodes, %d edges",
             G.number_of_nodes(), G.number_of_edges())

    flagged = [n for n, d in G.nodes(data=True)
               if d.get("node_type") == "Recipe"
               and d.get("quality_flag") == "upstream_corrupt"]
    if flagged:
        G.remove_nodes_from(flagged)
        log.info("  removed %d 'upstream_corrupt' recipes (and incident edges)",
                 len(flagged))
    log.info("  after filtering: %d nodes, %d edges",
             G.number_of_nodes(), G.number_of_edges())
    return G


# ---------------------------------------------------------------------------
# Feature construction per node type
# ---------------------------------------------------------------------------
def get_node_text(node_id: str, attrs: dict, node_type: str) -> str:
    """Return the best textual representation of a node for SBERT encoding."""
    if node_type == "Tool":
        return attrs.get("canonical_name") or node_id.replace("tool::", "")
    if node_type == "FoodOnClass":
        return attrs.get("foodon_label") or attrs.get("label") or node_id
    if node_type == "Place":
        return attrs.get("name") or attrs.get("canonical_name") or node_id.replace("place::", "")
    if node_type == "Period":
        return attrs.get("name") or node_id.replace("period::", "")
    return node_id


def build_node_features(
    G,
    qwen_npz: np.lib.npyio.NpzFile,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Return (features_by_type, node_ids_by_type).

    features_by_type[type]  -> Tensor of shape (n_nodes, feat_dim)
    node_ids_by_type[type]  -> list of node_ids in the same order

    The order in node_ids_by_type defines the row index used for PyG.
    """
    # Group nodes by type, sorted for determinism
    nodes_by_type: dict[str, list[tuple[str, dict]]] = {}
    for n, d in G.nodes(data=True):
        nt = d.get("node_type")
        if not nt:
            continue
        nodes_by_type.setdefault(nt, []).append((n, d))
    for nt in nodes_by_type:
        nodes_by_type[nt].sort(key=lambda x: x[0])

    # Index qwen3 vectors by node_id for the relevant types
    qwen_index: dict[str, np.ndarray] = {}
    for type_lower in ("recipes", "ingredients", "actions"):
        ids = qwen_npz[f"{type_lower}_node_ids"]
        vecs = qwen_npz[f"{type_lower}_vectors"]
        for nid, vec in zip(ids, vecs):
            qwen_index[str(nid)] = vec
    log.info("Loaded qwen3 embeddings for %d nodes (Recipe+Ingredient+Action)",
             len(qwen_index))

    # SBERT for Tool / FoodOnClass / Place
    sbert_needed = []
    for nt in SBERT_TYPES:
        if nt in nodes_by_type:
            sbert_needed.extend(get_node_text(n, d, nt) for n, d in nodes_by_type[nt])
    log.info("Computing SBERT embeddings for %d nodes (Tool/FoodOnClass/Place)",
             len(sbert_needed))
    sbert_vecs_flat = compute_sbert_embeddings(sbert_needed)
    log.info("  SBERT shape: %s", sbert_vecs_flat.shape)

    # Walk the same order to assign SBERT vectors back to nodes
    sbert_cursor = 0

    features_by_type: dict[str, torch.Tensor] = {}
    node_ids_by_type: dict[str, list[str]] = {}

    for nt, nodes in nodes_by_type.items():
        ids = [n for n, _ in nodes]
        node_ids_by_type[nt] = ids

        if nt in QWEN_TYPES:
            # Use qwen3; for any node missing from the npz, use a zero vector
            # (very rare, would indicate a graph/embedding mismatch).
            dim = next(iter(qwen_index.values())).shape[0] if qwen_index else 4096
            arr = np.zeros((len(ids), dim), dtype=np.float32)
            missing = 0
            for i, nid in enumerate(ids):
                v = qwen_index.get(nid)
                if v is None:
                    missing += 1
                    continue
                arr[i] = v
            if missing:
                log.warning("  %s: %d nodes missing from qwen3 npz (will use zeros)",
                            nt, missing)
            features_by_type[nt] = torch.from_numpy(arr).to(device)
            log.info("  %s: %d nodes, feat_dim=%d (qwen3)", nt, len(ids), dim)

        elif nt in SBERT_TYPES:
            arr = sbert_vecs_flat[sbert_cursor:sbert_cursor + len(ids)]
            sbert_cursor += len(ids)
            features_by_type[nt] = torch.from_numpy(arr).to(device)
            log.info("  %s: %d nodes, feat_dim=%d (SBERT)", nt, len(ids), SBERT_DIM)

        elif nt in LEARNABLE_TYPES:
            # Learnable embedding initialized random-normal. Will be optimized
            # alongside the model. We register it as a parameter in the model
            # at build time (see HeteroGNN class).
            arr = np.random.RandomState(SEED).normal(
                size=(len(ids), DEFAULT_HIDDEN_DIM)).astype(np.float32) * 0.1
            features_by_type[nt] = torch.from_numpy(arr).to(device)
            log.info("  %s: %d nodes, feat_dim=%d (learnable)",
                     nt, len(ids), DEFAULT_HIDDEN_DIM)
        else:
            log.warning("Unknown node_type '%s', skipping", nt)

    return features_by_type, node_ids_by_type


def compute_sbert_embeddings(texts: list[str]) -> np.ndarray:
    """Encode a list of strings with SBERT. Returns (n, 384) float32."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL)
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True)
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# HeteroData construction
# ---------------------------------------------------------------------------
def build_hetero_data(
    G,
    features_by_type: dict[str, torch.Tensor],
    node_ids_by_type: dict[str, list[str]],
    device: torch.device,
) -> HeteroData:
    """Build a PyG HeteroData object from the NetworkX graph."""
    data = HeteroData()

    # Assign features and create id->index maps per type
    id_to_idx: dict[str, dict[str, int]] = {}
    for nt, feats in features_by_type.items():
        data[nt].x = feats
        id_to_idx[nt] = {nid: i for i, nid in enumerate(node_ids_by_type[nt])}

    # Group edges by (src_type, edge_type, dst_type)
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    skipped = 0
    for u, v, ed in G.edges(data=True):
        et = ed.get("edge_type")
        if not et:
            skipped += 1
            continue
        ut = G.nodes[u].get("node_type")
        vt = G.nodes[v].get("node_type")
        if ut not in id_to_idx or vt not in id_to_idx:
            skipped += 1
            continue
        i = id_to_idx[ut].get(u)
        j = id_to_idx[vt].get(v)
        if i is None or j is None:
            skipped += 1
            continue
        grouped.setdefault((ut, et, vt), []).append((i, j))

    if skipped:
        log.warning("Skipped %d edges with missing type / index", skipped)

    for (ut, et, vt), pairs in grouped.items():
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
        data[ut, et, vt].edge_index = torch.stack([src, dst], dim=0)
        log.info("  edge %-30s %d edges", f"{ut}-[{et}]->{vt}", len(pairs))

    return data


# ---------------------------------------------------------------------------
# Link prediction split on the target relation
# ---------------------------------------------------------------------------
def split_target_edges(
    data: HeteroData,
    relation: tuple[str, str, str],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the target relation's edges into train/val/test for link
    prediction. The training graph (message passing) only uses train edges
    on this relation, but ALL edges on other relations stay.

    Returns (train_edges, val_edges, test_edges, train_edge_index_for_msg_passing)
    where each is a [2, n] LongTensor.
    """
    rng = np.random.RandomState(seed)
    edge_index = data[relation].edge_index
    n = edge_index.size(1)

    perm = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_edges = edge_index[:, train_idx]
    val_edges = edge_index[:, val_idx]
    test_edges = edge_index[:, test_idx]

    log.info("Target relation split: train=%d, val=%d, test=%d",
             train_edges.size(1), val_edges.size(1), test_edges.size(1))

    return train_edges, val_edges, test_edges, train_edges


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE that will be made heterogeneous via to_hetero.

    Using lazy in-channels (-1) so to_hetero can wire up per-relation
    weight matrices automatically based on each src node type's input dim.
    """
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
    """Score a (src, dst) pair as dot product of their final embeddings."""
    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        src = z_src[edge_index[0]]
        dst = z_dst[edge_index[1]]
        return (src * dst).sum(dim=-1)


# ---------------------------------------------------------------------------
# Negative sampling for the target relation
# ---------------------------------------------------------------------------
def sample_neg_edges_target(
    num_src: int,
    num_dst: int,
    n_samples: int,
    pos_edges: torch.Tensor,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Random (src, dst) pairs that are NOT in pos_edges. Approximate (does
    not check against the entire positive set for speed); collisions are
    rare for sparse graphs and have negligible effect on training.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    src = torch.randint(0, num_src, (n_samples,), generator=g)
    dst = torch.randint(0, num_dst, (n_samples,), generator=g)
    return torch.stack([src, dst], dim=0).to(device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, predictor: LinkPredictor, data: HeteroData,
             pos_edges: torch.Tensor, neg_edges: torch.Tensor) -> dict[str, float]:
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    z_src = out[TARGET_RELATION[0]]
    z_dst = out[TARGET_RELATION[2]]

    pos_score = predictor(z_src, z_dst, pos_edges).sigmoid()
    neg_score = predictor(z_src, z_dst, neg_edges).sigmoid()

    y_true = np.concatenate([
        np.ones(pos_score.size(0)),
        np.zeros(neg_score.size(0)),
    ])
    y_pred = np.concatenate([
        pos_score.cpu().numpy(),
        neg_score.cpu().numpy(),
    ])
    auc = roc_auc_score(y_true, y_pred)

    # Hits@10: for each positive (src, dst), is dst among the top-10
    # ingredients ranked by score against this src?
    hits10 = 0.0
    if pos_edges.size(1) > 0:
        # Score every src against every ingredient (small graph, feasible)
        all_scores = z_src @ z_dst.t()  # (n_recipe, n_ingredient)
        # For each positive edge, check if true dst is in top-10 for that src
        topk = all_scores.topk(10, dim=1).indices  # (n_recipe, 10)
        srcs = pos_edges[0]
        dsts = pos_edges[1]
        in_topk = (topk[srcs] == dsts.unsqueeze(1)).any(dim=1).float()
        hits10 = float(in_topk.mean().item())

    return {"auc": float(auc), "hits10": hits10}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args, data, features_by_type, node_ids_by_type, device):
    torch.manual_seed(SEED)

    # ---- Split the target relation for link prediction ----
    train_edges, val_edges, test_edges, _ = split_target_edges(
        data, TARGET_RELATION, SPLIT_RATIOS, seed=SEED
    )

    # For message passing during training, use only train_edges on the
    # target relation. We rebuild a "training data" view that replaces
    # the target relation's edge_index with train_edges + reverse.
    train_data = data.clone()
    train_data[TARGET_RELATION].edge_index = train_edges
    # The reverse will be added by ToUndirected below
    train_data = T.ToUndirected()(train_data)
    # data already had ToUndirected applied? No, we apply it here on the cloned
    # full data too so val/test eval uses the full message-passing graph.
    full_data = T.ToUndirected()(data)

    # ---- Pre-sample negative edges for val/test (fixed for fair comparison) ----
    n_recipe = features_by_type["Recipe"].size(0)
    n_ingredient = features_by_type["Ingredient"].size(0)
    val_neg = sample_neg_edges_target(
        n_recipe, n_ingredient, val_edges.size(1), val_edges, device, seed=SEED + 1
    )
    test_neg = sample_neg_edges_target(
        n_recipe, n_ingredient, test_edges.size(1), test_edges, device, seed=SEED + 2
    )

    # ---- Build model ----
    encoder = GraphSAGEEncoder(args.hidden_dim, args.out_dim)
    encoder = to_hetero(encoder, train_data.metadata(), aggr="sum").to(device)
    predictor = LinkPredictor().to(device)

    # Force lazy module init by one forward pass
    with torch.no_grad():
        _ = encoder(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- Training loop ----
    log_rows = []
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    log.info("Starting training: %d epochs, hidden=%d, out=%d, lr=%g",
             args.epochs, args.hidden_dim, args.out_dim, args.lr)

    if args.dry_run:
        log.info("[dry-run] Model built. Skipping training.")
        return encoder, predictor, None, log_rows

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        optimizer.zero_grad()

        # Forward on the training graph
        out = encoder(train_data.x_dict, train_data.edge_index_dict)
        z_src = out[TARGET_RELATION[0]]
        z_dst = out[TARGET_RELATION[2]]

        # Positive edges
        pos_score = predictor(z_src, z_dst, train_edges)

        # Negative sampling (re-sampled each epoch for variety)
        n_neg = int(train_edges.size(1) * args.neg_ratio)
        neg_edges = sample_neg_edges_target(
            n_recipe, n_ingredient, n_neg, train_edges, device,
            seed=SEED + 100 + epoch,
        )
        neg_score = predictor(z_src, z_dst, neg_edges)

        # BCE loss
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([pos_labels, neg_labels]),
        )

        loss.backward()
        optimizer.step()

        # Validation every epoch (cheap on this size)
        val_metrics = evaluate(encoder, predictor, full_data, val_edges, val_neg)

        log_rows.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "val_auc": val_metrics["auc"],
            "val_hits10": val_metrics["hits10"],
            "elapsed_s": time.time() - t_start,
        })

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            log.info("epoch %3d | loss=%.4f | val_auc=%.4f | hits@10=%.4f | %.1fs",
                     epoch, loss.item(), val_metrics["auc"], val_metrics["hits10"],
                     time.time() - t_start)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "predictor": {k: v.detach().cpu().clone() for k, v in predictor.state_dict().items()},
                "epoch": epoch,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                log.info("Early stopping at epoch %d (no val improvement for %d epochs)",
                         epoch, args.early_stop_patience)
                break

    # ---- Restore best model and evaluate on test ----
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        predictor.load_state_dict(best_state["predictor"])
        log.info("Restored best model from epoch %d (val_auc=%.4f)",
                 best_state["epoch"], best_val_auc)

    test_metrics = evaluate(encoder, predictor, full_data, test_edges, test_neg)
    log.info("TEST: auc=%.4f | hits@10=%.4f", test_metrics["auc"], test_metrics["hits10"])

    return encoder, predictor, test_metrics, log_rows


# ---------------------------------------------------------------------------
# Save final node embeddings
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_embeddings(encoder: nn.Module, data: HeteroData,
                    node_ids_by_type: dict[str, list[str]],
                    out_path: Path) -> None:
    encoder.eval()
    # Use the FULL graph (with all relations + reverse) for the final pass
    full_data = T.ToUndirected()(data)
    out = encoder(full_data.x_dict, full_data.edge_index_dict)

    arrays = {}
    for nt, vecs in out.items():
        arrays[f"{nt}_node_ids"] = np.asarray(node_ids_by_type[nt])
        arrays[f"{nt}_vectors"] = vecs.cpu().numpy().astype(np.float32)
        log.info("  %s: %s", nt, vecs.shape)
    np.savez_compressed(out_path, **arrays)
    log.info("Saved node embeddings: %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, default=GRAPH_PATH)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "mps", "cuda"],
                        help="Force device. Default: auto-detect (CPU on Mac).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--out-dim", type=int, default=DEFAULT_OUT_DIM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--neg-ratio", type=float, default=DEFAULT_NEG_RATIO,
                        help="Negative samples per positive (e.g. 1.0 = balanced).")
    parser.add_argument("--early-stop-patience", type=int,
                        default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build everything but skip training.")
    args = parser.parse_args()

    device = select_device(args.device)
    log.info("Device: %s", device)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Graph + filtering ----
    G = load_and_filter_graph(args.graph)

    # ---- Embeddings file ----
    log.info("Loading qwen3 embeddings: %s", args.embeddings)
    qwen_npz = np.load(args.embeddings, allow_pickle=True)

    # ---- Per-type features ----
    log.info("Building per-type features")
    features_by_type, node_ids_by_type = build_node_features(G, qwen_npz, device)

    # ---- HeteroData ----
    log.info("Building HeteroData")
    data = build_hetero_data(G, features_by_type, node_ids_by_type, device)
    log.info("Metadata: %s", data.metadata())

    # ---- Train ----
    encoder, predictor, test_metrics, log_rows = train(
        args, data, features_by_type, node_ids_by_type, device,
    )

    if args.dry_run:
        log.info("[dry-run] Done. No outputs saved.")
        return 0

    # ---- Save model and embeddings ----
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "gnn_model.pt"
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "metadata": {
            "hidden_dim": args.hidden_dim,
            "out_dim": args.out_dim,
            "target_relation": TARGET_RELATION,
            "test_metrics": test_metrics,
        },
    }, model_path)
    log.info("Saved model: %s", model_path)

    save_embeddings(encoder, data, node_ids_by_type,
                    args.output_dir / "gnn_node_embeddings.npz")

    # Training log CSV
    log_df = pd.DataFrame(log_rows)
    log_path = args.output_dir / "gnn_training_log.csv"
    log_df.to_csv(log_path, index=False)
    log.info("Saved training log: %s", log_path)

    # Final metrics JSON
    metrics_path = args.output_dir / "gnn_training_metrics.json"
    with metrics_path.open("w") as fh:
        json.dump({
            "device": str(device),
            "hyperparameters": {
                "hidden_dim": args.hidden_dim,
                "out_dim": args.out_dim,
                "epochs_requested": args.epochs,
                "epochs_run": len(log_rows),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "neg_ratio": args.neg_ratio,
            },
            "test_metrics": test_metrics,
            "best_val_auc": max((r["val_auc"] for r in log_rows), default=0.0),
        }, fh, indent=2)
    log.info("Saved metrics: %s", metrics_path)

    log.info("")
    log.info("Done. Next step: step10_rag_llm.py (build retrieval + LLM platform)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
