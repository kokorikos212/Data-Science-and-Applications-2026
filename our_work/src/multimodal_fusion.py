#!/usr/bin/env python3
"""
multimodal_fusion.py — Late-Fusion Network for Absolute Magnitude Prediction
=============================================================================
Implements a multi-view "Late Fusion" architecture that combines:
  1. A **Tabular Branch** (MLP) processing distance-proxy features from Gaia DR3.
  2. A **Spectral Branch** (MLP) processing PCA-reduced BP/RP spectral coefficients.
  3. A **Fusion Regression Head** that concatenates both embeddings and predicts
     the continuous target: Absolute Magnitude (M).

Three model variants are trained and compared:
  • Tabular-only  — baseline (Phase-1 style)
  • Spectral-only — spectral envelope → M
  • Multimodal Fusion — combined tabular + spectral

Metrics reported: RMSE, R², MAE.

Usage
-----
  python multimodal_fusion.py

Dependencies (install if missing):
  pip install numpy pandas scikit-learn torch datasets
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Local utilities
from utils import (
    prepare_multimodal_data,
    compute_metrics,
    TABULAR_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multimodal_fusion")


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """Centralised hyper-parameters and dataset knobs."""

    # Data
    dataset_path: str = "MultimodalUniverse/gaia"
    n_samples: int = 50_000
    stream_batch_size: int = 500
    tabular_features: List[str] = field(default_factory=lambda: TABULAR_FEATURES)
    pca_components: int = 5
    test_frac: float = 0.15
    val_frac: float = 0.15

    # Training
    batch_size: int = 256
    epochs: int = 80
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    lr_patience: int = 8
    lr_factor: float = 0.5
    grad_clip_norm: float = 1.0
    random_seed: int = 42

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════════════

class TabularMLP(nn.Module):
    """
    Phase-1-style MLP that predicts Absolute Magnitude from tabular
    distance-proxy features alone (no parallax).

    Architecture
    ------------
    Input(tab_dim) → Linear(64) → ReLU → Dropout → Linear(32) → ReLU
    → Dropout → Linear(16) → ReLU → Linear(1)
    """

    def __init__(self, input_dim: int, dropout: float = 0.05):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x_tab: torch.Tensor) -> torch.Tensor:
        return self.body(x_tab)


class SpectralMLP(nn.Module):
    """
    MLP that predicts Absolute Magnitude purely from PCA-reduced spectral
    coefficients — the spectral envelope acts as a stellar-class proxy.

    Architecture
    ------------
    Input(spec_dim) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(8)
    → ReLU → Linear(1)
    """

    def __init__(self, input_dim: int, dropout: float = 0.05):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, x_spec: torch.Tensor) -> torch.Tensor:
        return self.body(x_spec)


class MultimodalFusionModel(nn.Module):
    """
    Late-fusion architecture combining tabular and spectral streams.

    Tabular Branch   →  tab_embed (16-D)
    Spectral Branch  →  spec_embed (16-D)
                         │
                 Concat (32-D)
                         │
            Fusion Head (MLP)
                         │
                  M_pred (scalar)

    The individual branch embeddings are also exposed for analysis
    (e.g. t-SNE visualisation of the joint latent space).
    """

    def __init__(
        self,
        tabular_dim: int,
        spectral_dim: int,
        tab_embed_dim: int = 16,
        spec_embed_dim: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()

        # ---- Tabular encoder ----
        self.tab_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, tab_embed_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Spectral encoder ----
        self.spec_encoder = nn.Sequential(
            nn.Linear(spectral_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, spec_embed_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Fusion regression head ----
        fusion_input_dim = tab_embed_dim + spec_embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(
        self,
        x_tab: torch.Tensor,
        x_spec: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_tab : (B, tabular_dim)
        x_spec : (B, spectral_dim)
        return_embeddings : bool
            If True, also return the intermediate branch embeddings.

        Returns
        -------
        y_pred : (B, 1)
            Predicted absolute magnitude.
        OR (y_pred, tab_embed, spec_embed) when return_embeddings=True.
        """
        tab_embed = self.tab_encoder(x_tab)
        spec_embed = self.spec_encoder(x_spec)
        fused = torch.cat([tab_embed, spec_embed], dim=1)
        y_pred = self.fusion_head(fused)

        if return_embeddings:
            return y_pred, tab_embed, spec_embed
        return y_pred


# ═══════════════════════════════════════════════════════════════════════
# Training Utilities
# ═══════════════════════════════════════════════════════════════════════

def build_dataloaders(
    data: Dict[str, np.ndarray],
    batch_size: int,
    mode: str = "tabular",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders for a given input mode.

    Parameters
    ----------
    data : dict
        Output of prepare_multimodal_data().
    batch_size : int
    mode : str
        "tabular" → X_train_tab only
        "spectral" → X_train_spec only
        "fusion" → (X_train_tab, X_train_spec) jointly

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    y_train = torch.tensor(data["y_train"], dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(data["y_val"], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32).unsqueeze(1)

    if mode == "tabular":
        X_tr = torch.tensor(data["X_train_tab"], dtype=torch.float32)
        X_va = torch.tensor(data["X_val_tab"], dtype=torch.float32)
        X_te = torch.tensor(data["X_test_tab"], dtype=torch.float32)
        ds_train = TensorDataset(X_tr, y_train)
        ds_val = TensorDataset(X_va, y_val)
        ds_test = TensorDataset(X_te, y_test)

    elif mode == "spectral":
        X_tr = torch.tensor(data["X_train_spec"], dtype=torch.float32)
        X_va = torch.tensor(data["X_val_spec"], dtype=torch.float32)
        X_te = torch.tensor(data["X_test_spec"], dtype=torch.float32)
        ds_train = TensorDataset(X_tr, y_train)
        ds_val = TensorDataset(X_va, y_val)
        ds_test = TensorDataset(X_te, y_test)

    elif mode == "fusion":
        Xt_tr = torch.tensor(data["X_train_tab"], dtype=torch.float32)
        Xs_tr = torch.tensor(data["X_train_spec"], dtype=torch.float32)
        Xt_va = torch.tensor(data["X_val_tab"], dtype=torch.float32)
        Xs_va = torch.tensor(data["X_val_spec"], dtype=torch.float32)
        Xt_te = torch.tensor(data["X_test_tab"], dtype=torch.float32)
        Xs_te = torch.tensor(data["X_test_spec"], dtype=torch.float32)
        ds_train = TensorDataset(Xt_tr, Xs_tr, y_train)
        ds_val = TensorDataset(Xt_va, Xs_va, y_val)
        ds_test = TensorDataset(Xt_te, Xs_te, y_test)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'tabular', 'spectral', or 'fusion'.")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    mode: str = "tabular",
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch. Returns average loss over all batches."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        if mode == "fusion":
            x_tab, x_spec, y = (b.to(device) for b in batch)
            pred = model(x_tab, x_spec)
            n_samples = x_tab.size(0)
        else:
            x, y = (b.to(device) for b in batch)
            pred = model(x)
            n_samples = x.size(0)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * n_samples

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    mode: str = "tabular",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader.

    Returns
    -------
    avg_loss : float
    y_true : np.ndarray   (N,)
    y_pred : np.ndarray   (N,)
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []

    for batch in loader:
        if mode == "fusion":
            x_tab, x_spec, y = (b.to(device) for b in batch)
            pred = model(x_tab, x_spec)
            n_samples = x_tab.size(0)
        else:
            x, y = (b.to(device) for b in batch)
            pred = model(x)
            n_samples = x.size(0)

        loss = criterion(pred, y)
        total_loss += loss.item() * n_samples

        all_preds.append(pred.cpu().numpy())
        all_trues.append(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.vstack(all_trues).ravel()
    y_pred = np.vstack(all_preds).ravel()

    return avg_loss, y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainConfig,
    mode: str = "tabular",
    model_name: str = "Model",
) -> Dict:
    """
    Full training loop with LR scheduling, logging, and final test evaluation.

    Returns a dictionary with: model state_dict, best_val_loss, test_metrics,
    train_losses, val_losses.
    """
    device = config.device
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )

    best_val_loss = float("inf")
    best_state: Optional[Dict] = None
    train_losses: List[float] = []
    val_losses: List[float] = []
    patience_counter = 0
    early_stop_patience = config.lr_patience * 3  # harder early-stop threshold

    logger.info("── %s training on %s ──", model_name, device)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, mode, config.grad_clip_norm,
        )
        val_loss, _, _ = evaluate(model, val_loader, criterion, device, mode)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Early-stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  Epoch %3d | Train MSE: %.5f | Val MSE: %.5f | LR: %.5f",
                epoch, train_loss, val_loss, optimizer.param_groups[0]["lr"],
            )

        if patience_counter >= early_stop_patience:
            logger.info("  Early stopping at epoch %d (best val MSE: %.5f)", epoch, best_val_loss)
            break

    # Restore best weights & evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device, mode)
    metrics = compute_metrics(y_true, y_pred)

    logger.info(
        "  ▶ Test  — MSE: %.5f | RMSE: %.4f | R²: %.4f | MAE: %.4f",
        test_loss, metrics["rmse"], metrics["r2"], metrics["mae"],
    )

    return {
        "model_name": model_name,
        "best_val_loss": best_val_loss,
        "test_metrics": metrics,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_state": best_state,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════════

def main():
    config = TrainConfig()

    # ------------------------------------------------------------------
    # 0. Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    logger.info("═" * 62)
    logger.info("  Multimodal Fusion — Absolute Magnitude Prediction")
    logger.info("  Device: %s  |  PCA components: %d  |  Epochs: %d",
                config.device, config.pca_components, config.epochs)
    logger.info("═" * 62)

    # ------------------------------------------------------------------
    # 1. Data preparation (shared across all three models)
    # ------------------------------------------------------------------
    logger.info("Step 1/4 — Loading & preparing multimodal data …")
    data = prepare_multimodal_data(
        path=config.dataset_path,
        n_wanted=config.n_samples,
        batch_size=config.stream_batch_size,
        tabular_features=config.tabular_features,
        pca_components=config.pca_components,
        test_size=config.test_frac,
        val_size=config.val_frac,
        random_state=config.random_seed,
        device=config.device,
    )

    tabular_dim = data["X_train_tab"].shape[1]
    spectral_dim = data["X_train_spec"].shape[1]
    logger.info(
        "Data ready — %d train / %d val / %d test  |  tab_dim=%d  spec_dim=%d",
        data["n_train"], data["n_val"], data["n_test"], tabular_dim, spectral_dim,
    )
    logger.info("Tabular features: %s", data["tabular_feature_names"])
    logger.info(
        "PCA explained variance: %s",
        np.round(data["pca_extractor"].explained_variance_ratio_, 4),
    )

    # ------------------------------------------------------------------
    # 2. Build models
    # ------------------------------------------------------------------
    logger.info("Step 2/4 — Building model variants …")

    model_tabular = TabularMLP(input_dim=tabular_dim)
    model_spectral = SpectralMLP(input_dim=spectral_dim)
    model_fusion = MultimodalFusionModel(
        tabular_dim=tabular_dim,
        spectral_dim=spectral_dim,
    )

    param_counts = {
        "Tabular-only": sum(p.numel() for p in model_tabular.parameters()),
        "Spectral-only": sum(p.numel() for p in model_spectral.parameters()),
        "Fusion": sum(p.numel() for p in model_fusion.parameters()),
    }
    for name, count in param_counts.items():
        logger.info("  %-16s : %d parameters", name, count)

    # ------------------------------------------------------------------
    # 3. Train all three variants
    # ------------------------------------------------------------------
    logger.info("Step 3/4 — Training three model variants …")

    # 3a. Tabular-only
    logger.info("\n▶ Variant 1/3: TABULAR-ONLY BASELINE")
    loaders_tab = build_dataloaders(data, config.batch_size, mode="tabular")
    results_tab = train_model(
        model_tabular, *loaders_tab, config, mode="tabular", model_name="Tabular-only",
    )

    # 3b. Spectral-only
    logger.info("\n▶ Variant 2/3: SPECTRAL-ONLY")
    loaders_spec = build_dataloaders(data, config.batch_size, mode="spectral")
    results_spec = train_model(
        model_spectral, *loaders_spec, config, mode="spectral", model_name="Spectral-only",
    )

    # 3c. Multimodal Fusion
    logger.info("\n▶ Variant 3/3: MULTIMODAL FUSION")
    loaders_fusion = build_dataloaders(data, config.batch_size, mode="fusion")
    results_fusion = train_model(
        model_fusion, *loaders_fusion, config, mode="fusion", model_name="Multimodal Fusion",
    )

    # ------------------------------------------------------------------
    # 4. Comparative evaluation
    # ------------------------------------------------------------------
    logger.info("\n" + "═" * 62)
    logger.info("  COMPARATIVE RESULTS")
    logger.info("═" * 62)

    all_results = [results_tab, results_spec, results_fusion]

    # Header
    print(f"\n{'Model':<22} {'RMSE ↓':>10} {'R²  ↑':>10} {'MAE ↓':>10}")
    print("-" * 54)

    best_rmse = float("inf")
    best_model = ""
    for r in all_results:
        m = r["test_metrics"]
        print(
            f"{r['model_name']:<22} {m['rmse']:>10.4f} {m['r2']:>10.4f} {m['mae']:>10.4f}"
        )
        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_model = r["model_name"]

    print("-" * 54)

    # Improvement over tabular baseline
    base_rmse = results_tab["test_metrics"]["rmse"]
    base_r2 = results_tab["test_metrics"]["r2"]

    for r in all_results:
        if r["model_name"] == "Tabular-only":
            continue
        delta_rmse = r["test_metrics"]["rmse"] - base_rmse
        delta_r2 = r["test_metrics"]["r2"] - base_r2
        print(
            f"\n{r['model_name']} vs Tabular baseline: "
            f"ΔRMSE = {delta_rmse:+.4f}  |  ΔR² = {delta_r2:+.4f}"
        )

    print(f"\n🏆 Best model: {best_model} (RMSE = {best_rmse:.4f})")
    print("═" * 62)

    # ------------------------------------------------------------------
    # 5. Optional: export best model weights
    # ------------------------------------------------------------------
    # torch.save(results_fusion["best_state"], "best_fusion_model.pth")
    # logger.info("Best fusion model weights saved to best_fusion_model.pth")

    return all_results


if __name__ == "__main__":
    main()
