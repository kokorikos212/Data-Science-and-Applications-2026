"""
utils.py — Spectral Feature Extraction & Evaluation Utilities
==============================================================
Provides dimensionality reduction for BP/RP spectral coefficients
(PCA + optional 1D Convolutional Autoencoder), regression metrics,
and a multimodal data-preparation pipeline for the Gaia stellar
parameter prediction task.

Dependencies are declared at module level; install any missing
packages with: pip install numpy pandas scikit-learn torch datasets
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. PCA-Based Spectral Feature Extractor
# ---------------------------------------------------------------------------

class SpectralPCAExtractor:
    """
    Reduces high-dimensional spectral coefficients (e.g. 110-D Chebyshev
    representations from Gaia BP/RP) into a compact set of principal
    components capturing the dominant variance of the spectral envelope.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain (default 5).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_components: int = 5, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._pca: Optional[PCA] = None
        self._fitted = False

    @property
    def pca(self) -> PCA:
        if self._pca is None:
            raise RuntimeError("PCA has not been fitted yet. Call fit() or fit_transform() first.")
        return self._pca

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Per-component explained-variance ratio (available after fitting)."""
        return self.pca.explained_variance_ratio_

    @property
    def total_explained_variance_(self) -> float:
        """Cumulative explained variance across all retained components."""
        return float(self.pca.explained_variance_ratio_.sum())

    def fit(self, spectral_data: np.ndarray) -> SpectralPCAExtractor:
        """
        Fit PCA on the spectral coefficient matrix.

        Parameters
        ----------
        spectral_data : np.ndarray of shape (n_samples, n_coefficients)
        """
        if spectral_data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {spectral_data.shape}")

        self._pca = PCA(
            n_components=min(self.n_components, spectral_data.shape[1]),
            random_state=self.random_state,
        )
        self._pca.fit(spectral_data)
        self._fitted = True

        logger.info(
            "PCA fitted: %d components explain %.2f%% of spectral variance.",
            self.n_components,
            self.total_explained_variance_ * 100,
        )
        return self

    def transform(self, spectral_data: np.ndarray) -> np.ndarray:
        """
        Project spectral coefficients into the PCA latent space.

        Parameters
        ----------
        spectral_data : np.ndarray of shape (n_samples, n_coefficients)

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
        """
        if not self._fitted:
            raise RuntimeError("PCA not fitted. Call fit() first.")
        return self._pca.transform(spectral_data)

    def fit_transform(self, spectral_data: np.ndarray) -> np.ndarray:
        """Fit PCA and transform in a single call."""
        self.fit(spectral_data)
        return self.transform(spectral_data)


# ---------------------------------------------------------------------------
# 2. 1D Convolutional Autoencoder (optional deep alternative to PCA)
# ---------------------------------------------------------------------------

class Conv1DSpectralAutoencoder(nn.Module):
    """
    1-D convolutional autoencoder for learning a compressed latent
    representation of BP/RP spectral coefficient sequences.

    Architecture
    ------------
    Encoder:  Conv1d(k=5,s=2) → ReLU → Conv1d(k=5,s=2) → Flatten → Linear(latent)
    Decoder:  Linear → Unflatten → ConvTranspose1d → ConvTranspose1d → Sigmoid

    Parameters
    ----------
    input_length : int
        Number of spectral coefficients (e.g. 110).
    latent_dim : int
        Size of the compressed latent vector (default 8).
    """

    def __init__(self, input_length: int = 110, latent_dim: int = 8):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

        # ---- Encoder ----
        self.enc_conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)

        # Compute size of feature map after two strided convolutions
        def conv_out_len(L_in: int, kernel: int, stride: int, padding: int) -> int:
            return (L_in + 2 * padding - kernel) // stride + 1

        L1 = conv_out_len(input_length, 5, 2, 2)
        L2 = conv_out_len(L1, 5, 2, 2)
        self._flattened_size = 32 * L2

        self.enc_fc = nn.Linear(self._flattened_size, latent_dim)

        # ---- Decoder ----
        self.dec_fc = nn.Linear(latent_dim, self._flattened_size)
        self._dec_L2 = L2
        self.dec_deconv1 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.dec_deconv2 = nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_length) → latent: (B, latent_dim)"""
        x = x.unsqueeze(1)                     # (B, 1, L)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)              # flatten
        return self.enc_fc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → reconstruction: (B, input_length)"""
        x = F.relu(self.dec_fc(z))
        x = x.view(-1, 32, self._dec_L2)
        x = F.relu(self.dec_deconv1(x))
        x = self.dec_deconv2(x)
        return x.squeeze(1)                    # (B, L)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


# ---------------------------------------------------------------------------
# 3. Regression Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values (same shape as y_true).

    Returns
    -------
    dict with keys 'rmse', 'r2', 'mae'.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(residuals)))

    return {"rmse": rmse, "r2": r2, "mae": mae}


# ---------------------------------------------------------------------------
# 4. Multimodal Data Preparation Pipeline
# ---------------------------------------------------------------------------

# Feature sets used across all experiments — chosen for astrophysical relevance
# and availability in the Gaia DR3 MultimodalUniverse stream.
TABULAR_FEATURES: List[str] = [
    "phot_g_mean_mag",      # Apparent G-band magnitude (m)
    "bp_rp",                # Color index (BP − RP) — temperature proxy
    "pseudocolour",         # Astrometric chromaticity
    "teff_gspphot",         # Effective temperature [K] from GSP-Phot
    "rv_template_fe_h",     # Metallicity [Fe/H] from RV template
    "rv_template_logg",     # Surface gravity log₁₀(g) from RV template
]

SPECTRAL_KEY: str = "coeff"        # Column storing Chebyshev coefficient arrays
TARGET_KEY: str = "abs_mag"        # Absolute magnitude M = m + 5 + 5·log₁₀(p/1000)


def prepare_multimodal_data(
    path: str = "MultimodalUniverse/gaia",
    n_wanted: int = 50_000,
    batch_size: int = 500,
    tabular_features: Optional[List[str]] = None,
    pca_components: int = 5,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    device: str = "cpu",
) -> Dict:
    """
    End-to-end pipeline: stream Gaia data, clean, impute, apply PCA to
    spectral coefficients, split, and scale.

    Returns a dictionary with NumPy arrays and metadata ready for PyTorch
    DataLoader construction.

    Parameters
    ----------
    path : str
        HuggingFace dataset path.
    n_wanted : int
        Target number of raw samples to pull from the stream.
    batch_size : int
        Records per streaming iteration.
    tabular_features : list of str, optional
        Columns to use as tabular input (defaults to TABULAR_FEATURES).
    pca_components : int
        Number of PCA components to retain from the spectral coefficients.
    test_size, val_size : float
        Fraction of data for test and validation splits (of the full set).
    random_state : int
        Random seed for reproducibility.
    device : str
        Torch device string (unused here but passed through for downstream).

    Returns
    -------
    dict with keys:
        - X_train_tab, X_val_tab, X_test_tab    : scaled tabular arrays
        - X_train_spec, X_val_spec, X_test_spec  : PCA-reduced spectral arrays
        - y_train, y_val, y_test                 : target (abs_mag)
        - tabular_scaler, pca_extractor           : fitted transformers
        - tabular_feature_names, pca_components
        - n_train, n_val, n_test
    """
    if tabular_features is None:
        tabular_features = TABULAR_FEATURES

    logger.info("Streaming data from %s (target ≈ %d samples) …", path, n_wanted)

    # ── 1. Stream & flatten ───────────────────────────────────────────
    ds = load_dataset(path, split="train", streaming=True)
    ds = ds.with_format("pandas")
    ds_iter = iter(ds)

    all_rows: List[pd.DataFrame] = []
    total = 0
    while total < n_wanted:
        batch: List[pd.DataFrame] = []
        for _ in range(batch_size):
            try:
                batch.append(next(ds_iter))
            except StopIteration:
                break
        if not batch:
            break

        clean = [
            ex.to_dict("records")[0] if isinstance(ex, pd.DataFrame) else ex
            for ex in batch
        ]
        df_batch = pd.json_normalize(clean)
        df_batch.columns = [c.split(".")[-1] for c in df_batch.columns]
        all_rows.append(df_batch)
        total += len(df_batch)

    df_raw = pd.concat(all_rows, ignore_index=True)
    logger.info("Raw stream: %d rows, %d columns.", len(df_raw), len(df_raw.columns))

    # ── 2. Compute target (requires valid parallax) ───────────────────
    if TARGET_KEY not in df_raw.columns:
        mask_plx = df_raw["parallax"].notna() & (df_raw["parallax"] > 0)
        df_raw[TARGET_KEY] = np.nan
        df_raw.loc[mask_plx, TARGET_KEY] = (
            df_raw.loc[mask_plx, "phot_g_mean_mag"]
            + 5.0
            + 5.0 * np.log10(df_raw.loc[mask_plx, "parallax"] / 1000.0)
        )

    # ── 3. Safety filters ─────────────────────────────────────────────
    # Keep only rows where target and spectral data are non-null
    required_present = [c for c in [TARGET_KEY, SPECTRAL_KEY] if c in df_raw.columns]
    df_clean = df_raw.dropna(subset=required_present).copy()

    # Remove infinite / implausible targets
    df_clean = df_clean[np.isfinite(df_clean[TARGET_KEY])]

    # ── 4. Impute missing tabular features ────────────────────────────
    available_tab = [c for c in tabular_features if c in df_clean.columns]
    if len(available_tab) < len(tabular_features):
        missing = set(tabular_features) - set(available_tab)
        logger.warning("Features not found in dataset: %s", missing)

    imputer = SimpleImputer(strategy="median")
    df_clean[available_tab] = imputer.fit_transform(df_clean[available_tab])

    # ── 5. Extract aligned arrays ─────────────────────────────────────
    X_tabular = df_clean[available_tab].values.astype(np.float32)
    X_spectral = np.stack(df_clean[SPECTRAL_KEY].values).astype(np.float32)
    y_target = df_clean[TARGET_KEY].values.astype(np.float32)

    logger.info(
        "Aligned data: tabular %s, spectral %s, target %s",
        X_tabular.shape, X_spectral.shape, y_target.shape,
    )

    # ── 6. Train / Val / Test split ───────────────────────────────────
    n = len(y_target)
    indices = np.arange(n)
    idx_train_val, idx_test = train_test_split(
        indices, test_size=test_size, random_state=random_state,
    )
    # val_size is relative to the FULL dataset → rescale for the train_val subset
    val_frac_of_train_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=val_frac_of_train_val, random_state=random_state,
    )

    # ── 7. Fit PCA on *training* spectral data ────────────────────────
    pca_extractor = SpectralPCAExtractor(
        n_components=pca_components, random_state=random_state,
    )
    X_train_spec_raw = X_spectral[idx_train]
    pca_extractor.fit(X_train_spec_raw)

    X_train_spec = pca_extractor.transform(X_train_spec_raw)
    X_val_spec = pca_extractor.transform(X_spectral[idx_val])
    X_test_spec = pca_extractor.transform(X_spectral[idx_test])

    # ── 8. Fit StandardScaler on *training* tabular data ──────────────
    tabular_scaler = StandardScaler()
    X_train_tab = tabular_scaler.fit_transform(X_tabular[idx_train])
    X_val_tab = tabular_scaler.transform(X_tabular[idx_val])
    X_test_tab = tabular_scaler.transform(X_tabular[idx_test])

    y_train = y_target[idx_train]
    y_val = y_target[idx_val]
    y_test = y_target[idx_test]

    logger.info(
        "Splits — train: %d  |  val: %d  |  test: %d",
        len(y_train), len(y_val), len(y_test),
    )
    logger.info(
        "PCA %d components explain %.2f%% variance.",
        pca_components, pca_extractor.total_explained_variance_ * 100,
    )

    return {
        "X_train_tab": X_train_tab,
        "X_val_tab": X_val_tab,
        "X_test_tab": X_test_tab,
        "X_train_spec": X_train_spec,
        "X_val_spec": X_val_spec,
        "X_test_spec": X_test_spec,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "tabular_scaler": tabular_scaler,
        "pca_extractor": pca_extractor,
        "tabular_feature_names": available_tab,
        "pca_components": pca_components,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }
