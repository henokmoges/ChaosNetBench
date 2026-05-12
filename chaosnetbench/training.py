"""
ChaosNetBench Training Pipeline

Complete training pipeline for coupled standard map forecasting:
  - PyTorch Dataset with sliding windows
  - Training loop with early stopping
  - Chaos-aware evaluation (VPT, adjacency F1, CME)
  - Model checkpointing and result logging

Designed to work with all benchmark models.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field, asdict

from chaosnetbench.dataset import (
    load_stgnn_data,
    load_benchmark_data,
    build_variable_adjacency,
    build_variable_adjacency_sincos,
)
from chaosnetbench.models import create_model
from chaosnetbench.metrics import (
    evaluate_forecast,
    mse,
    mae,
    smape,
    valid_prediction_time,
    vpt_from_nrmse_curve,
    autoregressive_rollout_np,
    adjacency_f1,
    adjacency_auc,
    adjacency_structure_analysis,
)


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Training configuration for ChaosNetBench-CML experiments."""

    # Data
    dataset_path: str = "data/chaosnetbench_cml.h5"
    K: float = 2.0
    epsilon: float = 0.05
    N: int = 4
    seq_len: int = 48
    pred_len: int = 12
    window_stride: int = 1  # sliding window stride (pred_len recommended)
    use_sincos: bool = False  # θ → (sin θ, cos θ) encoding
    split_mode: str = "ic_split"  # "ic_split" (IC-based, recommended) or "temporal"

    # Model
    model_name: str = "graph_wavenet"

    # Training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    milestones: list = field(default_factory=lambda: [20, 35, 45])
    gamma: float = 0.5
    patience: int = 10
    lambda_sparsity: float = 0.0
    lambda_residual: float = 0.0

    # Model-specific hyperparameters (override defaults in create_model)
    model_kwargs: dict = field(default_factory=dict)

    # Exposed training controls
    grad_clip_norm: float = 5.0

    # Evaluation
    eval_horizons: list = field(default_factory=lambda: [3, 6, 12])
    ar_eval_horizons: list = field(default_factory=lambda: [12, 24, 48, 96])
    vpt_threshold: float = 1.0
    adj_threshold: float = 0.1

    # Infrastructure
    seed: int = 42
    device: str = "auto"
    output_dir: str = "results"
    num_workers: int = 4
    pin_memory: bool = True
    use_compile: str = "auto"  # "auto" | "true" | "false"; auto enables for agcrn


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────


class CSMDataset(Dataset):
    """Sliding window dataset for coupled standard map time series.

    Extracts (input, target) pairs using sliding windows over
    the normalized time series from the HDF5 dataset.
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int, stride: int = 1):
        """
        Args:
            data: [T, N] normalized time series
            seq_len: input window length
            pred_len: prediction horizon
            stride: step between successive windows (1 = every position)
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = max(1, stride)
        max_start = len(data) - seq_len - pred_len
        if max_start < 0:
            raise ValueError(
                f"Not enough data: T={len(data)}, need at least "
                f"seq_len+pred_len={seq_len + pred_len}"
            )
        self.n_windows = max_start // self.stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        s_begin = idx * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        x = self.data[s_begin:s_end]  # [seq_len, N]
        y = self.data[r_begin:r_end]  # [pred_len, N]
        return x, y


# ─────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────


class ChaosNetBenchTrainer:
    """Training pipeline with chaos-aware evaluation.

    Handles:
      - Data loading and windowing
      - Model creation and training loop
      - Early stopping on validation loss
      - Chaos-specific evaluation (VPT, adjacency discovery, CME)
      - Result logging and checkpointing
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self._setup_seed()
        self._setup_device()
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_output()

    def _setup_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _setup_device(self):
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)
        print(f"Using device: {self.device}")

    def _setup_data(self):
        """Load HDF5 data and create DataLoaders."""
        split_label = self.config.split_mode
        print(
            f"\nLoading data: K={self.config.K}, ε={self.config.epsilon}, N={self.config.N}"
            f"{' [sin/cos]' if self.config.use_sincos else ''}"
            f" split={split_label}"
        )

        if self.config.split_mode == "ic_split":
            data = load_benchmark_data(
                self.config.dataset_path,
                K=self.config.K,
                epsilon=self.config.epsilon,
                N=self.config.N,
                seq_len=self.config.seq_len,
                pred_len=self.config.pred_len,
                use_sincos=self.config.use_sincos,
            )
        else:
            data = load_stgnn_data(
                self.config.dataset_path,
                K=self.config.K,
                epsilon=self.config.epsilon,
                N=self.config.N,
                seq_len=self.config.seq_len,
                pred_len=self.config.pred_len,
                use_sincos=self.config.use_sincos,
            )

        self.n_nodes = data["n_nodes"]
        self.scaler_mean = data["scaler_mean"]
        self.scaler_std = data["scaler_std"]
        self.A_true = data["adjacency_2N"]

        # Per-IC test data for autoregressive evaluation (IC-based split)
        self.test_ics_normed = data.get("test_ics_normed", [])
        self.test_ic_diagnostics = data.get("test_ic_diagnostics", {})
        self.config_lambda_max = data.get("config_lambda_max", float("nan"))
        self.test_ic_indices = data.get("test_ics", [])

        # Create datasets
        _stride = self.config.window_stride
        train_ds = CSMDataset(data["train"], self.config.seq_len, self.config.pred_len, stride=_stride)
        val_ds = CSMDataset(data["val"], self.config.seq_len, self.config.pred_len, stride=1)  # full eval
        test_ds = CSMDataset(data["test"], self.config.seq_len, self.config.pred_len, stride=1)  # full eval

        print(f"  n_nodes: {self.n_nodes}")
        print(f"  Train windows: {len(train_ds)}")
        print(f"  Val windows:   {len(val_ds)}")
        print(f"  Test windows:  {len(test_ds)}")

        _use_pin = self.config.pin_memory and self.device.type == "cuda"
        _nw = self.config.num_workers
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=_nw,
            pin_memory=_use_pin,
            persistent_workers=(_nw > 0),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=_use_pin,
            persistent_workers=(_nw > 0),
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=_nw,
            pin_memory=_use_pin,
            persistent_workers=(_nw > 0),
        )

    def _setup_model(self):
        """Create and initialize model."""
        # Models that require the ground-truth adjacency matrix
        _needs_adj = {"oracle_gcn"}
        A_true_tensor = None
        if self.config.model_name in _needs_adj:
            A_true_tensor = torch.FloatTensor(self.A_true)

        # Build kwargs from model_kwargs (HP protocol)
        model_kw = dict(self.config.model_kwargs)

        self.model = create_model(
            model_name=self.config.model_name,
            n_nodes=self.n_nodes,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            A_true=A_true_tensor,
            **model_kw,
        )
        self.model = self.model.to(self.device)

        # Enable TF32 on Ampere+ GPUs for faster matmul
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # torch.compile for models with Python-loop bottlenecks (e.g. AGCRN)
        _should_compile = (
            self.config.use_compile == "true"
            or (self.config.use_compile == "auto"
                and self.config.model_name == "agcrn"
                and hasattr(torch, "compile"))
        )
        if _should_compile:
            print("  Applying torch.compile (mode='reduce-overhead')...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        param_info = self.model.count_parameters()
        print(f"\nModel: {self.config.model_name}")
        for k, v in param_info.items():
            print(f"  {k}: {v}")

    def _setup_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config.milestones,
            gamma=self.config.gamma,
        )

    def _setup_output(self):
        """Create output directory for results."""
        tag = (
            f"{self.config.model_name}_K{self.config.K}_"
            f"eps{self.config.epsilon}_N{self.config.N}_"
            f"L{self.config.seq_len}_P{self.config.pred_len}"
        )
        self.output_dir = os.path.join(self.config.output_dir, tag)
        os.makedirs(self.output_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    # ─────────────────── Training Loop ───────────────────

    def _train_epoch(self) -> Tuple[float, float]:
        """Train one epoch. Returns (avg_loss, avg_grad_norm)."""
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            # Models with a dedicated compute_loss (e.g. diffusion models)
            # bypass the expensive forward() call during training.
            if hasattr(self.model, "compute_loss"):
                loss = self.model.compute_loss(x, y)
                y_hat = None
            else:
                y_hat = self.model(x)
                loss = F.mse_loss(y_hat, y)

            # NRI KL divergence loss
            if hasattr(self.model, "get_kl_loss"):
                kl = self.model.get_kl_loss()
                if kl is not None:
                    loss = loss + kl

            # Sparsity regularization on adjacency
            if (
                self.config.lambda_sparsity > 0
                and hasattr(self.model, "adj")
                and callable(self.model.adj)
            ):
                A = self.model.adj()
                sparsity = A.sum() / A.numel()
                loss = loss + self.config.lambda_sparsity * sparsity

            # Residual magnitude regularization
            if y_hat is not None and self.config.lambda_residual > 0 and hasattr(self.model, "backbone"):
                y_lin = self.model.backbone(self.model.pre_norm(x))
                res_mag = (y_hat - y_lin).pow(2).mean()
                loss = loss + self.config.lambda_residual * res_mag

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.grad_clip_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm.item() if torch.isfinite(grad_norm) else 0.0
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_grad_norm = total_grad_norm / max(n_batches, 1)
        return avg_loss, avg_grad_norm

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """Validate. Returns (val_mse, val_mae).

        For diffusion models (those with compute_loss), uses the training
        loss (noise-prediction MSE) as a proxy for validation quality.
        Full generative inference (forward()) is too slow for per-epoch
        validation and is only used at test time.
        """
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_count = 0

        use_loss_val = hasattr(self.model, "compute_loss")

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            if use_loss_val:
                # Diffusion models: validate via noise-prediction loss
                loss_val = self.model.compute_loss(x, y)
                batch_size = y.shape[0]
                total_mse += loss_val.item() * batch_size
                total_mae += loss_val.item() * batch_size  # proxy
                total_count += batch_size
            else:
                y_hat = self.model(x)
                total_mse += F.mse_loss(y_hat, y, reduction="sum").item()
                total_mae += F.l1_loss(y_hat, y, reduction="sum").item()
                total_count += y.numel()

        return total_mse / total_count, total_mae / total_count

    def train(self) -> Dict:
        """Full training loop with early stopping.

        Returns:
            Dict with training history and best validation metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Training {self.config.model_name}")
        print(f"{'=' * 60}")

        history = {
            "train_loss": [],
            "val_mse": [],
            "val_mae": [],
            "grad_norm": [],
        }

        best_val_mse = float("inf")
        best_epoch = 0
        patience_counter = 0

        t0 = time.time()

        for epoch in range(self.config.epochs):
            # Train
            train_loss, grad_norm = self._train_epoch()

            # Validate
            val_mse, val_mae = self._validate()

            # Step scheduler
            self.scheduler.step()

            # Record
            history["train_loss"].append(train_loss)
            history["val_mse"].append(val_mse)
            history["val_mae"].append(val_mae)
            history["grad_norm"].append(grad_norm)

            # Check improvement
            improved = ""
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                patience_counter = 0
                improved = " *"
                # Save best model
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, "best_model.pt"),
                )
            else:
                patience_counter += 1

            # Print progress (every 5 epochs or on improvement)
            if epoch % 5 == 0 or improved:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:3d}: "
                    f"train={train_loss:.6f}  "
                    f"val_mse={val_mse:.6f}  "
                    f"val_mae={val_mae:.6f}  "
                    f"lr={lr:.2e}  ||g||={grad_norm:.2f}{improved}"
                )

            # Early stopping
            if patience_counter >= self.config.patience:
                print(
                    f"\n  Early stopping at epoch {epoch} "
                    f"(best: epoch {best_epoch}, val_mse={best_val_mse:.6f})"
                )
                break

        train_time = time.time() - t0
        print(f"\n  Training complete in {train_time:.1f}s")
        print(f"  Best epoch: {best_epoch}, val_mse: {best_val_mse:.6f}")

        # Load best model for evaluation
        best_path = os.path.join(self.output_dir, "best_model.pt")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, weights_only=True))

        # Convergence diagnostics
        converged = True
        convergence_flags = []
        if best_epoch < 3 and self.config.epochs > 10:
            convergence_flags.append("best_epoch_too_early")
            converged = False
        if patience_counter == 0 and epoch == self.config.epochs - 1:
            convergence_flags.append("no_early_stop_hit_max_epochs")
        if len(history["train_loss"]) >= 2:
            if history["train_loss"][-1] > history["train_loss"][0] * 0.9:
                convergence_flags.append("minimal_training_progress")
                converged = False
        if any(g > 100 for g in history["grad_norm"]):
            convergence_flags.append("large_gradient_norms")

        if not converged:
            print(f"  WARNING: convergence issue(s): {', '.join(convergence_flags)}")

        # Save training curves for post-hoc analysis
        curves_path = os.path.join(self.output_dir, "training_curves.json")
        with open(curves_path, "w") as f:
            json.dump({
                "train_loss": history["train_loss"],
                "val_mse": history["val_mse"],
                "val_mae": history["val_mae"],
                "grad_norm": history["grad_norm"],
                "converged": converged,
                "convergence_flags": convergence_flags,
            }, f, indent=2)

        history["best_epoch"] = best_epoch
        history["best_val_mse"] = best_val_mse
        history["train_time_s"] = train_time
        history["converged"] = converged
        history["convergence_flags"] = convergence_flags

        return history

    # ─────────────────── Evaluation ───────────────────

    @torch.no_grad()
    def _autoregressive_predict(
        self, x_init: np.ndarray, n_future: int, step_size: int = None
    ) -> np.ndarray:
        """Autoregressive rollout: chain model predictions.

        Args:
            x_init: [seq_len, N] initial input window (normalized)
            n_future: number of future steps to predict
            step_size: steps per model call (default: pred_len)

        Returns:
            y_pred: [n_future, N] autoregressive predictions
        """
        if step_size is None:
            step_size = self.config.pred_len

        self.model.eval()
        seq_len = self.config.seq_len
        n_nodes = x_init.shape[-1]

        y_pred = np.zeros((n_future, n_nodes), dtype=np.float32)
        window = torch.FloatTensor(x_init).unsqueeze(0).to(self.device)  # [1, seq_len, N]

        n_done = 0
        while n_done < n_future:
            pred = self.model(window)  # [1, pred_len, N]
            pred_np = pred.cpu().numpy()[0]  # [pred_len, N]

            end = min(n_done + step_size, n_future)
            n_store = end - n_done
            y_pred[n_done:end] = pred_np[:n_store]

            # Shift window
            pred_steps = pred[:, :n_store, :]  # [1, n_store, N]
            if n_store < seq_len:
                window = torch.cat([window[:, n_store:, :], pred_steps], dim=1)
            else:
                window = pred_steps[:, -seq_len:, :]

            n_done = end

        return y_pred

    @torch.no_grad()
    def _evaluate_autoregressive(self) -> Dict:
        """Autoregressive evaluation over per-IC test trajectories.

        For each test IC:
          1. Use first seq_len steps as input seed
          2. Roll out autoregressively for remaining steps
          3. Compute NRMSE(t) curve → extract VPT
          4. Normalize VPT by λ_max if available
          5. Stratify results by SALI-based orbit classification

        Returns:
            Dict with per-IC VPT, aggregated stats, Lyapunov-normalized VPT,
            and stratified results (regular vs chaotic)
        """
        self.model.eval()
        seq_len = self.config.seq_len

        # Build a torch model fn for rollout
        device = self.device

        def model_fn(x_np):
            """[1, seq_len, N] numpy -> [1, pred_len, N] numpy"""
            x_t = torch.FloatTensor(x_np).to(device)
            return self.model(x_t).cpu().numpy()

        per_ic_results = []
        vpt_steps_all = []
        vpt_lyapunov_all = []
        # Collect per-step curves across ICs for AR horizon metrics
        ar_horizons = self.config.ar_eval_horizons
        ar_mse_at_h = {h: [] for h in ar_horizons}
        ar_smape_at_h = {h: [] for h in ar_horizons}
        ar_nrmse_at_h = {h: [] for h in ar_horizons}

        for ic_i, ic_data in enumerate(self.test_ics_normed):
            T = ic_data.shape[0]
            if T <= seq_len + 10:
                continue  # Not enough data for meaningful rollout

            x_init = ic_data[:seq_len]  # [seq_len, N]
            y_true = ic_data[seq_len:]  # [T - seq_len, N]

            rollout = autoregressive_rollout_np(
                model_fn=model_fn,
                x_init=x_init,
                y_true_full=y_true,
                step_size=self.config.pred_len,
            )

            # Extract metrics at AR evaluation horizons
            for h in ar_horizons:
                if h <= len(rollout["mse_t"]):
                    ar_mse_at_h[h].append(float(rollout["mse_t"][h - 1]))
                    ar_smape_at_h[h].append(float(rollout["smape_t"][h - 1]))
                    ar_nrmse_at_h[h].append(float(rollout["nrmse_t"][h - 1]))

            # Extract VPT from NRMSE curve
            ic_idx = self.test_ic_indices[ic_i] if ic_i < len(self.test_ic_indices) else ic_i
            lambda_max = None
            # Try per-IC λ_max first, fall back to config-level mean
            if ic_idx in self.test_ic_diagnostics:
                lm = self.test_ic_diagnostics[ic_idx].get("lambda_max", np.nan)
                if not np.isnan(lm) and lm > 0:
                    lambda_max = lm
            if lambda_max is None and not np.isnan(self.config_lambda_max) and self.config_lambda_max > 0:
                lambda_max = self.config_lambda_max

            vpt_info = vpt_from_nrmse_curve(
                rollout["nrmse_t"],
                threshold=self.config.vpt_threshold,
                lambda_max=lambda_max,
            )

            ic_result = {
                "ic_idx": int(ic_idx),
                "vpt_steps": vpt_info["vpt_steps"],
                "mse_final": float(rollout["mse_t"][-1]),
                "smape_final": float(rollout["smape_t"][-1]),
                "n_rollout_steps": int(len(y_true)),
            }
            if "vpt_lyapunov_times" in vpt_info:
                ic_result["vpt_lyapunov_times"] = vpt_info["vpt_lyapunov_times"]
                vpt_lyapunov_all.append(vpt_info["vpt_lyapunov_times"])
            if ic_idx in self.test_ic_diagnostics:
                ic_result["is_chaotic"] = self.test_ic_diagnostics[ic_idx]["is_chaotic"]
                ic_result["chaos_regime"] = self.test_ic_diagnostics[ic_idx]["chaos_regime"]
                ic_result["sali_final"] = self.test_ic_diagnostics[ic_idx]["sali_final"]

            per_ic_results.append(ic_result)
            vpt_steps_all.append(vpt_info["vpt_steps"])

        # Aggregate
        vpt_arr = np.array(vpt_steps_all, dtype=float)
        ar_results = {
            "n_ics": len(per_ic_results),
            "vpt_mean": float(np.mean(vpt_arr)) if len(vpt_arr) > 0 else 0.0,
            "vpt_median": float(np.median(vpt_arr)) if len(vpt_arr) > 0 else 0.0,
            "vpt_std": float(np.std(vpt_arr)) if len(vpt_arr) > 0 else 0.0,
            "vpt_min": float(np.min(vpt_arr)) if len(vpt_arr) > 0 else 0.0,
            "vpt_max": float(np.max(vpt_arr)) if len(vpt_arr) > 0 else 0.0,
            "per_ic": per_ic_results,
        }

        # Lyapunov-normalized VPT
        if len(vpt_lyapunov_all) > 0:
            ly_arr = np.array(vpt_lyapunov_all, dtype=float)
            ar_results["vpt_lyapunov_mean"] = float(np.mean(ly_arr))
            ar_results["vpt_lyapunov_std"] = float(np.std(ly_arr))
            ar_results["lambda_max_used"] = float(
                self.config_lambda_max if not np.isnan(self.config_lambda_max) else 0.0
            )

        # SALI-stratified results (orbit-level difficulty labels from CNB dataset)
        stratified = {}
        for regime_label, filter_fn in [
            ("chaotic", lambda r: r.get("is_chaotic", True)),
            ("regular", lambda r: not r.get("is_chaotic", True)),
        ]:
            regime_ics = [r for r in per_ic_results if filter_fn(r)]
            if len(regime_ics) > 0:
                regime_vpts = [r["vpt_steps"] for r in regime_ics]
                regime_entry = {
                    "n_ics": len(regime_ics),
                    "vpt_mean": float(np.mean(regime_vpts)),
                    "vpt_std": float(np.std(regime_vpts)),
                    "vpt_median": float(np.median(regime_vpts)),
                }
                if any("vpt_lyapunov_times" in r for r in regime_ics):
                    ly_vals = [r["vpt_lyapunov_times"] for r in regime_ics if "vpt_lyapunov_times" in r]
                    regime_entry["vpt_lyapunov_mean"] = float(np.mean(ly_vals))
                    regime_entry["vpt_lyapunov_std"] = float(np.std(ly_vals))
                stratified[regime_label] = regime_entry

        if stratified:
            ar_results["stratified"] = stratified

        # AR horizon metrics (MSE/sMAPE/NRMSE at extended horizons from rollout)
        ar_horizon_metrics = {}
        for h in ar_horizons:
            if len(ar_mse_at_h[h]) > 0:
                ar_horizon_metrics[f"ar_mse_h{h}"] = float(np.mean(ar_mse_at_h[h]))
                ar_horizon_metrics[f"ar_smape_h{h}"] = float(np.mean(ar_smape_at_h[h]))
                ar_horizon_metrics[f"ar_nrmse_h{h}"] = float(np.mean(ar_nrmse_at_h[h]))
        if ar_horizon_metrics:
            ar_results["horizon_metrics"] = ar_horizon_metrics

        return ar_results

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Comprehensive evaluation on test set.

        Returns:
            Dict with MSE, MAE, VPT, adjacency metrics, per-horizon metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating {self.config.model_name}")
        print(f"{'=' * 60}")

        self.model.eval()

        # Collect all predictions
        all_y_true = []
        all_y_pred = []

        for x, y in self.test_loader:
            x = x.to(self.device)
            y_hat = self.model(x)
            all_y_true.append(y.numpy())
            all_y_pred.append(y_hat.cpu().numpy())

        y_true = np.concatenate(all_y_true, axis=0)  # [B, pred_len, N]
        y_pred = np.concatenate(all_y_pred, axis=0)

        print(f"  Test samples: {y_true.shape[0]}")

        # Learned adjacency (if available)
        A_learned = None
        if hasattr(self.model, "get_learned_adjacency"):
            A_learned_raw = self.model.get_learned_adjacency()
            if A_learned_raw is not None:
                A_learned = A_learned_raw

        # Comprehensive evaluation
        results = evaluate_forecast(
            y_true,
            y_pred,
            A_learned=A_learned,
            A_true=self.A_true,
            N=self.config.N,
            vpt_threshold=self.config.vpt_threshold,
            adj_threshold=self.config.adj_threshold,
            horizons=self.config.eval_horizons,
        )

        # Print results
        print(f"\n  Forecasting Metrics:")
        print(f"    MSE: {results['mse']:.6f}")
        print(f"    MAE: {results['mae']:.6f}")

        for h in self.config.eval_horizons:
            key_mse = f"mse_h{h}"
            key_mae = f"mae_h{h}"
            if key_mse in results:
                print(
                    f"    H{h}: MSE={results[key_mse]:.6f}  MAE={results[key_mae]:.6f}"
                )

        if "vpt_mean" in results:
            print(f"\n  Valid Prediction Time:")
            print(f"    VPT mean:   {results['vpt_mean']:.1f} steps")
            print(f"    VPT median: {results['vpt_median']:.1f} steps")
            print(
                f"    VPT range:  [{results['vpt_min']:.0f}, {results['vpt_max']:.0f}]"
            )

        if "adj_f1" in results:
            print(f"\n  Adjacency Recovery:")
            print(f"    F1:        {results['adj_f1']:.4f}")
            print(f"    Precision: {results['adj_precision']:.4f}")
            print(f"    Recall:    {results['adj_recall']:.4f}")
            print(f"    AUC:       {results['adj_auc']:.4f}")

            if "adj_blocks" in results:
                print(f"\n  Block-wise Adjacency Analysis:")
                blocks = results["adj_blocks"]
                for block_name in ["xx", "xp", "px", "pp"]:
                    key = f"block_{block_name}"
                    if key in blocks:
                        b = blocks[key]
                        auc_key = f"block_{block_name}_auc"
                        print(
                            f"    {block_name}: F1={b['f1']:.3f}  "
                            f"P={b['precision']:.3f}  R={b['recall']:.3f}  "
                            f"AUC={blocks.get(auc_key, 0):.3f}"
                        )

        # ─── Autoregressive evaluation (per-IC) ───
        if len(self.test_ics_normed) > 0:
            ar_results = self._evaluate_autoregressive()
            results["autoregressive"] = ar_results

            # Promote AR VPT to top-level (AR VPT is primary metric)
            results["ar_vpt_mean"] = ar_results["vpt_mean"]
            results["ar_vpt_median"] = ar_results["vpt_median"]
            results["ar_vpt_std"] = ar_results["vpt_std"]
            results["ar_n_test_ics"] = ar_results["n_ics"]

            # Print autoregressive summary
            print(f"\n  Autoregressive Evaluation ({len(self.test_ics_normed)} test ICs):")
            print(f"    VPT mean:   {ar_results['vpt_mean']:.1f} steps")
            print(f"    VPT median: {ar_results['vpt_median']:.1f} steps")
            if not np.isnan(ar_results.get("vpt_lyapunov_mean", float("nan"))):
                print(f"    VPT (λ⁻¹):  {ar_results['vpt_lyapunov_mean']:.2f} Lyapunov times")
            if "horizon_metrics" in ar_results:
                hm = ar_results["horizon_metrics"]
                h_keys = sorted(set(int(k.split("_h")[1]) for k in hm if k.startswith("ar_mse_h")))
                for h in h_keys:
                    print(f"    AR h={h}: MSE={hm[f'ar_mse_h{h}']:.6f}  "
                          f"sMAPE={hm[f'ar_smape_h{h}']:.2f}  "
                          f"NRMSE={hm[f'ar_nrmse_h{h}']:.4f}")
            if "stratified" in ar_results:
                strat = ar_results["stratified"]
                for regime, stats in strat.items():
                    print(
                        f"    {regime}: n={stats['n_ics']}, "
                        f"VPT={stats['vpt_mean']:.1f}±{stats['vpt_std']:.1f} steps"
                    )

        # Save results
        results_path = os.path.join(self.output_dir, "test_results.json")
        # Convert numpy types for JSON serialization
        results_serializable = self._make_serializable(results)
        with open(results_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        # Save learned adjacency
        if A_learned is not None:
            adj_path = os.path.join(self.output_dir, "learned_adjacency.npy")
            np.save(adj_path, A_learned)

            # Also save as readable text
            adj_txt_path = os.path.join(self.output_dir, "learned_adjacency.txt")
            N = self.config.N
            n_nodes = self.n_nodes
            n_adj = A_learned.shape[0]  # may differ from n_nodes (e.g. SpecSTG)
            if self.config.use_sincos:
                labels_full = (
                    [f"sx{i + 1}" for i in range(N)]
                    + [f"cx{i + 1}" for i in range(N)]
                    + [f"sp{i + 1}" for i in range(N)]
                    + [f"cp{i + 1}" for i in range(N)]
                )
            else:
                labels_full = [f"x{i + 1}" for i in range(N)] + [
                    f"p{i + 1}" for i in range(N)
                ]
            # Truncate labels to match learned adjacency size
            labels_adj = labels_full[:n_adj] if n_adj <= len(labels_full) else [
                f"n{i}" for i in range(n_adj)
            ]
            with open(adj_txt_path, "w") as f:
                if n_adj != n_nodes:
                    f.write(f"Learned Adjacency Matrix ({n_adj}x{n_adj} physical-node graph)\n")
                else:
                    f.write("Learned Adjacency Matrix (raw, before normalization)\n")
                f.write("    " + " ".join(f"{l:>6}" for l in labels_adj) + "\n")
                for i, row in enumerate(A_learned):
                    f.write(
                        f"{labels_adj[i]:>3} " + " ".join(f"{v:6.3f}" for v in row) + "\n"
                    )
                f.write(f"\nGround-Truth Adjacency Matrix\n")
                f.write("    " + " ".join(f"{l:>6}" for l in labels_full) + "\n")
                for i, row in enumerate(self.A_true):
                    f.write(
                        f"{labels_full[i]:>3} "
                        + " ".join(f"{int(v):6d}" for v in row)
                        + "\n"
                    )

        print(f"\n  Results saved to: {self.output_dir}")

        return results

    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        return obj

    def run(self) -> Tuple[Dict, Dict]:
        """Complete training + evaluation pipeline.

        Returns:
            (training_history, test_results)
        """
        history = self.train()
        results = self.evaluate()

        # Save combined summary
        summary = {
            "config": asdict(self.config),
            "training": {
                "best_epoch": history["best_epoch"],
                "best_val_mse": history["best_val_mse"],
                "train_time_s": history["train_time_s"],
            },
            "test": self._make_serializable(results),
        }
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return history, results


# ─────────────────────────────────────────────────────────
# Multi-configuration experiment runner
# ─────────────────────────────────────────────────────────


def run_experiment(
    dataset_path: str = "data/chaosnetbench_cml.h5",
    output_dir: str = "results/sweep",
    models: Optional[List[str]] = None,
    K_values: Optional[List[float]] = None,
    epsilon_values: Optional[List[float]] = None,
    seq_len: int = 48,
    pred_len: int = 12,
    epochs: int = 50,
    seed: int = 42,
    use_sincos: bool = False,
    seeds: Optional[List[int]] = None,
) -> Dict:
    """Run experiment across configurations.

    Args:
        dataset_path: path to HDF5 dataset
        output_dir: base output directory
        models: list of model names to evaluate
        K_values: K values to evaluate
        epsilon_values: epsilon values to evaluate
        seq_len: input sequence length
        pred_len: prediction horizon
        epochs: training epochs
        seed: random seed (used if seeds is None)
        use_sincos: whether to use sin/cos encoding
        seeds: list of seeds for multi-seed runs

    Returns:
        Dict mapping config -> results
    """
    if models is None:
        models = ["dlinear", "tcn", "lstm"]
    if K_values is None:
        K_values = [0.5, 2.0, 6.5]
    if epsilon_values is None:
        epsilon_values = [0.05, 0.3]
    if seeds is None:
        seeds = [seed]

    all_results = {}

    for model_name in models:
        for K in K_values:
            for eps in epsilon_values:
                for s in seeds:
                    seed_tag = f"_s{s}" if len(seeds) > 1 else ""
                    tag = f"{model_name}_K{K}_eps{eps}{seed_tag}"
                    print(f"\n{'#' * 60}")
                    print(f"# Experiment: {tag}")
                    print(f"{'#' * 60}")

                    config = TrainConfig(
                        dataset_path=dataset_path,
                        K=K,
                        epsilon=eps,
                        N=8,
                        seq_len=seq_len,
                        pred_len=pred_len,
                        model_name=model_name,
                        epochs=epochs,
                        seed=s,
                        output_dir=output_dir,
                        use_sincos=use_sincos,
                    )

                    try:
                        trainer = ChaosNetBenchTrainer(config)
                        history, results = trainer.run()
                        all_results[tag] = {
                            "history": {
                                "best_epoch": history["best_epoch"],
                                "best_val_mse": history["best_val_mse"],
                            },
                            "results": trainer._make_serializable(results),
                        }
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback

                        traceback.print_exc()
                        all_results[tag] = {"error": str(e)}

    # Save combined results
    combined_path = os.path.join(output_dir, "all_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("PILOT EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Config':<35} {'MSE':>10} {'MAE':>10} {'VPT':>8} {'Adj F1':>8}")
    print("-" * 80)
    for tag, res in all_results.items():
        if "error" in res:
            print(f"{tag:<35} ERROR: {res['error'][:40]}")
        else:
            r = res["results"]
            vpt = r.get("vpt_mean", "N/A")
            adj = r.get("adj_f1", "N/A")
            vpt_str = f"{vpt:.1f}" if isinstance(vpt, (int, float)) else vpt
            adj_str = f"{adj:.4f}" if isinstance(adj, (int, float)) else adj
            print(
                f"{tag:<35} {r['mse']:>10.6f} {r['mae']:>10.6f} {vpt_str:>8} {adj_str:>8}"
            )
    print(f"{'=' * 80}")

    return all_results


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChaosNetBench-CML Training")
    parser.add_argument("--mode", choices=["single", "sweep"], default="single")

    # Single mode args
    parser.add_argument("--model", default="graph_wavenet")
    parser.add_argument("--K", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Multiple seeds for multi-seed runs",
    )
    parser.add_argument("--dataset", default="data/chaosnetbench_cml.h5")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--use-sincos",
        action="store_true",
        help="Use sin/cos encoding for angular variables",
    )

    # Pilot mode args
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--K-values", nargs="+", type=float, default=None)
    parser.add_argument("--eps-values", nargs="+", type=float, default=None)

    args = parser.parse_args()

    if args.mode == "single":
        config = TrainConfig(
            dataset_path=args.dataset,
            K=args.K,
            epsilon=args.epsilon,
            N=args.N,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            model_name=args.model,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            seed=args.seed,
            output_dir=args.output_dir,
            use_sincos=args.use_sincos,
        )
        trainer = ChaosNetBenchTrainer(config)
        trainer.run()

    else:  # sweep
        run_experiment(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            models=args.models,
            K_values=args.K_values,
            epsilon_values=args.eps_values,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            epochs=args.epochs,
            seed=args.seed,
            use_sincos=args.use_sincos,
            seeds=args.seeds,
        )


# Backward-compatible alias for older imports.
ChaosBenchTrainer = ChaosNetBenchTrainer
