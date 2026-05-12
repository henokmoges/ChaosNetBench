"""
ChaosNetBench-CML Evaluation Metrics

Chaos-aware evaluation metrics for coupled map lattice forecasting:
- Valid Prediction Time (VPT): time until prediction diverges
- Cumulative Max Error (CME): sensitivity to initial conditions
- Adjacency recovery: F1, AUC, precision, recall for graph discovery
- Standard forecasting: MSE, MAE
"""

import numpy as np
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────
# Forecasting Metrics
# ─────────────────────────────────────────────────────────


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error.

    sMAPE = 200 * mean(|y_pred - y_true| / (|y_pred| + |y_true| + ε))

    This is Gilpin (2023)'s primary metric for chaotic time series forecasting.
    Range: [0, 200], where 0 = perfect prediction.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        sMAPE score (percentage, 0-200)

    Reference:
        Gilpin (2023) Phys Rev Research 5, 043252
    """
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    return float(200.0 * np.mean(numerator / denominator))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Normalized Root Mean Squared Error.

    NRMSE = RMSE / std(y_true)

    Used for VPT computation — threshold of 1.0 means error equals signal variability.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        NRMSE score (>0, where 1.0 means error = signal std)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    std = np.std(y_true)
    return float(rmse / (std + epsilon))


def mse_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MSE at each prediction horizon.

    Args:
        y_true: [pred_len, n_nodes] or [B, pred_len, n_nodes]
        y_pred: same shape

    Returns:
        MSE array of shape [pred_len]
    """
    if y_true.ndim == 2:
        return np.mean((y_true - y_pred) ** 2, axis=-1)
    else:
        return np.mean((y_true - y_pred) ** 2, axis=(0, -1))


def smape_per_horizon(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """sMAPE at each prediction horizon.

    Args:
        y_true: [pred_len, n_nodes] or [B, pred_len, n_nodes]
        y_pred: same shape

    Returns:
        sMAPE array of shape [pred_len]
    """
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
    smape_values = 200.0 * numerator / denominator

    if y_true.ndim == 2:
        return np.mean(smape_values, axis=-1)
    else:
        return np.mean(smape_values, axis=(0, -1))


def nrmse_per_horizon(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """NRMSE at each prediction horizon.

    Args:
        y_true: [pred_len, n_nodes] or [B, pred_len, n_nodes]
        y_pred: same shape

    Returns:
        NRMSE array of shape [pred_len]
    """
    if y_true.ndim == 2:
        # [pred_len]
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=-1))
        std = np.std(y_true) + epsilon
    else:
        # [B, pred_len, n_nodes] -> [pred_len]
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=(0, -1)))
        std = np.std(y_true) + epsilon
    return rmse / std


# ─────────────────────────────────────────────────────────
# Valid Prediction Time (VPT)
# ─────────────────────────────────────────────────────────


def valid_prediction_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    metric: str = "normalized_rmse",
) -> int:
    """Compute Valid Prediction Time (VPT).

    VPT is the number of time steps before the prediction error
    exceeds a threshold. This is the primary metric for chaotic
    system forecasting — it captures the Lyapunov timescale.

    Args:
        y_true: [pred_len, n_nodes] ground truth
        y_pred: [pred_len, n_nodes] prediction
        threshold: error threshold (relative to signal std)
        metric: "normalized_rmse" or "absolute_rmse"

    Returns:
        VPT: number of valid prediction steps (0 to pred_len)
    """
    pred_len = y_true.shape[0]

    for t in range(pred_len):
        if metric == "normalized_rmse":
            # RMSE normalized by signal standard deviation
            rmse_t = np.sqrt(np.mean((y_true[t] - y_pred[t]) ** 2))
            std_t = np.std(y_true[: t + 1]) if t > 0 else np.std(y_true)
            std_t = max(std_t, 1e-8)
            error = rmse_t / std_t
        else:
            error = np.sqrt(np.mean((y_true[t] - y_pred[t]) ** 2))

        if error > threshold:
            return t

    return pred_len


def vpt_batch(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute VPT statistics over a batch.

    Args:
        y_true: [B, pred_len, n_nodes]
        y_pred: [B, pred_len, n_nodes]
        threshold: error threshold

    Returns:
        Dict with vpt_mean, vpt_std, vpt_median
    """
    B = y_true.shape[0]
    vpts = np.array(
        [valid_prediction_time(y_true[i], y_pred[i], threshold) for i in range(B)]
    )
    return {
        "vpt_mean": float(np.mean(vpts)),
        "vpt_std": float(np.std(vpts)),
        "vpt_median": float(np.median(vpts)),
        "vpt_min": float(np.min(vpts)),
        "vpt_max": float(np.max(vpts)),
    }


# ─────────────────────────────────────────────────────────
# Autoregressive Evaluation
# ─────────────────────────────────────────────────────────


def autoregressive_rollout_np(
    model_fn,
    x_init: np.ndarray,
    y_true_full: np.ndarray,
    step_size: int = 1,
) -> Dict:
    """Autoregressive rollout: chain 1-step (or step_size-step) predictions.

    Produces continuous error(t) for proper VPT extraction, following
    Gilpin (2023) evaluation protocol.

    Args:
        model_fn: callable that takes [1, seq_len, N] numpy -> [1, step_size, N] numpy
        x_init: [seq_len, N] initial input window
        y_true_full: [T_future, N] full ground-truth continuation
        step_size: number of steps the model predicts per call

    Returns:
        Dict with:
            'y_pred': [T_future, N] autoregressive predictions
            'nrmse_t': [T_future] per-step NRMSE
            'mse_t': [T_future] per-step MSE
            'smape_t': [T_future] per-step sMAPE
    """
    T_future = y_true_full.shape[0]
    n_nodes = x_init.shape[-1]
    seq_len = x_init.shape[0]

    y_pred_full = np.zeros_like(y_true_full)
    window = x_init.copy()  # [seq_len, N]

    n_steps_done = 0
    while n_steps_done < T_future:
        # Predict next step_size steps
        pred = model_fn(window[np.newaxis])  # [1, step_size, N]
        pred = pred[0]  # [step_size, N]

        # Store predictions
        end = min(n_steps_done + step_size, T_future)
        n_to_store = end - n_steps_done
        y_pred_full[n_steps_done:end] = pred[:n_to_store]

        # Shift window: append predictions, drop oldest
        if n_to_store < seq_len:
            window = np.concatenate([window[n_to_store:], pred[:n_to_store]], axis=0)
        else:
            window = pred[-seq_len:]

        n_steps_done = end

    # Compute per-step error metrics
    mse_t = np.mean((y_true_full - y_pred_full) ** 2, axis=-1)  # [T_future]
    std_signal = np.std(y_true_full) + 1e-8
    nrmse_t = np.sqrt(mse_t) / std_signal

    numerator = np.abs(y_pred_full - y_true_full)
    denominator = np.abs(y_pred_full) + np.abs(y_true_full) + 1e-8
    smape_t = 200.0 * np.mean(numerator / denominator, axis=-1)

    return {
        "y_pred": y_pred_full,
        "nrmse_t": nrmse_t,
        "mse_t": mse_t,
        "smape_t": smape_t,
    }


def vpt_from_nrmse_curve(
    nrmse_t: np.ndarray,
    threshold: float = 1.0,
    lambda_max: Optional[float] = None,
) -> Dict[str, float]:
    """Extract VPT from a continuous NRMSE(t) curve.

    Args:
        nrmse_t: [T] per-step NRMSE from autoregressive rollout
        threshold: NRMSE threshold for valid prediction
        lambda_max: if provided, also return VPT in Lyapunov times

    Returns:
        Dict with vpt_steps (raw), and optionally vpt_lyapunov_times
    """
    exceed = np.where(nrmse_t > threshold)[0]
    vpt_steps = int(exceed[0]) if len(exceed) > 0 else len(nrmse_t)

    result = {"vpt_steps": vpt_steps}
    if lambda_max is not None and lambda_max > 0:
        # VPT in Lyapunov times: number of e-folding times the prediction
        # remains valid.  VPT_λ = VPT_steps × λ_max  (standard definition,
        # cf. Gilpin 2023 Physical Review Research 5, 043252).
        result["vpt_lyapunov_times"] = float(vpt_steps * lambda_max)
        result["lambda_max"] = float(lambda_max)

    return result


# ─────────────────────────────────────────────────────────
# Cumulative Max Error (CME)
# ─────────────────────────────────────────────────────────


def cumulative_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Cumulative Maximum Error across nodes.

    CME(t) = max_{s<=t} max_i |y_true(s,i) - y_pred(s,i)|

    This captures sensitivity to initial conditions — for chaotic
    systems, CME grows exponentially with rate ~ max Lyapunov exponent.

    Args:
        y_true: [pred_len, n_nodes]
        y_pred: [pred_len, n_nodes]

    Returns:
        CME array of shape [pred_len]
    """
    pointwise_error = np.abs(y_true - y_pred)
    max_over_nodes = np.max(pointwise_error, axis=-1)  # [pred_len]
    cme = np.maximum.accumulate(max_over_nodes)
    return cme


def estimate_lyapunov_from_cme(
    cme: np.ndarray, fit_range: Optional[Tuple[int, int]] = None
) -> float:
    """Estimate max Lyapunov exponent from CME growth rate.

    Fits log(CME) ~ λ_max * t in the exponential growth regime.

    Args:
        cme: CME array [pred_len]
        fit_range: (start, end) indices for fitting. If None,
                   auto-detect the exponential growth regime.

    Returns:
        Estimated λ_max (base-e, per iteration)
    """
    log_cme = np.log(cme + 1e-15)

    if fit_range is None:
        # Auto-detect: find region where log(CME) is roughly linear
        # Skip initial transient (first 5%) and saturation (last 20%)
        n = len(cme)
        start = max(1, n // 20)
        end = max(start + 10, int(0.8 * n))
        fit_range = (start, end)

    t = np.arange(fit_range[0], fit_range[1])
    y = log_cme[fit_range[0] : fit_range[1]]

    if len(t) < 2:
        return 0.0

    # Linear regression: log(CME) = λ*t + c
    coeffs = np.polyfit(t, y, 1)
    return float(coeffs[0])  # slope = λ_max


# ─────────────────────────────────────────────────────────
# Adjacency Recovery Metrics
# ─────────────────────────────────────────────────────────


def adjacency_f1(
    A_learned: np.ndarray,
    A_true: np.ndarray,
    threshold: float = 0.1,
    exclude_diagonal: bool = True,
) -> Dict[str, float]:
    """Compute F1, precision, recall for adjacency recovery.

    Binarizes the learned adjacency using threshold and compares
    to ground-truth binary adjacency.

    Args:
        A_learned: [N, N] learned adjacency (may have continuous values)
        A_true: [N, N] ground-truth binary adjacency
        threshold: binarization threshold for learned adjacency
        exclude_diagonal: if True, exclude diagonal (self-loops) from
            evaluation. Set to False for off-diagonal blocks (e.g. xp, px)
            where sub-block diagonal entries are legitimate edges.

    Returns:
        Dict with precision, recall, f1, accuracy
    """
    A_pred_binary = (A_learned > threshold).astype(float)
    A_true_binary = (A_true > 0).astype(float)

    n = A_true.shape[0]
    if exclude_diagonal:
        # Remove diagonal (self-loops) from evaluation
        mask = ~np.eye(n, dtype=bool)
        pred_flat = A_pred_binary[mask]
        true_flat = A_true_binary[mask]
    else:
        pred_flat = A_pred_binary.ravel()
        true_flat = A_true_binary.ravel()

    TP = float(np.sum((pred_flat == 1) & (true_flat == 1)))
    FP = float(np.sum((pred_flat == 1) & (true_flat == 0)))
    FN = float(np.sum((pred_flat == 0) & (true_flat == 1)))
    TN = float(np.sum((pred_flat == 0) & (true_flat == 0)))

    precision = TP / max(TP + FP, 1e-8)
    recall = TP / max(TP + FN, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
    }


def adjacency_auc(
    A_learned: np.ndarray,
    A_true: np.ndarray,
    n_thresholds: int = 100,
    exclude_diagonal: bool = True,
) -> float:
    """Compute AUC-ROC for adjacency recovery.

    Sweeps threshold over learned adjacency values and computes
    ROC curve against ground-truth binary adjacency.

    Args:
        A_learned: [N, N] learned adjacency (continuous values)
        A_true: [N, N] ground-truth binary adjacency
        n_thresholds: number of threshold values to sweep
        exclude_diagonal: if True, exclude diagonal from evaluation.

    Returns:
        AUC-ROC score (0 to 1, higher is better)
    """
    n = A_true.shape[0]
    if exclude_diagonal:
        mask = ~np.eye(n, dtype=bool)
        pred_flat = A_learned[mask]
        true_flat = (A_true[mask] > 0).astype(float)
    else:
        pred_flat = A_learned.ravel()
        true_flat = (A_true.ravel() > 0).astype(float)

    # Edge case: all same label or constant predictor → AUC undefined, return 0.5
    if true_flat.sum() == 0 or true_flat.sum() == len(true_flat):
        return 0.5
    if pred_flat.max() == pred_flat.min():
        return 0.5

    thresholds = np.linspace(pred_flat.max(), pred_flat.min(), n_thresholds)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        pred_binary = (pred_flat >= thresh).astype(float)
        TP = np.sum((pred_binary == 1) & (true_flat == 1))
        FP = np.sum((pred_binary == 1) & (true_flat == 0))
        FN = np.sum((pred_binary == 0) & (true_flat == 1))
        TN = np.sum((pred_binary == 0) & (true_flat == 0))

        tpr = TP / max(TP + FN, 1e-8)
        fpr = FP / max(FP + TN, 1e-8)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Build ROC curve: thresholds are in descending order so FPR/TPR are
    # monotone non-decreasing. Anchor at (0,0) and (1,1).
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    fpr_sorted = np.concatenate([[0.0], fpr_arr, [1.0]])
    tpr_sorted = np.concatenate([[0.0], tpr_arr, [1.0]])

    # np.trapezoid replaces deprecated np.trapz in NumPy 2.x
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_val = float(_trapz(tpr_sorted, fpr_sorted))
    return abs(auc_val)  # ensure positive


def adjacency_structure_analysis(
    A_learned: np.ndarray, A_true: np.ndarray, N: int, threshold: float = 0.1
) -> Dict:
    """Detailed analysis of learned adjacency structure.

    Decomposes the 2N×2N adjacency into physically meaningful blocks
    and evaluates each block separately.

    For CSM with blocked ordering [x₁,...,xₙ, p₁,...,pₙ]:
      A = [[A_xx, A_xp],    x_i depends on: x (via coupling), p (via x'=x+p')
           [A_px, A_pp]]    p_i depends on: x (via K*sin + coupling), p (none)

    Args:
        A_learned: [2N, 2N] learned adjacency
        A_true: [2N, 2N] ground-truth adjacency
        N: number of lattice sites
        threshold: binarization threshold

    Returns:
        Dict with block-wise metrics and overall analysis
    """
    blocks = {
        "xx": (slice(0, N), slice(0, N)),  # x←x (ring coupling)
        "xp": (slice(0, N), slice(N, 2 * N)),  # x←p (identity, x'=x+p')
        "px": (slice(N, 2 * N), slice(0, N)),  # p←x (K*sin + coupling)
        "pp": (slice(N, 2 * N), slice(N, 2 * N)),  # p←p (should be zero)
    }

    # Off-diagonal blocks (xp, px) should NOT exclude their sub-block
    # diagonal, since those entries (e.g. x1→p1, p1→x1) are legitimate
    # physical edges, not self-loops in the full 2N×2N matrix.
    is_off_diagonal_block = {"xx": False, "xp": True, "px": True, "pp": False}

    result = {}
    for name, (row_slice, col_slice) in blocks.items():
        A_l_block = A_learned[row_slice, col_slice]
        A_t_block = A_true[row_slice, col_slice]
        keep_diag = is_off_diagonal_block[name]

        result[f"block_{name}"] = adjacency_f1(
            A_l_block, A_t_block, threshold, exclude_diagonal=not keep_diag
        )
        result[f"block_{name}_auc"] = adjacency_auc(
            A_l_block, A_t_block, exclude_diagonal=not keep_diag
        )
        result[f"block_{name}_density_learned"] = float(
            (A_l_block > threshold).sum() / max(A_l_block.size, 1)
        )
        result[f"block_{name}_density_true"] = float(
            (A_t_block > 0).sum() / max(A_t_block.size, 1)
        )

    # Overall metrics (exclude diagonal = self-loops)
    result["overall"] = adjacency_f1(A_learned, A_true, threshold)
    result["overall_auc"] = adjacency_auc(A_learned, A_true)

    return result


# ─────────────────────────────────────────────────────────
# Graph Fidelity (Interpretability)
# ─────────────────────────────────────────────────────────


def graph_fidelity(
    A_learned: np.ndarray,
    A_true: np.ndarray,
    N: int,
    threshold: float = 0.1,
) -> Dict[str, float]:
    """Compute graph fidelity score measuring interpretability.

    Fidelity quantifies how faithfully the learned adjacency
    captures the true physical coupling structure. Combines:

    1. Topology fidelity: F1 score of edge recovery
    2. Weight fidelity: Rank correlation of learned vs true weights
    3. Block consistency: Adherence to physics-expected block structure
       - xx: ring coupling (should be sparse, off-diagonal)
       - xp: identity-like (x' = x + p')
       - px: coupling + K·sin (should match xx pattern)
       - pp: should be near-zero (no direct p-p coupling in CSM)
    4. Sparsity fidelity: Match between learned and true edge density

    Args:
        A_learned: [2N, 2N] learned adjacency matrix
        A_true: [2N, 2N] ground-truth adjacency
        N: number of lattice sites
        threshold: binarization threshold for learned adjacency

    Returns:
        Dict with component fidelities and composite score (0 to 1)
    """
    n_total = A_true.shape[0]
    mask = ~np.eye(n_total, dtype=bool)

    # 1. Topology fidelity (F1)
    f1_metrics = adjacency_f1(A_learned, A_true, threshold)
    topology_fidelity = f1_metrics["f1"]

    # 2. Weight fidelity (Spearman rank correlation on off-diagonal)
    pred_flat = A_learned[mask]
    true_flat = A_true[mask].astype(float)

    # Only compute correlation where true edges exist
    true_edges = true_flat > 0
    if true_edges.sum() > 1:
        try:
            from scipy.stats import spearmanr
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = spearmanr(pred_flat[true_edges], true_flat[true_edges])
            weight_fidelity = max(0.0, float(rho))  # clamp negative to 0
        except ImportError:
            # Fall back to numpy correlation if scipy unavailable
            weight_fidelity = max(
                0.0,
                float(np.corrcoef(pred_flat[true_edges], true_flat[true_edges])[0, 1]),
            )
    else:
        weight_fidelity = 0.0

    # 3. Block consistency score
    blocks = {
        "xx": (slice(0, N), slice(0, N)),
        "xp": (slice(0, N), slice(N, 2 * N)),
        "px": (slice(N, 2 * N), slice(0, N)),
        "pp": (slice(N, 2 * N), slice(N, 2 * N)),
    }

    is_off_diagonal = {"xx": False, "xp": True, "px": True, "pp": False}
    block_scores = {}
    for name, (rs, cs) in blocks.items():
        A_l = A_learned[rs, cs]
        A_t = A_true[rs, cs]
        bf1 = adjacency_f1(
            A_l, A_t, threshold, exclude_diagonal=not is_off_diagonal[name]
        )["f1"]
        block_scores[name] = bf1

    # Physics-weighted block consistency:
    # xx and px blocks carry the coupling information (most important)
    # xp carries momentum-position link
    # pp should be zero (penalize if model puts edges there)
    block_consistency = (
        0.35 * block_scores.get("xx", 0.0)
        + 0.25 * block_scores.get("px", 0.0)
        + 0.20 * block_scores.get("xp", 0.0)
        + 0.20 * block_scores.get("pp", 0.0)
    )

    # 4. Sparsity fidelity: how well learned density matches true density
    pred_density = float((A_learned[mask] > threshold).mean())
    true_density = float((A_true[mask] > 0).mean())
    sparsity_fidelity = 1.0 - min(abs(pred_density - true_density), 1.0)

    # Composite fidelity score (weighted average)
    composite = (
        0.35 * topology_fidelity
        + 0.25 * weight_fidelity
        + 0.25 * block_consistency
        + 0.15 * sparsity_fidelity
    )

    return {
        "fidelity_composite": float(composite),
        "fidelity_topology": float(topology_fidelity),
        "fidelity_weight": float(weight_fidelity),
        "fidelity_block_consistency": float(block_consistency),
        "fidelity_sparsity": float(sparsity_fidelity),
        "block_xx_f1": block_scores.get("xx", 0.0),
        "block_xp_f1": block_scores.get("xp", 0.0),
        "block_px_f1": block_scores.get("px", 0.0),
        "block_pp_f1": block_scores.get("pp", 0.0),
        "learned_density": pred_density,
        "true_density": true_density,
    }


# ─────────────────────────────────────────────────────────
# Comprehensive Evaluation
# ─────────────────────────────────────────────────────────


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    A_learned: Optional[np.ndarray] = None,
    A_true: Optional[np.ndarray] = None,
    N: Optional[int] = None,
    vpt_threshold: float = 0.5,
    adj_threshold: float = 0.1,
    horizons: Optional[list] = None,
) -> Dict:
    """Comprehensive evaluation combining all metrics.

    Args:
        y_true: [B, pred_len, n_nodes] or [pred_len, n_nodes]
        y_pred: same shape
        A_learned: [n_nodes, n_nodes] learned adjacency (optional)
        A_true: [n_nodes, n_nodes] ground-truth adjacency (optional)
        N: number of lattice sites (for block analysis)
        vpt_threshold: threshold for VPT computation
        adj_threshold: binarization threshold for adjacency
        horizons: specific horizons to evaluate MSE/MAE at

    Returns:
        Comprehensive evaluation dict
    """
    is_batch = y_true.ndim == 3

    results = {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
    }

    # Per-horizon metrics
    if horizons is not None:
        mse_h = mse_per_horizon(y_true, y_pred)
        smape_h = smape_per_horizon(y_true, y_pred)
        nrmse_h = nrmse_per_horizon(y_true, y_pred)
        for h in horizons:
            if is_batch:
                y_t_h = y_true[:, :h, :]
                y_p_h = y_pred[:, :h, :]
            else:
                y_t_h = y_true[:h, :]
                y_p_h = y_pred[:h, :]
            results[f"mse_h{h}"] = mse(y_t_h, y_p_h)
            results[f"mae_h{h}"] = mae(y_t_h, y_p_h)
            results[f"smape_h{h}"] = smape(y_t_h, y_p_h)

    # VPT
    if is_batch:
        vpt_stats = vpt_batch(y_true, y_pred, vpt_threshold)
        results.update(vpt_stats)
    else:
        results["vpt"] = valid_prediction_time(y_true, y_pred, vpt_threshold)

    # CME
    if not is_batch:
        cme_arr = cumulative_max_error(y_true, y_pred)
        results["cme_final"] = float(cme_arr[-1])
        results["lambda_max_estimated"] = estimate_lyapunov_from_cme(cme_arr)

    # Adjacency recovery
    if A_learned is not None and A_true is not None:
        # Handle shape mismatch: model may learn N×N (physical node)
        # adjacency while A_true is 2N×2N (phase space).  E.g. SpecSTG
        # operates on the ring Laplacian of N physical lattice sites.
        A_true_cmp = A_true
        _do_block_analysis = True
        if A_learned.shape[0] != A_true.shape[0]:
            if A_learned.shape[0] * 2 == A_true.shape[0]:
                n_phys = A_learned.shape[0]
                A_true_cmp = A_true[:n_phys, :n_phys]  # xx-block
                _do_block_analysis = False
            else:
                # Unknown mismatch — skip adjacency metrics entirely
                _do_block_analysis = False
                A_true_cmp = None

        if A_true_cmp is not None:
            adj_metrics = adjacency_f1(A_learned, A_true_cmp, adj_threshold)
            results["adj_f1"] = adj_metrics["f1"]
            results["adj_precision"] = adj_metrics["precision"]
            results["adj_recall"] = adj_metrics["recall"]
            results["adj_auc"] = adjacency_auc(A_learned, A_true_cmp)

        if N is not None and _do_block_analysis:
            block_metrics = adjacency_structure_analysis(
                A_learned, A_true, N, adj_threshold
            )
            results["adj_blocks"] = block_metrics

            # Graph fidelity (interpretability)
            fidelity = graph_fidelity(A_learned, A_true, N, adj_threshold)
            results["fidelity"] = fidelity["fidelity_composite"]
            results["fidelity_details"] = fidelity

    return results
