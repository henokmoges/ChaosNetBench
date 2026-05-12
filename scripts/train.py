#!/usr/bin/env python3
"""
ChaosNetBench — Training Entry Point

Train a single forecasting model on one ChaosNetBench-CML system instance.

Usage:
  # Single experiment
  python scripts/train.py --model graph_wavenet --K 2.0 --rho 0.20 --N 16 --seed 42

  # With explicit GPU
  python scripts/train.py --model d2stgnn --K 0.97 --rho 0.15 --N 8 --seed 42 --gpu 0

  # Run the 'test' reproduction tier (~10 min, one model, one instance)
  python scripts/train.py --model graph_wavenet --K 2.0 --rho 0.20 --N 8 --seed 42

Available models:
  Non-graph baselines: dlinear, tcn, lstm, nbeats, patchtst, itransformer, dsformer, stid
  Graph-based STGNNs:  agcrn, graph_wavenet, d2stgnn, staeformer
  Diagnostic control:  oracle_gcn
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaosnetbench.training import TrainConfig, ChaosNetBenchTrainer


# ──────────────────────────────────────────────────────────
# Benchmark constants (paper parameter grid)
# ──────────────────────────────────────────────────────────

DATASET_PATH = "data/chaosnetbench_cml.h5"

K_VALUES    = [0.5, 0.97, 2.0, 6.5]
RHO_VALUES  = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N_VALUES    = [8, 16, 32]

BENCHMARK_MODELS = [
    # Non-graph baselines
    "dlinear",       # Linear baseline
    "tcn",           # Temporal Convolutional Network
    "lstm",          # Long Short-Term Memory
    "nbeats",        # N-BEATS
    "patchtst",      # PatchTST
    "itransformer",  # iTransformer (cross-variate attention)
    "dsformer",      # DSformer (double-sampling transformer)
    "stid",          # STID (spatial identity embeddings)
    # Graph-based STGNNs
    "agcrn",         # Adaptive Graph Convolutional Recurrent Network
    "graph_wavenet", # Graph WaveNet (diffusion convolution)
    "d2stgnn",       # D2STGNN (diffusion + inherent dynamics)
    "staeformer",    # STAEformer (spatial attention)
    # Diagnostic control
    "oracle_gcn",    # GCN with ground-truth ring adjacency
]

# Per-model architecture hyperparameters (locked after pilot tuning)
MODEL_HP_REGISTRY = {
    "oracle_gcn":    {"d_model": 64, "n_gcn_layers": 2, "dropout": 0.1},
    "lstm":          {"hidden_dim": 64, "n_layers": 2},
    "dlinear":       {},
    "tcn":           {"hidden_channels": 32, "kernel_size": 3, "n_layers": 4, "dropout": 0.1},
    "nbeats":        {"n_stacks": 2, "n_blocks": 3, "hidden_dim": 128, "theta_dim": 16},
    "patchtst":      {"patch_len": 8, "stride": 4, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "itransformer":  {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 128},
    "agcrn":         {"embed_dim": 10, "rnn_units": 32, "n_layers": 2, "cheb_k": 2},
    "staeformer":    {"d_model": 64, "n_heads": 4, "n_t_layers": 2, "n_s_layers": 1,
                      "adaptive_embed_dim": 16},
    "graph_wavenet": {"residual_channels": 32, "dilation_channels": 32,
                      "skip_channels": 64, "end_channels": 128, "n_layers": 6},
    "stid":          {"embed_dim": 32, "hidden_dim": 128, "n_layers": 3},
    "d2stgnn":       {"hidden_dim": 32, "node_dim": 10, "n_layers": 3, "k_s": 2, "k_t": 3,
                      "n_heads": 4},
    "dsformer":      {"num_layer": 1, "dropout": 0.15, "num_head": 2, "num_samp": 2,
                      "use_node_embed": True},
}

# Shared training hyperparameters (frozen after pilot tuning on K=0.5, ρ=0.30, N=8)
BASE_CONFIG = {
    "learning_rate": 1e-3,
    "batch_size":    32,
    "epochs":        50,
    "milestones":    [20, 35, 45],
    "gamma":         0.5,
    "patience":      10,
    "grad_clip_norm": 5.0,
    "weight_decay":  0.0,
    "window_stride": 12,
}

# Per-model learning rate overrides (selected on positive control)
LR_OVERRIDES = {
    "dlinear":       3e-4,
    "tcn":           3e-3,
    "lstm":          3e-3,
    "nbeats":        3e-4,
    "itransformer":  1e-3,
    "oracle_gcn":    1e-3,
    "agcrn":         3e-4,
    "graph_wavenet": 3e-4,
    "stid":          1e-3,
    "d2stgnn":       3e-3,
    "dsformer":       1e-3,
}


def eps_from_rho(K: float, rho: float) -> float:
    """Compute ε = ρ × K, rounded to avoid floating point drift."""
    return round(rho * K, 6)


def make_tag(model: str, K: float, rho: float, N: int, seed: int) -> str:
    """Create unique experiment tag for result naming."""
    return f"{model}_K{K}_rho{rho}_N{N}_s{seed}"


def run_single(model: str, K: float, rho: float, N: int, seed: int,
               output_dir: str, gpu: int = None,
               dataset_path: str = DATASET_PATH,
               epochs: int = None) -> dict:
    """
    Train one model on one system instance and return results.

    Args:
        model:      Model name (see BENCHMARK_MODELS)
        K:          Kick strength (chaos parameter)
        rho:        Coupling-to-chaos ratio ρ = ε/K
        N:          Number of lattice sites
        seed:       Random seed
        output_dir: Directory to save results
        gpu:        CUDA device index (None = auto)

    Returns:
        Result dict with VPT, MSE, and training history.
    """
    epsilon = eps_from_rho(K, rho)
    tag = make_tag(model, K, rho, N, seed)
    exp_dir = os.path.join(output_dir, tag)

    model_kwargs = dict(MODEL_HP_REGISTRY.get(model, {}))
    lr = LR_OVERRIDES.get(model, BASE_CONFIG["learning_rate"])
    device = "auto" if gpu is None else f"cuda:{gpu}"
    num_workers = int(os.environ.get("CNB_NUM_WORKERS", 4))

    config = TrainConfig(
        dataset_path=dataset_path,
        K=K,
        epsilon=epsilon,
        N=N,
        seq_len=48,
        pred_len=12,
        window_stride=BASE_CONFIG["window_stride"],
        model_name=model,
        epochs=epochs if epochs is not None else BASE_CONFIG["epochs"],
        seed=seed,
        output_dir=exp_dir,
        model_kwargs=model_kwargs,
        learning_rate=lr,
        batch_size=BASE_CONFIG["batch_size"],
        milestones=BASE_CONFIG["milestones"],
        gamma=BASE_CONFIG["gamma"],
        patience=BASE_CONFIG["patience"],
        grad_clip_norm=BASE_CONFIG["grad_clip_norm"],
        weight_decay=BASE_CONFIG["weight_decay"],
        vpt_threshold=1.0,
        device=device,
        num_workers=num_workers,
    )

    trainer = ChaosNetBenchTrainer(config)
    history, results = trainer.run()

    result = {
        "config": {"model": model, "K": K, "rho": rho, "epsilon": epsilon,
                   "N": N, "seed": seed},
        "history": {
            "best_epoch":    history["best_epoch"],
            "best_val_mse":  history["best_val_mse"],
            "train_time_s":  history["train_time_s"],
        },
        "ar_vpt_mean": results.get("ar_vpt_mean"),
        "ar_vpt_std":  results.get("ar_vpt_std"),
        "mse":         results.get("mse"),
        "mae":         results.get("mae"),
    }

    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ChaosNetBench — train a single model on one system instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",  required=True, choices=BENCHMARK_MODELS,
                        help="Model to train")
    parser.add_argument("--K",     type=float, required=True,
                        help=f"Kick strength. Benchmark values: {K_VALUES}")
    parser.add_argument("--rho",   type=float, required=True,
                        help=f"Coupling ratio ρ=ε/K. Benchmark values: {RHO_VALUES}")
    parser.add_argument("--N",     type=int,   required=True, choices=N_VALUES,
                        help="Number of lattice sites")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--gpu",   type=int,   default=None,
                        help="CUDA device index (default: auto)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH,
                        help=f"Path to HDF5 dataset (default: {DATASET_PATH})")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (default: 50)")
    args = parser.parse_args()

    epsilon = eps_from_rho(args.K, args.rho)
    print(f"ChaosNetBench | model={args.model} K={args.K} ρ={args.rho} "
          f"ε={epsilon:.4f} N={args.N} seed={args.seed}")

    result = run_single(
        model=args.model,
        K=args.K,
        rho=args.rho,
        N=args.N,
        seed=args.seed,
        output_dir=args.output_dir,
        gpu=args.gpu,
        dataset_path=args.dataset,
        epochs=args.epochs,
    )

    if "error" in result:
        print(f"\nFAILED: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n--- Results ---")
        print(f"AR VPT (mean):  {result.get('ar_vpt_mean', 'N/A')}")
        print(f"AR VPT (std):   {result.get('ar_vpt_std', 'N/A')}")
        print(f"Test MSE:       {result.get('mse', 'N/A')}")
        print(f"Best val MSE:   {result['history']['best_val_mse']:.4f} "
              f"(epoch {result['history']['best_epoch']})")
        print(f"Train time:     {result['history']['train_time_s']:.0f}s")
        print(f"Results saved:  {args.output_dir}/")


if __name__ == "__main__":
    main()
