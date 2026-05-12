#!/usr/bin/env python3
"""
ChaosNetBench — Full Benchmark Sweep

Runs the complete ChaosNetBench-CML evaluation:
  - All 13 benchmark models
  - All 96 system instances (4K × 8ρ × 3N)
  - 3 seeds per instance (confirmation sweep)

Reproduction tiers:
  test    ~10 min,        1 model × 1 instance × 1 seed
  medium  ~2 GPU-hours,   1 model × 1 K-slice  × 3 seeds
  full    ~4,500 GPU-hrs, 13 models × 96 instances × 3 seeds

Usage:
  # Test tier (Graph WaveNet, K=2.0, all ρ, N=8, 1 seed)
  python scripts/sweep.py --tier test

  # Medium tier (Graph WaveNet, K=2.0 slice, 3 seeds)
  python scripts/sweep.py --tier medium

  # Full sweep (serial, ~4500 GPU-hrs)
  python scripts/sweep.py --tier full

  # Custom: specific models and GPU
  python scripts/sweep.py --models graph_wavenet d2stgnn --seeds 42 43 44 --gpu 0

  # Full sweep, continue from existing results
  python scripts/sweep.py --tier full --resume
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import constants from train.py
from scripts.train import (
    BENCHMARK_MODELS, K_VALUES, RHO_VALUES, N_VALUES,
    eps_from_rho, make_tag, run_single,
)


# ──────────────────────────────────────────────────────────
# Reproduction tier definitions
# ──────────────────────────────────────────────────────────

TIERS = {
    "test": {
        "description": "Single model, single instance, 1 seed (~10 min on A100)",
        "models": ["graph_wavenet"],
        "K_values": [2.0],
        "rho_values": [0.20],
        "N_values": [8],
        "seeds": [42],
        "output_dir": "results/test_run",
    },
    "medium": {
        "description": "1 model × 1 K-slice × 3 seeds (~2 GPU-hours on A100)",
        "models": ["graph_wavenet"],
        "K_values": [2.0],
        "rho_values": RHO_VALUES,
        "N_values": N_VALUES,
        "seeds": [42, 43, 44],
        "output_dir": "results/medium_run",
    },
    "full": {
        "description": "Complete benchmark: 13 models × 96 instances × 3 seeds (~4,500 GPU-hrs)",
        "models": BENCHMARK_MODELS,
        "K_values": K_VALUES,
        "rho_values": RHO_VALUES,
        "N_values": N_VALUES,
        "seeds": [42, 43, 44],
        "output_dir": "results/full_sweep",
    },
}


def build_grid(models, K_values, rho_values, N_values, seeds):
    """Build the experiment grid."""
    grid = []
    for model in models:
        for K in K_values:
            for rho in rho_values:
                eps = eps_from_rho(K, rho)
                if eps < 0.01 or eps > 5.0:
                    continue
                for N in N_values:
                    for seed in seeds:
                        grid.append({
                            "model": model, "K": K, "rho": rho,
                            "epsilon": eps, "N": N, "seed": seed,
                        })
    return grid


def run_sweep(grid, output_dir, gpu=None, resume=True):
    """Run all experiments in the grid serially."""
    results_path = os.path.join(output_dir, "all_results.json")
    os.makedirs(output_dir, exist_ok=True)

    # Load existing results for resume
    all_results = {}
    if resume and os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} experiments already completed")

    total = len(grid)
    completed = 0
    errors = 0
    start_time = time.time()

    print(f"\nSweep: {total} experiments, output: {output_dir}")
    print(f"{'─' * 60}")

    for i, exp in enumerate(grid):
        tag = make_tag(exp["model"], exp["K"], exp["rho"], exp["N"], exp["seed"])

        if tag in all_results and "error" not in all_results[tag]:
            completed += 1
            continue

        try:
            result = run_single(
                model=exp["model"], K=exp["K"], rho=exp["rho"],
                N=exp["N"], seed=exp["seed"],
                output_dir=output_dir, gpu=gpu,
            )
        except Exception as e:
            result = {"config": exp, "error": str(e)}
            errors += 1

        all_results[tag] = result
        completed += 1

        vpt = result.get("ar_vpt_mean", "N/A")
        status = "ERR" if "error" in result else f"VPT={vpt:.1f}" if isinstance(vpt, float) else "OK"
        elapsed = time.time() - start_time
        eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
        print(f"  [{completed}/{total}] {tag} → {status}  "
              f"(elapsed {elapsed/60:.0f}m, ETA {eta/60:.0f}m)")

        # Save incrementally for resume support
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'─' * 60}")
    print(f"Sweep complete: {completed} experiments, {errors} errors, {elapsed/60:.1f} min")
    print(f"Results: {results_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="ChaosNetBench benchmark sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tier", choices=list(TIERS.keys()), default=None,
                        help="Reproduction tier (test/medium/full)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=BENCHMARK_MODELS,
                        help="Models to evaluate (default: tier setting)")
    parser.add_argument("--K", nargs="+", type=float, default=None,
                        help="K values to include (default: tier setting)")
    parser.add_argument("--rho", nargs="+", type=float, default=None,
                        help="ρ values to include (default: tier setting)")
    parser.add_argument("--N", nargs="+", type=int, default=None,
                        help="N values to include (default: tier setting)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds (default: tier setting)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA device index")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: tier setting)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing results (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    # Resolve settings: tier defaults overridden by explicit args
    if args.tier:
        tier = TIERS[args.tier]
        print(f"Tier: {args.tier} — {tier['description']}")
    else:
        tier = TIERS["full"]

    models     = args.models     or tier["models"]
    K_values   = args.K          or tier["K_values"]
    rho_values = args.rho        or tier["rho_values"]
    N_values   = args.N          or tier["N_values"]
    seeds      = args.seeds      or tier["seeds"]
    output_dir = args.output_dir or tier["output_dir"]

    grid = build_grid(models, K_values, rho_values, N_values, seeds)

    print(f"\nChaosNetBench Sweep Configuration:")
    print(f"  Models:       {models}")
    print(f"  K values:     {K_values}")
    print(f"  ρ values:     {rho_values}")
    print(f"  N values:     {N_values}")
    print(f"  Seeds:        {seeds}")
    print(f"  Experiments:  {len(grid)}")
    print(f"  Output:       {output_dir}")
    print(f"  GPU:          {args.gpu if args.gpu is not None else 'auto'}")

    run_sweep(grid, output_dir, gpu=args.gpu, resume=args.resume)


if __name__ == "__main__":
    main()
