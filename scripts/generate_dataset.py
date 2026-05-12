#!/usr/bin/env python3
"""
ChaosNetBench — Dataset Generation Script

Generates the ChaosNetBench-CML dataset from the Coupled Standard Map (CSM).
The generated HDF5 file contains 9,600 trajectories across 96 system instances.

This script generates the full benchmark dataset. Expected output: ~26 GB.
For a quick test (subset), use --mode mini.

Usage:
  # Generate full benchmark dataset
  python scripts/generate_dataset.py --output-dir data/

    # Quick test (small subset, ~8-12 MB, ~1-2 min)
  python scripts/generate_dataset.py --mode mini --output-dir data/

  # Inspect an existing dataset
  python scripts/generate_dataset.py --mode inspect --filepath data/chaosnetbench_cml.h5

Full dataset specification:
  K values:   {0.5, 0.97, 2.0, 6.5}           (4 kick strengths)
  ρ values:   {0.05, 0.075, 0.10, ..., 0.50}   (8 coupling ratios)
  N values:   {8, 16, 32}                       (3 lattice sizes)
  ICs/config: 100                               (independent initial conditions)
  Steps:      10,000 (+ 1,000 transient)
  Total:      96 system instances × 100 ICs = 9,600 trajectories

Reference:
  Chirikov, B.V. (1979). A universal instability of many-dimensional
  oscillator systems. Physics Reports, 52(5), 263-379.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chaosnetbench.dataset import (
    BenchmarkConfig,
    DatasetConfig,
    MiniConfig,
    generate_dataset,
    inspect_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="ChaosNetBench dataset generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["full", "mini", "inspect"], default="full",
        help="Generation mode (default: full)"
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Directory to save the HDF5 file (default: data/)"
    )
    parser.add_argument(
        "--filepath", default=None,
        help="Path to HDF5 file for inspect mode"
    )
    args = parser.parse_args()

    if args.mode == "inspect":
        path = args.filepath or "data/chaosnetbench_cml.h5"
        inspect_dataset(path)

    elif args.mode == "mini":
        config = MiniConfig(output_dir=args.output_dir)
        print("ChaosNetBench Mini Dataset (quick test)")
        print(f"  Configurations: {config.n_configs}")
        print(f"  ICs/config:     {config.n_ics}")
        print(f"  Total:          {config.n_configs * config.n_ics} trajectories")
        print(f"  Output:         {args.output_dir}/{config.filename}")
        print()
        filepath = generate_dataset(config)
        print(f"\nDone: {filepath}")

    elif args.mode == "full":
        config = BenchmarkConfig(output_dir=args.output_dir)
        print("ChaosNetBench Full Benchmark Dataset")
        print(f"  K values:       {config.K_values}")
        print(f"  ρ values:       {config.rho_values}")
        print(f"  N values:       {config.N_values}")
        print(f"  ICs/config:     {config.n_ics}")
        print(f"  Configurations: {config.n_configs}")
        print(f"  Total:          {config.n_configs * config.n_ics} trajectories")
        print(f"  Expected size:  ~26 GB")
        print(f"  Output:         {args.output_dir}/{config.filename}")
        print()
        print("  Note: Full generation takes several hours on CPU.")
        print("  The precomputed dataset is available at:")
        print("    Hugging Face: https://huggingface.co/datasets/htmoges/chaosnetbench-cml")
        print()

        filepath = generate_dataset(config)
        print(f"\nDone: {filepath}")


if __name__ == "__main__":
    main()
