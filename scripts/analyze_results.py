#!/usr/bin/env python3
"""
ChaosNetBench — Results Aggregation

Aggregates multi-seed experiment results into a summary CSV and prints
ranked comparisons.

Input:  results/<sweep_dir>/  (per-experiment result.json files)
Output: results/<sweep_dir>/chaosnetbench_cml_results.csv

Usage:
  # Aggregate a 3-seed confirmation sweep
  python scripts/analyze_results.py --results-dir results/full_sweep

  # Specify output path explicitly
  python scripts/analyze_results.py --results-dir results/full_sweep --output results/chaosnetbench_cml_results.csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────────────────
# Model display names and classification
# ──────────────────────────────────────────────────────────

MODEL_DISPLAY = {
    "dlinear":       "DLinear",
    "tcn":           "TCN",
    "lstm":          "LSTM",
    "nbeats":        "N-BEATS",
    "patchtst":      "PatchTST",
    "itransformer":  "iTransformer",
    "dsformer":      "DSformer",
    "stid":          "STID",
    "oracle_gcn":    "Oracle GCN",
    "agcrn":         "AGCRN",
    "graph_wavenet": "Graph WaveNet",
    "d2stgnn":       "D2STGNN",
    "staeformer":    "STAEformer",
    "diffstg":       "DiffSTG",
}

MODEL_TIER = {
    "dlinear":       "Local temporal",
    "tcn":           "Local temporal",
    "nbeats":        "Local temporal",
    "patchtst":      "Local temporal",
    "stid":          "Spatial identity",
    "lstm":          "Cross-variate implicit",
    "itransformer":  "Cross-variate implicit",
    "dsformer":      "Cross-variate implicit",
    "oracle_gcn":    "Diagnostic (oracle)",
    "agcrn":         "Graph-based STGNN",
    "graph_wavenet": "Graph-based STGNN",
    "d2stgnn":       "Graph-based STGNN",
    "staeformer":    "Graph-based STGNN",
    "diffstg":       "Extended probe",
}

CONVERGENCE_THRESHOLD = 0.95  # test MSE < 0.95 → convergent


def load_results(results_dir: str) -> pd.DataFrame:
    """
    Load all result.json files from the sweep directory into a DataFrame.

    Handles both:
      - Per-model JSON:  results_dir/<model>_results.json
      - Per-experiment:  results_dir/<tag>/result.json
    """
    records = []
    results_path = Path(results_dir)

    # Strategy 1: look for per-model aggregated JSONs
    for json_path in results_path.glob("*_results.json"):
        model = json_path.stem.replace("_results", "")
        with open(json_path) as f:
            data = json.load(f)
        for key, val in data.items():
            if not isinstance(val, dict) or "error" in val:
                continue
            _append_record(records, val)

    # Strategy 2: look for all_results.json (sweep output)
    all_results_path = results_path / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path) as f:
            data = json.load(f)
        for key, val in data.items():
            if not isinstance(val, dict) or "error" in val:
                continue
            _append_record(records, val)

    # Strategy 3: per-experiment result.json files
    if not records:
        for result_json in results_path.rglob("result.json"):
            try:
                with open(result_json) as f:
                    val = json.load(f)
                if "error" not in val:
                    _append_record(records, val)
            except (json.JSONDecodeError, IOError):
                continue

    if not records:
        print(f"Warning: no results found in {results_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["model", "K", "rho", "N", "seed"])
    return df


def _append_record(records: list, val: dict):
    """Extract fields from a result dict and append to records."""
    cfg = val.get("config", {})
    history = val.get("history", {})
    records.append({
        "model":          cfg.get("model"),
        "model_display":  MODEL_DISPLAY.get(cfg.get("model", ""), cfg.get("model", "")),
        "tier":           MODEL_TIER.get(cfg.get("model", ""), ""),
        "K":              cfg.get("K"),
        "rho":            cfg.get("rho"),
        "N":              cfg.get("N"),
        "seed":           cfg.get("seed"),
        "epsilon":        cfg.get("epsilon"),
        "test_mse":       val.get("mse"),
        "test_mae":       val.get("mae"),
        "ar_vpt_mean":    val.get("ar_vpt_mean"),
        "ar_vpt_std":     val.get("ar_vpt_std"),
        "best_val_mse":   history.get("best_val_mse"),
        "train_time_s":   history.get("train_time_s"),
        "convergent":     (val.get("mse") or 1.0) < CONVERGENCE_THRESHOLD,
    })


def compute_aggregated_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-config statistics (mean ± std over seeds).

    Output has one row per (model, K, rho, N), with mean and std
    for each metric, plus seed count and convergence rate.
    """
    group_cols = ["model", "model_display", "tier", "K", "rho", "N", "epsilon"]

    rows = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row["n_seeds"] = len(grp)
        for col in ["test_mse", "test_mae", "ar_vpt_mean", "best_val_mse", "train_time_s"]:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else float("nan")
            row[f"{col}_std"]  = float(vals.std())  if len(vals) > 1 else float("nan")
        row["convergence_rate"] = float(grp["convergent"].mean())
        row["n_convergent"] = int(grp["convergent"].sum())
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, agg: pd.DataFrame):
    """Print a concise benchmark summary."""
    print("\n" + "=" * 70)
    print("  ChaosNetBench-CML Results Summary")
    print("=" * 70)

    print(f"\n  Total run records:  {len(df)}")
    print(f"  Unique models:      {df['model'].nunique()}")
    print(f"  Overall convergence: {df['convergent'].mean() * 100:.1f}%")

    # Fractional win counts (per-config winner by mean VPT, 3-seed mean)
    wins = defaultdict(float)
    n_scored = 0
    vpt_grp = agg[agg["convergence_rate"] > 0.0].copy()
    for (K, rho, N), grp in vpt_grp.groupby(["K", "rho", "N"]):
        if len(grp) < 2:
            continue
        max_vpt = grp["ar_vpt_mean_mean"].max()
        winners = grp[grp["ar_vpt_mean_mean"] == max_vpt]
        for _, w in winners.iterrows():
            wins[w["model_display"]] += 1.0 / len(winners)
        n_scored += 1

    print(f"\n  Fractional win counts ({n_scored} scored configs, Oracle GCN excluded):")
    print(f"  {'Model':20s} {'Wins':>8s}")
    print(f"  {'─' * 30}")
    oracle_excluded = [(m, w) for m, w in sorted(wins.items(), key=lambda x: -x[1])
                       if "Oracle" not in m]
    oracle_only     = [(m, w) for m, w in wins.items() if "Oracle" in m]
    for m, w in oracle_excluded:
        print(f"  {m:20s} {w:8.1f}")
    if oracle_only:
        print(f"  {'─' * 30}")
        for m, w in oracle_only:
            print(f"  {m:20s} {w:8.1f}  (diagnostic, excluded from rankings)")

    print(f"\n  Per-K convergence summary:")
    print(f"  {'Model':20s} {'K=0.5':>8s} {'K=0.97':>8s} {'K=2.0':>8s} {'K=6.5':>8s}")
    print(f"  {'─' * 60}")
    for model, grp in agg.groupby("model"):
        display = MODEL_DISPLAY.get(model, model)
        parts = []
        for K_val in [0.5, 0.97, 2.0, 6.5]:
            sub = grp[grp["K"] == K_val]
            n_c = int(sub["n_convergent"].sum())
            n_t = int(sub["n_seeds"].sum())
            parts.append(f"{n_c}/{n_t}" if n_t > 0 else "—")
        print(f"  {display:20s} {'  '.join(f'{p:>8s}' for p in parts)}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="ChaosNetBench results aggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing sweep results")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <results_dir>/chaosnetbench_cml_results.csv)")
    args = parser.parse_args()

    output = args.output or os.path.join(args.results_dir, "chaosnetbench_cml_results.csv")

    print(f"Loading results from: {args.results_dir}")
    df = load_results(args.results_dir)

    if df.empty:
        print("No results found. Exiting.")
        sys.exit(1)

    agg = compute_aggregated_csv(df)
    agg.to_csv(output, index=False)
    print(f"Saved aggregated CSV: {output}  ({len(agg)} rows)")

    print_summary(df, agg)


if __name__ == "__main__":
    main()
