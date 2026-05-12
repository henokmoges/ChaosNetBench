"""
Chaos Diagnostics for Coupled Standard Map

Python implementation of:
    - Maximal Lyapunov exponent (mLE) via tangent map
  - SALI (= GALI_2) — Smaller Alignment Index

Following standard chaos indicator methodology (Benettin et al., 1980; Skokos 2001):
  - mLE via renormalization of deviation vectors
  - SALI from alignment of two deviation vectors

The tangent map of the coupled standard map:
  δp'_n = δp_n + K cos(q_n) δq_n
           - ε[cos(q_{n+1}-q_n)(δq_{n+1}-δq_n) + cos(q_{n-1}-q_n)(δq_{n-1}-δq_n)]
  δq'_n = δq_n + δp'_n

Maintainer: H. T. Moges
"""

import numpy as np
from typing import Dict, Optional, Tuple


def tangent_map_csm(
    x: np.ndarray,
    p: np.ndarray,
    dx: np.ndarray,
    dp: np.ndarray,
    K: float,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply tangent map of the coupled standard map to deviation vectors.

    Args:
        x: [N] current positions
        p: [N] current momenta
        dx: [n_vectors, N] position deviations
        dp: [n_vectors, N] momentum deviations
        K: nonlinearity parameter
        epsilon: coupling strength

    Returns:
        dx_new: [n_vectors, N] updated position deviations
        dp_new: [n_vectors, N] updated momentum deviations
    """
    N = len(x)
    n_vec = dx.shape[0]

    dp_new = dp.copy()

    # Kicker part: K cos(x_i) δx_i
    dp_new += K * np.cos(x)[np.newaxis, :] * dx

    # Coupling part: -ε[cos(x_{i+1}-x_i)(δx_{i+1}-δx_i) + cos(x_{i-1}-x_i)(δx_{i-1}-δx_i)]
    for i in range(N):
        i_next = (i + 1) % N
        i_prev = (i - 1) % N

        c_next = np.cos(x[i_next] - x[i])
        c_prev = np.cos(x[i_prev] - x[i])

        dp_new[:, i] -= epsilon * (
            c_next * (dx[:, i_next] - dx[:, i]) +
            c_prev * (dx[:, i_prev] - dx[:, i])
        )

    dx_new = dx + dp_new
    return dx_new, dp_new


def step_csm(
    x: np.ndarray, p: np.ndarray, K: float, epsilon: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Single step of coupled standard map. Returns (x_new, p_new)."""
    N = len(x)
    coupling = np.zeros(N)
    for i in range(N):
        i_next = (i + 1) % N
        i_prev = (i - 1) % N
        coupling[i] = -epsilon * (
            np.sin(x[i_next] - x[i]) + np.sin(x[i_prev] - x[i])
        )
    p_new = p + K * np.sin(x) + coupling
    x_new = np.mod(x + p_new, 2 * np.pi)
    return x_new, p_new


def compute_mle(
    K: float,
    epsilon: float,
    N: int = 1,
    n_iterations: int = 100000,
    n_transient: int = 1000,
    x0: Optional[np.ndarray] = None,
    p0: Optional[np.ndarray] = None,
    seed: int = 42,
    renorm_every: int = 1,
) -> Dict[str, float]:
    """Compute the maximal Lyapunov exponent (mLE) for the coupled standard map.

    Uses a single deviation vector evolved by the tangent map with periodic
    normalization, following the Benettin et al. procedure for the largest
    Lyapunov exponent.

    Args:
        K: nonlinearity parameter
        epsilon: coupling strength
        N: number of sites (1 for single map)
        n_iterations: total iterations (after transient)
        n_transient: transient iterations to discard
        x0, p0: initial conditions (random if None)
        seed: random seed
        renorm_every: renormalization interval (1 = every step)

    Returns:
        Dict with 'mle' (maximal Lyapunov exponent) and 'converged' flag
    """
    rng = np.random.RandomState(seed)

    if x0 is None:
        x0 = rng.uniform(0, 2 * np.pi, N)
    if p0 is None:
        p0 = rng.uniform(-np.pi, np.pi, N)

    x, p = x0.copy(), p0.copy()

    # Initialize deviation vector (random unit vector in 2N-dim tangent space)
    dv = rng.randn(1, 2 * N)
    dv /= np.linalg.norm(dv)
    dx = dv[:, :N]
    dp = dv[:, N:]

    # Transient
    for _ in range(n_transient):
        dx, dp = tangent_map_csm(x, p, dx, dp, K, epsilon)
        x, p = step_csm(x, p, K, epsilon)
        # Renormalize
        full_vec = np.concatenate([dx, dp], axis=1)
        norm = np.linalg.norm(full_vec)
        if norm > 0:
            full_vec /= norm
        dx, dp = full_vec[:, :N], full_vec[:, N:]

    # Main computation
    sum_log_norms = 0.0
    n_renorms = 0

    for it in range(n_iterations):
        dx, dp = tangent_map_csm(x, p, dx, dp, K, epsilon)
        x, p = step_csm(x, p, K, epsilon)

        if (it + 1) % renorm_every == 0:
            full_vec = np.concatenate([dx, dp], axis=1)
            norm = np.linalg.norm(full_vec)
            if norm > 1e-300:
                sum_log_norms += np.log(norm)
                full_vec /= norm
            dx, dp = full_vec[:, :N], full_vec[:, N:]
            n_renorms += 1

    mle = sum_log_norms / max(n_renorms * renorm_every, 1)

    return {
        "mle": float(mle),
        "n_iterations": n_iterations,
        "converged": n_renorms > 100,
    }


def compute_sali(
    K: float,
    epsilon: float,
    N: int = 1,
    n_iterations: int = 100000,
    n_transient: int = 1000,
    x0: Optional[np.ndarray] = None,
    p0: Optional[np.ndarray] = None,
    seed: int = 42,
    sali_threshold: float = 1e-12,
) -> Dict:
    """Compute SALI (= GALI_2) for the coupled standard map.

    SALI = min(||v1_hat + v2_hat||, ||v1_hat - v2_hat||)
    where v1_hat, v2_hat are unit deviation vectors.

    For chaotic orbits: SALI → 0 exponentially (both vectors align)
    For regular orbits: SALI oscillates, stays O(1)

    Reference: Moges et al. (2022), Physica D

    Args:
        K, epsilon: CSM parameters
        N: number of sites
        n_iterations: total iterations after transient
        n_transient: transient to discard
        x0, p0: initial conditions
        seed: random seed
        sali_threshold: convergence threshold (below = chaotic)

    Returns:
        Dict with 'sali_final', 'mle', 'sali_history', 'is_chaotic', etc.
    """
    rng = np.random.RandomState(seed)

    if x0 is None:
        x0 = rng.uniform(0, 2 * np.pi, N)
    if p0 is None:
        p0 = rng.uniform(-np.pi, np.pi, N)

    x, p = x0.copy(), p0.copy()

    # Initialize TWO linearly independent deviation vectors
    dv = rng.randn(2, 2 * N)
    # Gram-Schmidt to ensure independence
    dv[0] /= np.linalg.norm(dv[0])
    dv[1] -= np.dot(dv[1], dv[0]) * dv[0]
    dv[1] /= np.linalg.norm(dv[1])

    dx = dv[:, :N]
    dp = dv[:, N:]

    # Transient (evolve but don't record)
    for _ in range(n_transient):
        dx, dp = tangent_map_csm(x, p, dx, dp, K, epsilon)
        x, p = step_csm(x, p, K, epsilon)
        # Renormalize each vector independently (NOT Gram-Schmidt — preserves alignment info)
        for k in range(2):
            full_k = np.concatenate([dx[k:k+1], dp[k:k+1]], axis=1)
            norm_k = np.linalg.norm(full_k)
            if norm_k > 1e-300:
                full_k /= norm_k
            dx[k:k+1] = full_k[:, :N]
            dp[k:k+1] = full_k[:, N:]

    # Main computation
    sum_log_norms_1 = 0.0
    sum_log_norms_2 = 0.0
    n_renorms = 0

    # Log-spaced recording of SALI history
    n_log_points = min(500, n_iterations)
    log_indices = np.unique(np.geomspace(1, n_iterations, n_log_points).astype(int))
    sali_history = []
    mle_history = []
    sali_converged = False
    convergence_iteration = n_iterations

    for it in range(n_iterations):
        # Evolve deviation vectors through tangent map
        dx, dp = tangent_map_csm(x, p, dx, dp, K, epsilon)
        x, p = step_csm(x, p, K, epsilon)

        # Renormalize individually (preserving alignment)
        for k in range(2):
            full_k = np.concatenate([dx[k:k+1], dp[k:k+1]], axis=1)
            norm_k = np.linalg.norm(full_k)
            if norm_k > 1e-300:
                if k == 0:
                    sum_log_norms_1 += np.log(norm_k)
                else:
                    sum_log_norms_2 += np.log(norm_k)
                full_k /= norm_k
            dx[k:k+1] = full_k[:, :N]
            dp[k:k+1] = full_k[:, N:]

        n_renorms += 1

        # Compute SALI at log-spaced intervals
        if (it + 1) in log_indices or sali_converged:
            v1 = np.concatenate([dx[0], dp[0]])
            v2 = np.concatenate([dx[1], dp[1]])
            v1_hat = v1 / (np.linalg.norm(v1) + 1e-300)
            v2_hat = v2 / (np.linalg.norm(v2) + 1e-300)

            sali = min(
                np.linalg.norm(v1_hat + v2_hat),
                np.linalg.norm(v1_hat - v2_hat),
            )

            mle_current = sum_log_norms_1 / max(n_renorms, 1)

            if (it + 1) in log_indices:
                sali_history.append({
                    "iteration": int(it + 1),
                    "sali": float(sali),
                    "mle": float(mle_current),
                })

            if sali < sali_threshold and not sali_converged:
                sali_converged = True
                convergence_iteration = it + 1
                break  # Early termination: saves time for strongly chaotic orbits

    mle_final = sum_log_norms_1 / max(n_renorms, 1)
    sali_final = sali_history[-1]["sali"] if sali_history else 0.0

    return {
        "sali_final": float(sali_final),
        "mle": float(mle_final),
        "is_chaotic": sali_final < sali_threshold,
        "convergence_iteration": int(convergence_iteration),
        "sali_converged": sali_converged,
        "n_iterations": n_iterations,
        "sali_history": sali_history,
        "K": K,
        "epsilon": epsilon,
        "N": N,
    }


def compute_diagnostics_grid(
    K_values: list,
    epsilon_values: list,
    N: int = 1,
    n_iterations: int = 100000,
    n_transient: int = 1000,
    n_ics: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Compute mLE and SALI across a (K, ε) grid.

    Returns results keyed by f"K_{K:.2f}_eps_{eps:.2f}_N_{N:02d}"
    matching the HDF5 dataset structure.

    Args:
        K_values, epsilon_values: parameter grid
        N: number of sites
        n_iterations: iterations for mLE/SALI
        n_transient: transient to discard
        n_ics: number of initial conditions to average over
        seed: base seed
        verbose: print progress

    Returns:
        Dict[config_key -> Dict[ic_idx -> diagnostics]]
    """
    results = {}
    total = len(K_values) * len(epsilon_values)
    count = 0

    for K in K_values:
        for eps in epsilon_values:
            config_key = f"K_{K:.2f}_eps_{eps:.2f}_N_{N:02d}"
            ic_results = {}

            mle_values = []
            for ic_idx in range(n_ics):
                ic_seed = seed + count * n_ics + ic_idx
                diag = compute_sali(
                    K=K, epsilon=eps, N=N,
                    n_iterations=n_iterations,
                    n_transient=n_transient,
                    seed=ic_seed,
                )
                ic_results[ic_idx] = {
                    "mle": diag["mle"],
                    "sali_final": diag["sali_final"],
                    "is_chaotic": diag["is_chaotic"],
                }
                mle_values.append(diag["mle"])

            results[config_key] = {
                "ics": ic_results,
                "mle_mean": float(np.mean(mle_values)),
                "mle_std": float(np.std(mle_values)),
                "lambda_max": float(np.mean(mle_values)),  # alias
                "K": K,
                "epsilon": eps,
                "N": N,
            }

            count += 1
            if verbose:
                print(
                    f"  [{count}/{total}] {config_key}: "
                    f"λ_max = {np.mean(mle_values):.6f} ± {np.std(mle_values):.6f}"
                )

    return results


def inject_diagnostics_to_hdf5(
    filepath: str,
    diagnostics: Dict,
    verbose: bool = True,
) -> None:
    """Inject computed λ_max and SALI values into existing HDF5 dataset.

    Updates the `diagnostics/<config_key>/ic_XX` groups with:
      - lambda_max (float)
      - sali_final (float)
      - is_chaotic (bool)
      - chaos_regime (updated from heuristic to SALI-based)

    Args:
        filepath: path to HDF5 file
        diagnostics: output of compute_diagnostics_grid()
        verbose: print progress
    """
    import h5py

    with h5py.File(filepath, "a") as f:
        diag_group = f["diagnostics"]

        for config_key, config_data in diagnostics.items():
            if config_key not in diag_group:
                if verbose:
                    print(f"  Skipping {config_key} (not in HDF5)")
                continue

            cfg_diag = diag_group[config_key]
            n_ics_in_file = len(cfg_diag)

            for ic_idx, ic_data in config_data["ics"].items():
                ic_key = f"ic_{int(ic_idx):02d}"
                if ic_key not in cfg_diag:
                    continue

                ic_grp = cfg_diag[ic_key]
                ic_grp.attrs["lambda_max"] = ic_data["mle"]
                ic_grp.attrs["sali_final"] = ic_data["sali_final"]
                ic_grp.attrs["is_chaotic"] = ic_data["is_chaotic"]
                ic_grp.attrs["lyapunov_spectrum_computed"] = True

                # Update chaos regime based on SALI
                if ic_data["is_chaotic"]:
                    if ic_data["mle"] > 0.5:
                        ic_grp.attrs["chaos_regime"] = "strong_chaos"
                    else:
                        ic_grp.attrs["chaos_regime"] = "weak_chaos"
                else:
                    ic_grp.attrs["chaos_regime"] = "regular"

            # Store config-level mean λ_max
            cfg_diag.attrs["lambda_max_mean"] = config_data["mle_mean"]
            cfg_diag.attrs["lambda_max_std"] = config_data["mle_std"]

            if verbose:
                print(
                    f"  Updated {config_key}: "
                    f"λ_max = {config_data['mle_mean']:.6f}"
                )


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compute chaos diagnostics (mLE + SALI) for CSM"
    )
    parser.add_argument("--K", nargs="+", type=float,
                        default=[0.5, 0.97, 2.0, 4.0, 6.5])
    parser.add_argument("--epsilon", nargs="+", type=float,
                        default=[0.0, 0.1, 0.3])
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=100000)
    parser.add_argument("--n-transient", type=int, default=1000)
    parser.add_argument("--n-ics", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject", type=str, default=None,
                        help="Path to HDF5 file to inject results into")
    parser.add_argument("--output", type=str, default="data/chaos_diagnostics.json",
                        help="Output JSON file")
    args = parser.parse_args()

    print("Computing chaos diagnostics (mLE + SALI)...")
    print(f"  K: {args.K}")
    print(f"  ε: {args.epsilon}")
    print(f"  N: {args.N}")
    print(f"  Iterations: {args.n_iter}")
    print()

    results = compute_diagnostics_grid(
        K_values=args.K,
        epsilon_values=args.epsilon,
        N=args.N,
        n_iterations=args.n_iter,
        n_transient=args.n_transient,
        n_ics=args.n_ics,
        seed=args.seed,
    )

    # Save results
    # Strip sali_history for JSON (too large)
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: vv for kk, vv in v.items() if kk != "sali_history"}

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Inject into HDF5 if requested
    if args.inject:
        print(f"\nInjecting into {args.inject}...")
        inject_diagnostics_to_hdf5(args.inject, results)
        print("Done.")
