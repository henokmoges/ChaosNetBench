"""
ChaosNetBench Dataset Generator

Generates multivariate time series from the Coupled Standard Map (CSM)
for STGNN benchmarking.

Data format:
  - State variables: [q₁, ..., qₙ, p₁, ..., pₙ] (blocked ordering)
  - Each variable is treated as a separate node (N_nodes = 2N)
  - Both wrapped (mod 2π) and unwrapped coordinates are stored
  - Ground-truth adjacency matrices at variable-level (2N×2N) and site-level (N×N)

Reference:
  Moges et al. (2022) Physica D: "Anomalous diffusion in single and coupled
  standard maps with extensive chaotic phase spaces"

MIT License
"""

import numpy as np
import h5py
import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Import our CSM implementation
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chaosnetbench.systems.standard_map import CoupledStandardMap
from chaosnetbench.systems.chaos_diagnostics import compute_sali


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    # Parameter grid (benchmark defaults: 4K × 8ρ × 3N = 96 instances)
    K_values: List[float] = field(default_factory=lambda: [0.5, 0.97, 2.0, 6.5])
    epsilon_values: List[float] = field(
        default_factory=lambda: [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
    )
    N_values: List[int] = field(default_factory=lambda: [8, 16, 32])
    n_ics: int = 100  # Number of initial conditions per configuration (benchmark: 100)

    # Trajectory parameters
    n_steps: int = 10000  # Steps to record (after transient)
    transient: int = 1000  # Steps to discard

    # Variable ordering
    ordering: str = "blocked"  # "blocked" = [x₁,...,xₙ,p₁,...,pₙ] or "interleaved" = [x₁,p₁,...,xₙ,pₙ]

    # Reproducibility
    base_seed: int = 42

    # SALI-based orbit classification (Moges et al. 2022)
    # The SALI screen runs diag_n_iterations of the tangent map (variational
    # equations). Early termination when SALI < 10⁻⁸ saves time for strongly
    # chaotic orbits.
    compute_diagnostics: bool = False  # Run SALI screen per orbit
    diag_n_iterations: int = 1000      # Fixed for all N; early termination handles the rest
    diag_n_transient: int = 0          # Transient for SALI (0 = use trajectory IC directly)

    # Storage options
    store_unwrapped: bool = True  # Store p_unwrapped array (for diffusion analysis)

    # Deterministic seed: seed = f(base_seed, K, ε, N, ic_idx) — same config always gets same seed
    seed_mode: str = "deterministic"

    # Output
    output_dir: str = "data"
    filename: str = "chaosnetbench_cml.h5"

    @property
    def n_configs(self) -> int:
        return len(self.K_values) * len(self.epsilon_values) * len(self.N_values)

    @property
    def n_trajectories(self) -> int:
        return self.n_configs * self.n_ics


@dataclass
class BenchmarkConfig(DatasetConfig):
    """
    ChaosNetBench benchmark configuration.

    Parameter grid (96 system instances):
      - K ∈ {0.5, 0.97, 2.0, 6.5}          — local chaos intensity
      - ρ = ε/K ∈ {0.05 … 0.50} (8 values) — coupling-to-chaos ratio
      - N ∈ {8, 16, 32}                     — lattice size
      - 100 ICs per configuration (70 train / 10 val / 20 test)
      - SALI classification enabled with n=1000 screening horizon
    """

    K_values: List[float] = field(default_factory=lambda: [0.5, 0.97, 2.0, 6.5])
    N_values: List[int] = field(default_factory=lambda: [8, 16, 32])
    n_ics: int = 100
    n_steps: int = 10000
    transient: int = 1000
    compute_diagnostics: bool = True
    diag_n_iterations: int = 1000
    store_unwrapped: bool = False  # Skip p_unwrapped re-integration (not used in ML training)
    filename: str = "chaosnetbench_cml.h5"
    seed_mode: str = "deterministic"

    # ρ grid: 8 coupling-to-chaos ratios
    rho_values: List[float] = field(
        default_factory=lambda: [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    )
    epsilon_min: float = 0.01
    epsilon_max: float = 5.0

    def __post_init__(self):
        """Derive epsilon_values from ρ × K, filtered to valid range."""
        self._eps_for_K = {}
        eps_set = set()
        for K in self.K_values:
            k_eps = []
            for rho in self.rho_values:
                eps = round(rho * K, 6)
                if self.epsilon_min <= eps <= self.epsilon_max:
                    k_eps.append(eps)
                    eps_set.add(eps)
            self._eps_for_K[K] = sorted(k_eps)
        self.epsilon_values = sorted(eps_set)

    def eps_for_K(self, K: float) -> List[float]:
        """Return only the epsilon values valid for this K."""
        return self._eps_for_K.get(K, self.epsilon_values)

    @property
    def n_configs(self) -> int:
        """Count only valid (K, eps, N) combinations."""
        total = 0
        for K in self.K_values:
            total += len(self.eps_for_K(K)) * len(self.N_values)
        return total


@dataclass
class MiniConfig(DatasetConfig):
    """Small dataset subset for quick end-to-end testing (~8-12 MB, ~1-2 min on CPU).

    Covers 2 K values × 2 ρ values × 1 N value × 6 ICs = 24 trajectories.
    Sufficient to verify the pipeline end-to-end without downloading the full dataset.
    """

    K_values: List[float] = field(default_factory=lambda: [0.5, 2.0])
    N_values: List[int] = field(default_factory=lambda: [8])
    n_ics: int = 6
    n_steps: int = 2000
    transient: int = 500
    compute_diagnostics: bool = True
    diag_n_iterations: int = 1000
    store_unwrapped: bool = False
    filename: str = "chaosnetbench_cml_mini.h5"
    seed_mode: str = "deterministic"

    rho_values: List[float] = field(default_factory=lambda: [0.10, 0.30])
    epsilon_min: float = 0.01
    epsilon_max: float = 5.0

    def __post_init__(self):
        self._eps_for_K = {}
        eps_set = set()
        for K in self.K_values:
            k_eps = []
            for rho in self.rho_values:
                eps = round(rho * K, 6)
                if self.epsilon_min <= eps <= self.epsilon_max:
                    k_eps.append(eps)
                    eps_set.add(eps)
            self._eps_for_K[K] = sorted(k_eps)
        self.epsilon_values = sorted(eps_set)

    def eps_for_K(self, K: float) -> List[float]:
        return self._eps_for_K.get(K, self.epsilon_values)


# ============================================================================
# Ground-truth adjacency construction
# ============================================================================


def build_site_adjacency(N: int) -> np.ndarray:
    """
    Build N×N site-level ring adjacency matrix.

    A[i,j] = 1 if sites i and j are nearest neighbors on the ring.
    """
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        A[i, (i - 1) % N] = 1.0
    return A


def build_variable_adjacency(N: int, include_self_loops: bool = False) -> np.ndarray:
    """
    Build 2N×2N variable-level adjacency from the CSM Jacobian structure.

    With blocked ordering [x₁,...,xₙ, p₁,...,pₙ]:

    The Jacobian of the CSM map gives the dependency structure:
      p'_i depends on: p_i (self), x_i (K·sin), x_{i±1} (coupling)
      x'_i depends on: x_i (self), p'_i (which depends on above)

    At the one-step level (direct dependencies):
      x'_i ← x_i, p_i, x_{i±1}  (through p'_i)
      p'_i ← p_i, x_i, x_{i±1}

    The 2N×2N matrix has block structure:
      A = [ A_xx  A_xp ]
          [ A_px  A_pp ]

    where:
      A_xx = ring (x_i depends on x_{i±1} via coupling through p')
      A_xp = I (x_i depends on p_i directly via x' = x + p')
      A_px = ring + I (p_i depends on x_i and x_{i±1})
      A_pp = 0 (p_i doesn't directly depend on p_j≠i, excluding self)

    Parameters
    ----------
    N : int
        Number of lattice sites
    include_self_loops : bool
        Whether to include self-loops (diagonal entries)

    Returns
    -------
    A : np.ndarray, shape (2N, 2N)
        Variable-level adjacency matrix
    """
    ring = build_site_adjacency(N)  # N×N ring
    I_N = np.eye(N, dtype=np.float32)
    Z_N = np.zeros((N, N), dtype=np.float32)

    # Block construction
    A_xx = ring.copy()  # x_i depends on x_{i±1} (through coupling in p')
    A_xp = I_N.copy()  # x_i depends on p_i (via x' = x + p')
    A_px = ring + I_N  # p_i depends on x_i (K·sin) and x_{i±1} (coupling)
    A_pp = Z_N.copy()  # no direct p-p dependencies (excluding self)

    # Assemble 2N×2N matrix
    A = np.block([[A_xx, A_xp], [A_px, A_pp]])

    if not include_self_loops:
        np.fill_diagonal(A, 0.0)

    # Binarize (some entries may be > 1 from ring + I)
    A = (A > 0).astype(np.float32)

    return A


def build_coupling_only_adjacency(N: int) -> np.ndarray:
    """
    Build 2N×2N adjacency with ONLY the inter-site coupling edges.

    These are the edges that depend on ε (vanish when ε=0).
    Useful for evaluating whether the model specifically learns the coupling.
    """
    ring = build_site_adjacency(N)
    Z_N = np.zeros((N, N), dtype=np.float32)

    A = np.block([[ring, Z_N], [ring, Z_N]])
    np.fill_diagonal(A, 0.0)
    return A


def build_variable_adjacency_sincos(
    N: int, include_self_loops: bool = False
) -> np.ndarray:
    """
    Build 4N×4N variable-level adjacency for sin/cos encoded data.

    With sin/cos encoding, the variable ordering becomes:
      [sin(x₁),...,sin(xₙ), cos(x₁),...,cos(xₙ),
       sin(p₁),...,sin(pₙ), cos(p₁),...,cos(pₙ)]

    Each original variable θ maps to (sin θ, cos θ), so dependencies
    are duplicated across sin/cos pairs. If A_2N is the 2N×2N Jacobian
    adjacency, the 4N×4N version tiles each NxN block into a 2Nx2N block:

      A_4N[i·N:(i+1)·N, j·N:(j+1)·N] for each (sin_i,cos_i) × (sin_j,cos_j)

    Parameters
    ----------
    N : int
        Number of lattice sites
    include_self_loops : bool
        Whether to include self-loops

    Returns
    -------
    A : np.ndarray, shape (4N, 4N)
    """
    # Get the base 2N×2N adjacency blocks
    ring = build_site_adjacency(N)
    I_N = np.eye(N, dtype=np.float32)
    Z_N = np.zeros((N, N), dtype=np.float32)

    # Original blocks (see build_variable_adjacency)
    A_xx = ring.copy()
    A_xp = I_N.copy()
    A_px = ring + I_N
    A_pp = Z_N.copy()

    # Each block B_{ij} in the 2×2 block structure becomes a 2×2 tile:
    # [[B, B], [B, B]] because sin(θ) and cos(θ) share the same dependencies
    def tile_block(B):
        return np.block([[B, B], [B, B]])

    # 4N×4N: 4 super-blocks, each is a 2N×2N tile of the original N×N block
    A = np.block(
        [
            [tile_block(A_xx), tile_block(A_xp)],
            [tile_block(A_px), tile_block(A_pp)],
        ]
    )

    if not include_self_loops:
        np.fill_diagonal(A, 0.0)

    A = (A > 0).astype(np.float32)
    return A


# ============================================================================
# Data generation
# ============================================================================


def generate_trajectory(
    N: int,
    K: float,
    epsilon: float,
    n_steps: int,
    transient: int,
    seed: int,
    ordering: str = "blocked",
    compute_unwrapped: bool = True,
) -> Dict:
    """
    Generate a single CSM trajectory with both wrapped and unwrapped coordinates.

    Parameters
    ----------
    N : int
        Number of lattice sites
    K, epsilon : float
        CSM parameters
    n_steps : int
        Steps to record after transient
    transient : int
        Steps to discard
    seed : int
        Random seed for reproducibility
    ordering : str
        "blocked" for [x₁,...,xₙ,p₁,...,pₙ] or
        "interleaved" for [x₁,p₁,...,xₙ,pₙ]

    Returns
    -------
    result : dict
        Contains state_wrapped, p_unwrapped, initial_conditions, etc.
    """
    csm = CoupledStandardMap(N=N, K=K, epsilon=epsilon, seed=seed)

    # Generate initial conditions
    x0, p0 = csm.random_initial_conditions()

    # --- Wrapped trajectory (standard integration, x mod 2π) ---
    x_traj, p_traj = csm.integrate(x0, p0, n_steps, transient=transient)

    # Wrap p to [0, 2π) for the bounded dataset
    p_traj_wrapped = np.mod(p_traj, 2 * np.pi)

    # --- Unwrapped p trajectory (for diffusion analysis) ---
    # Re-integrate WITHOUT modding p to get unwrapped momentum
    # Note: x is still mod 2π (position is always on the circle)
    if compute_unwrapped:
        x_unwrap, p_unwrap = x0.copy(), p0.copy()

        # Burn transient (tracking unwrapped p)
        for _ in range(transient):
            coupling_force = np.zeros(N)
            for i in range(N):
                i_next = (i + 1) % N
                i_prev = (i - 1) % N
                coupling_force[i] = -epsilon * (
                    np.sin(x_unwrap[i_next] - x_unwrap[i])
                    + np.sin(x_unwrap[i_prev] - x_unwrap[i])
                )
            p_unwrap = p_unwrap + K * np.sin(x_unwrap) + coupling_force
            x_unwrap = np.mod(x_unwrap + p_unwrap, 2 * np.pi)

        # Record unwrapped p trajectory
        p_unwrapped = np.zeros((n_steps, N))
        for t in range(n_steps):
            p_unwrapped[t] = p_unwrap
            coupling_force = np.zeros(N)
            for i in range(N):
                i_next = (i + 1) % N
                i_prev = (i - 1) % N
                coupling_force[i] = -epsilon * (
                    np.sin(x_unwrap[i_next] - x_unwrap[i])
                    + np.sin(x_unwrap[i_prev] - x_unwrap[i])
                )
            p_unwrap = p_unwrap + K * np.sin(x_unwrap) + coupling_force
            x_unwrap = np.mod(x_unwrap + p_unwrap, 2 * np.pi)
    else:
        p_unwrapped = None

    # --- Assemble state tensor ---
    if ordering == "blocked":
        # [x₁,...,xₙ, p₁,...,pₙ]  shape: [T, 2N]
        state_wrapped = np.concatenate([x_traj, p_traj_wrapped], axis=1)
    elif ordering == "interleaved":
        # [x₁,p₁, x₂,p₂, ..., xₙ,pₙ]  shape: [T, 2N]
        state_wrapped = np.empty((n_steps, 2 * N))
        state_wrapped[:, 0::2] = x_traj
        state_wrapped[:, 1::2] = p_traj_wrapped
    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    # --- Compute diffusion exponent from unwrapped p ---
    if p_unwrapped is not None:
        delta_p = p_unwrapped - p_unwrapped[0:1]  # displacement from initial
        mean_sq_disp = np.mean(delta_p**2, axis=1)  # average over sites

        # Fit μ from log-log slope of ⟨(Δp)²⟩ vs n
        # Use last half for stability (skip transient initial behaviour)
        n_half = n_steps // 2
        if n_half > 10:
            log_n = np.log(np.arange(n_half, n_steps) + 1)
            log_msd = np.log(mean_sq_disp[n_half:] + 1e-30)  # avoid log(0)
            # Simple linear regression
            if np.std(log_n) > 0 and np.isfinite(log_msd).all():
                slope, intercept = np.polyfit(log_n, log_msd, 1)
                diffusion_exponent = slope
            else:
                diffusion_exponent = np.nan
        else:
            diffusion_exponent = np.nan
    else:
        mean_sq_disp = None
        diffusion_exponent = np.nan

    result = {
        "state_wrapped": state_wrapped.astype(np.float64),  # [T, 2N]
        "initial_conditions": np.concatenate([x0, p0]),  # [2N]
        "diffusion_exponent": float(diffusion_exponent),
    }
    if p_unwrapped is not None:
        result["p_unwrapped"] = p_unwrapped.astype(np.float64)  # [T, N]
    if mean_sq_disp is not None:
        result["mean_sq_displacement"] = mean_sq_disp.astype(np.float64)  # [T]
    return result


def compute_regime_label(K: float, epsilon: float, diffusion_exponent: float) -> str:
    """
    Assign chaos regime label based on K value and computed diagnostics.

    This is a heuristic classification. The definitive labeling will be
    based on Lyapunov exponents computed via Fortran.

    Categories:
      - "regular": near-integrable dynamics (K < K_cr)
      - "weak_chaos": mixed phase space (K_cr < K < 3)
      - "moderate_chaos": extensive chaos (3 ≤ K < 5)
      - "strong_chaos": fully chaotic (K ≥ 5)
      - "anomalous_transport": accelerator mode regime (K near 2πm)
    """
    # Period-1 AM intervals: K ∈ (2πm - δ, 2πm + δ) for m = 1, 2, ...
    TWO_PI = 2 * np.pi
    near_am = False
    for m in range(1, 12):
        if abs(K - TWO_PI * m) < 0.5 * TWO_PI:
            near_am = True
            break

    if K < 0.97:
        return "regular"
    elif K < 3.0:
        return "weak_chaos"
    elif K < 5.0:
        return "moderate_chaos"
    elif near_am and diffusion_exponent > 1.5:
        return "anomalous_transport"
    else:
        return "strong_chaos"


# SALI thresholds following Moges et al. (2022), Physica D
# At n=50: chaotic orbits have GALI₂ < 1e-8 (exponential collapse);
#          regular orbits have GALI₂ > 1e-4 (power-law n⁻²);
#          4+ orders of magnitude gap separates regimes.
SALI_CHAOTIC_THRESHOLD = 1e-8
SALI_REGULAR_THRESHOLD = 1e-4


def config_seed(base_seed: int, K: float, epsilon: float, N: int, ic_idx: int) -> int:
    """
    Deterministic seed for a specific (K, ε, N, ic_idx) configuration.

    Uses a hash so the same config always produces the same seed,
    regardless of what other configs are in the grid.

    Returns a positive integer in [base_seed, base_seed + 2^31).
    """
    key = f"{base_seed}|{K:.10f}|{epsilon:.10f}|{N}|{ic_idx}"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return base_seed + (h % (2**31))


def classify_orbit_sali(
    K: float,
    epsilon: float,
    N: int,
    x0: np.ndarray,
    p0: np.ndarray,
    n_iterations: int = 50,
    n_transient: int = 0,
    seed: int = 42,
) -> Dict:
    """
    Classify a single orbit using a short GALI₂ (=SALI) screen.

    IMPORTANT: This runs n_iterations of the TANGENT MAP (variational
    equations) from the orbit's initial condition, NOT n_iterations of the
    trajectory itself. The trajectory is generated separately with n_steps
    timesteps. This screen is a lightweight chaos indicator only.

    Three-tier orbit classification scheme (Moges et al. 2022):
      - "chaotic":  GALI₂ < 1e-8  (exponential collapse, definitively chaotic)
      - "regular":  GALI₂ > 1e-4  (power-law n⁻² regime, KAM torus)
      - "sticky":   1e-8 ≤ GALI₂ ≤ 1e-4  (ambiguous, near separatrices)

    Parameters
    ----------
    K, epsilon : float
        CSM parameters
    N : int
        Number of lattice sites
    x0, p0 : np.ndarray, shape (N,)
        Initial conditions (the actual ICs used for the trajectory)
    n_iterations : int
        Tangent map iterations (50 is sufficient for standard map)
    n_transient : int
        Pre-screen transient (0 = classify from the same IC)
    seed : int
        RNG seed for deviation vectors only

    Returns
    -------
    dict with keys:
        'sali_value': float — final GALI₂ value
        'chaos_regime': str — "chaotic", "regular", or "sticky"
        'n_iterations': int — iterations used
    """
    result = compute_sali(
        K=K,
        epsilon=epsilon,
        N=N,
        n_iterations=n_iterations,
        n_transient=n_transient,
        x0=x0.copy(),
        p0=p0.copy(),
        seed=seed,
    )

    sali_val = result["sali_final"]

    # Handle machine-precision zeros: SALI can collapse to exactly 0.0
    # when deviation vectors align perfectly. Treat as below chaotic threshold.
    if sali_val <= 0.0:
        sali_val = SALI_CHAOTIC_THRESHOLD * 1e-4  # Store as ~1e-12, well below threshold

    if sali_val < SALI_CHAOTIC_THRESHOLD:
        regime = "chaotic"
    elif sali_val > SALI_REGULAR_THRESHOLD:
        regime = "regular"
    else:
        regime = "sticky"

    return {
        "sali_value": float(sali_val),
        "chaos_regime": regime,
        "n_iterations": n_iterations,
    }


# ============================================================================
# HDF5 packaging
# ============================================================================


def generate_dataset(config: DatasetConfig, verbose: bool = True) -> str:
    """
    Generate the full ChaosNetBench-CML dataset and save to HDF5.

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration
    verbose : bool
        Print progress

    Returns
    -------
    filepath : str
        Path to the generated HDF5 file
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / config.filename

    total = config.n_trajectories
    count = 0

    if verbose:
        print(f"ChaosNetBench Dataset Generator")
        print(f"  Configurations: {config.n_configs}")
        print(f"  Trajectories per config: {config.n_ics}")
        print(f"  Total trajectories: {total}")
        print(
            f"  Steps per trajectory: {config.n_steps} (+ {config.transient} transient)"
        )
        print(f"  Variable ordering: {config.ordering}")
        print(f"  Output: {filepath}")
        print()

    with h5py.File(filepath, "w") as f:
        # ---- Metadata ----
        meta = f.create_group("metadata")
        meta.attrs["system_name"] = "coupled_standard_map"
        meta.attrs["reference"] = (
            "Moges et al. (2022) — SALI chaos indicator"
        )
        meta.attrs["variable_ordering"] = config.ordering
        meta.attrs["n_steps"] = config.n_steps
        meta.attrs["transient"] = config.transient
        meta.attrs["base_seed"] = config.base_seed
        meta.attrs["seed_mode"] = config.seed_mode
        meta.attrs["n_ics"] = config.n_ics
        meta.attrs["version"] = "1.0"
        meta.create_dataset("K_values", data=config.K_values)
        meta.create_dataset("epsilon_values", data=config.epsilon_values)
        meta.create_dataset("N_values", data=config.N_values)

        if config.ordering == "blocked":
            # Document column labels
            meta.attrs["column_description"] = (
                "Columns: [x_1, x_2, ..., x_N, p_1, p_2, ..., p_N]. "
                "Positions first, then momenta. Both mod 2pi."
            )
        else:
            meta.attrs["column_description"] = (
                "Columns: [x_1, p_1, x_2, p_2, ..., x_N, p_N]. "
                "Position-momentum pairs interleaved. Both mod 2pi."
            )

        # ---- Adjacency matrices (one per N) ----
        adj_group = f.create_group("adjacency")
        for N in config.N_values:
            n_group = adj_group.create_group(f"N_{N:02d}")
            n_group.create_dataset("ring_NxN", data=build_site_adjacency(N))
            n_group.create_dataset("jacobian_2Nx2N", data=build_variable_adjacency(N))
            n_group.create_dataset(
                "coupling_only_2Nx2N", data=build_coupling_only_adjacency(N)
            )
            n_group.attrs["n_sites"] = N
            n_group.attrs["n_variables"] = 2 * N

        # ---- Trajectories ----
        traj_group = f.create_group("trajectories")
        diag_group = f.create_group("diagnostics")

        seed_counter = config.base_seed  # Used only for seed_mode="counter"

        for N in config.N_values:
            for K in config.K_values:
                # Use per-K epsilon list if available (RatioSweepConfig),
                # otherwise fall back to the global epsilon_values list.
                eps_list = (config.eps_for_K(K)
                            if hasattr(config, 'eps_for_K')
                            else config.epsilon_values)
                for epsilon in eps_list:
                    config_key = f"K_{K:.2f}_eps_{epsilon:.2f}_N_{N:02d}"
                    cfg_traj = traj_group.create_group(config_key)
                    cfg_diag = diag_group.create_group(config_key)

                    # Store config-level attributes
                    cfg_traj.attrs["K"] = K
                    cfg_traj.attrs["epsilon"] = epsilon
                    cfg_traj.attrs["N"] = N

                    for ic_idx in range(config.n_ics):
                        if config.seed_mode == "deterministic":
                            traj_seed = config_seed(config.base_seed, K, epsilon, N, ic_idx)
                        else:
                            seed_counter += 1
                            traj_seed = seed_counter

                        result = generate_trajectory(
                            N=N,
                            K=K,
                            epsilon=epsilon,
                            n_steps=config.n_steps,
                            transient=config.transient,
                            seed=traj_seed,
                            ordering=config.ordering,
                            compute_unwrapped=config.store_unwrapped,
                        )

                        # Store trajectory
                        ic_key = f"ic_{ic_idx:02d}"
                        ic_traj = cfg_traj.create_group(ic_key)
                        ic_traj.create_dataset(
                            "state_wrapped",
                            data=result["state_wrapped"],
                            compression="gzip",
                            compression_opts=4,
                        )
                        if "p_unwrapped" in result:
                            ic_traj.create_dataset(
                                "p_unwrapped",
                                data=result["p_unwrapped"],
                                compression="gzip",
                                compression_opts=4,
                            )
                        ic_traj.create_dataset(
                            "initial_conditions", data=result["initial_conditions"]
                        )
                        ic_traj.attrs["seed"] = traj_seed

                        # Store diagnostics
                        ic_diag = cfg_diag.create_group(ic_key)
                        ic_diag.attrs["diffusion_exponent"] = result[
                            "diffusion_exponent"
                        ]

                        # SALI-based orbit classification (if enabled)
                        if config.compute_diagnostics:
                            ic_x0 = result["initial_conditions"][:N]
                            ic_p0 = result["initial_conditions"][N:]
                            sali_result = classify_orbit_sali(
                                K=K,
                                epsilon=epsilon,
                                N=N,
                                x0=ic_x0,
                                p0=ic_p0,
                                n_iterations=config.diag_n_iterations,
                                n_transient=config.diag_n_transient,
                                seed=traj_seed,
                            )
                            regime = sali_result["chaos_regime"]
                            ic_diag.attrs["sali_screen_value"] = sali_result["sali_value"]
                            ic_diag.attrs["sali_screen_iterations"] = sali_result["n_iterations"]
                        else:
                            # Fallback: K-based heuristic (legacy)
                            regime = compute_regime_label(
                                K, epsilon, result["diffusion_exponent"]
                            )

                        ic_diag.attrs["chaos_regime"] = regime
                        if "mean_sq_displacement" in result:
                            ic_diag.create_dataset(
                                "mean_sq_displacement",
                                data=result["mean_sq_displacement"],
                                compression="gzip",
                                compression_opts=4,
                            )
                        # Placeholder for Fortran-computed Lyapunov/GALI
                        ic_diag.attrs["lambda_max"] = np.nan
                        ic_diag.attrs["lyapunov_spectrum_computed"] = False

                        count += 1
                        if verbose and (count % 5 == 0 or count == total):
                            print(
                                f"  [{count}/{total}] {config_key} ic={ic_idx} "
                                f"regime={regime} μ={result['diffusion_exponent']:.3f}"
                            )

    if verbose:
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"\nDataset saved: {filepath} ({file_size:.1f} MB)")
        print(f"Total trajectories: {count}")

    return str(filepath)


# ============================================================================
# Data loading for STGNN training
# ============================================================================


def load_stgnn_data(
    filepath: str,
    K: float,
    epsilon: float,
    N: int,
    seq_len: int = 96,
    pred_len: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    ic_indices: Optional[List[int]] = None,
    normalize: bool = True,
    use_sincos: bool = False,
) -> Dict:
    """
    Load CSM data from HDF5 in Lite-STGNN compatible format.

    Concatenates all requested trajectories and creates sliding window samples.
    Applies StandardScaler fitted on the training split only.

    Parameters
    ----------
    filepath : str
        Path to HDF5 dataset
    K, epsilon : float
        CSM parameters to load
    N : int
        System size
    seq_len : int
        Input window length
    pred_len : int
        Prediction horizon
    train_ratio, val_ratio : float
        Train/validation split ratios (test = 1 - train - val)
    ic_indices : list of int, optional
        Which IC indices to load (default: all)
    normalize : bool
        Whether to apply StandardScaler
    use_sincos : bool
        If True, encode angles θ → (sin θ, cos θ), doubling nodes from 2N to 4N.
        This resolves the 0/2π discontinuity for angular variables.

    Returns
    -------
    data : dict with keys:
        'train': np.ndarray [T_train, n_nodes]
        'val': np.ndarray [T_val, n_nodes]
        'test': np.ndarray [T_test, n_nodes]
        'scaler_mean': np.ndarray [n_nodes]
        'scaler_std': np.ndarray [n_nodes]
        'adjacency_2N': np.ndarray [n_nodes, n_nodes]  (2N×2N or 4N×4N)
        'adjacency_N': np.ndarray [N, N]
        'n_nodes': int  (2N or 4N)
        'config': dict
    """
    config_key = f"K_{K:.2f}_eps_{epsilon:.2f}_N_{N:02d}"

    with h5py.File(filepath, "r") as f:
        # Load adjacency
        adj_N = f[f"adjacency/N_{N:02d}/ring_NxN"][:]

        # Load trajectories
        traj_group = f[f"trajectories/{config_key}"]

        if ic_indices is None:
            ic_indices = list(range(len(traj_group)))

        all_data = []
        for ic_idx in ic_indices:
            ic_key = f"ic_{ic_idx:02d}"
            state = traj_group[ic_key]["state_wrapped"][:]
            all_data.append(state)

        # Concatenate all trajectories
        data = np.concatenate(all_data, axis=0)  # [T_total, 2N]

    # Apply sin/cos encoding if requested
    if use_sincos:
        # data is [T, 2N] with blocked [x₁,...,xₙ, p₁,...,pₙ]
        x_part = data[:, :N]  # [T, N] positions
        p_part = data[:, N:]  # [T, N] momenta (wrapped to [0, 2π))

        # Encode: θ → (sin θ, cos θ)
        # New ordering: [sin(x₁),...,sin(xₙ), cos(x₁),...,cos(xₙ),
        #                sin(p₁),...,sin(pₙ), cos(p₁),...,cos(pₙ)]
        data = np.concatenate(
            [
                np.sin(x_part),
                np.cos(x_part),
                np.sin(p_part),
                np.cos(p_part),
            ],
            axis=1,
        )  # [T, 4N]

        adj_2N = build_variable_adjacency_sincos(N)
        n_nodes = 4 * N
    else:
        adj_2N = build_variable_adjacency(N)
        n_nodes = 2 * N

    T = len(data)
    T_train = int(T * train_ratio)
    T_val = int(T * val_ratio)

    train_data = data[:T_train]
    val_data = data[T_train : T_train + T_val]
    test_data = data[T_train + T_val :]

    # StandardScaler (fit on train only)
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std < 1e-8] = 1.0  # prevent division by zero

    if normalize:
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std
        test_data = (test_data - mean) / std

    return {
        "train": train_data.astype(np.float32),
        "val": val_data.astype(np.float32),
        "test": test_data.astype(np.float32),
        "scaler_mean": mean.astype(np.float32),
        "scaler_std": std.astype(np.float32),
        "adjacency_2N": adj_2N,
        "adjacency_N": adj_N,
        "n_nodes": n_nodes,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "config": {"K": K, "epsilon": epsilon, "N": N, "use_sincos": use_sincos},
    }


def load_benchmark_data(
    filepath: str,
    K: float,
    epsilon: float,
    N: int,
    seq_len: int = 96,
    pred_len: int = 100,
    train_ic_indices: Optional[List[int]] = None,
    test_ic_indices: Optional[List[int]] = None,
    val_ratio: float = 0.125,
    normalize: bool = True,
    use_sincos: bool = False,
) -> Dict:
    """
    Load CSM data using the ChaosNetBench IC-based train/test split.

    Unlike load_stgnn_data, this uses DIFFERENT initial conditions for
    train vs test. Different initial conditions are allocated to train vs test,
    following best practices for chaotic systems.

    Parameters
    ----------
    filepath : str
        Path to HDF5 dataset
    K, epsilon : float
        CSM parameters to load
    N : int
        System size
    seq_len : int
        Input window length
    pred_len : int
        Prediction horizon (for sliding windows)
    train_ic_indices : list of int, optional
        IC indices for training (default: first 80%)
    test_ic_indices : list of int, optional
        IC indices for testing (default: last 20%)
    val_ratio : float
        Fraction of train ICs to use for validation
    normalize : bool
        Whether to apply StandardScaler
    use_sincos : bool
        If True, encode angles θ → (sin θ, cos θ)

    Returns
    -------
    data : dict with keys:
        'train': np.ndarray [T_train, n_nodes]
        'val': np.ndarray [T_val, n_nodes]
        'test': np.ndarray [T_test, n_nodes]
        'train_ics': list of IC indices used for training
        'test_ics': list of IC indices used for testing
        ... (same as load_stgnn_data)
    """
    config_key = f"K_{K:.2f}_eps_{epsilon:.2f}_N_{N:02d}"

    with h5py.File(filepath, "r") as f:
        # Load adjacency
        adj_N = f[f"adjacency/N_{N:02d}/ring_NxN"][:]

        # Get available ICs
        traj_group = f[f"trajectories/{config_key}"]
        n_ics = len(traj_group)

        # Default: 80% train, 20% test (different ICs)
        if train_ic_indices is None and test_ic_indices is None:
            n_train_ics = int(0.8 * n_ics)
            train_ic_indices = list(range(n_train_ics))
            test_ic_indices = list(range(n_train_ics, n_ics))
        elif train_ic_indices is None:
            # All non-test ICs for training
            train_ic_indices = [i for i in range(n_ics) if i not in test_ic_indices]
        elif test_ic_indices is None:
            # All non-train ICs for testing
            test_ic_indices = [i for i in range(n_ics) if i not in train_ic_indices]

        # Split train into train/val
        n_val = max(1, int(len(train_ic_indices) * val_ratio))
        val_ic_indices = train_ic_indices[-n_val:]
        train_ic_indices = train_ic_indices[:-n_val]

        # Load train trajectories
        train_data = []
        for ic_idx in train_ic_indices:
            ic_key = f"ic_{ic_idx:02d}"
            state = traj_group[ic_key]["state_wrapped"][:]
            train_data.append(state)
        train_data = np.concatenate(train_data, axis=0)

        # Load val trajectories
        val_data = []
        for ic_idx in val_ic_indices:
            ic_key = f"ic_{ic_idx:02d}"
            state = traj_group[ic_key]["state_wrapped"][:]
            val_data.append(state)
        val_data = np.concatenate(val_data, axis=0)

        # Load test trajectories (DIFFERENT ICs!)
        test_data = []
        test_ics_raw = []  # Per-IC trajectories for autoregressive eval
        for ic_idx in test_ic_indices:
            ic_key = f"ic_{ic_idx:02d}"
            state = traj_group[ic_key]["state_wrapped"][:]
            test_data.append(state)
            test_ics_raw.append(state.copy())
        test_data = np.concatenate(test_data, axis=0)

        # Load per-IC diagnostics (λ_max, SALI, regime) if available
        test_ic_diagnostics = {}
        diag_key = f"diagnostics/{config_key}"
        if diag_key in f:
            diag_group = f[diag_key]
            for ic_idx in test_ic_indices:
                ic_key = f"ic_{ic_idx:02d}"
                if ic_key in diag_group:
                    attrs = dict(diag_group[ic_key].attrs)
                    test_ic_diagnostics[ic_idx] = {
                        "lambda_max": float(attrs.get("lambda_max", np.nan)),
                        "sali_final": float(attrs.get("sali_final", np.nan)),
                        "is_chaotic": bool(attrs.get("is_chaotic", True)),
                        "chaos_regime": str(attrs.get("chaos_regime", "unknown")),
                    }
            # Also load config-level mean λ_max
            config_lambda_max = float(
                diag_group.attrs.get("lambda_max_mean", np.nan)
            )
        else:
            config_lambda_max = np.nan

    # Apply sin/cos encoding if requested
    def apply_sincos(data, N):
        x_part = data[:, :N]
        p_part = data[:, N:]
        return np.concatenate(
            [np.sin(x_part), np.cos(x_part), np.sin(p_part), np.cos(p_part)], axis=1
        )

    if use_sincos:
        train_data = apply_sincos(train_data, N)
        val_data = apply_sincos(val_data, N)
        test_data = apply_sincos(test_data, N)
        adj_2N = build_variable_adjacency_sincos(N)
        n_nodes = 4 * N
    else:
        adj_2N = build_variable_adjacency(N)
        n_nodes = 2 * N

    # StandardScaler (fit on train only)
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std < 1e-8] = 1.0

    if normalize:
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std
        test_data = (test_data - mean) / std

    # Normalize per-IC test trajectories too
    test_ics_normed = []
    for ic_data in test_ics_raw:
        if use_sincos:
            ic_data = apply_sincos(ic_data, N)
        ic_normed = (ic_data - mean) / std if normalize else ic_data
        test_ics_normed.append(ic_normed.astype(np.float32))

    return {
        "train": train_data.astype(np.float32),
        "val": val_data.astype(np.float32),
        "test": test_data.astype(np.float32),
        "test_ics_normed": test_ics_normed,  # Per-IC normalized trajectories
        "test_ic_diagnostics": test_ic_diagnostics,  # Per-IC chaos diagnostics
        "config_lambda_max": config_lambda_max,  # Config-level mean λ_max
        "train_ics": train_ic_indices,
        "val_ics": val_ic_indices,
        "test_ics": test_ic_indices,
        "scaler_mean": mean.astype(np.float32),
        "scaler_std": std.astype(np.float32),
        "adjacency_2N": adj_2N,
        "adjacency_N": adj_N,
        "n_nodes": n_nodes,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "config": {"K": K, "epsilon": epsilon, "N": N, "use_sincos": use_sincos},
        "split_type": "cnb_ic_split",
    }


# ============================================================================
# Dataset summary / inspection
# ============================================================================


def inspect_dataset(filepath: str) -> None:
    """Print summary of an HDF5 dataset."""
    with h5py.File(filepath, "r") as f:
        meta = f["metadata"]
        print("=" * 60)
        print("ChaosNetBench-CML Dataset Summary")
        print("=" * 60)
        print(f"System: {meta.attrs['system_name']}")
        print(f"Reference: {meta.attrs['reference']}")
        print(f"Version: {meta.attrs['version']}")
        print(f"Ordering: {meta.attrs['variable_ordering']}")
        print(f"Steps: {meta.attrs['n_steps']} (transient: {meta.attrs['transient']})")
        print(f"ICs per config: {meta.attrs['n_ics']}")
        print()

        K_values = meta["K_values"][:]
        eps_values = meta["epsilon_values"][:]
        N_values = meta["N_values"][:]
        print(f"K values: {K_values}")
        print(f"ε values: {eps_values}")
        print(f"N values: {N_values}")
        print(f"Configurations: {len(K_values) * len(eps_values) * len(N_values)}")
        print(
            f"Total trajectories: {len(K_values) * len(eps_values) * len(N_values) * meta.attrs['n_ics']}"
        )
        print()

        # Adjacency info
        print("Adjacency matrices:")
        for N in N_values:
            n_key = f"N_{int(N):02d}"
            adj = f[f"adjacency/{n_key}/jacobian_2Nx2N"][:]
            n_edges = int(adj.sum())
            print(f"  N={int(N):2d}: {2 * int(N)}×{2 * int(N)} matrix, {n_edges} edges")
        print()

        # Diagnostics summary
        print("Regime distribution:")
        regime_counts = {}
        for config_key in f["diagnostics"]:
            for ic_key in f[f"diagnostics/{config_key}"]:
                regime = f[f"diagnostics/{config_key}/{ic_key}"].attrs["chaos_regime"]
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
        for regime, count in sorted(regime_counts.items()):
            print(f"  {regime}: {count}")

        # File size
        print()
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"File size: {file_size:.1f} MB")
        print("=" * 60)


# Public alias


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChaosNetBench Dataset Generator")
    parser.add_argument(
        "--mode",
        choices=["mini", "full", "inspect"],
        default="mini",
        help="Generation mode: mini (quick test), full (complete benchmark dataset), or inspect existing",
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--filepath", default=None, help="Path for inspect mode")
    args = parser.parse_args()

    if args.mode == "inspect":
        path = args.filepath or os.path.join(args.output_dir, "chaosnetbench_cml_mini.h5")
        inspect_dataset(path)
    elif args.mode == "mini":
        config = MiniConfig(output_dir=args.output_dir)
        generate_dataset(config)
    elif args.mode == "full":
        config = BenchmarkConfig(output_dir=args.output_dir)
        generate_dataset(config)

