"""
Reference:
  Moges et al. (2022) Physica D: "Anomalous diffusion in single and coupled standard maps with extensive chaotic phase spaces"

NumPy implementation of the Coupled Standard Map (Chirikov-Taylor map on a ring lattice)
for the ChaosNetBench benchmark. See the CNB paper (Eq. 1) for dynamics.
"""

import numpy as np
from typing import Tuple, Optional


class CoupledStandardMap:
    """
    Coupled Chirikov-Taylor standard map on a ring lattice.

    Dynamics (Eq. 1 of the CNB paper):
        p_{n+1}^{(i)} = p_n^{(i)} + K*sin(q_n^{(i)})
                        - ε[sin(q_n^{(i+1)} - q_n^{(i)}) + sin(q_n^{(i-1)} - q_n^{(i)})]
        q_{n+1}^{(i)} = q_n^{(i)} + p_{n+1}^{(i)}  (mod 2π)

    where n is the map iteration (discrete time) and i = 1,...,N indexes sites on the ring.

    Parameters
    ----------
    N : int
        Number of lattice sites (default: 50)
    K : float
        Nonlinearity parameter (default: 1.0)
        - K = 0.5:  Near-integrable (KAM tori mostly intact)
        - K = 0.97: Mixed phase space, regular dominant
        - K = 2.0:  Mixed phase space, chaos dominant
        - K = 6.5:  Extended chaos
    epsilon : float
        Coupling strength (default: 0.2)

    Attributes
    ----------
    adjacency : np.ndarray
        Ground-truth ring topology (N x N sparse matrix)
    lambda_max : float
        finite-time Maximum Lyapunov exponent (computed if diagnostics are run)

    References
    ----------
    [1] Chirikov, B.V. (1979). A universal instability of many-dimensional
        oscillator systems. Physics Reports, 52(5), 263-379.
    [2] Kantz, H. and Grassberger, P. (1988). Internal Arnold diffusion and
        chaos thresholds in coupled symplectic maps.
    [3] Moges, et.al, Physica D (2022). Anomalous diffusion in single and coupled standard maps with extensive chaotic phase spaces.
    """

    def __init__(
        self,
        N: int = 50,
        K: float = 1.0,
        epsilon: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.N = N
        self.K = K
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Ground-truth ring adjacency (for graph learning evaluation)
        self.adjacency = self._build_ring_adjacency()

        # Chaos diagnostics (computed during integration)
        self.lambda_max = None
        self.gali = None

    def _build_ring_adjacency(self) -> np.ndarray:
        """Construct ring topology adjacency matrix."""
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for i in range(self.N):
            A[i, (i + 1) % self.N] = 1.0  # Right neighbor
            A[i, (i - 1) % self.N] = 1.0  # Left neighbor
        return A

    def step(self, x: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single timestep evolution.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Position coordinates q (mod 2π)
        p : np.ndarray, shape (N,)
            Momentum coordinates

        Returns
        -------
        x_new : np.ndarray, shape (N,)
        p_new : np.ndarray, shape (N,)
        """
        N = self.N

        # Compute coupling forces
        # -ε·[sin(q_{n}^{(i+1)} - q_{n}^{(i)}) + sin(q_{n}^{(i-1)} - q_{n}^{(i)})]
        # Eq. 1 of the CNB paper; Chirikov (1979)
        coupling_force = np.zeros(N)
        for i in range(N):
            i_next = (i + 1) % N
            i_prev = (i - 1) % N
            coupling_force[i] = -self.epsilon * (
                np.sin(x[i_next] - x[i]) + np.sin(x[i_prev] - x[i])
            )

        # Update momentum
        p_new = p + self.K * np.sin(x) + coupling_force

        # Update position (mod 2π)
        x_new = np.mod(x + p_new, 2 * np.pi)

        return x_new, p_new

    def integrate(
        self, x0: np.ndarray, p0: np.ndarray, n_steps: int, transient: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate trajectory for n_steps.

        Parameters
        ----------
        x0 : np.ndarray, shape (N,)
            Initial positions q₀
        p0 : np.ndarray, shape (N,)
            Initial momenta
        n_steps : int
            Number of timesteps to integrate
        transient : int
            Number of initial steps to discard (default: 1000)

        Returns
        -------
        x_traj : np.ndarray, shape (n_steps, N)
        p_traj : np.ndarray, shape (n_steps, N)
        """
        x_traj = np.zeros((n_steps, self.N))
        p_traj = np.zeros((n_steps, self.N))

        x, p = x0.copy(), p0.copy()

        # Discard transient
        for _ in range(transient):
            x, p = self.step(x, p)

        # Record trajectory
        for t in range(n_steps):
            x_traj[t] = x
            p_traj[t] = p
            x, p = self.step(x, p)

        return x_traj, p_traj

    def random_initial_conditions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random initial conditions.

        Returns
        -------
        x0 : np.ndarray, shape (N,)
        p0 : np.ndarray, shape (N,)
        """
        x0 = self.rng.uniform(0, 2 * np.pi, size=self.N)
        p0 = self.rng.uniform(-np.pi, np.pi, size=self.N)
        return x0, p0

    def compute_cumulative_max_error(
        self, x_true: np.ndarray, x_pred: np.ndarray
    ) -> np.ndarray:
        """
        Cumulative maximum error (Schötz et al. 2024).

        CME(T) = max_{t ≤ T} ||x_pred(t) - x_true(t)||

        Parameters
        ----------
        x_true : np.ndarray, shape (T, N)
        x_pred : np.ndarray, shape (T, N)

        Returns
        -------
        cme : np.ndarray, shape (T,)
            Cumulative maximum error at each timestep
        """
        errors = np.linalg.norm(x_pred - x_true, axis=1)
        cme = np.maximum.accumulate(errors)
        return cme

    def __repr__(self) -> str:
        regime = (
            "near-integrable"
            if self.K <= 0.5
            else ("mixed (regular dominant)" if self.K <= 1.2 else
                  ("mixed (chaos dominant)" if self.K <= 4.0 else "extended chaos"))
        )
        return (
            f"CoupledStandardMap(N={self.N}, K={self.K}, ε={self.epsilon})\n"
            f"  Regime: {regime}\n"
            f"  Ring topology: {self.N} nodes, 2-nearest-neighbor"
        )


def generate_dataset(
    N: int = 50,
    K_values: list = [0.5, 1.0, 3.0],
    epsilon_values: list = [0.1, 0.2, 0.3],
    n_trajectories: int = 100,
    n_steps: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Generate benchmark dataset for Paper 1.

    Parameters
    ----------
    N : int
        Lattice size
    K_values : list
        Nonlinearity parameters (chaos control)
    epsilon_values : list
        Coupling strengths
    n_trajectories : int
        Number of trajectories per (K, ε) pair
    n_steps : int
        Trajectory length
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dataset : dict
        {
            'trajectories': list of (x_traj, p_traj, K, epsilon),
            'adjacency': ground-truth ring matrix,
            'metadata': parameter info
        }
    """
    rng = np.random.default_rng(seed)

    trajectories = []

    for K in K_values:
        for epsilon in epsilon_values:
            sm = CoupledStandardMap(
                N=N,
                K=K,
                epsilon=epsilon,
                seed=int(rng.integers(0, 2**32 - 1)),
            )

            for _ in range(n_trajectories):
                x0, p0 = sm.random_initial_conditions()
                x_traj, p_traj = sm.integrate(x0, p0, n_steps)

                trajectories.append(
                    {"x": x_traj, "p": p_traj, "K": K, "epsilon": epsilon, "N": N}
                )

    dataset = {
        "trajectories": trajectories,
        "adjacency": CoupledStandardMap(N=N).adjacency,
        "metadata": {
            "N": N,
            "K_values": K_values,
            "epsilon_values": epsilon_values,
            "n_trajectories_per_config": n_trajectories,
            "n_steps": n_steps,
            "system": "coupled_standard_map",
            "reference": "Moges et al. (2022), Physica D",
        },
    }

    return dataset
