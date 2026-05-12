"""
ChaosNetBench: Benchmarking Spatio-Temporal Graph Neural Networks on Chaotic Lattice Dynamics

Public benchmark code release
MIT License
"""

__version__ = "1.0.0"
__author__ = "H. T. Moges"

from chaosnetbench.systems.standard_map import CoupledStandardMap
from chaosnetbench.dataset import BenchmarkConfig, MiniConfig, load_benchmark_data, load_stgnn_data
from chaosnetbench.metrics import valid_prediction_time, autoregressive_rollout_np
from chaosnetbench.training import TrainConfig, ChaosNetBenchTrainer, ChaosBenchTrainer

__all__ = [
    "CoupledStandardMap",
    "BenchmarkConfig",
    "MiniConfig",
    "load_benchmark_data",
    "load_stgnn_data",
    "valid_prediction_time",
    "autoregressive_rollout_np",
    "TrainConfig",
    "ChaosNetBenchTrainer",
    "ChaosBenchTrainer",
]
