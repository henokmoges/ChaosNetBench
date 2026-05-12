# ChaosNetBench Dataset

## Quick Summary

| Option | Size     | Description |
|--------|----------|-------------|
| HuggingFace (recommended) | ~27.3 GB | Full benchmark dataset, precomputed and ready to use |
| Mini (included) | ~8-12 MB | Small subset for smoke testing, no download required |

---

## Option A — Download from Hugging Face (Recommended)

The precomputed full benchmark dataset is hosted on Hugging Face:

```
https://huggingface.co/datasets/htmoges/chaosnetbench-cml
```

```bash
# Download using the HuggingFace Hub CLI
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='htmoges/chaosnetbench-cml', \
    filename='data/chaosnetbench_cml.h5', repo_type='dataset', \
    local_dir='data/')"
```

MD5 checksum: `<checksum published with dataset>`

Public metadata files include:

- `data/chaosnetbench_cml.croissant.json` for full-dataset Croissant metadata
- `data/multiseed_aggregated.csv` for a lightweight benchmark-results preview

---

## Option B — Mini Dataset (included, no download)

A small subset for smoke-testing the benchmark locally is bundled at
`data/chaosnetbench_cml_mini.h5` (K ∈ {0.5, 2.0}, ρ ∈ {0.10, 0.30}, N=8, 6 ICs).
No generation step is needed.

```bash
python scripts/train.py \
    --model dlinear --K 0.5 --rho 0.10 --N 8 --seed 42 \
    --dataset data/chaosnetbench_cml_mini.h5
```

---

## Dataset Format

The HDF5 file is organized into four top-level groups:

```
chaosnetbench_cml.h5
├── metadata/
│   ├── K_values        float64  (4,)    — kick strengths [0.5, 0.97, 2.0, 6.5]
│   ├── epsilon_values  float64  (8,)    — coupling strengths ε = ρ·K
│   └── N_values        int64    (3,)    — lattice sizes [8, 16, 32]
├── adjacency/
│   └── N_{NN}/
│       ├── ring_NxN        float32  (N, N)    — nearest-neighbor ring topology
│       └── jacobian_2Nx2N  float32  (2N, 2N)  — Jacobian-based (q,p) adjacency
├── trajectories/
│   └── K_{K:.2f}_eps_{ε:.2f}_N_{NN}/
│       └── ic_{ii}/
│           ├── state_wrapped       float64  (T, 2N)  — cols 0..N-1: q (mod 2π), N..2N-1: p
│           └── initial_conditions  float64  (2N,)    — [q₀₁,…,q₀ₙ, p₀₁,…,p₀ₙ]
└── diagnostics/
    └── K_{K:.2f}_eps_{ε:.2f}_N_{NN}/
        └── ic_{ii}/   (attrs: sali_screen_value, chaos_regime, lambda_max, …)
```

### Loading in Python

```python
from chaosnetbench.dataset import load_benchmark_data, inspect_dataset

# Inspect file structure
inspect_dataset("data/chaosnetbench_cml.h5")

# Load train/val/test splits for one (K, ε, N) instance
data = load_benchmark_data(
    "data/chaosnetbench_cml.h5",
    K=2.0, epsilon=0.40, N=8
)
# data["train"].shape  → (T_train, 16)  [2N=16 for N=8]
# Columns: 0..N-1  = q coordinates (angle, mod 2π)
#          N..2N-1 = p coordinates (momentum)
```
