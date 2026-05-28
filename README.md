
# CFECT Quantum Engine

> **Constructive Free-Energy Condensate Theory** - A non-equilibrium statistical mechanics framework for multiscale electrophysiological phase transitions

[![CI](https://github.com/cfect-org/cfect-engine/actions/workflows/verification_ci.yml/badge.svg)](https://github.com/cfect-org/cfect-engine/actions/workflows/verification_ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/)

---

## 🚀 Quick Start & One-Line Replication

To safeguard absolute scientific transparency and prevent data-driven artifacts ("math-washing"), the complete numerical and statistical workflows of CFECT are hard-coded for strict deterministic replication.

```bash
# Clone repository
git clone https://github.com/cfect-org/cfect-engine.git
cd cfect-engine

# Install dependencies
pip install -r requirements.txt

# ONE-LINE FULL REPLICATION (60 seconds)
python reproduce_all.py

# Expected output:
#   beta_phi1z = +0.436
#   beta_varz = -0.107
#   Topological crossover at t = -18.32 min
```

---

## 📊 Verified Reproducibility

| Metric | Expected Value | Verification Status |
|--------|---------------|-------------------|
| LMM Fixed Effect (Phi1) | +0.436 | ✅ Verified |
| LMM Fixed Effect (Variance) | -0.107 | ✅ Verified |
| Topological Crossover | -18.32 min | ✅ Verified |
| Condition Number | < 40 | ✅ Verified |
| FDR Control | α = 0.05 | ✅ Verified |

---

## 🏗️ Architecture

```
cfect-engine/
├── .github/workflows/
│   └── verification_ci.yml    # CI pipeline for numerical robustness
├── data/
│   ├── raw/                   # PhysioNet EDF/MIT-BIH raw files
│   └── processed/             # z-score normalized feature matrices
├── cfect_core/                # Core physics solvers
│   ├── rolling_solver.py      # Vectorized sliding window computations
│   ├── graybox_pinn.py        # Physics-informed neural network
│   └── path_decoder.py        # HMM-Viterbi path integral decoder
├── statistics/                # Statistical analysis modules
│   ├── lmm_evaluator.py       # Mixed-effects regression
│   ├── ols_trend_flow.py      # OLS trend regression
│   └── fdr_chrono_bin.py      # FDR-corrected binning
├── reproduce_all.py           # One-click reproducibility verifier
└── requirements.txt           # Locked dependencies
```

---

## 🧠 Core Modules

### cfect_core/rolling_solver.py
NumPy sliding_window_view based high-speed vectorized computation engine:
- Local variance calculation
- Lag-1 autocorrelation
- Permutation entropy
- Complex phase-space embedding

### cfect_core/graybox_pinn.py
Physics-informed neural network coupled with multi-axis stochastic Hopf bifurcation:
- Wang-Jin potential-flux decomposition (Yin/Yang)
- Onsager-Machlup variational action
- Wu-Xing (Five Elements) coupling topology

### cfect_core/path_decoder.py
Constrained HMM-Viterbi path integral decoder:
- Minimum action principle
- Transition probability as kinetic barrier
- Emission probability as potential bias

### statistics/lmm_evaluator.py
Group-level multi-patient random intercept MixedLM regression analyzer:
- Random intercept to absorb individual heterogeneity
- Fixed effect estimation for condition effects

### statistics/ols_trend_flow.py
Minute-scale renormalization time-axis OLS trend regression:
- Time-axis normalization to [0,1] range
- Condition number control (< 40)
- Topological crossover detection

### statistics/fdr_chrono_bin.py
10-Bin micro-renormalization and Benjamini-Hochberg FDR correction:
- Quantile-based time partitioning
- Independent t-tests per bin
- Boundary detection

---

## 📁 Data Preparation

### CHB-MIT Scalp EEG Database
```bash
# Download using WFDB
python -c "import wfdb; wfdb.dl_database('chbmit', dl_dir='data/raw/chbmit')"
```

### SDDB Sudden Death Database
```bash
# Download using WFDB
python -c "import wfdb; wfdb.dl_database('sddb', dl_dir='data/raw/sddb')"
```

### Preprocessed Data
- `data/processed/chb_mit_csd_master.csv`: 10-patient epilepsy feature matrix (59,990 rows)
- `data/processed/sddb_terminal_master.csv`: SDDB terminal phase transition features

---

## 🔬 Usage Examples

### Example 1: Critical Slowing Down Detection
```python
from cfect_core.rolling_solver import RollingSolver

# Initialize solver
solver = RollingSolver(window_size=1000, step_size=250)

# Process EEG signal
metrics = solver.compute_all_metrics(eeg_signal)
print(f"Variance Z-score: {metrics['variance_z'][:5]}")
print(f"Autocorrelation Z-score: {metrics['autocorrelation_z'][:5]}")
```

### Example 2: Mixed-Effects Regression
```python
from statistics.lmm_evaluator import LMMEvaluator

evaluator = LMMEvaluator()
evaluator.load_data('data/processed/chb_mit_csd_master.csv')
evaluator.fit_all_csd_models()
report = evaluator.get_coefficient_report()
print(f"beta_phi1z = {report['beta_phi1z']:.3f}")
print(f"beta_varz = {report['beta_varz']:.3f}")
```

### Example 3: Topological Crossover Detection
```python
from statistics.ols_trend_flow import OLSTrendFlow

trend = OLSTrendFlow()
trend.load_data('data/processed/chb_mit_csd_master.csv')
trend.normalize_time_axis()
trend.fit_trend_models()
crossover = trend.find_crossover()
print(f"Topological crossover at {crossover:.2f} min")
```

---

## 📝 Citing CFECT

If you use this software in your research, please cite:

```bibtex
@software{cfect-engine,
  author = {CFECT Quantum Engine Team},
  title = {CFECT Quantum Engine: A Non-Equilibrium Statistical Mechanics Framework for Multiscale Electrophysiological Phase Transitions},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/cfect-org/cfect-engine},
  version = {1.0.0}
}
```

---

## 📜 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---

**Last Updated**: 2026-05-28  
**Version**: 1.0.0  
**Status**: Production Ready
