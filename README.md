
# Constructive Free Energy Condensation Theory (CFECT)

> **A Non-Equilibrium Statistical Mechanics Framework for Tracking Macroscopic State Transitions in Living Networks**

[![CI](https://github.com/9981na/CFECT-Engine-XZ/actions/workflows/verification_ci.yml/badge.svg)](https://github.com/9981na/CFECT-Engine-XZ/actions/workflows/verification_ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/)

---

## 📋 Overview

CFECT introduces a **dual-axis inverse bifurcation** signature for detecting approaching critical transitions in living non-equilibrium networks. The framework operationalises systemic resilience as a complex order parameter Z = σ² + iΦ₁ on a Wang-Jin potential-flux landscape, resolving the clinical paradox where catastrophic physiological transitions occur without detectable structural abnormalities.

**Key empirical finding**: Across 187,181 time-series windows from 243 subjects spanning 3 organ systems (brain, heart, sleep-state), we demonstrate a universal pattern: lag-1 autocorrelation (Φ₁, Yang) increases while variance (σ², Yin) paradoxically condenses before critical transitions — the opposite of classical Critical Slowing Down predictions.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/9981na/CFECT-Engine-XZ.git
cd CFECT-Engine-XZ

# Install dependencies
pip install -r requirements.txt

# Run the three-gateway benchmark (Sleep-EDF real data)
python pipelines/run_three_gateways.py

# Or run the full CSD feature extraction pipeline
python pipelines/run_sleep_edf_csd.py
```

---

## 📊 Verified Core Results (v11 Manuscript)

| Metric | Value | 95% CI | Verification |
|--------|-------|--------|-------------|
| LMM Fixed Effect Φ₁ (Yang) | β = **+0.436** | [0.367, 0.505] | ✅ |
| LMM Fixed Effect σ² (Yin) | β = **−0.107** | [−0.138, −0.076] | ✅ |
| Pre-ictal Φ₁ shift (CHB-MIT) | ΔΦ₁ = **0.2359** | Cohen's d = 0.874 [0.802, 0.946] | ✅ |
| Pre-ictal σ² shift (CHB-MIT) | Δσ² = **−0.127** | Cohen's d = −0.532 [−0.614, −0.450] | ✅ |
| Topological crossover | **−18.32 min** | — | ✅ |
| N1 Sleep Stage F1 | **0.402** | Human inter-rater: [0.30, 0.45] | ✅ |
| REM-Wake Spectral Separation | **θ/α d = 1.84** | [1.62, 2.06] | ✅ |
| Negative Shear Zone (SDDB) | **ρ = −0.128** | [−0.167, −0.089] | ✅ |
| Cross-subject Macro F1 | **0.482** | — | ✅ |

---

## 📁 Datasets

| Database | System | Subjects | Windows | Transition |
|----------|--------|----------|---------|------------|
| [CHB-MIT Scalp EEG](https://physionet.org/content/chbmit/) | Brain (cortical) | 8 | 59,990 | Inter-ictal → Pre-ictal |
| [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/) | Brain (sleep) | 197 | 112,633 | W/N1/N2/N3/REM |
| [BUT-PDB ECG](https://physionet.org/content/butpdb/) | Heart (rhythm) | 15 | 12,485 | Sinus → AFIB/Bigeminy |
| [SDDB Holter](https://physionet.org/content/sddb/) | Heart (terminal) | 23 | 2,073 | Sinus → VF/Asystole |
| **Total** | **3 organ systems** | **243** | **187,181** | **6 transition types** |

All data sourced from [PhysioNet](https://physionet.org/) under ODC-By v1.0.

---

## 🏗️ Architecture

```
CFECT-Engine-XZ/
├── cfect_core/                  # Core physics solvers
│   ├── rolling_solver.py        # Vectorized sliding window CSD computations
│   ├── graybox_pinn.py          # Physics-informed neural network (Wang-Jin decomposition)
│   ├── path_decoder.py          # HMM-Viterbi path integral decoder
│   ├── fluctuation_theorem.py   # Gallavotti-Cohen fluctuation symmetry verification
│   ├── integrated_info.py       # Causal emergence ratio computation
│   └── spatial_ews.py           # Spatial early warning signal analysis
├── pipelines/                   # Data processing pipelines
│   ├── run_sleep_edf_csd.py     # Sleep-EDF feature extraction
│   ├── run_three_gateways.py    # Three-gateway benchmark suite
│   ├── sddb_extract_afib.py     # SDDB AFIB extraction
│   ├── sleep_edf_lmm_engine.py  # Linear mixed-model engine
│   └── chbmit_spatial_routing.py # CHB-MIT spatial analysis
├── statistics/                  # Statistical analysis
│   ├── lmm_evaluator.py         # Mixed-effects regression
│   ├── ols_trend_flow.py        # OLS trend regression & crossover detection
│   ├── fdr_chrono_bin.py        # FDR-corrected time binning
│   ├── critical_slowing_test.py # Classical CSD baseline
│   └── stationarity_prescreen.py# Stationarity verification
├── src/                         # Shared utilities
│   ├── features.py              # Feature extraction
│   ├── models.py                # ML models
│   └── phase_velocity.py        # Phase-space velocity computation
├── visualization/               # Publication-grade figures
│   ├── generate_nature_figures_v5.py  # Main figure generation (10 panels)
│   ├── cfect_multi_panel_viz.py       # Multi-panel visualisation
│   └── supplementary_plots.py         # Extended data plots
├── data/                        # Data management
│   ├── create_precomputed.py    # Precomputed feature generation
│   └── archive_cfect_data.py    # Data archiving
├── reproduce_all.py             # Legacy reproducibility verifier
├── reproducibility_harness.py   # Stress-testing & sensitivity analysis
└── requirements.txt             # Locked dependencies
```

---

## 🧠 Core Theoretical Framework

### Wang-Jin Potential-Flux Decomposition

For a biological network described by a state vector **X**(t):

$$\dot{\mathbf{X}} = \mathbf{f}(\mathbf{X}) + \boldsymbol{\xi}(t)$$

The steady-state probability distribution P_ss satisfies the stationary Fokker-Planck equation. The deterministic drift decomposes as:

$$\mathbf{f}(\mathbf{X}) = -\mathbf{D}\nabla U(\mathbf{X}) + \mathbf{v}(\mathbf{X})$$

where:
- **−D∇U**: Conservative dissipative force (Yin) — drives system toward potential minima
- **v(X)**: Non-conservative rotational force (Yang) — drives limit cycles & oscillations
- A non-zero flux velocity field (v ≠ 0) breaks detailed balance — the signature of living systems

### The CFECT Complex Order Parameter

Z(t) = σ²_Z(t) + iΦ₁_Z(t), projected onto the dominant slow manifold via:

$$\dot{Z} = (\alpha_{\text{eff}} + i\omega)Z - (1 + i\beta)|Z|^2Z + \Gamma(t) + \xi(t)$$

The dual-axis inverse bifurcation — σ²↓ (Yin condensation) coupled with Φ₁↑ (Yang rigidification) — constitutes a universal dynamical signature of approaching critical transitions in non-equilibrium biological networks.

---

## 🔬 Pipeline Verification

### One-command reproducibility check
```bash
python reproducibility_harness.py
```

### Pipeline-specific benchmarks
```bash
# Sleep staging benchmark (197 subjects)
python pipelines/run_three_gateways.py

# CSD feature extraction
python pipelines/run_sleep_edf_csd.py

# Spectral separation verification
python _verify_spectral_separation.py
```

---

## 📝 Citation

If you use this work in your research, please cite the associated manuscript:

```bibtex
@article{zhang2026cfect,
  author = {Zhang, Xu},
  title = {The Architecture of Non-Equilibrium Phase Transitions in Living Networks: 
           A Unified Potential-Flux Paradigm for Macroscopic State Tracking},
  journal = {Nature},
  year = {2026},
  note = {Under review}
}
```

For the software:

```bibtex
@software{cfect-engine-xz,
  author = {Zhang, Xu},
  title = {CFECT Engine: Non-Equilibrium Statistical Mechanics Framework for 
           Multiscale Electrophysiological Phase Transitions},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/9981na/CFECT-Engine-XZ},
  version = {1.0.0}
}
```

---

## 📜 License

This project is licensed under the Apache-2.0 License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on maintaining reproducibility and scientific integrity.

---

**Version**: 1.0.0  
**Status**: Manuscript under review at Nature
