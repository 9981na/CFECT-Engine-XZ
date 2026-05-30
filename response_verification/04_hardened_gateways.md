# Hardened Gateway Modules — API Reference

---

## 1. `statistics/critical_slowing_test.py`

**Purpose**: Mann-Kendall trend test + confidence interval + threshold verification for critical slowing down (CSD) indicators.

```python
from statistics.critical_slowing_test import verify_csd

result = verify_csd(
    ac_series,          # array-like: lag-1 autocorrelation over time
    var_series,         # array-like: variance over time
    ar1_threshold=0.15, # minimum monotonic AR(1) growth
    var_increase_pct=50,# minimum variance increase %
    window_size=100,    # sliding window size
    alpha=0.05          # significance level
)
# Returns dict with: passed, p_value_trend, ci_lower, ci_upper,
#                    ar1_growth, var_increase_pct
```

**Validation command**: `python -c "from statistics.critical_slowing_test import verify_csd; help(verify_csd)"`

---

## 2. `statistics/stationarity_prescreen.py`

**Purpose**: Augmented Dickey-Fuller (ADF) test to verify stationarity before CSD analysis. Addresses Agent 2's criticism.

```python
from statistics.stationarity_prescreen import check_stationarity

result = check_stationarity(
    series,         # array-like
    method='adf',   # 'adf' | 'kpss'
    alpha=0.05
)
# Returns dict with: is_stationary, adf_statistic, p_value,
#                    critical_values
```

**Validation command**: `python -c "from statistics.stationarity_prescreen import check_stationarity; help(check_stationarity)"`

---

## 3. `cfect_core/fluctuation_theorem.py`

**Purpose**: Gallavotti-Cohen fluctuation theorem verification for complex embedding Z = X + iY.

```python
from cfect_core.fluctuation_theorem import verify_gallavotti_cohen

result = verify_gallavotti_cohen(
    trajectory_pairs,   # list of (forward, backward) trajectory pairs
    beta=1.0,           # inverse temperature
    n_bins=50           # histogram bins for KS test
)
# Returns dict with: ks_statistic, p_value, entropy_production_rate,
#                    passes_symmetry_check
```

**Note**: This module is available for optional verification. The manuscript has been updated to retract "thermodynamic consistency" in favor of "complex dynamical coordinate parameterization."

---

## 4. `cfect_core/integrated_info.py`

**Purpose**: Φ_E (Effective Information) computation to compare causal structure preservation under coarse-graining.

```python
from cfect_core.integrated_info import compute_phi_e

result = compute_phi_e(
    full_ts,            # original time series
    coarse_ts=None,     # pre-computed coarse-grained (auto if None)
    tau=5,              # coarse-graining factor
    n_surrogates=1000,  # surrogate count for significance
)
# Returns dict with: phi_full, phi_coarse, phi_ratio,
#                    passes_threshold (phi_ratio > 0.5)
```

---

## 5. `cfect_core/spatial_ews.py`

**Purpose**: Spatial early warning signals — Moran's I, spatial variance, skewness across multi-channel data.

```python
from cfect_core.spatial_ews import compute_spatial_ews

result = compute_spatial_ews(
    data_2d,        # (n_channels, n_timepoints)
    method='moran_i'
)
# Returns dict with: moran_i, spatial_variance, skewness,
#                    p_value (for Moran's I)
```

---

## 6. `pipelines/n1_baseline_comparison.py`

**Purpose**: N1 spectral baseline comparison — theta/delta ratio vs CFECT Φ₁_Z, comparing against human inter-rater F1.

```python
from pipelines.n1_baseline_comparison import compare_n1_baseline

result = compare_n1_baseline(
    sleep_edf_csv_path,     # path to sleep_csd_features.csv
    n_surrogates=1000
)
# Returns dict with: n1_f1_from_features, theta_delta_baseline_f1,
#                    human_interrater_f1_benchmark
```

---

## Verification Command Cheatsheet

```bash
# Quick module smoke tests (no data required)
python -c "from statistics.critical_slowing_test import verify_csd; print('✅ critical_slowing_test loaded')"
python -c "from statistics.stationarity_prescreen import check_stationarity; print('✅ stationarity_prescreen loaded')"
python -c "from cfect_core.fluctuation_theorem import verify_gallavotti_cohen; print('✅ fluctuation_theorem loaded')"
python -c "from cfect_core.integrated_info import compute_phi_e; print('✅ integrated_info loaded')"
python -c "from cfect_core.spatial_ews import compute_spatial_ews; print('✅ spatial_ews loaded')"

# End-to-end data verification
python -c "
import pandas as pd
df = pd.read_csv(r'E:\MEM\paper\real\output2\main\sleep_csd_features.csv')
print(f'Sleep-EDF: {len(df):,} windows, {df.columns.tolist()}')
print(f'SC Wake Phi1_Z: {df[(df.Study_Type==\"SC\") & (df.Sleep_Stage==\"W\")][\"Phi1_Z\"].mean():.4f}')
print(f'ST Placebo Wake Phi1_Z: {df[(df.Study_Type==\"ST\") & (df.Drug_Type==\"Placebo\") & (df.Sleep_Stage==\"W\")][\"Phi1_Z\"].mean():.4f}')
"
```
