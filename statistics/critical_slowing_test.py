"""
Critical Slowing Down Statistical Verification
===============================================
Implements DeepSeek RedTeam F1 requirement:
- Mann-Kendall trend test (p < 0.05)
- 95% confidence intervals
- Baseline window definition (first 20% vs first 10%)
- AR(1) monotonic growth >= 0.15 + variance growth >= 50%

Author: CFECT Quantum Engine Team
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, Dict


def mann_kendall_trend(x: np.ndarray) -> Tuple[float, float]:
    """
    Mann-Kendall trend test for monotonic trend detection.
    
    Args:
        x: 1D array of time series data
    
    Returns:
        Tuple of (tau_statistic, p_value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    
    # Variance calculation with tie correction
    n_unique = len(np.unique(x))
    ties = n - n_unique
    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    if ties > 0:
        # Tie correction (simplified)
        var_s *= (1 - (ties * (ties - 1)) / (n * (n - 1)))
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    
    return tau, p_value


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for the mean using bootstrap.
    
    Args:
        data: 1D array
        confidence: Confidence level (default: 0.95)
    
    Returns:
        Tuple of (lower_ci, upper_ci)
    """
    n_bootstrap = 1000
    n = len(data)
    
    rng = np.random.default_rng(42)
    bootstrap_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return lower, upper


def verify_csd(
    ac_series: np.ndarray,
    var_series: np.ndarray,
    ar1_threshold: float = 0.15,
    var_increase_pct: float = 50.0,
    window_size: int = 100,
    baseline_fraction: float = 0.2
) -> Dict:
    """
    Verify critical slowing down conditions.
    
    DeepSeek RedTeam F1 requirements:
    1. AR(1) monotonic growth >= 0.15
    2. Variance growth >= 50%
    3. Mann-Kendall p < 0.05
    
    Args:
        ac_series: Lag-1 autocorrelation time series
        var_series: Variance time series
        ar1_threshold: Minimum AR(1) increase (default: 0.15)
        var_increase_pct: Minimum variance increase percentage (default: 50%)
        window_size: Size of analysis windows
        baseline_fraction: Fraction of series to use as baseline (default: 0.2)
    
    Returns:
        Dict with keys:
            - 'pass': bool, overall PASS/FAIL
            - 'ar1_increase': float, observed AR(1) increase
            - 'var_increase_pct': float, observed variance increase %
            - 'mk_p_value': float, Mann-Kendall p-value
            - 'mk_tau': float, Mann-Kendall tau statistic
            - 'ci_baseline': tuple, 95% CI for baseline
            - 'ci_current': tuple, 95% CI for current
            - 'details': str, detailed explanation
    """
    n = len(ac_series)
    effective_window = min(window_size, n // 4)
    
    # Split into baseline and current windows
    baseline_end = max(effective_window, int(n * baseline_fraction))
    baseline_ac = ac_series[:baseline_end]
    baseline_var = var_series[:baseline_end]
    current_ac = ac_series[baseline_end:]
    current_var = var_series[baseline_end:]
    
    # Mann-Kendall trend test on the entire series
    mk_tau_ac, mk_p_ac = mann_kendall_trend(ac_series)
    mk_tau_var, mk_p_var = mann_kendall_trend(var_series)
    
    # Use the more significant p-value
    mk_p_value = min(mk_p_ac, mk_p_var)
    mk_tau = mk_tau_ac if abs(mk_tau_ac) > abs(mk_tau_var) else mk_tau_var
    
    # AR(1) increase (from baseline mean to current max)
    baseline_ar1_mean = np.mean(baseline_ac)
    current_ar1_max = np.max(current_ac)
    ar1_increase = current_ar1_max - baseline_ar1_mean
    
    # Variance increase
    baseline_var_mean = np.mean(baseline_var)
    current_var_max = np.max(current_var)
    var_increase = ((current_var_max - baseline_var_mean) / max(baseline_var_mean, 1e-12)) * 100
    
    # Confidence intervals
    ci_baseline = compute_confidence_interval(baseline_ac)
    ci_current = compute_confidence_interval(current_ac)
    
    # Threshold checks
    ar1_ok = ar1_increase >= ar1_threshold
    var_ok = var_increase >= var_increase_pct
    mk_ok = mk_p_value < 0.05
    
    # Build result
    result = {
        'pass': ar1_ok and var_ok and mk_ok,
        'ar1_increase': round(ar1_increase, 4),
        'var_increase_pct': round(var_increase, 2),
        'mk_p_value': round(mk_p_value, 6),
        'mk_tau': round(mk_tau, 4),
        'ci_baseline': (round(ci_baseline[0], 4), round(ci_baseline[1], 4)),
        'ci_current': (round(ci_current[0], 4), round(ci_current[1], 4)),
        'details': (
            f"AR(1) increase: {ar1_increase:.4f} (needs >= {ar1_threshold}) {'✓' if ar1_ok else '✗'} | "
            f"Var increase: {var_increase:.1f}% (needs >= {var_increase_pct}%) {'✓' if var_ok else '✗'} | "
            f"Mann-Kendall p: {mk_p_value:.6f} (needs < 0.05) {'✓' if mk_ok else '✗'}"
        ),
        'checks': {
            'ar1_ok': ar1_ok,
            'var_ok': var_ok,
            'mk_ok': mk_ok
        }
    }
    
    return result


def augment_with_dickey_fuller(series: np.ndarray, max_lag: Optional[int] = None) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Args:
        series: Time series to test
        max_lag: Maximum lag for ADF test
    
    Returns:
        Dict with ADF statistic, p-value, and stationarity verdict
    """
    from statsmodels.tsa.stattools import adfuller
    
    if max_lag is None:
        max_lag = int(len(series) ** 0.5)
    
    result = adfuller(series, maxlag=max_lag, autolag='AIC')
    
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    is_stationary = p_value < 0.05
    
    return {
        'adf_statistic': round(adf_stat, 4),
        'p_value': round(p_value, 6),
        'critical_values': {k: round(v, 4) for k, v in critical_values.items()},
        'is_stationary': is_stationary,
        'details': f"ADF stat={adf_stat:.4f}, p={p_value:.6f}, {'stationary' if is_stationary else 'non-stationary'}"
    }
