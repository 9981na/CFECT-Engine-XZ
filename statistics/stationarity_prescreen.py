"""
Stationarity Prescreening Module
=================================
Augmented Dickey-Fuller test for stationarity verification.
Addresses Agent 2 criticism: MSE tau=5 violates Stam stationarity assumptions.

Author: CFECT Quantum Engine Team
"""

import numpy as np
from typing import Optional, Tuple, Dict


def adf_stationarity_test(series: np.ndarray, verbose: bool = False) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary
    
    Args:
        series: Time series to test
        verbose: Whether to print detailed results
    
    Returns:
        Dict with ADF stat, p-value, critical values, and verdict
    """
    from statsmodels.tsa.stattools import adfuller
    
    max_lag = min(int(len(series) ** 0.5), len(series) // 4)
    
    try:
        result = adfuller(series, maxlag=max_lag, autolag='AIC', regression='c')
        
        adf_stat = float(result[0])
        p_value = float(result[1])
        critical_values = {k: float(v) for k, v in result[4].items()}
        used_lag = int(result[2])
        n_obs = int(result[3])
        
        is_stationary = p_value < 0.05
        
        verdict = {
            'adf_statistic': round(adf_stat, 4),
            'p_value': round(p_value, 6),
            'critical_values': critical_values,
            'used_lag': used_lag,
            'n_obs': n_obs,
            'is_stationary': is_stationary,
            'reject_unit_root': is_stationary,
            'details': (
                f"ADF={adf_stat:.4f}, p={p_value:.6f}, "
                f"lag={used_lag}, n={n_obs}, "
                f"{'STATIONARY' if is_stationary else 'NON-STATIONARY'}"
            )
        }
    except Exception as e:
        verdict = {
            'adf_statistic': None,
            'p_value': None,
            'critical_values': {},
            'used_lag': None,
            'n_obs': len(series),
            'is_stationary': False,
            'reject_unit_root': False,
            'details': f"ADF test failed: {str(e)}"
        }
    
    if verbose:
        print(f"[Stationarity] {verdict['details']}")
    
    return verdict


def check_epoch_stationarity(
    signal: np.ndarray,
    epoch_length: int = 300,
    fs: int = 100,
    n_epochs_to_check: int = 10,
    verbose: bool = False
) -> Dict:
    """
    Check stationarity across multiple epochs of a signal.
    
    Args:
        signal: Full signal
        epoch_length: Length of each epoch in seconds
        fs: Sampling frequency
        n_epochs_to_check: Number of random epochs to check
        verbose: Whether to print results
    
    Returns:
        Dict with stationarity summary across epochs
    """
    samples_per_epoch = epoch_length * fs
    n_epochs = len(signal) // samples_per_epoch
    n_check = min(n_epochs_to_check, n_epochs)
    
    if n_check == 0:
        return {'stationary_epochs': 0, 'total_checked': 0, 'stationarity_ratio': 0.0}
    
    rng = np.random.RandomState(42)
    indices = rng.choice(n_epochs, size=n_check, replace=False)
    
    stationarity_results = []
    for idx in indices:
        epoch = signal[idx * samples_per_epoch:(idx + 1) * samples_per_epoch]
        result = adf_stationarity_test(epoch, verbose=False)
        stationarity_results.append(result['is_stationary'])
    
    n_stationary = sum(stationarity_results)
    ratio = n_stationary / n_check
    
    summary = {
        'stationary_epochs': n_stationary,
        'total_checked': n_check,
        'stationarity_ratio': round(ratio, 3),
        'all_stationary': ratio == 1.0,
        'mostly_stationary': ratio >= 0.8,
        'recommend_max_scale': int(epoch_length / 5) if ratio >= 0.8 else 1
    }
    
    if verbose:
        print(f"[Stationarity] {n_stationary}/{n_check} epochs stationary "
              f"(ratio={ratio:.3f}). "
              f"Recommended max MSE scale: {summary['recommend_max_scale']}")
    
    return summary


def adaptive_max_scale(signal: np.ndarray, fs: int = 100, verbose: bool = False) -> int:
    """
    Determine adaptive maximum MSE scale factor based on stationarity.
    
    More stationary -> larger max_scale is safe.
    Less stationary -> restrict max_scale to avoid confounds.
    
    Args:
        signal: 1D signal
        fs: Sampling frequency
        verbose: Whether to print results
    
    Returns:
        Recommended max_scale (1-20)
    """
    # Quick ADF test on full signal
    result = adf_stationarity_test(signal, verbose=False)
    
    if result['is_stationary']:
        # Stationary signal: allow up to scale 10
        recommended = min(10, len(signal) // 100)
    else:
        # Non-stationary signal: restrict to scale 2 (MSE tau=2)
        recommended = 2
    
    if verbose:
        print(f"[Adaptive Max Scale] Signal is "
              f"{'STATIONARY' if result['is_stationary'] else 'NON-STATIONARY'}, "
              f"max_scale recommended = {recommended}")
    
    return recommended
