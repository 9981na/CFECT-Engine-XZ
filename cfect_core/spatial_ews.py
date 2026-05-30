"""
Spatial Early Warning Signals Module
======================================
Implements spatial extension of critical slowing down detection.
Addresses DeepSeek RedTeam observation about spatial EWS being absent.

Spatial EWS indicators:
1. Moran's I (spatial autocorrelation)
2. Spatial variance
3. Cross-correlation length
4. Spatial skewness/kurtosis

Author: CFECT Quantum Engine Team
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple


def spatial_variance(data_matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Compute spatial variance across channels/sensors.
    
    Args:
        data_matrix: (n_timepoints, n_channels) or (n_channels, n_timepoints)
        axis: Axis along channels (0=channels in rows, 1=channels in columns)
    
    Returns:
        Spatial variance time series
    """
    return np.var(data_matrix, axis=axis)


def morans_i(data_matrix: np.ndarray, 
             connectivity: Optional[np.ndarray] = None,
             axis: int = 1,
             window: int = 10) -> np.ndarray:
    """
    Compute Moran's I (spatial autocorrelation) over time.
    
    Uses spatial adjacency to measure how similar neighboring channels are.
    High I = strong spatial structure (early warning for spatial critical transitions).
    
    Args:
        data_matrix: (n_timepoints, n_channels) or (n_channels, n_timepoints)
        connectivity: Adjacency matrix (n_channels, n_channels). If None, uses
                      sequential adjacency (ring topology).
        axis: 1 if time x channels, 0 if channels x time
        window: Rolling window for local Moran's I estimation
    
    Returns:
        Moran's I time series (length = n_timepoints - window + 1)
    """
    if axis == 0:
        data_matrix = data_matrix.T
    
    n_time, n_chan = data_matrix.shape
    
    if connectivity is None:
        # Sequential ring topology: each channel connected to neighbors
        connectivity = np.zeros((n_chan, n_chan))
        for i in range(n_chan):
            connectivity[i, (i-1) % n_chan] = 1
            connectivity[i, (i+1) % n_chan] = 1
    
    # Row-normalize connectivity
    row_sums = connectivity.sum(axis=1, keepdims=True)
    connectivity = connectivity / np.maximum(row_sums, 1e-10)
    
    # Nearest-neighbor connectivity: sum of absolute differences
    W = connectivity + connectivity.T
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-10)
    
    n_windows = n_time - window + 1
    morans_i_vals = np.zeros(n_windows)
    
    for t in range(n_windows):
        segment = data_matrix[t:t+window, :]
        
        # Center the data
        z = segment - np.mean(segment, axis=0, keepdims=True)
        
        # Compute Moran's I for each channel, then average
        n = n_chan
        s0 = np.sum(W)
        
        if s0 == 0 or np.std(z) < 1e-10:
            morans_i_vals[t] = 0.0
            continue
        
        # Simplified: compute spatial autocorrelation as mean pairwise correlation
        z_std = np.std(z, axis=0)
        z_std = np.maximum(z_std, 1e-10)
        z_norm = z / z_std
        
        # Weighted sum of neighbor products (vectorized)
        # Avoids O(n² * n_chan²) nested loop — uses matrix multiplication
        z_corr = z_norm.T @ z_norm  # (n_chan, n_chan) correlation matrix
        weighted_sum = np.sum(W * z_corr)
        
        # Normalize
        total_var = np.sum(z_norm ** 2)
        morans_i_vals[t] = (n / s0) * weighted_sum / max(total_var, 1e-10)
    
    return morans_i_vals


def spatial_skewness_kurtosis(data_matrix: np.ndarray, axis: int = 1) -> Dict:
    """
    Compute spatial skewness and kurtosis across channels.
    
    Increased skewness/kurtosis can indicate approaching critical transition.
    
    Args:
        data_matrix: (n_timepoints, n_channels) or (n_channels, n_timepoints)
        axis: 1 if time x channels
    
    Returns:
        Dict with spatial skewness and kurtosis time series
    """
    if axis == 0:
        data_matrix = data_matrix.T
    
    skewness = stats.skew(data_matrix, axis=1)
    kurtosis = stats.kurtosis(data_matrix, axis=1, fisher=True)  # Excess kurtosis
    
    return {
        'spatial_skewness': skewness,
        'spatial_kurtosis': kurtosis,
        'skewness_mean': float(np.mean(skewness)),
        'skewness_std': float(np.std(skewness)),
        'kurtosis_mean': float(np.mean(kurtosis)),
        'kurtosis_std': float(np.std(kurtosis))
    }


def verify_spatial_ews(
    data_matrix: np.ndarray,
    use_mann_kendall: bool = True
) -> Dict:
    """
    Full spatial early warning signals verification.
    
    Checks for:
    1. Increasing spatial variance (precursor to spatial critical transition)
    2. Increasing Moran's I (increasing spatial autocorrelation)
    3. Spatial skewness/kurtosis trends
    
    Args:
        data_matrix: (n_timepoints, n_channels)
        use_mann_kendall: Whether to use Mann-Kendall trend test
    
    Returns:
        Dict with all spatial EWS indicators and verdict
    """
    from statistics.critical_slowing_test import mann_kendall_trend
    
    var_series = spatial_variance(data_matrix)
    skew_kurt = spatial_skewness_kurtosis(data_matrix)
    morans_i_series = morans_i(data_matrix, window=min(10, data_matrix.shape[1]//2))
    
    results = {}
    
    # 1. Spatial variance trend
    if use_mann_kendall:
        tau, p_val = mann_kendall_trend(var_series)
        var_trend = {'tau': tau, 'p_value': p_val, 'significant': p_val < 0.05 and abs(tau) > 0.1}
    else:
        slope = np.polyfit(np.arange(len(var_series)), var_series, 1)[0]
        var_trend = {'tau': slope, 'p_value': 0.0, 'significant': abs(slope) > 0.01}
    results['spatial_variance'] = var_trend
    
    # 2. Moran's I trend
    if len(morans_i_series) > 5:
        if use_mann_kendall:
            tau, p_val = mann_kendall_trend(morans_i_series)
            morans_trend = {'tau': tau, 'p_value': p_val, 'significant': p_val < 0.05 and abs(tau) > 0.1}
        else:
            slope = np.polyfit(np.arange(len(morans_i_series)), morans_i_series, 1)[0]
            morans_trend = {'tau': slope, 'p_value': 0.0, 'significant': abs(slope) > 0.01}
        results['morans_i'] = morans_trend
    
    # 3. Skewness/Kurtosis
    results['spatial_skewness'] = skew_kurt
    
    # Verdict
    var_ok = var_trend.get('significant', False)
    morans_ok = results.get('morans_i', {}).get('significant', False)
    
    checks_passed = sum([var_ok, morans_ok])
    
    results['verdict'] = {
        'pass': checks_passed >= 1,
        'checks_passed': checks_passed,
        'checks_total': 2,
        'spatial_variance_trend_ok': var_ok,
        'morans_i_trend_ok': morans_ok,
        'details': (
            f"Spatial EWS: variance_trend={'OK' if var_ok else 'FAIL'}, "
            f"morans_i={'OK' if morans_ok else 'FAIL'}, "
            f"overall: {'PASS' if checks_passed >= 1 else 'FAIL'}"
        )
    }
    
    return results
