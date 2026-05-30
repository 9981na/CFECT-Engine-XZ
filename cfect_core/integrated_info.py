"""
Integrated Information (Phi_E) Computation Module
==================================================
Implements DeepSeek RedTeam F3 requirement:
- Compare causal structure preservation before/after coarse-graining
- Phi_E(coarse) > 0.5 * Phi_E(full)

Uses nearest-neighbor entropy estimation (compatible with PyPhi philosophy
but lightweight for time series analysis).

Note: Full PyPhi installation is heavy and requires discrete state spaces.
This module implements continuous-valued Phi_E via effective information.

Author: CFECT Quantum Engine Team
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Dict, List, Optional, Tuple


def effective_info(
    time_series: np.ndarray,
    n_neighbors: int = 5,
    n_surrogates: int = 100
) -> Dict:
    """
    Compute effective information (Phi_E inspired) for a time series.
    
    Uses nearest-neighbor based entropy estimation to compute:
    Phi_E = I(X; Y) - sum(I(X_i; Y_i))  (simplified)
    
    Where I is mutual information estimated via kNN.
    
    Args:
        time_series: 1D or 2D time series (n_timepoints, n_dims)
        n_neighbors: Number of neighbors for entropy estimation
        n_surrogates: Number of surrogate shuffles for significance
    
    Returns:
        Dict with Phi_E value, significance, and details
    """
    if time_series.ndim == 1:
        ts = time_series.reshape(-1, 1)
    else:
        ts = time_series
    
    n, d = ts.shape
    if n < n_neighbors + 10:
        return {'phi_e': 0.0, 'significant': False,
                'details': 'Time series too short for Phi_E estimation'}
    
    # Split into two halves for mutual information
    mid = n // 2
    X = ts[:mid]
    Y = ts[mid:2*mid]
    
    # Ensure equal lengths
    min_len = min(len(X), len(Y))
    X = X[:min_len]
    Y = Y[:min_len]
    
    # Joint space [X, Y]
    XY = np.hstack([X, Y])
    
    # kNN entropy estimation for I(X; Y) = H(X) + H(Y) - H(X,Y)
    h_x = _knn_entropy(X, n_neighbors)
    h_y = _knn_entropy(Y, n_neighbors)
    h_xy = _knn_entropy(XY, n_neighbors)
    
    mi = h_x + h_y - h_xy
    
    # Surrogate testing
    surr_mis = []
    for s in range(n_surrogates):
        Y_shuffled = Y[np.random.permutation(len(Y))]
        XY_surr = np.hstack([X, Y_shuffled])
        h_y_surr = _knn_entropy(Y_shuffled, n_neighbors)
        h_xy_surr = _knn_entropy(XY_surr, n_neighbors)
        surr_mi = h_x + h_y_surr - h_xy_surr
        surr_mis.append(surr_mi)
    
    surr_mis = np.array(surr_mis)
    z_score = (mi - np.mean(surr_mis)) / (np.std(surr_mis) + 1e-10)
    p_value = 2 * (1 - _norm_cdf(abs(z_score)))
    significant = p_value < 0.05
    
    return {
        'phi_e': float(mi),
        'mi_value': float(mi),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'significant': bool(significant),
        'n_neighbors': n_neighbors,
        'n_surrogates': n_surrogates,
        'details': (
            f"Phi_E={mi:.4f} bits, z={z_score:.2f}, p={p_value:.4f}, "
            f"{'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'}"
        )
    }


def _knn_entropy(data: np.ndarray, k: int = 5) -> float:
    """
    k-nearest neighbor entropy estimator.
    
    Kozachenko-Leonenko estimator:
    H = psi(N) - psi(k) + d/N * sum(log(epsilon_i))
    
    where psi = digamma function, epsilon_i = distance to k-th neighbor
    """
    from scipy import special
    from scipy.spatial import KDTree
    
    n, d = data.shape
    if n <= k:
        return 0.0
    
    tree = KDTree(data)
    # Get distances to k-th nearest neighbor (excluding self)
    distances, _ = tree.query(data, k=k+1)
    epsilon_k = distances[:, -1]  # k-th neighbor distance
    epsilon_k = np.maximum(epsilon_k, 1e-12)
    
    h = special.digamma(n) - special.digamma(k) + d * np.mean(np.log(epsilon_k))
    
    # Add log volume of d-dimensional unit ball
    h += _log_unit_ball_volume(d)
    
    return float(h)


def _log_unit_ball_volume(d: int) -> float:
    """Log volume of d-dimensional unit ball."""
    from scipy import special
    return d/2 * np.log(np.pi) - special.gammaln(d/2 + 1)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    from scipy import stats
    return float(stats.norm.cdf(x))


def renorm(ts: np.ndarray, tau: int = 5) -> np.ndarray:
    """
    Coarse-graining via renormalization (time-domain averaging).
    
    Args:
        ts: 1D time series
        tau: Coarse-graining factor
    
    Returns:
        Coarse-grained time series
    """
    n = len(ts)
    m = n // tau
    if m == 0:
        return ts
    
    coarse = np.zeros(m)
    for i in range(m):
        coarse[i] = np.mean(ts[i*tau:(i+1)*tau])
    
    return coarse


def compute_phi_e(
    full_ts: np.ndarray,
    coarse_ts: Optional[np.ndarray] = None,
    tau: int = 5,
    n_surrogates: int = 100
) -> Dict:
    """
    Compare causal structure preservation before/after coarse-graining.
    
    Implements DeepSeek RedTeam F3:
    Phi_E(coarse) > 0.5 * Phi_E(full)
    
    Args:
        full_ts: Original (fine-grained) time series
        coarse_ts: Pre-computed coarse-grained series (optional)
        tau: Coarse-graining factor (used if coarse_ts is None)
        n_surrogates: Number of surrogates for significance
    
    Returns:
        Dict with Phi_E comparison and pass/fail verdict
    """
    if coarse_ts is None:
        coarse_ts = renorm(full_ts, tau)
    
    # Compute Phi_E for both
    phi_full = effective_info(full_ts, n_surrogates=n_surrogates)
    phi_coarse = effective_info(coarse_ts, n_surrogates=n_surrogates)
    
    phi_full_val = phi_full['phi_e']
    phi_coarse_val = phi_coarse['phi_e']
    
    # Check condition: Phi_E(coarse) > 0.5 * Phi_E(full)
    if phi_full_val > 0:
        ratio = phi_coarse_val / phi_full_val
        condition_met = ratio > 0.5
    else:
        ratio = 0.0
        condition_met = False
    
    return {
        'phi_e_full': phi_full_val,
        'phi_e_coarse': phi_coarse_val,
        'ratio': float(ratio),
        'tau': tau,
        'condition_met': bool(condition_met),
        'full_significant': phi_full['significant'],
        'coarse_significant': phi_coarse['significant'],
        'details': (
            f"Phi_E(full)={phi_full_val:.4f}, Phi_E(coarse)={phi_coarse_val:.4f}, "
            f"ratio={ratio:.4f}, "
            f"condition Phi_E(coarse) > 0.5*Phi_E(full): "
            f"{'PASS' if condition_met else 'FAIL'} "
            f"(threshold: ratio > 0.5)"
        )
    }
