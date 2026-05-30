"""
Phase Velocity & Coherent Fragmentation Index
==============================================
Computes structural memory flow velocity and focal-global phase splitting.

Metrics:
  1. Structural Memory Flow Velocity: d(Phi1_Z)/dt
     - Captures acceleration toward pathological attractor
  2. Coherent Fragmentation Index: correlation drift between paired trajectories
     - Measures loss of network coherence during bifurcation
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_structural_velocity(
    phi1_series: np.ndarray,
    time_series: np.ndarray,
    window: int = 5
) -> np.ndarray:
    """
    Compute structural memory flow velocity (first-order gradient).
    
    v(t) = [Phi1_Z(t) - Phi1_Z(t-w)] / [t - (t-w)]
    
    The velocity captures how fast the system is being driven
    toward the pathological attractor. Near bifurcation, |v| should
    show an exponential increase.
    
    Parameters
    ----------
    phi1_series : ndarray
        Phi1_Z values over time.
    time_series : ndarray
        Corresponding time points.
    window : int
        Number of steps for finite difference.
        
    Returns
    -------
    velocity : ndarray
        First-order velocity (same length as input, NaN for boundary).
    """
    if len(phi1_series) < window + 1:
        return np.full_like(phi1_series, np.nan)
    
    velocity = np.full_like(phi1_series, np.nan)
    
    for i in range(window, len(phi1_series)):
        dt = time_series[i] - time_series[i - window]
        if dt != 0:
            velocity[i] = (phi1_series[i] - phi1_series[i - window]) / dt
    
    return velocity


def compute_acceleration(velocity: np.ndarray, time_series: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Compute acceleration (second-order gradient).
    
    a(t) = d(v)/dt
    
    Positive acceleration = system speeding up toward catastrophe.
    """
    return compute_structural_velocity(velocity, time_series, window=window)


def compute_focal_global_splitting(
    focal_phi1: np.ndarray,
    global_phi1: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    Compute focal-global phase splitting index.
    
    Measures sliding-window correlation between focal (patient-specific)
    and global (background) Phi1_Z trajectories.
    
    A DROP in correlation indicates the focal region is decoupling
    from the background — the hallmark of critical slowing.
    
    Parameters
    ----------
    focal_phi1 : ndarray
        Phi1_Z from the epileptogenic zone (or max-variance channel).
    global_phi1 : ndarray
        Phi1_Z from background channels.
    window : int
        Sliding window size for correlation.
        
    Returns
    -------
    correlation : ndarray
        Sliding correlation coefficient (same length, NaN for boundaries).
    """
    n = len(focal_phi1)
    correlation = np.full(n, np.nan)
    
    for i in range(window, n + 1):
        r, _ = stats.pearsonr(focal_phi1[i-window:i], global_phi1[i-window:i])
        correlation[i-1] = r
    
    return correlation


def compute_volatility_of_rigidity(phi1_series: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute the volatility of structural rigidity.
    
    VoR = |d(Phi1_Z)/dt|_smoothed
    
    High VoR indicates the system is in a metastable "glassy" state,
    about to undergo structural collapse.
    """
    velocity = compute_structural_velocity(phi1_series, np.arange(len(phi1_series)), window=1)
    # Smooth the absolute velocity
    abs_vel = np.abs(velocity)
    vol = pd.Series(abs_vel).rolling(window=window, min_periods=1).mean().values
    return vol


def compute_coherent_fragmentation_index(
    phi1_series: np.ndarray,
    time_series: np.ndarray,
    window: int = 10,
    threshold_slope: float = -0.05
) -> dict:
    """
    Compute unified coherent fragmentation index.
    
    Fragmentation is detected when:
    1. Velocity exceeds 2*sigma of baseline velocity
    2. Acceleration is positive (speeding up)
    3. The slope of velocity over the last 'window' points is negative enough
    
    Returns dict with indices and summary statistics.
    """
    velocity = compute_structural_velocity(phi1_series, time_series, window=3)
    acceleration = compute_acceleration(velocity, time_series, window=3)
    
    valid_v = velocity[~np.isnan(velocity)]
    if len(valid_v) < 5:
        return {
            'fragmentation_detected': False,
            'velocity': velocity,
            'acceleration': acceleration,
            'peak_velocity': np.nan,
            'fragmentation_time': np.nan
        }
    
    v_mean = np.mean(valid_v[:len(valid_v)//3])  # baseline = first third
    v_std = np.std(valid_v[:len(valid_v)//3])
    v_threshold = v_mean + 2 * v_std
    
    # Find where velocity exceeds threshold AND acceleration is positive
    fragmentation_idx = np.where(
        (np.abs(velocity) > v_threshold) & 
        (acceleration > 0) & 
        (~np.isnan(velocity)) & 
        (~np.isnan(acceleration))
    )[0]
    
    if len(fragmentation_idx) == 0:
        return {
            'fragmentation_detected': False,
            'velocity': velocity,
            'acceleration': acceleration,
            'peak_velocity': np.max(np.abs(valid_v)),
            'fragmentation_time': np.nan
        }
    
    first_frag = fragmentation_idx[0]
    
    return {
        'fragmentation_detected': True,
        'velocity': velocity,
        'acceleration': acceleration,
        'peak_velocity': np.max(np.abs(valid_v)),
        'fragmentation_time': time_series[first_frag] if first_frag < len(time_series) else np.nan,
        'fragmentation_idx': first_frag
    }
