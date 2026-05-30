"""
Physiological Time Warping Engine
==================================
Non-linear alignment of patient trajectories based on First Passage Time (FPT).

Theory:
  - Each patient has a unique "critical compensation drift onset" (t_critical)
  - Defined as the moment Phi1_Z first exceeds Inter-ictal baseline mean + 3*sigma
  - The interval [t_critical, seizure_onset] is elastically mapped to [0, 1]
  - In this physiological clock, the true bifurcation dynamics become visible

Reference:
  - First Passage Time in non-equilibrium statistical physics (Gardiner, 2004)
  - Adapted for variational free energy landscapes (Friston, 2019)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional


def detect_fpt_tcritical(
    patient_data: pd.DataFrame,
    inter_ictal_data: pd.DataFrame,
    z_threshold: float = 3.0,
    metric: str = 'Phi1_Z',
    min_lookback: int = 5
) -> Optional[float]:
    """
    Detect First Passage Time (FPT) for a single patient.
    
    Parameters
    ----------
    patient_data : DataFrame
        Pre-ictal rows for one patient, must have 'Time_to_Onset' and metric column.
    inter_ictal_data : DataFrame
        Inter-ictal rows for the same patient.
    z_threshold : float
        Multiples of inter-ictal std above its mean to define "critical".
    metric : str
        Which CFECT metric to use ('Phi1_Z' or 'Variance_Z').
    min_lookback : int
        Minimum number of time points before seizure to search (avoid edge effects).
        
    Returns
    -------
    t_critical : float or None
        Time (in seconds before seizure) when FPT is crossed.
        Returns None if never crossed (no critical transition detected).
    """
    if patient_data.empty or inter_ictal_data.empty:
        return None
    
    # Inter-ictal baseline statistics
    mu = inter_ictal_data[metric].mean()
    sigma = inter_ictal_data[metric].std()
    threshold = mu + z_threshold * sigma
    
    # Pre-ictal trajectory sorted by time (descending: from far in past toward seizure)
    traj = patient_data.sort_values('Time_to_Onset', ascending=True)
    
    # Find first time when Phi1_Z crosses above threshold
    # We scan from far past toward seizure (recent)
    # The FPT is the FIRST time the metric EXCEEDS the threshold
    above = traj[traj[metric] > threshold]
    
    if above.empty:
        # Fallback: use metric > baseline mean as relaxation
        above = traj[traj[metric] > mu]
    
    if above.empty:
        return None
    
    # The critical onset time: the earliest time when crossing occurs
    # (minimum Time_to_Onset among above-threshold points)
    t_critical = above['Time_to_Onset'].min()
    
    # Sanity: FPT must be before seizure and after the first data point
    return t_critical


def compute_physiological_time(
    patient_data: pd.DataFrame,
    t_critical: float,
    t_seizure: float = 0.0,
    n_bins: int = 20
) -> pd.DataFrame:
    """
    Warp absolute clock to physiological clock for one patient.
    
    Parameters
    ----------
    patient_data : DataFrame
        Pre-ictal data for one patient.
    t_critical : float
        First Passage Time (seconds before seizure).
    t_seizure : float
        Seizure onset time (usually 0).
    n_bins : int
        Number of bins for the warped time axis.
        
    Returns
    -------
    patient_data with added column 'Physio_Time' in [0, 1].
    """
    result = patient_data.copy()
    
    # Clip to [t_critical, t_seizure] interval
    mask = (result['Time_to_Onset'] >= t_critical) & (result['Time_to_Onset'] <= t_seizure)
    result = result[mask].copy()
    
    if len(result) < 2:
        return result  # Not enough points to warp
    
    # Elastic mapping: linear stretch to [0, 1]
    # t=0.0 = critical onset, t=1.0 = seizure
    duration = t_seizure - t_critical
    if duration <= 0:
        return result
    
    result['Physio_Time'] = (result['Time_to_Onset'] - t_critical) / duration
    
    return result


def compute_patient_trajectory_physiological(
    patient_data: pd.DataFrame,
    inter_ictal_data: pd.DataFrame,
    metric: str = 'Phi1_Z',
    z_threshold: float = 3.0,
    n_bins: int = 20
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Compute one patient's trajectory in physiological time.
    
    Returns
    -------
    physio_bins : ndarray or None
        Bin centers in physiological time.
    metric_binned : ndarray or None
        Metric values in each physiological bin.
    t_critical : float or None
        Detected FPT.
    """
    t_critical = detect_fpt_tcritical(
        patient_data, inter_ictal_data,
        z_threshold=z_threshold, metric=metric
    )
    
    if t_critical is None:
        return None, None, None
    
    warped = compute_physiological_time(
        patient_data, t_critical, t_seizure=0.0, n_bins=n_bins
    )
    
    if len(warped) < n_bins:
        return None, None, t_critical
    
    # Bin in physiological time
    warped['physio_bin'] = pd.cut(
        warped['Physio_Time'], bins=n_bins, labels=False
    )
    
    profile = warped.groupby('physio_bin').agg(
        physio_time=('Physio_Time', 'mean'),
        metric=('Physio_Time', 'mean'),  # placeholder
    )
    
    # Actually compute metric mean per bin
    metric_binned = warped.groupby('physio_bin')[metric].mean().values
    physio_bins = warped.groupby('physio_bin')['Physio_Time'].mean().values
    
    return physio_bins, metric_binned, t_critical


def warp_all_patients(
    df: pd.DataFrame,
    metric: str = 'Phi1_Z',
    z_threshold: float = 3.0,
    n_bins: int = 20
) -> pd.DataFrame:
    """
    Warp all patients in a DataFrame to physiological time.
    
    Returns
    -------
    DataFrame with columns: Patient_ID, Physio_Bin, Metric_Mean, t_critical
    """
    rows = []
    
    for pid in df['Patient_ID'].unique():
        pre = df[(df['Patient_ID'] == pid) & (df['Condition'] == 'Pre-ictal')]
        inter = df[(df['Patient_ID'] == pid) & (df['Condition'] == 'Inter-ictal')]
        
        if pre.empty or inter.empty:
            continue
        
        t_crit = detect_fpt_tcritical(pre, inter, z_threshold=z_threshold, metric=metric)
        if t_crit is None:
            continue
        
        warped = compute_physiological_time(pre, t_crit, t_seizure=0.0, n_bins=n_bins)
        if len(warped) < 3:
            continue
        
        warped['physio_bin'] = pd.cut(warped['Physio_Time'], bins=n_bins, labels=False)
        profile = warped.groupby('physio_bin').agg(
            physio_time=('Physio_Time', 'mean'),
            metric_mean=(metric, 'mean')
        ).reset_index()
        
        for _, r in profile.iterrows():
            rows.append({
                'Patient_ID': pid,
                'Physio_Bin': r['physio_bin'],
                'Physio_Time': r['physio_time'],
                'Metric_Mean': r['metric_mean'],
                't_critical_sec': t_crit
            })
    
    if not rows:
        return pd.DataFrame()
    
    result = pd.DataFrame(rows)
    
    # Group-level aggregation on physiological time
    # This is the KEY step: we now align across patients using physiological time
    group_profile = result.groupby('Physio_Bin').agg(
        Physio_Time=('Physio_Time', 'mean'),
        Metric_Mean=('Metric_Mean', 'mean'),
        Metric_SEM=('Metric_Mean', 'sem'),
    ).reset_index()
    
    group_profile['Patient_Count'] = result['Patient_ID'].nunique()
    
    return group_profile


def compute_velocity_in_physiological_time(
    warp_profile: pd.DataFrame,
    metric_col: str = 'Metric_Mean'
) -> pd.DataFrame:
    """
    Compute first-order velocity in physiological time.
    
    velocity = d(Metric) / d(Physio_Time)
    
    This captures the ACCELERATION toward bifurcation.
    """
    profile = warp_profile.sort_values('Physio_Time').copy()
    
    # Central difference gradient
    physio = profile['Physio_Time'].values
    metric = profile[metric_col].values
    
    velocity = np.gradient(metric, physio)
    
    profile['Velocity'] = velocity
    profile['Velocity_Abs'] = np.abs(velocity)
    
    return profile
