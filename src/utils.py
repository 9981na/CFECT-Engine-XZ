"""
CFECT-Quantum-Engine: Signal Processing and Utility Module
Provides filtering, normalization, and file I/O utilities.
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(
    data: np.ndarray, 
    lowcut: float, 
    highcut: float, 
    fs: float, 
    order: int = 4
) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter to the input signal.
    
    Mathematical Formalism:
        H(s) = (b0 + b1*s^-1 + ... + bn*s^-n) / (1 + a1*s^-1 + ... + an*s^-n)
        
        where coefficients are computed using Butterworth polynomial approximation.
    
    Args:
        data: Input signal (shape: [N_samples])
        lowcut: Lower cutoff frequency (Hz)
        highcut: Upper cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (default: 4)
    
    Returns:
        np.ndarray: Filtered signal (shape: [N_samples])
    """
    nyquist: float = 0.5 * fs
    low: float = lowcut / nyquist
    high: float = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    filtered: np.ndarray = filtfilt(b, a, data)
    
    return filtered

def apply_eeg_filter(signal: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Applies standard EEG bandpass filter (0.5-40 Hz) to remove noise and baseline drift.
    
    Filter Specifications:
        - Low cutoff: 0.5 Hz (removes DC drift and respiratory artifacts)
        - High cutoff: 40 Hz (removes EMG and line noise)
        - Filter type: 4th-order Butterworth
    
    Args:
        signal: Raw EEG signal (shape: [N_samples])
        fs: Sampling frequency (default: 100 Hz)
    
    Returns:
        np.ndarray: Bandpass-filtered EEG signal (shape: [N_samples])
    """
    return butter_bandpass_filter(signal, 0.5, 40.0, fs)

def normalize_features(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalizes feature matrix using specified method.
    
    Mathematical Formalism (Standard):
        X_norm = (X - μ) / σ
        
        where μ = mean(X), σ = standard deviation(X)
    
    Args:
        X: Feature matrix (shape: [N_samples, N_features])
        method: Normalization method ('standard' or 'minmax')
    
    Returns:
        np.ndarray: Normalized feature matrix (shape: [N_samples, N_features])
    """
    if method == 'standard':
        mean: np.ndarray = np.mean(X, axis=0)
        std: np.ndarray = np.std(X, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        return (X - mean) / std
    
    elif method == 'minmax':
        min_vals: np.ndarray = np.min(X, axis=0)
        max_vals: np.ndarray = np.max(X, axis=0)
        range_vals: np.ndarray = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        return (X - min_vals) / range_vals
    
    return X

def load_precomputed_features(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads pre-computed features from NumPy .npz file.
    
    Args:
        filepath: Path to .npz file containing X and y arrays
    
    Returns:
        Tuple of (X, y) where:
            X: Feature matrix (shape: [N_samples, N_features])
            y: Label array (shape: [N_samples])
    """
    data: dict = np.load(filepath)
    return data['X'], data['y']

def save_features(X: np.ndarray, y: np.ndarray, filepath: str) -> None:
    """
    Saves feature matrix and labels to NumPy .npz file.
    
    Args:
        X: Feature matrix (shape: [N_samples, N_features])
        y: Label array (shape: [N_samples])
        filepath: Output .npz file path
    
    Returns:
        None
    """
    np.savez(filepath, X=X, y=y)

def ensure_dir(path: str) -> None:
    """
    Ensures directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
    
    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
