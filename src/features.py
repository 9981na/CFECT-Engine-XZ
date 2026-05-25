"""
CFECT-Quantum-Engine: Topological and Renormalization Operator Module
Implements Multi-Scale Entropy (MSE) coarse-graining and frequency extraction.
"""

import numpy as np
from scipy.signal import welch
from numba import jit

# Physical constants from the original CFECT paper
EMBEDDING_DIMENSION: int = 2  # m=2, standard for sample entropy
TOLERANCE_COEFFICIENT: float = 0.15  # r=0.15*std, empirical thermodynamic limit
MAX_SCALE_FACTOR: int = 5  # Maximum renormalization group scale
SAMPLING_FREQUENCY: int = 100  # Standard EEG sampling rate (Hz)

@jit(nopython=True, fastmath=True)
def _compute_sample_entropy(coarse_signal: np.ndarray, embedding_dim: int, tolerance: float) -> tuple[int, int]:
    """
    Core Sample Entropy calculation (Numba-accelerated).
    
    Mathematical Formalism:
        A^m = number of matching (m+1)-length vectors
        B^m = number of matching m-length vectors
        SampEn = -ln(A^m / B^m)
    
    Args:
        coarse_signal: 1D array of coarse-grained physiological signal
        embedding_dim: Embedding dimension m (typically 2)
        tolerance: Matching threshold (typically 0.15 * std)
    
    Returns:
        Tuple of (B^m, A^m) counts for entropy computation
    """
    signal_length: int = len(coarse_signal)
    count_B: int = 0  # Matching m-length vectors
    count_A: int = 0  # Matching (m+1)-length vectors
    
    for i in range(signal_length - embedding_dim):
        for j in range(i + 1, signal_length - embedding_dim):
            match_m: bool = True
            for k in range(embedding_dim):
                if abs(coarse_signal[i + k] - coarse_signal[j + k]) > tolerance:
                    match_m = False
                    break
            if match_m:
                count_B += 1
                if abs(coarse_signal[i + embedding_dim] - coarse_signal[j + embedding_dim]) <= tolerance:
                    count_A += 1
    
    return count_B, count_A

def extract_multiscale_entropy(
    signal: np.ndarray, 
    max_scale: int = MAX_SCALE_FACTOR,
    embedding_dim: int = EMBEDDING_DIMENSION,
    tolerance_coeff: float = TOLERANCE_COEFFICIENT
) -> np.ndarray:
    """
    Executes the Renormalization Group coarse-graining operation and evaluates Sample Entropy.
    
    Mathematical Formalism:
        y_j^(τ) = (1/τ) * Σ_{i=(j-1)τ+1}^{jτ} x_i  (coarse-graining)
        SampEn(m, r, τ) = -ln(A^m(τ) / B^m(τ))
    
    Args:
        signal: 1D array of the microvolt physiological epoch (shape: [N_samples])
        max_scale: Maximum coarse-graining scale factor τ (default: 5)
        embedding_dim: Embedding dimension m (default: 2)
        tolerance_coeff: Tolerance threshold coefficient (default: 0.15 * std)
    
    Returns:
        np.ndarray: Multi-scale entropy values across scales 1 to max_scale (shape: [max_scale])
    """
    std_dev: float = np.std(signal)
    
    if std_dev < 1e-8:
        return np.zeros(max_scale)
    
    tolerance: float = tolerance_coeff * std_dev
    entropy_values: list[float] = []
    
    for scale in range(1, max_scale + 1):
        if scale == 1:
            coarse_signal: np.ndarray = signal
        else:
            truncated_length: int = len(signal) - (len(signal) % scale)
            coarse_signal = signal[:truncated_length].reshape(-1, scale).mean(axis=1)
        
        if len(coarse_signal) <= embedding_dim + 1:
            entropy_values.append(0.0)
            continue
        
        count_B, count_A = _compute_sample_entropy(coarse_signal, embedding_dim, tolerance)
        
        if count_B > 0 and count_A > 0:
            entropy: float = -np.log(count_A / count_B)
        else:
            entropy = 0.0
        
        entropy_values.append(entropy)
    
    return np.array(entropy_values)

def extract_spectral_powers(
    signal: np.ndarray, 
    sampling_freq: int = SAMPLING_FREQUENCY
) -> np.ndarray:
    """
    Extracts relative power in five canonical EEG frequency bands using Welch's method.
    
    Mathematical Formalism:
        Pxx(f) = (1/(fs*N)) * |Σ_{n=0}^{N-1} x[n] * e^(-j2πfn/N)|^2
        Relative Power = ∫_{f_low}^{f_high} Pxx(f) df / ∫_{0.5}^{40} Pxx(f) df
    
    Args:
        signal: 1D array of EEG epoch (shape: [N_samples])
        sampling_freq: Sampling frequency in Hz (default: 100)
    
    Returns:
        np.ndarray: Relative power in [Delta, Theta, Alpha, Sigma, Beta] bands (shape: [5])
    """
    frequencies, power_spectral_density = welch(
        signal, 
        fs=sampling_freq, 
        nperseg=sampling_freq * 2
    )
    
    # Frequency bands defined per AASM standards
    frequency_bands = [
        (0.5, 4),   # Delta: Deep sleep (N3)
        (4, 8),     # Theta: Light sleep (N1/N2)
        (8, 12),    # Alpha: Relaxed wakefulness
        (12, 15),   # Sigma: Sleep spindles (N2)
        (15, 30)    # Beta: Active wakefulness
    ]
    
    total_power: float = np.sum(power_spectral_density[
        (frequencies >= 0.5) & (frequencies <= 40)
    ])
    
    if total_power == 0:
        return np.zeros(5)
    
    band_powers: list[float] = []
    for (freq_min, freq_max) in frequency_bands:
        band_power: float = np.sum(power_spectral_density[
            (frequencies >= freq_min) & (frequencies <= freq_max)
        ])
        band_powers.append(band_power / total_power)
    
    return np.array(band_powers)

def extract_feature_vector(
    signal: np.ndarray, 
    sampling_freq: int = SAMPLING_FREQUENCY,
    max_scale: int = MAX_SCALE_FACTOR
) -> np.ndarray:
    """
    Constructs complete feature vector: variance + spectral powers + multi-scale entropy.
    
    Feature Composition:
        [0]    : Temporal variance (σ²)
        [1-5]  : Spectral powers (Delta, Theta, Alpha, Sigma, Beta)
        [6-10] : Multi-scale entropy (Scales 1-5)
    
    Args:
        signal: 1D array of EEG epoch (shape: [N_samples])
        sampling_freq: Sampling frequency in Hz (default: 100)
        max_scale: Maximum MSE scale (default: 5)
    
    Returns:
        np.ndarray: Complete feature vector (shape: [11])
    """
    variance: float = np.var(signal)
    spectral_features: np.ndarray = extract_spectral_powers(signal, sampling_freq)
    entropy_features: np.ndarray = extract_multiscale_entropy(signal, max_scale)
    
    return np.concatenate([[variance], spectral_features, entropy_features])
