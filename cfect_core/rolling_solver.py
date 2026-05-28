
"""
Rolling Solver Module
NumPy sliding_window_view based high-speed vectorized computation engine
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class RollingSolver:
    """
    Spatial manifold high-speed vectorized computation solver.
    
    Implements:
    - Local variance calculation
    - Lag-1 autocorrelation
    - Permutation entropy
    - Complex phase-space embedding
    """
    
    def __init__(self, window_size: int = 1000, step_size: int = 250):
        """
        Initialize rolling solver with window parameters.
        
        Args:
            window_size: Number of samples per window (default: 1000)
            step_size: Step between consecutive windows (default: 250)
        """
        self.window_size = window_size
        self.step_size = step_size
    
    def _validate_input(self, signal: np.ndarray) -> None:
        """Validate input signal dimensions."""
        if signal.ndim != 1:
            raise ValueError("Input signal must be 1-dimensional")
        if len(signal) < self.window_size:
            raise ValueError(f"Signal length ({len(signal)}) must exceed window size ({self.window_size})")
    
    def rolling_window(self, signal: np.ndarray) -> np.ndarray:
        """
        Generate sliding windows from input signal.
        
        Args:
            signal: 1D input signal
            
        Returns:
            2D array of shape (n_windows, window_size)
        """
        self._validate_input(signal)
        return sliding_window_view(signal, window_shape=self.window_size)[::self.step_size]
    
    def local_variance(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute local variance using vectorized sliding window.
        
        Args:
            signal: 1D input signal
            
        Returns:
            Array of variance values for each window
        """
        windows = self.rolling_window(signal)
        return np.var(windows, axis=1)
    
    def lag1_autocorrelation(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute lag-1 autocorrelation using vectorized sliding window.
        
        Args:
            signal: 1D input signal
            
        Returns:
            Array of autocorrelation values for each window
        """
        windows = self.rolling_window(signal)
        
        # Compute mean for each window
        means = np.mean(windows, axis=1, keepdims=True)
        
        # Numerator: E[(X_t - mu)(X_{t+1} - mu)]
        centered = windows - means
        numerator = np.sum(centered[:, :-1] * centered[:, 1:], axis=1)
        
        # Denominator: E[(X_t - mu)^2]
        denominator = np.sum(centered ** 2, axis=1)
        
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
        
        return result
    
    def permutation_entropy(self, signal: np.ndarray, order: int = 3) -> np.ndarray:
        """
        Compute permutation entropy using vectorized operations.
        
        Args:
            signal: 1D input signal
            order: Embedding dimension (default: 3)
            
        Returns:
            Array of permutation entropy values for each window
        """
        windows = self.rolling_window(signal)
        n_windows, window_size = windows.shape
        
        if window_size < order:
            raise ValueError(f"Window size ({window_size}) must be >= order ({order})")
        
        # Generate permutation patterns
        n_patterns = window_size - (order - 1)
        patterns = np.zeros((n_windows, n_patterns, order), dtype=int)
        
        for i in range(order):
            patterns[:, :, i] = windows[:, i:i + n_patterns]
        
        # Get indices that would sort each pattern
        sorted_indices = np.argsort(patterns, axis=2)
        
        # Convert to unique tuples and count
        entropy = np.zeros(n_windows)
        max_entropy = np.log2(np.math.factorial(order))
        
        for i in range(n_windows):
            unique, counts = np.unique(sorted_indices[i], axis=0, return_counts=True)
            probs = counts / n_patterns
            entropy[i] = -np.sum(probs * np.log2(probs)) / max_entropy
        
        return entropy
    
    def complex_embedding(self, signal: np.ndarray) -> np.ndarray:
        """
        Embed signal into complex phase space.
        
        Args:
            signal: 1D input signal
            
        Returns:
            Complex array where real part is variance and imaginary part is autocorrelation
        """
        variance = self.local_variance(signal)
        autocorr = self.lag1_autocorrelation(signal)
        
        # Normalize both components
        var_norm = (variance - np.mean(variance)) / np.std(variance)
        ac_norm = (autocorr - np.mean(autocorr)) / np.std(autocorr)
        
        return var_norm + 1j * ac_norm
    
    def compute_all_metrics(self, signal: np.ndarray) -> dict:
        """
        Compute all CSD metrics in one pass.
        
        Args:
            signal: 1D input signal
            
        Returns:
            Dictionary containing variance, autocorrelation, permutation entropy, and complex embedding
        """
        windows = self.rolling_window(signal)
        n_windows = windows.shape[0]
        
        # Compute variance
        variance = np.var(windows, axis=1)
        
        # Compute lag-1 autocorrelation
        means = np.mean(windows, axis=1, keepdims=True)
        centered = windows - means
        numerator = np.sum(centered[:, :-1] * centered[:, 1:], axis=1)
        denominator = np.sum(centered ** 2, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            autocorr = numerator / denominator
        
        # Normalize
        var_z = (variance - np.mean(variance)) / np.std(variance)
        ac_z = (autocorr - np.mean(autocorr)) / np.std(autocorr)
        
        # Complex embedding
        complex_state = var_z + 1j * ac_z
        
        return {
            'variance': variance,
            'autocorrelation': autocorr,
            'variance_z': var_z,
            'autocorrelation_z': ac_z,
            'complex_state': complex_state
        }
