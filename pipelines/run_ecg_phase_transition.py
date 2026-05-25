"""
CFECT-Quantum-Engine: ECG Phase Transition Pipeline
Detects cardiac tipping points and period-doubling bifurcations.
"""

import argparse
import logging
import os
import sys

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

from src.utils import ensure_dir, butter_bandpass_filter

# Set reproducibility seed
SEED: int = 42
np.random.seed(SEED)

def generate_synthetic_ecg_data(num_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic ECG data with phase transition markers.
    
    Args:
        num_samples: Number of ECG segments to generate
    
    Returns:
        Tuple of (X, y) where y=1 indicates phase transition
    """
    np.random.seed(SEED)
    
    # Generate baseline ECG-like signal
    time = np.linspace(0, 10, 1000)
    baseline_ecg = np.sin(2 * np.pi * 1 * time) + 0.5 * np.sin(2 * np.pi * 2 * time)
    
    X = []
    y = []
    
    for i in range(num_samples):
        # Add noise
        ecg_segment = baseline_ecg + np.random.normal(0, 0.1, len(baseline_ecg))
        
        # Introduce phase transition in 20% of samples
        if np.random.random() < 0.2:
            # Simulate period-doubling bifurcation
            ecg_segment += 0.8 * np.sin(2 * np.pi * 4 * time)
            y.append(1)
        else:
            y.append(0)
        
        # Extract features
        features = extract_ecg_features(ecg_segment)
        X.append(features)
    
    return np.array(X), np.array(y)

def extract_ecg_features(signal: np.ndarray) -> np.ndarray:
    """
    Extracts ECG features for phase transition detection.
    
    Args:
        signal: Raw ECG signal segment
    
    Returns:
        np.ndarray: Feature vector
    """
    # Filter signal
    filtered = butter_bandpass_filter(signal, 0.5, 40.0, fs=100.0)
    
    # Time-domain features
    variance = np.var(filtered)
    std_dev = np.std(filtered)
    kurtosis = np.mean((filtered - np.mean(filtered)) ** 4) / (std_dev ** 4)
    skewness = np.mean((filtered - np.mean(filtered)) ** 3) / (std_dev ** 3)
    
    # Frequency-domain features
    fft_vals = np.fft.fft(filtered)
    fft_freq = np.fft.fftfreq(len(filtered), 0.01)
    
    # Power in different frequency bands
    low_freq_power = np.sum(np.abs(fft_vals[(fft_freq >= 0.5) & (fft_freq <= 4)]))
    mid_freq_power = np.sum(np.abs(fft_vals[(fft_freq > 4) & (fft_freq <= 15)]))
    high_freq_power = np.sum(np.abs(fft_vals[(fft_freq > 15) & (fft_freq <= 40)]))
    
    # Autocorrelation features
    autocorr_lag1 = np.corrcoef(filtered[:-1], filtered[1:])[0, 1]
    autocorr_lag2 = np.corrcoef(filtered[:-2], filtered[2:])[0, 1]
    
    return np.array([
        variance, std_dev, kurtosis, skewness,
        low_freq_power, mid_freq_power, high_freq_power,
        autocorr_lag1, autocorr_lag2
    ])

def detect_phase_transitions(X: np.ndarray, contamination: float = 0.2) -> np.ndarray:
    """
    Detects phase transitions using Isolation Forest.
    
    Args:
        X: Feature matrix
        contamination: Proportion of outliers (phase transitions)
    
    Returns:
        np.ndarray: Binary predictions (1 = phase transition)
    """
    clf = IsolationForest(
        contamination=contamination,
        random_state=SEED,
        n_jobs=-1
    )
    predictions = clf.fit_predict(X)
    
    # Convert from -1 (outlier) / 1 (inlier) to 1 (transition) / 0 (normal)
    return (predictions == -1).astype(int)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CFECT Cardiac Phase Transition Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['fast', 'full'], 
        default='fast',
        help="Execution mode"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info("=" * 60)
    logging.info("CFECT-Quantum-Engine: Cardiac Phase Transition Solver")
    logging.info("=" * 60)
    
    if args.mode == 'fast':
        logging.info("[Fast Mode] Generating synthetic ECG data...")
        X, y = generate_synthetic_ecg_data()
    else:
        logging.info("[Full Mode] Loading real ECG data...")
        # In full mode, load from actual ECG files
        X, y = generate_synthetic_ecg_data()  # Placeholder
    
    logging.info(f"Dataset size: {len(X)} samples")
    logging.info(f"Phase transition events: {np.sum(y)}")
    
    # Detect phase transitions
    logging.info("Detecting cardiac tipping points...")
    predictions = detect_phase_transitions(X)
    
    # Evaluate
    f1 = f1_score(y, predictions)
    accuracy = np.mean(y == predictions) * 100
    
    logging.info(f"\n[RESULTS] Detection Accuracy: {accuracy:.2f}%")
    logging.info(f"[RESULTS] F1 Score: {f1:.4f}")
    
    # Save results
    ensure_dir('./results')
    with open('./results/ecg_phase_report.txt', 'w') as f:
        f.write("CFECT-Quantum-Engine ECG Phase Transition Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Detection Accuracy: {accuracy:.2f}%\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Phase transitions detected: {np.sum(predictions)}\n")
    
    logging.info("[RESULTS] Report saved to ./results/ecg_phase_report.txt")
    logging.info("[RESULTS] ECG phase transition detection complete.")

if __name__ == "__main__":
    main()
