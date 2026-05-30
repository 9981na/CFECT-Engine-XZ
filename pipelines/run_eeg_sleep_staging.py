"""
CFECT-Quantum-Engine: EEG Sleep Staging Pipeline

⚠️  FILE SYSTEMATICALLY DEPRECATED — DeepSeek RedTeam F4 Audit.
   All synthetic data code paths have been removed following the audit
   (Death Boundary F4: "N1 F1=0.16 reflects physiological ambiguity — FAULTED").

   Removed:
     - generate_synthetic_eeg_data()   → hardcoded stage patterns
     - run_fast_mode() synthetic fallback  → precomputed features only
     - run_full_mode() accuracy         → 93.3 + N(0,1.5) random noise

   ✅ Use INSTEAD:
     pipelines/run_sleep_edf_csd.py        — full Sleep-EDF CSD pipeline
     pipelines/n1_baseline_comparison.py   — N1 spectral baseline comparison
     pipelines/run_three_gateways.py       — three-gateway benchmark
"""

import warnings
warnings.simplefilter('error', DeprecationWarning)
warnings.warn(
    "[FATAL] run_eeg_sleep_staging.py is systematically DEPRECATED. "
    "All synthetic code paths removed following DeepSeek RedTeam F4 audit. "
    "Use pipelines/run_sleep_edf_csd.py or pipelines/run_three_gateways.py.",
    DeprecationWarning, stacklevel=2
)

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut

from src.models import learn_transition_matrix, viterbi_path_integral
from src.utils import ensure_dir, load_precomputed_features

SEED: int = 42
np.random.seed(SEED)


def generate_synthetic_eeg_data(*args, **kwargs):
    """
    [LEGACY REMOVED] Synthetic data generation removed following DeepSeek RedTeam F4 audit.

    Raises:
        NotImplementedError: Always -- synthetic data has been removed.
    """
    raise NotImplementedError(
        "[LEGACY REMOVED] Synthetic data generation removed following "
        "DeepSeek RedTeam F4 audit. "
        "Use 'pipelines/run_sleep_edf_csd.py' for real Sleep-EDF analysis."
    )


def run_fast_mode() -> None:
    """
    Reviewer Fast-Pass mode using pre-computed features only.
    No synthetic data fallback -- errors if precomputed features missing.
    """
    logging.info("[Reviewer Fast-Pass] Loading pre-extracted CFECT feature subspace...")

    try:
        X, y = load_precomputed_features('./data/precomputed_features.npz')
    except FileNotFoundError:
        raise FileNotFoundError(
            "[LEGACY REMOVED] Pre-computed features not found and synthetic fallback "
            "has been removed. Use 'pipelines/run_sleep_edf_csd.py' for real data."
        )
    else:
        groups = np.array([i // 100 for i in range(len(y))])

    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Label distribution: {np.bincount(y)}")

    logging.info("Training static emitter (Random Forest)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)

    logo = LeaveOneGroupOut()
    fold_accuracies_raw = []
    fold_accuracies_hmm = []
    all_true = []
    all_pred_hmm = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        rf.fit(X_train, y_train)
        pred_raw = rf.predict(X_test)
        prob_emissions = rf.predict_proba(X_test)
        transition_matrix = learn_transition_matrix(y_train)
        pred_hmm = viterbi_path_integral(prob_emissions, transition_matrix)

        fold_accuracies_raw.append(accuracy_score(y_test, pred_raw))
        fold_accuracies_hmm.append(accuracy_score(y_test, pred_hmm))
        all_true.extend(y_test)
        all_pred_hmm.extend(pred_hmm)

    mean_raw_acc = np.mean(fold_accuracies_raw) * 100
    mean_hmm_acc = np.mean(fold_accuracies_hmm) * 100

    logging.info(f"\n[RESULTS] Static RF Accuracy: {mean_raw_acc:.2f}%")
    logging.info(f"[RESULTS] CFECT HMM Accuracy: {mean_hmm_acc:.2f}%")

    ensure_dir('./results')
    report = classification_report(
        all_true, all_pred_hmm,
        target_names=['Wake', 'N1', 'N2', 'N3', 'REM']
    )

    with open('./results/classification_report.txt', 'w') as f:
        f.write("CFECT-Quantum-Engine Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean Static RF Accuracy: {mean_raw_acc:.2f}%\n")
        f.write(f"Mean CFECT HMM Accuracy: {mean_hmm_acc:.2f}%\n")
        f.write("\nDetailed Classification Report:\n")
        f.write(report)

    logging.info("[RESULTS] Classification report saved to ./results/classification_report.txt")


def run_full_mode(data_dir: str = './data/sleep-edf') -> None:
    """
    [LEGACY REMOVED] Full mode execution removed.

    Previously contained synthetic placeholder accuracy (93.3 + N(0,1.5)).
    Now raises error: use pipelines/run_sleep_edf_csd.py or run_three_gateways.py.
    """
    raise NotImplementedError(
        "[LEGACY REMOVED] Full mode execution removed following "
        "DeepSeek RedTeam F4 audit. "
        "Use 'pipelines/run_sleep_edf_csd.py' for real Sleep-EDF analysis "
        "or 'pipelines/run_three_gateways.py' for the benchmark suite."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CFECT Cortical Landscape Decoupling Solver [DEPRECATED]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', type=str, choices=['fast', 'full'], default='fast')
    parser.add_argument('--data_dir', type=str, default='./data/sleep-edf')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("=" * 60)
    logging.info("CFECT-Quantum-Engine [DEPRECATED]")
    logging.info("=" * 60)
    logging.info(f"Execution Mode: {args.mode.upper()}")

    if args.mode == 'fast':
        run_fast_mode()
    elif args.mode == 'full':
        run_full_mode(args.data_dir)


if __name__ == "__main__":
    main()
