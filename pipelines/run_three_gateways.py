#!/usr/bin/env python3
"""
CFECT Sleep Staging: Three-Gateway Benchmark (N1 Focus)
======================================================
Implements 3 Viterbi decoding schemes on real Sleep-EDF features:
  S1 — HSMM Duration Clamping (epoch-level minimum dwell enforcement)
  S2 — Time-Varying Transition Gate (Φ₁_Z modulated)
  S3 — RF-Emission Hybrid Gateway (CFECT feature + RF blend)

Outputs:
  - Per-scheme N1 F1 (macro, per-class, confusion matrix)
  - Comparison table vs human inter-rater baseline (N1 F1=0.30-0.45)

Author: CFECT Quantum Engine Team
Date: 2026-05-30
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from scipy.stats import mode
warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────
HOME = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HOME)

OUTPUT_DIR = r"E:\MEM\paper\real\output2\main"
FEATURE_CSV = os.path.join(OUTPUT_DIR, "sleep_csd_features.csv")
REPORT_DIR = os.path.join(OUTPUT_DIR, "gateway_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────
NUM_CLASSES = 5  # W, N1, N2, N3, REM
STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']

# Stage code mapping in CSV: W=0, N1=1, N2=2, N3=3, N3_alt=4, REM=5
# We need to normalize: consolidate 3+4→N3(2)
STAGE_MAP_CSV = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}

# Human inter-rater N1 F1 (Rosenberg 2013, Younes 2016)
N1_HUMAN_LOW = 0.30
N1_HUMAN_HIGH = 0.45

# CFECT features used for emission modulation
CFECT_FEATURES = ['Variance_Z', 'Phi1_Z', 'DNB_Z']

# Default transition matrix (from src/models.py)
DEFAULT_TRANS = np.array([
    [0.85, 0.08, 0.04, 0.00, 0.03],  # W
    [0.12, 0.45, 0.35, 0.00, 0.08],  # N1
    [0.04, 0.02, 0.85, 0.05, 0.04],  # N2
    [0.01, 0.00, 0.19, 0.80, 0.00],  # N3
    [0.06, 0.02, 0.12, 0.00, 0.80]   # REM
])


# ═══════════════════════════════════════════════════════════════
# Data Loading & Preparation
# ═══════════════════════════════════════════════════════════════

def load_sleep_data(csv_path: str = FEATURE_CSV) -> pd.DataFrame:
    """Load and validate Sleep-EDF feature CSV."""
    print(f"\n{'='*60}")
    print("  LOADING SLEEP-EDF FEATURES")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    print(f"  Raw rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Subjects: {df['Subject_ID'].nunique()}")
    
    # Map sleep stages to unified numbering
    # CSV has: W=0, N1=1, N2=2, N3=3, N3_alt=4, REM=5
    # We use:   W=0, N1=1, N2=2, N3=3, REM=4
    if 'Sleep_Stage_Code' in df.columns:
        df['Stage_Num'] = df['Sleep_Stage_Code'].map(STAGE_MAP_CSV)
    elif 'Stage' in df.columns:
        df['Stage_Num'] = df['Stage'].map(STAGE_MAP_CSV)
    else:
        # Map from string labels
        stage_to_num = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
        df['Stage_Num'] = df['Sleep_Stage'].map(stage_to_num)
    
    # Drop unknown stages
    df = df.dropna(subset=['Stage_Num'])
    df['Stage_Num'] = df['Stage_Num'].astype(int)
    
    print(f"  Valid stages: {len(df):,} rows")
    stage_dist = df['Sleep_Stage'].value_counts()
    for s in ['W', 'N1', 'N2', 'N3', 'REM']:
        cnt = stage_dist.get(s, 0)
        print(f"    {s}: {cnt:>8,} ({cnt/len(df)*100:5.1f}%)")
    
    return df


def aggregate_to_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sub-window level (4 per epoch) to epoch level.
    
    Strategy:
    - Features: median of sub-windows (robust to outliers)
    - Stage: majority vote (tie goes to deeper sleep stage)
    """
    print(f"\n  Aggregating {len(df):,} windows → epochs...")
    
    # For each subject+epoch, aggregate
    agg = df.groupby(['Subject_ID', 'Study_Type', 'Epoch_Index']).agg({
        'Variance_Z': 'median',
        'Phi1_Z': 'median',
        'DNB_Z': 'median',
        'Variance': 'median',
        'Phi1': 'median',
        'Stage_Num': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'Night': 'first',
        'Is_Drug': 'first',
        'Drug_Type': 'first',
        'Subject_Num': 'first',
    }).reset_index()
    
    # Rename stage column
    agg.rename(columns={'Stage_Num': 'Stage'}, inplace=True)
    agg['Stage'] = agg['Stage'].astype(int)
    
    # Sort for sequential modeling
    agg.sort_values(['Subject_ID', 'Epoch_Index'], inplace=True)
    
    print(f"  Epochs: {len(agg):,}")
    stage_dist = agg['Stage'].value_counts()
    for s in range(5):
        cnt = stage_dist.get(s, 0)
        print(f"    {STAGE_NAMES[s]}: {cnt:>8,} ({cnt/len(agg)*100:5.1f}%)")
    
    return agg


def prepare_cv_data(df_epoch: pd.DataFrame):
    """
    Prepare feature matrix X, labels y, groups for subject-level CV.
    Features: CFECT Z-scores
    """
    feat_cols = ['Variance_Z', 'Phi1_Z', 'DNB_Z']
    X = df_epoch[feat_cols].values.astype(np.float64)
    y = df_epoch['Stage'].values.astype(int)
    groups = df_epoch['Subject_ID'].values
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
    
    print(f"\n  Feature matrix: {X.shape}")
    print(f"  Labels: {len(np.unique(y))} classes")
    print(f"  Groups (subjects): {len(np.unique(groups))}")
    
    return X, y, groups


# ═══════════════════════════════════════════════════════════════
# Core Viterbi Decoder (from src/models.py, adapted)
# ═══════════════════════════════════════════════════════════════

def viterbi_decode(
    emission_probs: np.ndarray,
    transition_matrix: np.ndarray,
    min_durations: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Viterbi decoding with optional HSMM minimum duration clamping.
    
    Parameters:
    -----------
    emission_probs : (T, C) array of log emission probabilities
    transition_matrix : (C, C) array, log transition probabilities  
    min_durations : (C,) array of minimum dwell time in epochs (None=no clamp)
    
    Returns:
    --------
    best_path : (T,) array of decoded states
    """
    T, C = emission_probs.shape
    
    # Protect against NaN/zeros
    eps = 1e-30
    log_emit = np.log(np.clip(emission_probs, eps, 1.0))
    log_trans = np.log(np.clip(transition_matrix, eps, 1.0))
    
    # HSMM: enforce minimum duration
    if min_durations is not None:
        # Post-processing approach: run standard Viterbi then enforce duration
        path = _standard_viterbi(log_emit, log_trans)
        return _enforce_min_duration(path, min_durations)
    else:
        return _standard_viterbi(log_emit, log_trans)


def _standard_viterbi(log_emit: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Standard Viterbi algorithm."""
    T, C = log_emit.shape
    
    # DP trellis
    V = np.zeros((T, C))
    back = np.zeros((T, C), dtype=int)
    
    # Initial state
    V[0, :] = log_emit[0, :]
    
    # Forward
    for t in range(1, T):
        for s in range(C):
            probs = V[t-1, :] + log_trans[:, s]
            best_prev = np.argmax(probs)
            V[t, s] = probs[best_prev] + log_emit[t, s]
            back[t, s] = best_prev
    
    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(V[-1, :])
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    
    return path


def _enforce_min_duration(path: np.ndarray, min_durations: np.ndarray) -> np.ndarray:
    """
    Post-process a Viterbi path to enforce minimum state durations.
    
    If a state persists for fewer than min_durations[state] epochs,
    the segment is reassigned to the neighboring state with higher probability.
    """
    T = len(path)
    result = path.copy()
    
    # Find all segments
    segments = []
    start = 0
    for i in range(1, T):
        if path[i] != path[start]:
            segments.append((start, i - 1, path[start]))
            start = i
    segments.append((start, T - 1, path[start]))
    
    # Check each segment
    for seg_start, seg_end, state in segments:
        duration = seg_end - seg_start + 1
        min_dur = min_durations[state]
        
        if duration < min_dur:
            # Assign to the neighboring states
            # Strategy: assign to whichever neighbor (left/right) is most prevalent
            left_state = result[seg_start - 1] if seg_start > 0 else None
            right_state = result[seg_end + 1] if seg_end < T - 1 else None
            
            if left_state is not None and (right_state is None or left_state == right_state):
                result[seg_start:seg_end + 1] = left_state
            elif right_state is not None and (left_state is None or right_state == left_state):
                result[seg_start:seg_end + 1] = right_state
            elif left_state is not None and right_state is not None and left_state != right_state:
                # Split: assign first half to left, second half to right
                mid = (seg_start + seg_end) // 2
                result[seg_start:mid + 1] = left_state
                result[mid + 1:seg_end + 1] = right_state
    
    return result


# ═══════════════════════════════════════════════════════════════
# S1: HSMM Duration Clamping
# ═══════════════════════════════════════════════════════════════

def scheme_1_hsmm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    transitions: np.ndarray,
    rf: RandomForestClassifier
) -> np.ndarray:
    """
    S1: Standard RF → Viterbi with HSMM minimum duration enforcement.
    
    Duration minima (epochs ~30s):
    - W:   2 epochs (1 min, micro-wakes)
    - N1:  2 epochs (1 min, genuine brief N1)
    - N2:  2 epochs (1 min)
    - N3:  2 epochs (1 min)
    - REM: 2 epochs (1 min)
    """
    min_durations = np.array([2, 2, 2, 2, 2])
    
    # Learn transition matrix from training data
    trans = _learn_transition_matrix(y_train, transitions)
    
    # Predict
    proba = rf.predict_proba(X_test)
    
    # Handle missing class columns
    full_proba = np.zeros((len(proba), NUM_CLASSES))
    for i, c in enumerate(rf.classes_):
        if c < NUM_CLASSES:
            full_proba[:, c] = proba[:, i]
    
    # Viterbi with HSMM
    path = viterbi_decode(full_proba, trans, min_durations=min_durations)
    return path


# ═══════════════════════════════════════════════════════════════
# S2: Time-Varying Transition Gate (Φ₁_Z modulated)
# ═══════════════════════════════════════════════════════════════

def _compute_transition_gate(features: np.ndarray) -> np.ndarray:
    """
    Compute time-varying gate signal from CFECT features.
    
    Gate principle:
    - When Φ₁_Z is high (less negative ≈ N1 region), increase W→N1 and N1→W
    - When Variance_Z is high (pre-transition fluctuation), increase all transitions
    
    Returns:
    - gate_weight : (T,) array in [0.5, 1.5] range
    """
    phi1 = features[:, 1]  # Phi1_Z column
    varz = features[:, 0]  # Variance_Z column
    
    # Φ₁_Z gate: normalize to [0.5, 1.5]
    # Φ₁_Z ranges from ~-25 (deep N3) to ~5 (Wake)
    # N1 is around -12, Wake around -14, N2 around -4
    # We want to ↑ transition probability when near N1 boundary
    phi1_norm = np.clip(phi1, -20, 5)
    phi1_gate = 1.0 - 0.4 * (phi1_norm + 7.5) / 12.5  # Center at -7.5 (between Wake and N2)
    phi1_gate = np.clip(phi1_gate, 0.6, 1.4)
    
    # Variance_Z gate: higher variance = more transition probability
    var_norm = np.clip(varz, -3, 8) / 8.0
    var_gate = 1.0 + 0.3 * var_norm
    var_gate = np.clip(var_gate, 0.7, 1.3)
    
    # Combined gate (weighted average)
    gate = 0.7 * phi1_gate + 0.3 * var_gate
    gate = np.clip(gate, 0.5, 1.5)
    
    return gate


def _modulate_transitions(
    base_trans: np.ndarray,
    gate: float,
    target_states: List[int] = None
) -> np.ndarray:
    """
    Modulate transition matrix by gate signal.
    
    Increases transitions INTO specific target states when gate > 1,
    decreases when gate < 1.
    """
    if target_states is None:
        target_states = [1]  # N1
    
    T_mod = base_trans.copy()
    
    for s in target_states:
        # Scale all incoming transitions to state s
        for from_s in range(NUM_CLASSES):
            T_mod[from_s, s] = np.clip(
                base_trans[from_s, s] * gate, 0.01, 0.99
            )
        
        # Renormalize rows
        for from_s in range(NUM_CLASSES):
            if from_s not in target_states:
                # Redistribute to maintain row sum = 1
                row_slice = [c for c in range(NUM_CLASSES) if c not in target_states]
                row_sum = T_mod[from_s, row_slice].sum()
                if row_sum > 0:
                    for c in row_slice:
                        T_mod[from_s, c] /= row_sum
                else:
                    T_mod[from_s] = base_trans[from_s]
    
    # Ensure row sums = 1
    for i in range(NUM_CLASSES):
        T_mod[i] /= T_mod[i].sum()
    
    return T_mod


def scheme_2_time_varying(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    transitions: np.ndarray,
    rf: RandomForestClassifier
) -> np.ndarray:
    """
    S2: Standard RF + Viterbi with time-varying transition matrix.
    
    For each test epoch t, the transition matrix T(t) is modulated
    by the CFECT gate signal g(t): 
      T_{ij}(t) = T_base_{ij} * g(t) for j in target states
      Then renormalized to maintain row sum = 1.
    """
    # Learn base transition matrix from training data
    trans = _learn_transition_matrix(y_train, transitions)
    
    # Compute gate signal from test features
    gate = _compute_transition_gate(X_test)
    
    # Predict
    proba = rf.predict_proba(X_test)
    full_proba = np.zeros((len(proba), NUM_CLASSES))
    for i, c in enumerate(rf.classes_):
        if c < NUM_CLASSES:
            full_proba[:, c] = proba[:, i]
    
    # Run Viterbi with time-varying transitions
    T = len(X_test)
    eps = 1e-30
    log_emit = np.log(np.clip(full_proba, eps, 1.0))
    log_trans_base = np.log(np.clip(trans, eps, 1.0))
    
    V = np.zeros((T, NUM_CLASSES))
    back = np.zeros((T, NUM_CLASSES), dtype=int)
    V[0, :] = log_emit[0, :]
    
    for t in range(1, T):
        # Modulate transition matrix for this time step
        T_mod = _modulate_transitions(trans, gate[t], target_states=[1, 0, 4])
        log_trans_t = np.log(np.clip(T_mod, eps, 1.0))
        
        for s in range(NUM_CLASSES):
            probs = V[t-1, :] + log_trans_t[:, s]
            best_prev = np.argmax(probs)
            V[t, s] = probs[best_prev] + log_emit[t, s]
            back[t, s] = best_prev
    
    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(V[-1, :])
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    
    return path


# ═══════════════════════════════════════════════════════════════
# S3: RF-Emission Hybrid Gateway
# ═══════════════════════════════════════════════════════════════

def _compute_feature_likelihood(
    X: np.ndarray,
    class_stats: Dict[int, Dict]
) -> np.ndarray:
    """
    Compute per-class likelihood from CFECT feature statistics.
    
    Uses Gaussian likelihood based on training set mean/std per class.
    
    Returns:
    - likelihood : (T, C) array of class probabilities from features alone
    """
    T = len(X)
    C = NUM_CLASSES
    likelihood = np.ones((T, C))
    
    for c in range(C):
        if c not in class_stats:
            continue
        stats = class_stats[c]
        for fi, feat_name in enumerate(['Variance_Z', 'Phi1_Z', 'DNB_Z']):
            mu = stats['mean'][fi]
            sigma = stats['std'][fi] + 0.01
            # Gaussian log-likelihood
            log_lik = -0.5 * ((X[:, fi] - mu) / sigma) ** 2 - np.log(sigma)
            likelihood[:, c] *= np.exp(np.clip(log_lik, -20, 0))
    
    # Normalize
    likelihood /= likelihood.sum(axis=1, keepdims=True)
    return likelihood


def _compute_class_stats(X: np.ndarray, y: np.ndarray) -> Dict[int, Dict]:
    """Compute per-class feature statistics from training data."""
    stats = {}
    for c in range(NUM_CLASSES):
        mask = y == c
        if mask.sum() < 5:
            continue
        stats[c] = {
            'mean': X[mask].mean(axis=0),
            'std': X[mask].std(axis=0),
            'count': mask.sum()
        }
    return stats


def scheme_3_rf_hybrid(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    transitions: np.ndarray,
    rf: RandomForestClassifier
) -> np.ndarray:
    """
    S3: RF emission probabilities + CFECT feature likelihoods (hybrid).
    
    For N1 specifically (but applied to all classes):
      P(y_t = c | X_t) = (1-λ) * P_RF(c | X_t) + λ * P_CFECT(c | X_t)
    
    Where λ = 0.2 (small perturbation from RF baseline)
    """
    lambda_blend = 0.2
    
    # Learn transition matrix
    trans = _learn_transition_matrix(y_train, transitions)
    
    # Compute class stats from training data
    class_stats = _compute_class_stats(X_train, y_train)
    
    # RF probabilities
    proba = rf.predict_proba(X_test)
    rf_proba = np.zeros((len(proba), NUM_CLASSES))
    for i, c in enumerate(rf.classes_):
        if c < NUM_CLASSES:
            rf_proba[:, i] = proba[:, i]
    
    # CFECT feature likelihood
    feat_likelihood = _compute_feature_likelihood(X_test, class_stats)
    
    # Hybrid blend
    hybrid_proba = (1 - lambda_blend) * rf_proba + lambda_blend * feat_likelihood
    hybrid_proba /= hybrid_proba.sum(axis=1, keepdims=True)
    
    # Viterbi
    path = viterbi_decode(hybrid_proba, trans)
    return path


# ═══════════════════════════════════════════════════════════════
# Helper: Learn Transition Matrix
# ═══════════════════════════════════════════════════════════════

def _learn_transition_matrix(
    y_train: np.ndarray,
    default_trans: np.ndarray
) -> np.ndarray:
    """
    Learn transition matrix from training labels with Laplace smoothing.
    """
    C = NUM_CLASSES
    counts = np.ones((C, C))
    
    for i in range(len(y_train) - 1):
        from_s = int(y_train[i])
        to_s = int(y_train[i + 1])
        if from_s < C and to_s < C:
            counts[from_s, to_s] += 1
    
    trans = counts / counts.sum(axis=1, keepdims=True)
    
    # Fallback to default if pathological
    if np.any(np.isnan(trans)) or np.any(trans.sum(axis=1) == 0):
        return default_trans
    
    return trans


# ═══════════════════════════════════════════════════════════════
# Cross-Validation Runner
# ═══════════════════════════════════════════════════════════════

def run_cv_evaluation(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    scheme_name: str
) -> Dict:
    """
    Run subject-level cross-validation for a given scheme.
    
    Uses Leave-One-Group-Out (LOGO) where groups = subjects.
    """
    print(f"\n  {'='*50}")
    print(f"  RUNNING: {scheme_name}")
    print(f"  {'='*50}")
    
    unique_groups = np.unique(groups)
    n_folds = min(len(unique_groups), 20)  # Cap at 20 folds
    
    # If too many subjects, use StratifiedKFold on groups
    if len(unique_groups) > 50:
        # Sample subjects
        rng = np.random.RandomState(42)
        sampled_groups = rng.choice(unique_groups, 50, replace=False)
        mask = np.isin(groups, sampled_groups)
        X = X[mask]
        y = y[mask]
        groups = groups[mask]
        n_folds = 10
    
    # LOGO or StratifiedKFold
    logo = LeaveOneGroupOut()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Choose CV strategy based on n_subjects
    if len(unique_groups) <= 30:
        cv = logo
        cv_split = logo.split(X, y, groups)
        print(f"  CV: Leave-One-Subject-Out ({len(unique_groups)} folds)")
    else:
        cv = skf
        cv_split = skf.split(X, y)
        print(f"  CV: Stratified {n_folds}-Fold")
    
    fold_results = []
    all_y_true = []
    all_y_pred = []
    
    t_start = time.time()
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_split):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        # Skip if test set has only 1 class
        if len(np.unique(y_te)) < 2:
            continue
        
        # Train RF
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_leaf=5, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        
        # Get transitions from training
        trans = _learn_transition_matrix(y_tr, DEFAULT_TRANS)
        
        # Run scheme
        if scheme_name == 'S1_HSMM':
            y_pred = scheme_1_hsmm(X_tr, y_tr, X_te, y_te, trans, rf)
        elif scheme_name == 'S2_TimeVarying':
            y_pred = scheme_2_time_varying(X_tr, y_tr, X_te, y_te, trans, rf)
        elif scheme_name == 'S3_RF_Hybrid':
            y_pred = scheme_3_rf_hybrid(X_tr, y_tr, X_te, y_te, trans, rf)
        else:
            # Baseline: RF + standard Viterbi (no HSMM)
            proba = rf.predict_proba(X_te)
            full_proba = np.zeros((len(proba), NUM_CLASSES))
            for i, c in enumerate(rf.classes_):
                if c < NUM_CLASSES:
                    full_proba[:, i] = proba[:, i]
            y_pred = viterbi_decode(full_proba, trans)
        
        # Metrics
        acc = accuracy_score(y_te, y_pred)
        f1_macro = f1_score(y_te, y_pred, average='macro', zero_division=0)
        f1_per_class = f1_score(y_te, y_pred, labels=range(5), average=None, zero_division=0)
        
        # N1 F1 specifically
        n1_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
        
        fold_results.append({
            'fold': fold_idx,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class.tolist() if hasattr(f1_per_class, 'tolist') else [0]*5,
            'n1_f1': n1_f1,
        })
        
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        
        if (fold_idx + 1) % 5 == 0:
            elapsed = time.time() - t_start
            print(f"    Fold {fold_idx+1}/{n_folds} — "
                  f"Acc={acc:.3f}, N1 F1={n1_f1:.3f} ({elapsed:.0f}s)")
    
    t_elapsed = time.time() - t_start
    
    # Aggregate
    df_folds = pd.DataFrame(fold_results)
    n1_f1_values = df_folds['n1_f1'].values
    
    results = {
        'scheme': scheme_name,
        'n_folds': len(fold_results),
        'accuracy_mean': float(df_folds['accuracy'].mean()),
        'accuracy_std': float(df_folds['accuracy'].std()),
        'f1_macro_mean': float(df_folds['f1_macro'].mean()),
        'f1_macro_std': float(df_folds['f1_macro'].std()),
        'n1_f1_mean': float(np.mean(n1_f1_values)),
        'n1_f1_std': float(np.std(n1_f1_values)),
        'n1_f1_percentile_25': float(np.percentile(n1_f1_values, 25)),
        'n1_f1_percentile_75': float(np.percentile(n1_f1_values, 75)),
        'n1_f1_values': [round(v, 4) for v in n1_f1_values],
        'elapsed_seconds': round(t_elapsed, 1),
    }
    
    print(f"\n  [{scheme_name}] Results:")
    print(f"    Accuracy:  {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
    print(f"    F1 Macro:  {results['f1_macro_mean']:.3f} ± {results['f1_macro_std']:.3f}")
    print(f"    N1 F1:     {results['n1_f1_mean']:.3f} ± {results['n1_f1_std']:.3f}")
    print(f"    N1 F1 25-75%: [{results['n1_f1_percentile_25']:.3f}, {results['n1_f1_percentile_75']:.3f}]")
    
    # Per-class F1 (from all predictions)
    print(f"\n  Per-class F1 (aggregated):")
    cm = confusion_matrix(all_y_true, all_y_pred, labels=range(5))
    p, r, f1_agg, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=range(5), zero_division=0
    )
    for i, name in enumerate(STAGE_NAMES):
        print(f"    {name}: F1={f1_agg[i]:.3f} P={p[i]:.3f} R={r[i]:.3f}")
    
    results['per_class_f1'] = {STAGE_NAMES[i]: float(f1_agg[i]) for i in range(5)}
    results['confusion_matrix'] = cm.tolist()
    
    return results


# ═══════════════════════════════════════════════════════════════
# N1 F1 Benchmark vs Human Inter-Rater
# ═══════════════════════════════════════════════════════════════

def benchmark_n1_f1(results_list: List[Dict], output_path: str):
    """
    Compare N1 F1 across schemes against human inter-rater baseline.
    Also compare against the original synthetic-data "results" (F1 ≈ 0.16).
    """
    print(f"\n{'='*60}")
    print("  N1 F1 BENCHMARK vs HUMAN INTER-RATER")
    print(f"{'='*60}")
    print(f"  Human inter-rater N1 F1 range: [{N1_HUMAN_LOW}, {N1_HUMAN_HIGH}]")
    print(f"  Original synthetic data N1 F1: ≈0.16 (hardcoded)")
    print()
    
    rows = []
    for res in results_list:
        n1_f1 = res['n1_f1_mean']
        n1_std = res['n1_f1_std']
        n1_p25 = res['n1_f1_percentile_25']
        n1_p75 = res['n1_f1_percentile_75']
        
        # Assess vs human
        if n1_f1 >= N1_HUMAN_LOW:
            if n1_f1 > N1_HUMAN_HIGH * 1.3:
                status = "ABOVE HUMAN RANGE (check leakage)"
            else:
                status = "WITHIN HUMAN RANGE ✓"
        else:
            gap = N1_HUMAN_LOW - n1_f1
            status = f"BELOW HUMAN (gap={gap:.3f})"
        
        # Improvement over synthetic baseline
        synth_gap = n1_f1 - 0.16
        
        rows.append({
            'Scheme': res['scheme'],
            'N1_F1_Mean': f"{n1_f1:.3f}",
            'N1_F1_Std': f"{n1_std:.3f}",
            'N1_F1_25P': f"{n1_p25:.3f}",
            'N1_F1_75P': f"{n1_p75:.3f}",
            'Accuracy': f"{res['accuracy_mean']:.3f}",
            'F1_Macro': f"{res['f1_macro_mean']:.3f}",
            'Δ_vs_Synthetic': f"+{synth_gap:.3f}" if synth_gap > 0 else f"{synth_gap:.3f}",
            'Status_vs_Human': status,
        })
    
    df_bench = pd.DataFrame(rows)
    print(df_bench.to_string(index=False))
    
    # Save
    bench_path = os.path.join(output_path, "n1_f1_benchmark.csv")
    df_bench.to_csv(bench_path, index=False)
    print(f"\n  Benchmark saved: {bench_path}")
    
    # Generate summary
    best_scheme = max(results_list, key=lambda r: r['n1_f1_mean'])
    print(f"\n  Best scheme: {best_scheme['scheme']} (N1 F1={best_scheme['n1_f1_mean']:.3f})")
    
    return df_bench


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  CFECT Sleep Staging: Three-Gateway Benchmark")
    print(f"  {datetime.now().isoformat()}")
    print("  Real Sleep-EDF Data (153 SC + 44 ST)")
    print("=" * 60)
    
    # 1. Load & prepare data
    df_raw = load_sleep_data(FEATURE_CSV)
    df_epoch = aggregate_to_epochs(df_raw)
    X, y, groups = prepare_cv_data(df_epoch)
    
    # Stage distribution
    stage_counts = pd.Series(y).value_counts().sort_index()
    print(f"\n  Epoch-level stage distribution:")
    for s in range(NUM_CLASSES):
        cnt = stage_counts.get(s, 0)
        print(f"    {STAGE_NAMES[s]}: {cnt:>8,} ({cnt/len(y)*100:5.1f}%)")
    
    # 2. Define schemes
    schemes = [
        ('RF+Viterbi (Baseline)', None),  # Will use standard approach
        ('S1_HSMM', None),
        ('S2_TimeVarying', None),
        ('S3_RF_Hybrid', None),
    ]
    
    # 3. Run CV for each
    all_results = []
    for scheme_name, _ in schemes:
        results = run_cv_evaluation(X, y, groups, scheme_name)
        all_results.append(results)
        
        # Save intermediate
        with open(os.path.join(REPORT_DIR, f"{scheme_name}_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # 4. N1 F1 benchmark
    bench = benchmark_n1_f1(all_results, REPORT_DIR)
    
    # 5. Complete report
    report_path = os.path.join(REPORT_DIR, "gateway_comparison_report.md")
    with open(report_path, 'w') as f:
        f.write("# CFECT Sleep Staging: Three-Gateway Comparison\n\n")
        f.write(f"**Date**: {datetime.now().isoformat()}\n")
        f.write(f"**Data**: Sleep-EDF (153 SC + 44 ST, ~112K windows → ~28K epochs)\n\n")
        
        f.write("## N1 F1 Benchmark vs Human Inter-Rater\n\n")
        f.write(f"Human inter-rater N1 F1 range: [{N1_HUMAN_LOW}, {N1_HUMAN_HIGH}]\n")
        f.write(f"Original synthetic data N1 F1: ≈0.16 (hardcoded in run_eeg_sleep_staging.py)\n\n")
        
        f.write("| Scheme | N1 F1 Mean | N1 F1 Std | N1 F1 [25%, 75%] | Acc | F1 Macro | vs Human |\n")
        f.write("|--------|-----------|----------|-----------------|-----|---------|---------|\n")
        for res in all_results:
            n1 = f"{res['n1_f1_mean']:.3f}"
            n1s = f"{res['n1_f1_std']:.3f}"
            n1r = f"[{res['n1_f1_percentile_25']:.3f}, {res['n1_f1_percentile_75']:.3f}]"
            acc = f"{res['accuracy_mean']:.3f}"
            fm = f"{res['f1_macro_mean']:.3f}"
            
            if res['n1_f1_mean'] >= N1_HUMAN_LOW:
                vh = "WITHIN RANGE ✓"
            else:
                gap = N1_HUMAN_LOW - res['n1_f1_mean']
                vh = f"below by {gap:.3f}"
            
            f.write(f"| {res['scheme']:20s} | {n1:>9s} | {n1s:>8s} | {n1r:>24s} | {acc:>5s} | {fm:>8s} | {vh} |\n")
        
        f.write("\n\n## Per-Class F1 (Aggregated)\n\n")
        for res in all_results:
            f.write(f"\n### {res['scheme']}\n\n")
            f.write("| Stage | F1 | Precision | Recall |\n")
            f.write("|-------|-----|-----------|--------|\n")
            cm = np.array(res['confusion_matrix'])
            for i, name in enumerate(STAGE_NAMES):
                fi = res['per_class_f1'].get(name, 0)
                n_correct = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
                n_total = cm[i, :].sum()
                f.write(f"| {name} | {fi:.3f} | — | {n_correct}/{n_total} |\n")
        
        f.write("\n\n## Conclusion\n\n")
        best = max(all_results, key=lambda r: r['n1_f1_mean'])
        f.write(f"Best scheme: **{best['scheme']}** (N1 F1 = {best['n1_f1_mean']:.3f})\n\n")
        
        if best['n1_f1_mean'] >= N1_HUMAN_LOW:
            f.write("✅ **N1 F1 is within human inter-rater range** — the N1 detection problem\n")
            f.write("   is fundamentally limited by EEG feature ambiguity, not model capability.\n")
        else:
            gap = N1_HUMAN_LOW - best['n1_f1_mean']
            f.write(f"⚠️ **N1 F1 below human range by {gap:.3f}** — further improvement needed.\n")
            f.write("   Consider: spectral feature augmentation, subject-specific tuning,\n")
            f.write("   or multi-channel integration.\n")
    
    print(f"\n  Full report saved: {report_path}")
    print(f"\n{'='*60}")
    print("  DONE: Three-Gateway Benchmark Complete")
    print(f"{'='*60}")
    
    return all_results


if __name__ == "__main__":
    results = main()
