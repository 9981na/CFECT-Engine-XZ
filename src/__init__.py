"""
CFECT-Quantum-Engine: Package Initialization
Exports core modules with type-checked interfaces.
"""

from .features import (
    extract_multiscale_entropy,
    extract_spectral_powers,
    extract_feature_vector,
    _compute_sample_entropy,
    EMBEDDING_DIMENSION,
    TOLERANCE_COEFFICIENT,
    MAX_SCALE_FACTOR,
    SAMPLING_FREQUENCY
)

from .models import (
    learn_transition_matrix,
    viterbi_path_integral,
    viterbi_path_integral_dynamic,
    NUM_CLASSES,
    DEFAULT_TRANSITION_MATRIX
)

from .utils import (
    butter_bandpass_filter,
    apply_eeg_filter,
    normalize_features,
    load_precomputed_features,
    save_features,
    ensure_dir
)

__all__ = [
    # Features module
    'extract_multiscale_entropy',
    'extract_spectral_powers',
    'extract_feature_vector',
    '_compute_sample_entropy',
    'EMBEDDING_DIMENSION',
    'TOLERANCE_COEFFICIENT',
    'MAX_SCALE_FACTOR',
    'SAMPLING_FREQUENCY',
    
    # Models module
    'learn_transition_matrix',
    'viterbi_path_integral',
    'viterbi_path_integral_dynamic',
    'NUM_CLASSES',
    'DEFAULT_TRANSITION_MATRIX',
    
    # Utils module
    'butter_bandpass_filter',
    'apply_eeg_filter',
    'normalize_features',
    'load_precomputed_features',
    'save_features',
    'ensure_dir'
]
