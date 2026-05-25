"""
CFECT-Quantum-Engine: Dynamic Potential Barrier and Path Integral Module
Implements Maximum Likelihood transition matrix learning and Viterbi decoding.
"""

import numpy as np

# Number of sleep stages per AASM standards
NUM_CLASSES: int = 5  # Wake, N1, N2, N3, REM

# Default transition matrix from prior physiological knowledge
# Rows: from state, Columns: to state
DEFAULT_TRANSITION_MATRIX: np.ndarray = np.array([
    [0.85, 0.08, 0.04, 0.00, 0.03],  # Wake transitions
    [0.12, 0.45, 0.35, 0.00, 0.08],  # N1 transitions
    [0.04, 0.02, 0.85, 0.05, 0.04],  # N2 transitions
    [0.01, 0.00, 0.19, 0.80, 0.00],  # N3 transitions
    [0.06, 0.02, 0.12, 0.00, 0.80]   # REM transitions
])

def learn_transition_matrix(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Learns the topological potential barrier matrix via Maximum Likelihood Estimation.
    
    Mathematical Formalism:
        A_ij = (count(y_t = j | y_{t-1} = i) + 1) / (count(y_{t-1} = i) + C)
        where C = num_classes (Laplace smoothing factor)
    
    Args:
        labels: 1D array of sequential sleep stage labels (shape: [N_epochs])
        num_classes: Number of discrete states (default: 5)
    
    Returns:
        np.ndarray: Learned transition probability matrix (shape: [num_classes, num_classes])
    """
    # Initialize with Laplace smoothing (add 1 to prevent zero probabilities)
    transition_counts: np.ndarray = np.ones((num_classes, num_classes))
    
    for i in range(len(labels) - 1):
        from_state: int = labels[i]
        to_state: int = labels[i + 1]
        transition_counts[from_state, to_state] += 1
    
    # Normalize by row sums to convert counts to probabilities
    row_sums: np.ndarray = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix: np.ndarray = transition_counts / row_sums
    
    return transition_matrix

def viterbi_path_integral(
    emission_probs: np.ndarray, 
    transition_matrix: np.ndarray = DEFAULT_TRANSITION_MATRIX
) -> np.ndarray:
    """
    Minimizes the global trajectory energy functional via Lagrangian Path Integral constraint.
    
    Mathematical Formalism:
        S(Y) = -Σ_{t=1}^{T} [ ln B(y_t, X_t) + ln A(y_{t-1}, y_t) ]
        
        where:
            S(Y) = Action functional (path cost)
            B(y_t, X_t) = Emission probability from static classifier
            A(y_{t-1}, y_t) = Transition probability from learned barrier matrix
    
    Args:
        emission_probs: Conditional probabilities from static constructor (shape: [N_epochs, N_classes])
        transition_matrix: Topological potential barrier matrix (shape: [N_classes, N_classes])
    
    Returns:
        np.ndarray: Globally optimal continuous phase sequence (shape: [N_epochs])
    """
    num_epochs, num_classes = emission_probs.shape
    
    # Log-space computations for numerical stability
    log_transition: np.ndarray = np.log(np.clip(transition_matrix, 1e-12, 1.0))
    log_emission: np.ndarray = np.log(np.clip(emission_probs, 1e-12, 1.0))
    
    # Dynamic programming matrices
    trellis: np.ndarray = np.zeros((num_epochs, num_classes))
    backpointers: np.ndarray = np.zeros((num_epochs, num_classes), dtype=int)
    
    # Initial state: use emission probabilities at t=0
    trellis[0, :] = log_emission[0, :]
    
    # Forward pass: compute minimum action path to each state at each time step
    for t in range(1, num_epochs):
        for current_state in range(num_classes):
            # Find path with minimum action to current state
            transition_costs: np.ndarray = trellis[t - 1, :] + log_transition[:, current_state]
            best_prev_state: int = np.argmax(transition_costs)
            trellis[t, current_state] = transition_costs[best_prev_state] + log_emission[t, current_state]
            backpointers[t, current_state] = best_prev_state
    
    # Backward pass: find the optimal path
    best_path: np.ndarray = np.zeros(num_epochs, dtype=int)
    best_path[-1] = np.argmax(trellis[-1, :])
    
    for t in range(num_epochs - 2, -1, -1):
        best_path[t] = backpointers[t + 1, best_path[t + 1]]
    
    return best_path

def viterbi_path_integral_dynamic(
    emission_probs: np.ndarray, 
    transition_matrix: np.ndarray
) -> np.ndarray:
    """
    Dynamic version of Viterbi path integral that accepts learned transition matrix.
    
    Args:
        emission_probs: Conditional probabilities (shape: [N_epochs, N_classes])
        transition_matrix: Learned transition matrix (shape: [N_classes, N_classes])
    
    Returns:
        np.ndarray: Optimal state sequence (shape: [N_epochs])
    """
    return viterbi_path_integral(emission_probs, transition_matrix)
