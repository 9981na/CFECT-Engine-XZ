
"""
Path Decoder Module
HMM-Viterbi minimum action path integral decoder with macro-state continuity constraints
"""

import numpy as np
from scipy.special import logsumexp

class PathDecoder:
    """
    Constrained HMM-Viterbi path integral decoder.
    
    Implements:
    - Hidden Markov Model with state continuity constraints
    - Minimum action principle for path selection
    - Transition probability as kinetic barrier
    - Emission probability as potential bias
    """
    
    def __init__(self, n_states: int = 4):
        """
        Initialize path decoder with specified number of states.
        
        Args:
            n_states: Number of macro-states (default: 4 for wake/N1/N2/N3)
        """
        self.n_states = n_states
        self.transition_matrix = None
        self.emission_params = None
    
    def initialize_transition_matrix(self, barriers: np.ndarray = None):
        """
        Initialize transition probability matrix using kinetic barriers.
        
        Args:
            barriers: Matrix of transition barriers (n_states x n_states)
                      Higher barrier = lower transition probability
        """
        if barriers is None:
            # Default sleep architecture constraints
            # Rows: from state, Columns: to state
            barriers = np.array([
                [0.0, 1.0, 2.0, 3.0],   # Wake transitions
                [1.0, 0.0, 1.0, 2.0],   # N1 transitions
                [2.0, 1.0, 0.0, 1.0],   # N2 transitions
                [3.0, 2.0, 1.0, 0.0],   # N3 transitions
            ])
        
        # Convert barriers to probabilities using Boltzmann-like distribution
        self.transition_matrix = np.exp(-barriers)
        
        # Normalize rows
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
    
    def initialize_emission_params(self, means: np.ndarray = None, covs: np.ndarray = None):
        """
        Initialize emission parameters (Gaussian means and covariances).
        
        Args:
            means: Mean vectors for each state (n_states x n_features)
            covs: Covariance matrices for each state (n_states x n_features x n_features)
        """
        if means is None:
            # Default means based on typical EEG features
            means = np.array([
                [0.5, 0.8],   # Wake: high variance, high autocorrelation
                [0.2, 0.4],   # N1: moderate variance, moderate autocorrelation
                [-0.3, 0.6],  # N2: lower variance, higher autocorrelation
                [-0.8, 0.9],   # N3: low variance, very high autocorrelation
            ])
        
        if covs is None:
            # Default diagonal covariances
            n_features = means.shape[1]
            covs = np.array([np.eye(n_features) * 0.1 for _ in range(self.n_states)])
        
        self.emission_params = {
            'means': means,
            'covs': covs
        }
    
    def log_emission_prob(self, observation: np.ndarray, state: int) -> float:
        """
        Compute log emission probability for an observation given a state.
        
        Args:
            observation: Feature vector
            state: State index
            
        Returns:
            Log probability
        """
        mean = self.emission_params['means'][state]
        cov = self.emission_params['covs'][state]
        
        diff = observation - mean
        inv_cov = np.linalg.inv(cov)
        log_det = np.log(np.linalg.det(cov))
        
        # Multivariate Gaussian log probability
        log_prob = -0.5 * (diff @ inv_cov @ diff.T + log_det + len(mean) * np.log(2 * np.pi))
        
        return log_prob
    
    def viterbi_decode(self, observations: np.ndarray) -> np.ndarray:
        """
        Perform Viterbi decoding with minimum action principle.
        
        Args:
            observations: Sequence of observations (n_time_steps x n_features)
            
        Returns:
            Array of state indices for each time step
        """
        n_time = observations.shape[0]
        
        # Initialize Viterbi and backpointer matrices
        log_viterbi = np.zeros((n_time, self.n_states))
        backpointer = np.zeros((n_time, self.n_states), dtype=int)
        
        # Initial state probabilities (uniform)
        log_viterbi[0] = np.log(1.0 / self.n_states) + \
                         np.array([self.log_emission_prob(observations[0], s) for s in range(self.n_states)])
        
        # Forward pass
        for t in range(1, n_time):
            for s in range(self.n_states):
                # Compute log probabilities of transitioning from each previous state
                log_probs = log_viterbi[t-1] + np.log(self.transition_matrix[:, s])
                best_prev_state = np.argmax(log_probs)
                log_viterbi[t, s] = log_probs[best_prev_state] + self.log_emission_prob(observations[t], s)
                backpointer[t, s] = best_prev_state
        
        # Backward pass to find best path
        path = np.zeros(n_time, dtype=int)
        path[-1] = np.argmax(log_viterbi[-1])
        
        for t in range(n_time - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]
        
        return path
    
    def minimum_action_path(self, observations: np.ndarray, 
                           potential_weight: float = 1.0, 
                           kinetic_weight: float = 1.0) -> np.ndarray:
        """
        Find minimum action path integrating potential and kinetic costs.
        
        Args:
            observations: Sequence of observations
            potential_weight: Weight for emission (potential) cost
            kinetic_weight: Weight for transition (kinetic) cost
            
        Returns:
            Array of state indices for minimum action path
        """
        n_time = observations.shape[0]
        
        # Action cost matrices
        action = np.zeros((n_time, self.n_states))
        backpointer = np.zeros((n_time, self.n_states), dtype=int)
        
        # Initial action
        action[0] = np.array([self._potential_cost(observations[0], s) for s in range(self.n_states)])
        
        # Forward pass with action minimization
        for t in range(1, n_time):
            for s in range(self.n_states):
                # Compute total action from each previous state
                potential_cost = self._potential_cost(observations[t], s)
                transition_costs = kinetic_weight * self._kinetic_cost(np.arange(self.n_states), s)
                
                total_costs = action[t-1] + transition_costs + potential_cost
                best_prev = np.argmin(total_costs)
                
                action[t, s] = total_costs[best_prev]
                backpointer[t, s] = best_prev
        
        # Backward pass
        path = np.zeros(n_time, dtype=int)
        path[-1] = np.argmin(action[-1])
        
        for t in range(n_time - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]
        
        return path
    
    def _potential_cost(self, observation: np.ndarray, state: int) -> float:
        """Compute potential cost (negative log emission probability)."""
        return -self.log_emission_prob(observation, state)
    
    def _kinetic_cost(self, from_states: np.ndarray, to_state: int) -> np.ndarray:
        """Compute kinetic cost (transition barrier)."""
        return -np.log(self.transition_matrix[from_states, to_state])
