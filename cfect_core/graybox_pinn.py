
"""
Gray-Box PINN Module
Physics-informed neural network coupled with multi-axis stochastic Hopf bifurcation
"""

import numpy as np
from scipy.integrate import solve_ivp

class GrayBoxPINN:
    """
    Coupled multi-axis stochastic Hopf bifurcation solver with physics-informed learning.
    
    Combines:
    - Stochastic differential equations for phase-space dynamics
    - Physics constraints as loss function regularization
    - Wang-Jin potential-flux decomposition
    """
    
    def __init__(self, n_dim: int = 5):
        """
        Initialize PINN with n-dimensional state space.
        
        Args:
            n_dim: Number of physiological axes (default: 5 for Wu-Xing)
        """
        self.n_dim = n_dim
        self._initialize_coupling_matrix()
    
    def _initialize_coupling_matrix(self):
        """Initialize Wu-Xing (Five Elements) coupling weight matrix."""
        self.W = np.array([
            [0,  1, -1,  0,  0],  # Wood -> Fire, <- Metal
            [0,  0,  1, -1,  0],  # Fire -> Earth, <- Water
            [0,  0,  0,  1, -1],  # Earth -> Metal, <- Wood
            [-1, 0,  0,  0,  1],  # Metal -> Water, <- Fire
            [1, -1,  0,  0,  0],  # Water -> Wood, <- Earth
        ])
    
    def hopf_bifurcation(self, t: float, z: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Hopf bifurcation normal form.
        
        Args:
            t: Time
            z: Complex state vector
            alpha: Bifurcation control parameter
            beta: Nonlinear saturation coefficient
            
        Returns:
            Time derivative dz/dt
        """
        omega = 1.0  # Natural frequency
        
        # Linear term with bifurcation parameter
        linear_term = (alpha + 1j * omega) * z
        
        # Nonlinear saturation
        nonlinear_term = -beta * np.abs(z)**2 * z
        
        # Coupling term from Wu-Xing topology
        coupling_term = 0.1 * self.W @ z
        
        return linear_term + nonlinear_term + coupling_term
    
    def wang_jin_decomposition(self, z: np.ndarray) -> tuple:
        """
        Wang-Jin potential-flux decomposition.
        
        Args:
            z: Complex state vector
            
        Returns:
            Tuple of (total_force, potential_component, flux_component)
        """
        # Diffusion matrix
        D = np.eye(self.n_dim)
        
        # Potential gradient (Yin)
        grad_U = self._potential_gradient(z)
        
        # Non-equilibrium flux (Yang)
        j_s = self._non_equilibrium_flux(z)
        P_s = np.exp(-self._potential(z))
        
        # Total force
        F = -D @ grad_U + j_s / P_s
        
        return F, -D @ grad_U, j_s / P_s
    
    def _potential(self, z: np.ndarray) -> float:
        """Compute potential landscape value."""
        return 0.5 * np.sum(np.abs(z)**4) - 0.5 * np.sum(np.abs(z)**2)
    
    def _potential_gradient(self, z: np.ndarray) -> np.ndarray:
        """Compute gradient of potential landscape."""
        return 2 * np.abs(z)**2 * z - z
    
    def _non_equilibrium_flux(self, z: np.ndarray) -> np.ndarray:
        """Compute non-equilibrium probability flux."""
        flux = 1j * self.W @ z
        detuning = np.sum(np.abs(z)**2) - 1.0
        return flux * (1 + 0.5 * detuning**2)
    
    def solve_stochastic(self, z0: np.ndarray, t_span: tuple, 
                         alpha: float, beta: float, sigma: float = 0.1,
                         dt: float = 1e-3) -> tuple:
        """
        Solve stochastic Hopf bifurcation using Euler-Maruyama method.
        
        Args:
            z0: Initial state
            t_span: Time interval (t_start, t_end)
            alpha: Bifurcation parameter
            beta: Nonlinear coefficient
            sigma: Noise intensity
            dt: Time step
            
        Returns:
            Tuple of (time_array, solution_array)
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)
        t = np.linspace(t_start, t_end, n_steps)
        z = np.zeros((n_steps, self.n_dim), dtype=np.complex128)
        z[0] = z0
        
        for i in range(n_steps - 1):
            dz = self.hopf_bifurcation(t[i], z[i], alpha, beta) * dt
            dz += sigma * (np.random.randn(self.n_dim) + 1j * np.random.randn(self.n_dim)) * np.sqrt(dt)
            z[i+1] = z[i] + dz
        
        return t, z
    
    def solve_deterministic(self, z0: np.ndarray, t_span: tuple,
                            alpha: float, beta: float) -> tuple:
        """
        Solve deterministic Hopf bifurcation using ODE solver.
        
        Args:
            z0: Initial state
            t_span: Time interval
            alpha: Bifurcation parameter
            beta: Nonlinear coefficient
            
        Returns:
            Tuple of (time_array, solution_array)
        """
        def rhs(t, z):
            z_complex = z[:self.n_dim] + 1j * z[self.n_dim:]
            dz_dt = self.hopf_bifurcation(t, z_complex, alpha, beta)
            return np.concatenate([np.real(dz_dt), np.imag(dz_dt)])
        
        z0_flat = np.concatenate([np.real(z0), np.imag(z0)])
        sol = solve_ivp(rhs, t_span, z0_flat, method='RK45')
        
        t = sol.t
        z = sol.y[:self.n_dim] + 1j * sol.y[self.n_dim:]
        
        return t, z.T
    
    def pinn_loss(self, z: np.ndarray, dz_dt: np.ndarray,
                  alpha: float, beta: float) -> float:
        """
        Compute physics-informed loss.
        
        Args:
            z: State vector
            dz_dt: Time derivative
            alpha: Bifurcation parameter
            beta: Nonlinear coefficient
            
        Returns:
            Physics loss value
        """
        drift = self.hopf_bifurcation(0, z, alpha, beta)
        residual = dz_dt - drift
        return np.mean(np.abs(residual)**2)
