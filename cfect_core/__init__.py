
"""
CFECT Core Module
Constructive Free-Energy Condensate Theory - Core Solvers

Contains:
- rolling_solver.py: Spatial manifold vectorized computation
- graybox_pinn.py: Physics-informed neural network
- path_decoder.py: HMM-Viterbi path integral decoder
- fluctuation_theorem.py: Gallavotti-Cohen symmetry verification
- integrated_info.py: Phi_E integrated information computation
- spatial_ews.py: Spatial early warning signals
"""

from .rolling_solver import RollingSolver
from .graybox_pinn import GrayBoxPINN
from .path_decoder import PathDecoder
from .fluctuation_theorem import (
    compute_forward_backward_ratio,
    compute_entropy_production,
    verify_thermodynamic_consistency,
    honest_reparameterization
)
from .integrated_info import (
    effective_info,
    compute_phi_e,
    renorm
)
from .spatial_ews import (
    spatial_variance,
    morans_i,
    spatial_skewness_kurtosis,
    verify_spatial_ews
)

__all__ = [
    'RollingSolver', 'GrayBoxPINN', 'PathDecoder',
    'compute_forward_backward_ratio', 'compute_entropy_production',
    'verify_thermodynamic_consistency', 'honest_reparameterization',
    'effective_info', 'compute_phi_e', 'renorm',
    'spatial_variance', 'morans_i', 'spatial_skewness_kurtosis', 'verify_spatial_ews'
]
__version__ = '1.1.0'
__author__ = 'CFECT Quantum Engine Team'
