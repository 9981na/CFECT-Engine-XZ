
"""
CFECT Core Module
Constructive Free-Energy Condensate Theory - Core Solvers

Contains:
- rolling_solver.py: Spatial manifold vectorized computation
- graybox_pinn.py: Physics-informed neural network
- path_decoder.py: HMM-Viterbi path integral decoder
"""

from .rolling_solver import RollingSolver
from .graybox_pinn import GrayBoxPINN
from .path_decoder import PathDecoder

__all__ = ['RollingSolver', 'GrayBoxPINN', 'PathDecoder']
__version__ = '1.0.0'
__author__ = 'CFECT Quantum Engine Team'
