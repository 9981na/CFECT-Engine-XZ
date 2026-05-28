
"""
CFECT Statistics Module
Statistical analysis tools for critical slowing down detection

Contains:
- lmm_evaluator.py: Mixed-effects regression analyzer
- ols_trend_flow.py: OLS trend regression with renormalization
- fdr_chrono_bin.py: FDR-corrected chronological binning
"""

from .lmm_evaluator import LMMEvaluator
from .ols_trend_flow import OLSTrendFlow
from .fdr_chrono_bin import FDRChronoBin

__all__ = ['LMMEvaluator', 'OLSTrendFlow', 'FDRChronoBin']
__version__ = '1.0.0'
__author__ = 'CFECT Quantum Engine Team'
