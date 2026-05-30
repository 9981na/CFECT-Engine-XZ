
"""
CFECT Statistics Module
Statistical analysis tools for critical slowing down detection

Contains:
- lmm_evaluator.py: Mixed-effects regression analyzer
- ols_trend_flow.py: OLS trend regression with renormalization
- fdr_chrono_bin.py: FDR-corrected chronological binning
- critical_slowing_test.py: Mann-Kendall + CI + threshold verification (F1)
- stationarity_prescreen.py: Augmented Dickey-Fuller stationarity test
"""

from .lmm_evaluator import LMMEvaluator
from .ols_trend_flow import OLSTrendFlow
from .fdr_chrono_bin import FDRChronoBin
from .critical_slowing_test import (
    verify_csd,
    mann_kendall_trend,
    compute_confidence_interval,
    augment_with_dickey_fuller
)
from .stationarity_prescreen import (
    adf_stationarity_test,
    check_epoch_stationarity,
    adaptive_max_scale
)

__all__ = [
    'LMMEvaluator', 'OLSTrendFlow', 'FDRChronoBin',
    'verify_csd', 'mann_kendall_trend', 'compute_confidence_interval',
    'augment_with_dickey_fuller',
    'adf_stationarity_test', 'check_epoch_stationarity', 'adaptive_max_scale'
]
__version__ = '1.2.0'
__author__ = 'CFECT Quantum Engine Team'
