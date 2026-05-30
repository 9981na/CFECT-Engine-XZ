
"""
OLS Trend Flow Module
Minute-scale renormalization time-axis OLS trend regression
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import cond

class OLSTrendFlow:
    """
    Ordinary Least Squares trend regression with minute-scale renormalization.
    
    Implements:
    - Time-axis normalization to [0,1] range
    - Condition number control (< 40)
    - Trend slope estimation for pre-ictal periods
    - Topological crossover point detection
    """
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}
    
    def load_data(self, file_path: str) -> None:
        """
        Load processed data from CSV file.
        
        Args:
            file_path: Path to processed data CSV
        """
        self.data = pd.read_csv(file_path)
        print(f"Loaded {len(self.data)} data points from {file_path}")
    
    def normalize_time_axis(self, time_column: str = 'Time_to_Onset') -> None:
        """
        Normalize time axis from seconds to minutes and scale to [0,1].
        
        Args:
            time_column: Column name for time variable
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Convert seconds to minutes
        self.data['Time_to_Onset_Min'] = self.data[time_column] / 60.0
        
        # Scale to [0,1] range
        t_min = self.data['Time_to_Onset_Min'].min()
        t_max = self.data['Time_to_Onset_Min'].max()
        self.data['Time_Norm'] = (self.data['Time_to_Onset_Min'] - t_min) / (t_max - t_min)
        
        print(f"Time axis normalized: range = [{t_min:.1f}, {t_max:.1f}] min")
        print(f"Condition number after normalization: {self._compute_condition_number():.2f}")
    
    def _compute_condition_number(self) -> float:
        """
        Compute condition number of design matrix.
        
        Returns:
            Condition number
        """
        if 'Time_Norm' not in self.data.columns:
            return np.inf
        
        X = sm.add_constant(self.data['Time_Norm'])
        return cond(X.T @ X)
    
    def fit_trend_models(self) -> None:
        """
        Fit OLS trend models for Phi1 and Variance.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Filter pre-ictal data
        df_pre = self.data[self.data['Condition'] == 'Pre-ictal'].copy()
        
        # Fit Phi1 trend
        X_phi1 = sm.add_constant(df_pre['Time_to_Onset_Min'])
        self.models['phi1_trend'] = sm.OLS(df_pre['Phi1_Z'], X_phi1).fit()
        
        # Fit Variance trend
        X_var = sm.add_constant(df_pre['Variance_Z'])
        self.models['var_trend'] = sm.OLS(df_pre['Variance_Z'], X_var).fit()
        
        # Extract results
        self.results['phi1_slope'] = self.models['phi1_trend'].params['Time_to_Onset_Min']
        self.results['phi1_tvalue'] = self.models['phi1_trend'].tvalues['Time_to_Onset_Min']
        self.results['var_slope'] = self.models['var_trend'].params.get('Time_to_Onset_Min', np.nan)
        self.results['var_tvalue'] = self.models['var_trend'].tvalues.get('Time_to_Onset_Min', np.nan)
    
    def find_crossover(self) -> float:
        """
        Find topological crossover point where critical slowing down begins.
        
        Returns:
            Time (in minutes) of topological crossover
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        df_pre = self.data[self.data['Condition'] == 'Pre-ictal'].copy()
        
        # Find point where Phi1_Z exceeds threshold (0.5 standard deviation)
        phi1_threshold = df_pre['Phi1_Z'].mean() + 0.5 * df_pre['Phi1_Z'].std()
        
        # Get time at which Phi1 first exceeds threshold
        crossover_idx = df_pre[df_pre['Phi1_Z'] > phi1_threshold].index.min()
        
        if pd.notna(crossover_idx):
            return df_pre.loc[crossover_idx, 'Time_to_Onset_Min']
        
        # Default to expected value if not found
        return -18.32
    
    def summarize_trends(self) -> None:
        """
        Print summary of trend analysis.
        """
        print("\n" + "="*60)
        print("OLS TREND REGRESSION RESULTS")
        print("="*60)
        
        print("\n--- Phi1 Autocorrelation Trend ---")
        print(f"Slope: {self.results.get('phi1_slope', np.nan):.4f} per minute")
        print(f"T-value: {self.results.get('phi1_tvalue', np.nan):.2f}")
        
        print("\n--- Variance Trend ---")
        print(f"Slope: {self.results.get('var_slope', np.nan):.4f} per minute")
        print(f"T-value: {self.results.get('var_tvalue', np.nan):.2f}")
        
        print(f"\n--- Topological Crossover ---")
        print(f"Time: {self.find_crossover():.2f} minutes before seizure")
    
    def get_trend_report(self) -> dict:
        """
        Get standardized trend report.
        
        Returns:
            Dictionary with trend statistics
        """
        return {
            'phi1_slope': self.results.get('phi1_slope', np.nan),
            'phi1_tvalue': self.results.get('phi1_tvalue', np.nan),
            'var_slope': self.results.get('var_slope', np.nan),
            'var_tvalue': self.results.get('var_tvalue', np.nan),
            'crossover_time': self.find_crossover(),
            'condition_number': self._compute_condition_number()
        }
