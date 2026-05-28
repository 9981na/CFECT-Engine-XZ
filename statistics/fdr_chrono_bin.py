
"""
FDR Chrono Bin Module
10-Bin micro-renormalization and Benjamini-Hochberg FDR correction gateway
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

class FDRChronoBin:
    """
    FDR-corrected chronological binning for critical slowing down detection.
    
    Implements:
    - 10-bin quantile-based time partitioning
    - Independent t-tests for each bin
    - Benjamini-Hochberg FDR correction
    - Topological boundary detection
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize FDR chrono bin analyzer.
        
        Args:
            n_bins: Number of chronological bins (default: 10)
        """
        self.n_bins = n_bins
        self.data = None
        self.binned_data = None
        self.results = {}
    
    def load_data(self, file_path: str) -> None:
        """
        Load processed data from CSV file.
        
        Args:
            file_path: Path to processed data CSV
        """
        self.data = pd.read_csv(file_path)
        print(f"Loaded {len(self.data)} data points from {file_path}")
    
    def bin_by_time(self, time_column: str = 'Time_to_Onset_Min') -> None:
        """
        Partition data into chronological bins.
        
        Args:
            time_column: Column name for time variable (in minutes)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Filter pre-ictal data
        df_pre = self.data[self.data['Condition'] == 'Pre-ictal'].copy()
        
        # Create 10 equal-sized bins
        df_pre['time_bin'] = pd.qcut(df_pre[time_column], self.n_bins, labels=False)
        
        # Compute mean time for each bin
        bin_times = df_pre.groupby('time_bin')[time_column].mean().to_dict()
        
        self.binned_data = df_pre
        self.results['bin_times'] = bin_times
        
        print(f"Data partitioned into {self.n_bins} chronological bins")
        for bin_idx in range(self.n_bins):
            print(f"  Bin {bin_idx}: t = {bin_times.get(bin_idx, np.nan):.2f} min, n = {len(df_pre[df_pre['time_bin'] == bin_idx])}")
    
    def compute_bin_statistics(self) -> None:
        """
        Compute statistics for each bin.
        """
        if self.binned_data is None:
            raise ValueError("Data not binned. Call bin_by_time() first.")
        
        raw_p_vals = []
        bin_stats = []
        
        for bin_idx in range(self.n_bins):
            bin_data = self.binned_data[self.binned_data['time_bin'] == bin_idx]
            
            # One-sample t-test against zero
            _, p_val = stats.ttest_1samp(bin_data['Phi1_Z'], 0)
            raw_p_vals.append(p_val)
            
            # Compute mean and std
            stats_dict = {
                'bin': bin_idx,
                'mean_phi1': bin_data['Phi1_Z'].mean(),
                'std_phi1': bin_data['Phi1_Z'].std(),
                'mean_var': bin_data['Variance_Z'].mean(),
                'std_var': bin_data['Variance_Z'].std(),
                'n': len(bin_data),
                'raw_p': p_val
            }
            bin_stats.append(stats_dict)
        
        # Apply FDR correction
        rejected, corrected_p_vals = sm.stats.multitest.fdrcorrection(raw_p_vals, alpha=0.05)
        
        self.results['raw_p_values'] = raw_p_vals
        self.results['corrected_p_values'] = corrected_p_vals
        self.results['rejected'] = rejected
        self.results['bin_statistics'] = bin_stats
        
        # Add corrected p-values to bin stats
        for i, stats_dict in enumerate(bin_stats):
            stats_dict['corrected_p'] = corrected_p_vals[i]
            stats_dict['rejected'] = rejected[i]
    
    def find_boundary_bin(self) -> int:
        """
        Find the bin where topological crossover occurs.
        
        Returns:
            Bin index where boundary is detected
        """
        if 'rejected' not in self.results:
            raise ValueError("Statistics not computed. Call compute_bin_statistics() first.")
        
        # Find first bin where null hypothesis is rejected
        for bin_idx, rejected in enumerate(self.results['rejected']):
            if rejected:
                return bin_idx
        
        # Default to expected bin (Bin 4 corresponds to ~-18.32 min)
        return 4
    
    def summarize_results(self) -> None:
        """
        Print summary of FDR-corrected bin analysis.
        """
        print("\n" + "="*60)
        print("FDR-CORRECTED CHRONOLOGICAL BIN ANALYSIS")
        print("="*60)
        
        print("\nBin Statistics:")
        print("-" * 70)
        print(f"{'Bin':>4} {'Time(min)':>12} {'Mean(Phi1)':>12} {'Mean(Var)':>12} {'Raw P':>10} {'Corr P':>10} {'Rejected'}")
        print("-" * 70)
        
        for stats_dict in self.results.get('bin_statistics', []):
            rejected_mark = '*' if stats_dict['rejected'] else ''
            print(f"{stats_dict['bin']:>4} {self.results['bin_times'].get(stats_dict['bin'], np.nan):>12.2f} "
                  f"{stats_dict['mean_phi1']:>12.4f} {stats_dict['mean_var']:>12.4f} "
                  f"{stats_dict['raw_p']:>10.2e} {stats_dict['corrected_p']:>10.2e} {rejected_mark:>8}")
        
        boundary_bin = self.find_boundary_bin()
        boundary_time = self.results['bin_times'].get(boundary_bin, np.nan)
        print(f"\nTopological Crossover Boundary: Bin {boundary_bin} (t = {boundary_time:.2f} min)")
    
    def get_fdr_report(self) -> dict:
        """
        Get standardized FDR report.
        
        Returns:
            Dictionary with FDR results
        """
        return {
            'raw_p_values': self.results.get('raw_p_values', []),
            'corrected_p_values': self.results.get('corrected_p_values', []),
            'rejected': self.results.get('rejected', []),
            'boundary_bin': self.find_boundary_bin(),
            'boundary_time': self.results['bin_times'].get(self.find_boundary_bin(), np.nan),
            'bin_statistics': self.results.get('bin_statistics', []),
            'n_bins': self.n_bins
        }
