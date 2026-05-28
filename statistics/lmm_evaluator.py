
"""
LMM Evaluator Module
Group-level multi-patient random intercept MixedLM regression analyzer
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

class LMMEvaluator:
    """
    Mixed-effects model evaluator for group-level analysis.
    
    Implements:
    - Random intercept models to absorb individual heterogeneity
    - Fixed effect estimation for condition effects
    - Comprehensive model diagnostics
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
    
    def fit_model(self, formula: str, group_col: str = 'Patient_ID') -> dict:
        """
        Fit mixed-effects model with random intercept.
        
        Args:
            formula: Formula string for model specification
            group_col: Column name for grouping variable
            
        Returns:
            Dictionary containing model results
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        model = smf.mixedlm(formula, self.data, groups=self.data[group_col])
        result = model.fit()
        
        return {
            'model': model,
            'result': result,
            'params': result.params.to_dict(),
            'pvalues': result.pvalues.to_dict(),
            'tvalues': result.tvalues.to_dict(),
            'aic': result.aic,
            'bic': result.bic,
            'loglike': result.llf
        }
    
    def fit_all_csd_models(self) -> None:
        """
        Fit all CSD-related mixed-effects models.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Clean data
        df_clean = self.data.dropna(subset=['Variance_Z', 'Phi1_Z', 'Time_to_Onset']).copy()
        df_clean['condition_code'] = np.where(df_clean['Condition'] == 'Pre-ictal', 1, 0)
        
        # Fit models
        self.models['phi1_condition'] = self.fit_model(
            'Phi1_Z ~ condition_code', 
            'Patient_ID'
        )
        self.models['variance_condition'] = self.fit_model(
            'Variance_Z ~ condition_code',
            'Patient_ID'
        )
        self.models['phi1_time'] = self.fit_model(
            'Phi1_Z ~ Time_to_Onset',
            'Patient_ID'
        )
        self.models['variance_time'] = self.fit_model(
            'Variance_Z ~ Time_to_Onset',
            'Patient_ID'
        )
        
        # Extract key coefficients
        self.results['beta_phi1_condition'] = self.models['phi1_condition']['params'].get('condition_code', np.nan)
        self.results['beta_var_condition'] = self.models['variance_condition']['params'].get('condition_code', np.nan)
        self.results['beta_phi1_time'] = self.models['phi1_time']['params'].get('Time_to_Onset', np.nan)
        self.results['beta_var_time'] = self.models['variance_time']['params'].get('Time_to_Onset', np.nan)
    
    def summarize_results(self) -> None:
        """
        Print comprehensive summary of all fitted models.
        """
        print("\n" + "="*60)
        print("MIXED-EFFECTS MODEL RESULTS SUMMARY")
        print("="*60)
        
        for model_name, model_data in self.models.items():
            print(f"\n--- {model_name.upper()} ---")
            print(model_data['result'].summary().tables[1])
    
    def get_coefficient_report(self) -> dict:
        """
        Get standardized coefficient report.
        
        Returns:
            Dictionary with verified coefficients
        """
        return {
            'beta_phi1z': self.results.get('beta_phi1_condition', np.nan),
            'beta_varz': self.results.get('beta_var_condition', np.nan),
            'beta_phi1_time': self.results.get('beta_phi1_time', np.nan),
            'beta_var_time': self.results.get('beta_var_time', np.nan),
            'verified': self._verify_coefficients()
        }
    
    def _verify_coefficients(self) -> bool:
        """
        Verify coefficients match expected values.
        
        Expected:
        - beta_phi1z = +0.436
        - beta_varz = -0.107
        
        Returns:
            True if coefficients match within tolerance
        """
        target_phi1 = 0.436
        target_var = -0.107
        tolerance = 0.01
        
        phi1_ok = abs(self.results.get('beta_phi1_condition', np.nan) - target_phi1) < tolerance
        var_ok = abs(self.results.get('beta_var_condition', np.nan) - target_var) < tolerance
        
        return phi1_ok and var_ok
