#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFECT Visualization Module - Nature Publication Quality Figures

This module implements:
1. Permutation test with 1,000 random permutations of time labels
2. Nature-spec compliant dual-panel figure generation
3. Statistical verification with empirical p-value computation
4. Export to both PNG (300 dpi) and PDF formats
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import norm
import statsmodels.api as sm

def configure_nature_style():
    """Configure matplotlib for Nature publication quality."""
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    rcParams['font.size'] = 8.5
    rcParams['axes.linewidth'] = 0.75
    rcParams['lines.linewidth'] = 1.1
    rcParams['xtick.major.width'] = 0.75
    rcParams['ytick.major.width'] = 0.75
    rcParams['xtick.major.size'] = 3.5
    rcParams['ytick.major.size'] = 3.5
    rcParams['legend.fontsize'] = 6.8
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.unicode_minus'] = False
    
    COLORS = {
        'VAR': '#2c5282',
        'RIGID': '#9b2c2c',
        'NULL': '#cbd5e0',
        'TRUE': '#e53e3e'
    }
    return COLORS

def run_unified_permutation_and_plotting(master_csv_path, n_permutations=1000, output_dir='.'):
    """
    Run permutation test and generate Supplementary Figure 1.
    
    Args:
        master_csv_path: Path to chb_mit_csd_master.csv
        n_permutations: Number of permutations (default: 1000)
        output_dir: Output directory for figures
    
    Returns:
        empirical_p: Empirical p-value from permutation test
    """
    print("====== 启动 CFECT 核心置换检验与随附生图一体化引擎 ======")
    
    COLORS = configure_nature_style()
    np.random.seed(42)
    
    df = pd.read_csv(master_csv_path)
    df_pre = df[df['Condition'] == 'Pre-ictal'].dropna(subset=['Variance_Z', 'Phi1_Z', 'Time_to_Onset']).copy()
    df_pre['Time_Min'] = df_pre['Time_to_Onset'] / 60.0
    
    df_pre['time_bin_10'] = pd.qcut(df_pre['Time_Min'], 10, labels=False)
    true_profile = df_pre.groupby('time_bin_10').agg({
        'Time_Min': 'mean', 
        'Variance_Z': 'mean', 
        'Phi1_Z': 'mean'
    }).reset_index()
    
    X_true = sm.add_constant(df_pre['Time_Min'])
    ols_true_phi1 = sm.OLS(df_pre['Phi1_Z'], X_true).fit()
    true_slope = ols_true_phi1.params['Time_Min']
    print(f"[真实观测] 恢复力记忆项 Phi1_Z 随时间演进的真实线性爬升斜率: {true_slope:.5f}")
    
    print(f"[数据重整] 正在执行 {n_permutations} 次零假设流形无序置换...")
    permuted_slopes = []
    sample_shuffled_trajectories = []
    
    for perm in range(n_permutations):
        shuffled_times = np.random.permutation(df_pre['Time_Min'].values)
        ols_perm = sm.OLS(df_pre['Phi1_Z'].values, sm.add_constant(shuffled_times)).fit()
        permuted_slopes.append(ols_perm.params['x1'])
        
        if perm < 5:
            df_perm_tmp = pd.DataFrame({
                'Time_Min': shuffled_times,
                'Phi1_Z': df_pre['Phi1_Z'].values
            })
            df_perm_tmp['bin'] = pd.qcut(df_perm_tmp['Time_Min'], 10, labels=False)
            perm_profile = df_perm_tmp.groupby('bin').agg({'Time_Min': 'mean', 'Phi1_Z': 'mean'}).reset_index()
            sample_shuffled_trajectories.append(perm_profile)
            
        if (perm + 1) % 200 == 0:
            print(f"  置换进度: {perm + 1}/{n_permutations}")
    
    permuted_slopes = np.array(permuted_slopes)
    
    empirical_p = (np.sum(permuted_slopes >= true_slope) + 1) / (n_permutations + 1)
    print(f"[置换审计] 经验推断完成。P-value = {empirical_p:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2))
    
    for idx, perm_p in enumerate(sample_shuffled_trajectories):
        label = 'Time-permuted Null Paths' if idx == 0 else None
        ax1.plot(perm_p['Time_Min'], perm_p['Phi1_Z'], color=COLORS['NULL'],
                 linestyle='--', linewidth=0.75, alpha=0.7, label=label)
    
    ax1.plot(true_profile['Time_Min'], true_profile['Phi1_Z'], color=COLORS['RIGID'],
             marker='s', markersize=3, markerfacecolor='white', markeredgewidth=0.8,
             linestyle='-', linewidth=1.2, label=r'Observed $\Phi_{1_Z}$ (Rigidity Rise)', zorder=4)
    
    ax1.plot(true_profile['Time_Min'], true_profile['Variance_Z'], color=COLORS['VAR'],
             marker='o', markersize=3, markerfacecolor='white', markeredgewidth=0.8,
             linestyle='-', linewidth=1.2, label=r'Observed $Variance_Z$ (Energy Sink)', zorder=4)
    
    ax1.axvline(-18.32, color='black', linestyle=':', linewidth=0.8, alpha=0.6)
    ax1.text(-17.5, 0.22, r'$t = -18.32$ min' + '\nCrossover', fontsize=7, color='black', alpha=0.8)
    
    ax1.set_xlabel('Time to Seizure Onset (minutes)', fontsize=8)
    ax1.set_ylabel(r'Metric Fluctuations ($Z$-Score)', fontsize=8)
    ax1.set_xlim(-30.5, -0.5)
    ax1.set_ylim(-0.3, 0.3)
    ax1.tick_params(direction='in', labelsize=7.5)
    ax1.legend(frameon=False, loc='lower left', handletextpad=0.3)
    ax1.text(-0.18, 1.05, 'a', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
    
    counts, bins, patches = ax2.hist(permuted_slopes, bins=35, color='#e2e8f0',
                                     edgecolor='#cbd5e0', linewidth=0.5, density=True,
                                     label='Null Slopes ($\Delta t$ Shuffled)')
    
    mu, std = np.mean(permuted_slopes), np.std(permuted_slopes)
    xmin, xmax = ax2.get_xlim()
    x_axis = np.linspace(xmin, xmax, 100)
    ax2.plot(x_axis, norm.pdf(x_axis, mu, std), color='#718096', linestyle='-', linewidth=0.8, alpha=0.8)
    
    p_label = f'Observed Slope ($p = {empirical_p:.3f}$)'
    ax2.axvline(true_slope, color=COLORS['TRUE'], linestyle='-', linewidth=1.2, label=p_label)
    
    ax2.set_xlabel(r'Linear Trajectory Slope $\beta_{time}$', fontsize=8)
    ax2.set_ylabel('Probability Density', fontsize=8)
    ax2.tick_params(direction='in', labelsize=7.5)
    ax2.set_xlim(permuted_slopes.min() - 0.002, true_slope + 0.003)
    ax2.legend(frameon=False, loc='upper left', handletextpad=0.4)
    ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/Supplementary_Figure_1.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Supplementary_Figure_1.pdf', bbox_inches='tight')
    plt.close()
    
    pd.DataFrame({'permuted_slopes': permuted_slopes}).to_csv(f'{output_dir}/cfect_shuffled_slopes_matrix.csv', index=False)
    
    print(f"SUCCESS: Supplementary Figure 1 exported to {output_dir}")
    return empirical_p

if __name__ == "__main__":
    run_unified_permutation_and_plotting("data/processed/chb_mit_csd_master.csv")