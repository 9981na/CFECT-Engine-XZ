#!/usr/bin/env python3
"""Query CHB-MIT permutation test values from real data."""
import pandas as pd, numpy as np, os

path = r"E:\MEM\paper\real\output2\chb_mit_labeled.csv"
if not os.path.exists(path):
    print(f"[FAIL] CHB-MIT CSV not found: {path}")
    exit(1)

chb = pd.read_csv(path)
print(f"[OK] CHB-MIT: {len(chb):,} windows")

if 'Phi1_Z' not in chb.columns or 'Condition' not in chb.columns:
    print("Required columns not found")
    exit(1)

# Pre-ictal vs Inter-ictal
pre = chb[chb['Condition'].str.lower().str.contains('pre', na=False)]['Phi1_Z']
inter = chb[chb['Condition'].str.lower().str.contains('inter', na=False)]['Phi1_Z']

delta = pre.mean() - inter.mean()
from scipy import stats
t_stat, p_val = stats.ttest_ind(pre, inter, equal_var=False)
n1, n2 = len(pre), len(inter)
pooled = np.sqrt(((n1-1)*pre.std()**2 + (n2-1)*inter.std()**2) / (n1+n2-2))
d = delta / (pooled + 1e-10)

# Permutation test
combined = np.concatenate([pre.values, inter.values])
labels = np.concatenate([np.ones(n1), np.zeros(n2)])
true_delta = delta
np.random.seed(42)
perm_deltas = [combined[np.random.permutation(labels)==1].mean() - 
               combined[np.random.permutation(labels)==0].mean() for _ in range(1000)]
perm_p = np.mean(np.abs(perm_deltas) >= np.abs(true_delta))

print(f"\n--- Pre-ictal vs Inter-ictal ---")
print(f"  Pre-ictal:      {pre.mean():.4f} (n={n1})")
print(f"  Inter-ictal:    {inter.mean():.4f} (n={n2})")
print(f"  Delta = {delta:.4f}")
print(f"  t = {t_stat:.4f}, p = {p_val:.6f}")
print(f"  Cohen's d = {d:.4f}")
print(f"  Permutation p (1000) = {perm_p:.4f}")
