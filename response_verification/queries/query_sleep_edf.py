#!/usr/bin/env python3
"""Query Sleep-EDF verification values from real data."""
import pandas as pd, numpy as np, os

path = r"E:\MEM\paper\real\output2\main\sleep_csd_features.csv"
if not os.path.exists(path):
    print(f"[FAIL] Sleep-EDF CSV not found: {path}")
    exit(1)

df = pd.read_csv(path)
print(f"[OK] Sleep-EDF: {len(df):,} windows")

# Stage means
print("\n--- Stage-by-Stage Phi1_Z ---")
for stage in ['W','N1','N2','N3','REM']:
    sd = df[df['Sleep_Stage'] == stage]
    sc = sd[sd['Study_Type'] == 'SC']
    st = sd[sd['Study_Type'] == 'ST']
    print(f"  {stage:5s}: ALL={sd['Phi1_Z'].mean():.4f} (n={len(sd):,}), "
          f"SC={sc['Phi1_Z'].mean():.4f}, ST={st['Phi1_Z'].mean():.4f}")

# Tri-polar wake
print("\n--- Tri-Polar Wake ---")
for cond, mask in [
    ('SC Healthy', (df['Study_Type']=='SC') & (df['Sleep_Stage']=='W')),
    ('ST Placebo', (df['Study_Type']=='ST') & (df['Drug_Type']=='Placebo') & (df['Sleep_Stage']=='W')),
    ('ST Temazepam', (df['Study_Type']=='ST') & (df['Drug_Type']=='Temazepam') & (df['Sleep_Stage']=='W')),
]:
    sub = df[mask]['Phi1_Z']
    print(f"  {cond:20s}: {sub.mean():.4f} (n={len(sub)})" if len(sub) > 0 else f"  {cond:20s}: NO DATA")

# Range
print(f"\n--- Range ---")
print(f"  Min: {df['Phi1_Z'].min():.4f}")
print(f"  Max: {df['Phi1_Z'].max():.4f}")
print(f"  P1:  {df['Phi1_Z'].quantile(0.01):.4f}")
print(f"  P99: {df['Phi1_Z'].quantile(0.99):.4f}")
