#!/usr/bin/env python3
"""Query SDDB verification values from real data."""
import pandas as pd, os

path = r"E:\MEM\paper\real\output2\sddb_labeled.csv"
if not os.path.exists(path):
    print(f"[FAIL] SDDB CSV not found: {path}")
    exit(1)

df = pd.read_csv(path)
print(f"[OK] SDDB: {len(df):,} rows")
print(f"  Columns: {list(df.columns)}")
print(f"  Shape: {df.shape}")
