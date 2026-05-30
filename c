"""Quick check CSV columns."""
import pandas as pd, numpy as np
df = pd.read_csv("E:/MEM/paper/real/data/chb_mit_csd_master.csv")
print("Cols:", list(df.columns))
print("Shape:", df.shape)
print("Cond unique:", df["Condition"].unique())
print("Has Phi1_Z:", "Phi1_Z" in df.columns)
pre = df[df["Condition"] == "Pre-ictal"]
inter = df[df["Condition"] == "Inter-ictal"]
print(f"Pre: {len(pre)}, Inter: {len(inter)}")
col = "Phi1_Z" if "Phi1_Z" in df.columns else "Phi1"
pv = pre[col].dropna().values
iv = inter[col].dropna().values
print(f"Using col: {col}, Pre vals: {len(pv)}, Inter vals: {len(iv)}")
if len(pv) > 0 and len(iv) > 0:
    print(f"Pre mean: {pv.mean():.4f}, Inter mean: {iv.mean():.4f}")
    print(f"Diff: {pv.mean() - iv.mean():.4f}")

# Also check SDDB
df2 = pd.read_csv("E:/MEM/paper/real/data/sddb_terminal_master.csv")
print("\nSDDB shape:", df2.shape)
print("SDDB cols:", list(df2.columns))
