#!/usr/bin/env python3
"""Phase A: Set run_id=2 and generate all figures for Nature paper."""
import json, os, sys

# ── 1. Set run_id=2 ──────────────────────────────────────
cfg_path = "E:/MEM/paper/manuscript_config.json"
with open(cfg_path, encoding='utf-8') as f:
    cfg = json.load(f)
cfg['manuscript_config']['run_id'] = 2
with open(cfg_path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print(f"[OK] run_id=2 written to {cfg_path}")

# ── 2. Ensure output dir exists ──────────────────────────
output_dir = "E:/MEM/paper/generated_manuscript/run2/sp"
os.makedirs(output_dir, exist_ok=True)
print(f"[OK] Output dir: {output_dir}")

# ── 3. Copy over reusable figures from run1 that still apply ──
import shutil
run1_sp = "E:/MEM/paper/generated_manuscript/run1/sp"
run1_fig = "E:/MEM/paper/generated_manuscript/run1"
for fname in os.listdir(run1_sp):
    if fname.endswith(('.png', '.pdf', '.jpg')):
        shutil.copy2(os.path.join(run1_sp, fname), os.path.join(output_dir, fname))
for fname in os.listdir(run1_fig):
    if fname.endswith(('.png', '.pdf', '.jpg')) and 'Figure' in fname:
        shutil.copy2(os.path.join(run1_fig, fname), os.path.join(output_dir, f"..\\{fname}"))
print(f"[OK] Copied reusable figures from run1")
