#!/usr/bin/env python3
"""
SDDB AFib 漏洞修复engine v2.0
============================
Phase A: 从 WFDB .ari (rhythm annotations) 提取 4 条 AFib 记录
        解析 Rhythm Label '(AFIB' 的持续区间，计算 rolling features
        追加到 sddb_terminal_master.csv

架构笔记:
  SDDB 的 .atr = beat 级标签 (N, S, V) — 不含 rhythm 信息
  SDDB 的 .ari = rhythm 级标签 — aux_note 含 '(AFIB', '(N', '(ST1-' 等
  + 符号表示 rhythm 起始/切换
"""

import numpy as np
import pandas as pd
import os, sys, warnings
warnings.filterwarnings('ignore')

try:
    import wfdb
except ImportError:
    print("[FAIL] wfdb not installed. Run: pip install wfdb")
    sys.exit(1)

# -- 路径配置 ------------------------------------------------------
DATA_DIR = r"E:\MEM\paper\real\data"
SDDB_DIR = os.path.join(DATA_DIR, "sudden-cardiac-death-holter-database-1.0.0")
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = r"E:\MEM\paper\real\output2"

AFIB_RECORDS = [35, 36, 37, 50]
BASE_COLS = ['Record', 'Time_to_Event', 'DNB_Std', 'External_Correlation', 'Variance', 'Autocorrelation']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def extract_rhythm_segments_from_ari(ann, target_label='(AFIB'):
    """
    从 WFDB .ari (rhythm annotations) 中提取特定 rhythm 持续区间。
    
    SDDB 的 .ari 格式：
      - symbol='+' 表示 rhythm 标签起始点
      - aux_note[i] 包含标签名如 '(AFIB', '(N', '(ST1-'
      - 持续到下一个 '+' 或文件末尾
    
    Returns: list of (start_sample, end_sample)
    """
    segments = []
    current_start = None
    current_label = None
    
    for i in range(len(ann.sample)):
        sym = str(ann.symbol[i]) if i < len(ann.symbol) else ''
        
        # '+' 标记 rhythm 切换点
        if '+' in sym or sym.startswith('+'):
            # 如果当前有活跃 rhythm 且匹配目标，保存 segment
            if current_start is not None and current_label is not None:
                if target_label in current_label:
                    segments.append((current_start, ann.sample[i]))
            
            # 提取新 rhythm 标签
            if i < len(ann.aux_note) and ann.aux_note[i] is not None:
                aux = str(ann.aux_note[i]).strip()
                if aux:
                    current_label = aux
                else:
                    current_label = None
            else:
                current_label = None
            current_start = ann.sample[i]
    
    # 关闭最后的 segment
    if current_start is not None and current_label is not None:
        if target_label in current_label:
            segments.append((current_start, ann.sample[-1]))
    
    return segments


def compute_rolling_features(ecg, fs=250):
    """
    对整个 ECG 信号计算滑动窗口 CFECT 特征。
    窗口 = 7500 (30s @ 250Hz), 步长 = 1875 (75% 重叠)
    
    Returns: (features_array, time_axis)
    features_array: [n_windows, 4] = [Variance, Autocorrelation, DNB_Std, External_Correlation]
    """
    window_size = 7500   # 30s @ 250Hz
    step_size = 1875     # 7.5s (75% overlap)
    
    if len(ecg) < window_size:
        return np.array([]), np.array([])
    
    n_windows = (len(ecg) - window_size) // step_size + 1
    features = np.zeros((n_windows, 4))
    time_axis = np.zeros(n_windows)
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        win = ecg[start:end]
        time_axis[i] = (start + end) / 2 / fs  # center time in seconds
        
        # 1. Variance
        features[i, 0] = np.var(win)
        
        # 2. Lag-1 Autocorrelation
        ac = np.corrcoef(win[:-1], win[1:])[0, 1]
        features[i, 1] = ac if not np.isnan(ac) else 0.0
        
        # 3. DNB_Std (detrended std)
        trend = np.polyval(np.polyfit(np.arange(len(win)), win, 1), np.arange(len(win)))
        features[i, 2] = np.std(win - trend)
        
        # 4. External_Correlation (lag-50 ≈ 200ms)
        if len(win) > 50:
            ec = np.corrcoef(win[:-50], win[50:])[0, 1]
            features[i, 3] = ec if not np.isnan(ec) else 0.0
    
    return features, time_axis


def process_afib_record(record_num):
    """处理单个 AFib 记录并提取特征 DataFrame"""
    print(f"\n{'='*50}")
    print(f"  Record {record_num}")
    print(f"{'='*50}")
    
    rec_str = str(record_num)
    rec_path = os.path.join(SDDB_DIR, rec_str)
    
    # 1. 读取信号
    try:
        sig, fields = wfdb.rdsamp(rec_path, sampto=650000)
        print(f"  [OK] Signal: {sig.shape}, fs={fields.get('fs', '?')} Hz")
    except Exception as e:
        print(f"  [FAIL] Signal read failed: {e}")
        return None
    
    ecg = sig[:, 0]  # MLII 导联
    fs = fields.get('fs', 250)
    
    # 2. 读取 .ari (rhythm) 注释
    try:
        ann = wfdb.rdann(rec_path, 'ari')
        print(f"  [OK] .ari annotations: {ann.sample.shape[0]} total")
    except Exception as e:
        print(f"  [FAIL] .ari read failed: {e}")
        return None
    
    # 3. 提取 AFib rhythm 区间
    afib_segments = extract_rhythm_segments_from_ari(ann, '(AFIB')
    print(f"  [DATA] AFib segments found: {len(afib_segments)}")
    
    if not afib_segments:
        # Debug: 显示所有 rhythm 标签
        all_rhythms = []
        for i in range(len(ann.sample)):
            if i < len(ann.aux_note) and ann.aux_note[i] is not None:
                aux = str(ann.aux_note[i]).strip()
                if aux:
                    all_rhythms.append((ann.sample[i], aux))
        print(f"  [WARN]  All rhythm labels (first 15):")
        for s, aux in all_rhythms[:15]:
            dur = f"~{(all_rhythms[all_rhythms.index((s,aux))+1][0] - s)/fs/60:.0f}min" if (s,aux) != all_rhythms[-1] else "->end"
            print(f"    sample={s} ({s/fs:.1f}s) -> {dur}, label={repr(aux)}")
        return None
    
    # 4. 对每个 AFib 区间计算 CFECT rolling features
    all_rows = []
    time_counter = 0
    
    for seg_start, seg_end in afib_segments:
        dur_min = (seg_end - seg_start) / (fs * 60)
        print(f"  [SEARCH] Segment: [{seg_start}, {seg_end}) = {dur_min:.1f} min")
        
        # Extract segment
        ecg_seg = ecg[seg_start:seg_end]
        features, _ = compute_rolling_features(ecg_seg, fs)
        
        if len(features) == 0:
            print(f"     [WARN]  Segment too short for rolling window")
            continue
        
        print(f"     -> {len(features)} windows extracted")
        
        for feat in features:
            all_rows.append({
                'Record': record_num,
                'Time_to_Event': time_counter,
                'DNB_Std': feat[2],
                'External_Correlation': feat[3],
                'Variance': feat[0],
                'Autocorrelation': feat[1],
            })
            time_counter += 1
    
    if not all_rows:
        print(f"  [FAIL] No features extracted")
        return None
    
    df = pd.DataFrame(all_rows, columns=BASE_COLS)
    print(f"  [OK] Total: {len(df)} feature rows")
    return df


def main():
    print("=" * 60)
    print("  SDDB AFib 漏洞修复engine v2.0")
    print("  Phase A: 4 AFib 黄金记录 (.ari rhythm 解析)")
    print("=" * 60)
    
    print(f"\n[DIR] SDDB DIR: {SDDB_DIR}")
    print(f"[TARGET] AFib records: {AFIB_RECORDS}")
    
    if not os.path.exists(SDDB_DIR):
        print(f"[FAIL] SDDB directory not found")
        sys.exit(1)
    
    # Process each AFib record
    all_dfs = []
    for rec in AFIB_RECORDS:
        df = process_afib_record(rec)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
    
    if not all_dfs:
        print("\n[FAIL] No AFib data extracted. Aborting.")
        # Save empty result file instead
        print("  -> Creating empty result placeholder")
        df_empty = pd.DataFrame(columns=BASE_COLS)
        df_empty.to_csv(os.path.join(OUTPUT_DIR, "sddb_afib_extracted.csv"), index=False)
        return
    
    # Combine all AFib data
    df_afib = pd.concat(all_dfs, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"  [OK] Total AFib windows: {len(df_afib)}")
    print(f"  Records: {sorted(df_afib.Record.unique())}")
    
    # Save AFib-only data to output
    afib_out = os.path.join(OUTPUT_DIR, "sddb_afib_extracted.csv")
    df_afib.to_csv(afib_out, index=False)
    print(f"  [SAVE] Saved to: {afib_out}")
    
    # -- Merge into master CSV --
    existing_path = os.path.join(PROCESSED_DIR, "sddb_terminal_master.csv")
    alt_path = os.path.join(DATA_DIR, "sddb_terminal_master.csv")
    
    if os.path.exists(existing_path):
        df_existing = pd.read_csv(existing_path)
    elif os.path.exists(alt_path):
        df_existing = pd.read_csv(alt_path)
        existing_path = alt_path
    else:
        print(f"  [FAIL] No existing sddb_terminal_master.csv found")
        # Save only AFib data to processed dir as new base
        df_afib.to_csv(existing_path, index=False)
        print(f"  [SAVE] Created new base: {existing_path}")
        return df_afib
    
    print(f"\n[DATA] Existing data: {len(df_existing)} rows, records {sorted(df_existing.Record.unique())}")
    
    # Remove any existing rows from overlapping records (clean replace)
    existing_records = set(df_existing['Record'].unique())
    new_records = set(df_afib['Record'].unique())
    overlap = existing_records & new_records
    if overlap:
        print(f"  [SYNC] Overlap records: {overlap}. Replacing...")
        df_existing = df_existing[~df_existing['Record'].isin(overlap)]
        print(f"  -> Existing after removal: {len(df_existing)} rows")
    
    # Merge and sort
    df_combined = pd.concat([df_existing, df_afib], ignore_index=True)
    df_combined = df_combined.sort_values(['Record', 'Time_to_Event']).reset_index(drop=True)
    
    # Save
    out_path = os.path.join(PROCESSED_DIR, "sddb_terminal_master.csv")
    df_combined.to_csv(out_path, index=False)
    print(f"\n  [OK] Merged -> {out_path}")
    print(f"  [OK] Total: {len(df_combined)} rows")
    
    # Verify completeness
    expected_records = {30, 31, 35, 36, 37, 46, 50, 51, 52}
    actual = set(df_combined['Record'].unique())
    missing = expected_records - actual
    if missing:
        print(f"  [WARN]  Missing: {sorted(missing)}")
    else:
        print(f"  [OK] All {len(expected_records)} records present!")
    
    return df_combined


if __name__ == "__main__":
    df = main()
    print("\n[OK] Phase A complete: AFib records extracted and merged.")
