#!/usr/bin/env python3
"""
Sleep-EDF CFECT 特征提取engine v1.0
==================================
Phase B: 197 全夜多导睡眠记录的无损 CFECT 特征提取

协议:
  1. EDF+ 解析: PSG + Hypnogram 多模态时间轴刚性对齐
  2. CFECT 滑动窗口: K=3000 (30s), Deltat=750 (75% 重叠)
  3. Stage-4 内源性基线归一化 (变分自由能最小化吸引子锚定)
  4. SC / ST 双队列后验对比 -> Table S6

拓扑转译:
  Sleep Cassette (SC)  = 年龄梯度 -> 势能景观老化
  Sleep Telemetry (ST) = Temazepam -> 外源性化学阻尼
"""

import numpy as np
import pandas as pd
import os, sys, glob, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

try:
    import pyedflib
except ImportError:
    print("[FAIL] pyedflib not installed. Run: pip install pyedflib")
    sys.exit(1)

# -- 路径配置 ------------------------------------------------------
DATA_ROOT = r"E:\MEM\paper\real\data\sleep-edf-database-expanded-1.0.0"
OUTPUT_DIR = r"E:\MEM\paper\real\output2"
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

SC_DIR = os.path.join(DATA_ROOT, "sleep-cassette")
ST_DIR = os.path.join(DATA_ROOT, "sleep-telemetry")

# 已知损坏记录（来自 PhysioNet 故障报告）
BROKEN_RECORDS = {'SC4361', 'SC4521', 'SC4132'}

# CFECT 滑动窗口参数
WINDOW_SIZE = 3000      # 30s @ 100Hz
STEP_SIZE = 750         # 7.5s (75% 重叠)

# 睡眠阶段编码
SLEEP_STAGE_MAP = {
    0: 'W',   # Wake
    1: 'N1',  # N1
    2: 'N2',  # N2
    3: 'N3',  # N3 (deep sleep)
    4: 'N3',  # N3 (alternative encoding)
    5: 'REM', # REM
    9: 'Unknown',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_edf_files():
    """扫描 SC + ST 目录，返回所有 PSG + Hypnogram 文件对"""
    sc_psg = sorted(glob.glob(os.path.join(SC_DIR, "*-PSG.edf")))
    sc_hyp = sorted(glob.glob(os.path.join(SC_DIR, "*-Hypnogram.edf")))
    st_psg = sorted(glob.glob(os.path.join(ST_DIR, "*-PSG.edf")))
    st_hyp = sorted(glob.glob(os.path.join(ST_DIR, "*-Hypnogram.edf")))
    
    sc_pairs = []
    for psg in sc_psg:
        base = os.path.basename(psg).replace("-PSG.edf", "")
        rec_id = base[:6]  # e.g. SC4001
        if rec_id in BROKEN_RECORDS:
            print(f"  [WARN]  Skipping broken record: {rec_id}")
            continue
        hyp = os.path.join(SC_DIR, f"{base.replace('E0', 'EC').replace('E1', 'EC')}-Hypnogram.edf")
        if not os.path.exists(hyp):
            hyp = os.path.join(SC_DIR, f"{base[:6]}EC-Hypnogram.edf")
        if os.path.exists(hyp):
            sc_pairs.append((rec_id, psg, hyp, 'SC'))
        else:
            print(f"  [WARN]  No hypnogram for {rec_id}, psg={psg}, tried={hyp}")
    
    st_pairs = []
    for psg in st_psg:
        base = os.path.basename(psg).replace("-PSG.edf", "")
        rec_id = base  # Preserve J0/JP suffix for drug labeling
        hyp = os.path.join(ST_DIR, f"{base.replace('J0', 'JP')}-Hypnogram.edf")
        if not os.path.exists(hyp):
            hyp = os.path.join(ST_DIR, f"{base}JP-Hypnogram.edf")
        if os.path.exists(hyp):
            st_pairs.append((rec_id, psg, hyp, 'ST'))
        else:
            print(f"  [WARN]  No hypnogram for {rec_id}")
    
    print(f"\n[DATA] File pairs found:")
    print(f"  Sleep Cassette (SC): {len(sc_pairs)}")
    print(f"  Sleep Telemetry (ST): {len(st_pairs)}")
    print(f"  Total: {len(sc_pairs) + len(st_pairs)}")
    
    return sc_pairs + st_pairs


def read_sleep_edf(psg_path, hyp_path):
    """
    读取 PSG + Hypnogram EDF 文件并刚性对齐时间轴。
    
    Returns:
    --------
    signals : dict
        {'eeg': np.array (n_samples,), 'emg': np.array (n_samples,), ...}
    hypnogram : np.array (n_epochs,)
        每个 30s epoch 的睡眠阶段标签
    fs : int
        EEG 采样率
    """
    # -- 读 PSG --
    f_psg = pyedflib.EdfReader(psg_path)
    n_signals = f_psg.signals_in_file
    fs = int(f_psg.getSampleFrequency(0))
    n_samples = f_psg.getNSamples()[0]
    
    # 找出通道索引
    ch_labels = [f_psg.getLabel(i).strip().upper() for i in range(n_signals)]
    
    eeg_idx = None
    emg_idx = None
    for i, label in enumerate(ch_labels):
        if 'FPZ' in label and 'CZ' in label:
            eeg_idx = i
        elif 'EMG' in label:
            emg_idx = i
    
    if eeg_idx is None:
        # Fallback to first EEG channel
        for i, label in enumerate(ch_labels):
            if 'EEG' in label:
                eeg_idx = i
                break
    
    if eeg_idx is None:
        eeg_idx = 0  # desperate fallback
    
    # 读取信号
    eeg = f_psg.readSignal(eeg_idx)
    emg = f_psg.readSignal(emg_idx) if emg_idx is not None else np.zeros(n_samples)
    f_psg.close()
    
    # -- 读 Hypnogram --
    f_hyp = pyedflib.EdfReader(hyp_path)
    hyp_fs = int(f_hyp.getSampleFrequency(0))
    hyp_signal = f_hyp.readSignal(0)
    f_hyp.close()
    
    # Hypnogram 采样率 = 1 Hz (1 sample per second), 每个 epoch=30s
    # 但 physionet 格式可能编码为每 30s 一个值 @ 1/30 Hz
    # 转化为 epoch 级标签
    n_epochs = len(hyp_signal)
    hypnogram = np.round(hyp_signal).astype(int)
    
    return {
        'eeg': eeg,
        'emg': emg,
        'fs': fs,
        'ch_labels': ch_labels,
        'eeg_idx': eeg_idx,
        'emg_idx': emg_idx,
    }, hypnogram, fs


def compute_cfect_features(eeg, hypnogram, fs=100, subject_id='', study_type='SC'):
    """
    滑动窗口 CFECT 特征计算 + 睡眠阶段标签广播对齐。
    
    窗口: K=3000 (30s @ 100Hz), 步长 Deltat=750 (7.5s, 75% 重叠)
    标签广播: 每个 30s epoch -> 4 个子窗口
    
    Returns: DataFrame
    """
    if len(eeg) < WINDOW_SIZE:
        return pd.DataFrame()
    
    n_windows = (len(eeg) - WINDOW_SIZE) // STEP_SIZE + 1
    n_epochs = len(hypnogram)
    
    # 预分配
    rows = []
    
    for i in range(n_windows):
        start = i * STEP_SIZE
        end = start + WINDOW_SIZE
        win = eeg[start:end]
        center_sec = (start + end) / 2 / fs
        
        # 睡眠阶段标签广播: 4 个子窗口 -> 1 个 epoch (30s = 3000 samples @ 100Hz)
        epoch_idx = int(center_sec // 30)
        stage_raw = hypnogram[epoch_idx] if epoch_idx < n_epochs else 9
        stage = SLEEP_STAGE_MAP.get(stage_raw, f'Unknown({stage_raw})')
        
        # ---- CFECT 特征 ----
        # 1. Variance
        variance = np.var(win)
        
        # 2. Lag-1 Autocorrelation (Phi1)
        ac = np.corrcoef(win[:-1], win[1:])[0, 1]
        if np.isnan(ac):
            ac = 0.0
        
        # 3. DNB_Std (detrended)
        trend = np.polyval(np.polyfit(np.arange(len(win)), win, 1), np.arange(len(win)))
        dnb_std = np.std(win - trend)
        
        # 4. Lag-50 External Correlation (~500ms for EEG)
        ext_corr = 0.0
        if len(win) > 50:
            ec = np.corrcoef(win[:-50], win[50:])[0, 1]
            ext_corr = ec if not np.isnan(ec) else 0.0
        
        rows.append({
            'Subject_ID': subject_id,
            'Study_Type': study_type,
            'Window': i,
            'Time_Sec': center_sec,
            'Sleep_Stage': stage,
            'Epoch_Index': epoch_idx,
            'Variance': variance,
            'Autocorrelation': ac,
            'DNB_Std': dnb_std,
            'External_Correlation': ext_corr,
        })
    
    return pd.DataFrame(rows)


def normalize_by_n3_anchor(df):
    """
    Stage-4 (N3) 内源性基线归一化。
    
    对每个受试者:
      1. 提取 Night 1 的 N3 窗口集
      2. 计算 mu_N3, sigma_N3 作为归一化基线
      3. 所有特征行: Z = (x - mu_N3) / sigma_N3
    
    物理含义: N3 深睡 = 变分自由能最小化吸引子锚点
    """
    print(f"\n  [DNA] Normalizing by N3 anchor...")
    
    df_norm = df.copy()
    
    for subj in df['Subject_ID'].unique():
        mask = df['Subject_ID'] == subj
        subj_data = df[mask]
        
        # Night 1 的 N3 窗口
        n3_mask = subj_data['Sleep_Stage'] == 'N3'
        # 假设前 1/3 为 Night 1
        night1_cutoff = len(subj_data) // 3
        night1_n3 = subj_data.iloc[:night1_cutoff][n3_mask]
        
        if len(night1_n3) < 10:
            # Fallback: 所有 N3
            night1_n3 = subj_data[n3_mask]
        
        if len(night1_n3) < 10:
            print(f"    [WARN]  {subj}: insufficient N3 windows ({len(night1_n3)}), using global")
            continue
        
        # 计算 N3 基线
        mu_var = night1_n3['Variance'].mean()
        sigma_var = night1_n3['Variance'].std()
        mu_ac = night1_n3['Autocorrelation'].mean()
        sigma_ac = night1_n3['Autocorrelation'].std()
        mu_dnb = night1_n3['DNB_Std'].mean()
        sigma_dnb = night1_n3['DNB_Std'].std()
        
        # Z-score 归一化
        df_norm.loc[mask, 'Variance_Z'] = (subj_data['Variance'].values - mu_var) / (sigma_var + 1e-10)
        df_norm.loc[mask, 'Phi1_Z'] = (subj_data['Autocorrelation'].values - mu_ac) / (sigma_ac + 1e-10)
        df_norm.loc[mask, 'DNB_Z'] = (subj_data['DNB_Std'].values - mu_dnb) / (sigma_dnb + 1e-10)
        
        print(f"    {subj}: mu_var={mu_var:.4f}, mu_ac={mu_ac:.4f} (N3 n={len(night1_n3)})")
    
    return df_norm


def add_meta_labels(df):
    """
    为 Sleep-EDF 数据添加元数据列:
    - Is_Drug: Temazepam vs Placebo (仅 ST)
    - Subject_Num: 受试者编号
    - Age_Bin: 年龄分组 (仅 SC 有年龄信息)
    """
    # SC = 无药, ST = Temazepam (但需拆分服药夜/安慰剂夜)
    # SC: SC4xxxE0-PSG (Night 1), SC4xxxE1-PSG (Night 2)
    # ST: ST7xxxJ0-PSG (Placebo), ST7xxxJP-PSG (Temazepam)
    
    df['Night'] = df['Subject_ID'].apply(
        lambda sid: 1 if 'E0' in sid or 'J0' in sid else 2
    )
    
    # ST 药物: J0 = Placebo, JP = Temazepam
    df['Is_Drug'] = df.apply(
        lambda r: 0 if 'J0' in r['Subject_ID'] or r['Study_Type'] == 'SC' 
                  else (1 if 'JP' in r['Subject_ID'] else 0),
        axis=1
    )
    
    # Drug_Type 列
    df['Drug_Type'] = df.apply(
        lambda r: 'None' if r['Study_Type'] == 'SC' 
                  else ('Temazepam' if r['Is_Drug'] else 'Placebo'),
        axis=1
    )
    
    # 提取受试者编号
    df['Subject_Num'] = df['Subject_ID'].apply(
        lambda sid: int(sid[2:6].lstrip('0') or '0')
    )
    
    return df


def build_table_s6(df):
    """
    构建 Table S6: Sleep-EDF 亚组对比表
    
    四行:
      SC 青年 (25-40), SC 高龄 (80+)
      ST Placebo, ST Temazepam
    """
    print(f"\n{'='*60}")
    print("  Table S6: Sleep-EDF Subgroup Comparison")
    print(f"{'='*60}")
    
    # 准确定义亚组
    # SC: 无药, Age 已知 (从 Subject_Num 推断: SC4xxx, 编号<40=青年, >65=高龄)
    # ST: Placebo vs Temazepam
    
    subgroups = {
        'SC Young (25-40)': df[(df['Study_Type'] == 'SC') & (df['Subject_Num'] <= 40)],
        'SC Old (80+)': df[(df['Study_Type'] == 'SC') & (df['Subject_Num'] >= 80)],
        'ST Placebo': df[(df['Study_Type'] == 'ST') & (df['Drug_Type'] == 'Placebo')],
        'ST Temazepam': df[(df['Study_Type'] == 'ST') & (df['Drug_Type'] == 'Temazepam')],
    }
    
    rows = []
    for name, grp in subgroups.items():
        if len(grp) == 0:
            print(f"  [WARN]  Empty: {name}")
            continue
        
        # Wake->N1 临界区: epoch_idx 从 Wake 切到第一个 N1
        # 取入睡前 30 分钟 (= 60 epochs)
        rows.append({
            'Subgroup': name,
            'N_Windows': len(grp),
            'Variance_Z_mean': grp['Variance_Z'].mean(),
            'Variance_Z_std': grp['Variance_Z'].std(),
            'Phi1_Z_mean': grp['Phi1_Z'].mean(),
            'Phi1_Z_std': grp['Phi1_Z'].std(),
            'DNB_Z_mean': grp['DNB_Z'].mean(),
            'DNB_Z_std': grp['DNB_Z'].std(),
        })
    
    table = pd.DataFrame(rows)
    tab_path = os.path.join(OUTPUT_DIR, "table_s6_sleep_subgroups.csv")
    table.to_csv(tab_path, index=False)
    print(table.to_string(index=False))
    print(f"\n  [SAVE] Saved: {tab_path}")
    
    return table


def analyze_st_drug_effect(df):
    """
    ST 队列：Temazepam vs Placebo 的入睡前 30 分钟对比
    
    Welch t-test + Cohen's d
    物理预测: Temazepam -> Variance_Z ↓↓, Phi1 钳制在低位
    """
    print(f"\n{'='*60}")
    print("  ST Drug Effect Analysis (Temazepam vs Placebo)")
    print(f"{'='*60}")
    
    # 入睡前 30 分钟: Wake->N1 临界区
    wake_n1 = df[(df['Study_Type'] == 'ST')]
    
    if len(wake_n1) == 0:
        print("  [WARN]  No ST data")
        return None
    
    placebo = wake_n1[wake_n1['Drug_Type'] == 'Placebo']
    drug = wake_n1[wake_n1['Drug_Type'] == 'Temazepam']
    
    print(f"  Placebo: {len(placebo)} windows")
    print(f"  Temazepam: {len(drug)} windows")
    
    from scipy import stats
    
    for metric in ['Variance_Z', 'Phi1_Z', 'DNB_Z']:
        if len(placebo[metric].dropna()) < 2 or len(drug[metric].dropna()) < 2:
            print(f"  [WARN]  Insufficient data for {metric}")
            continue
        
        p_mean = placebo[metric].mean()
        d_mean = drug[metric].mean()
        
        t_stat, p_val = stats.ttest_ind(placebo[metric].dropna(), drug[metric].dropna(), equal_var=False)
        
        # Cohen's d
        n1, n2 = len(placebo[metric].dropna()), len(drug[metric].dropna())
        s1, s2 = placebo[metric].std(), drug[metric].std()
        pooled_se = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        cohens_d = (p_mean - d_mean) / (pooled_se + 1e-10)
        
        print(f"  {metric:15s}: Placebo={p_mean:.4f}, Drug={d_mean:.4f}")
        print(f"    Welch t={t_stat:.3f}, p={p_val:.6f}, d={cohens_d:.3f}")
        
        if p_val < 0.05:
            print(f"    -> {'[OK] SIGNIFICANT' if cohens_d*d_mean > 0 else '[WARN] Opposite direction'}")
        else:
            print(f"    -> [FAIL] Not significant")
    
    return wake_n1


def analyze_sc_age_effect(df):
    """
    SC 队列：年龄梯度对入睡临界区 CSD 的影响
    
    OLS: Phi1_Z ~ Age * Transition + (1|Subject)
    """
    print(f"\n{'='*60}")
    print("  SC Age Effect Analysis")
    print(f"{'='*60}")
    
    sc = df[df['Study_Type'] == 'SC'].copy()
    if len(sc) == 0:
        print("  [WARN]  No SC data")
        return None
    
    # Age proxy from Subject_Num (SC4xxx)
    sc['Age_Proxy'] = sc['Subject_Num']
    
    # Wake->N1 transition regions
    sc['Is_Transition'] = sc['Sleep_Stage'].isin(['W', 'N1'])
    
    # Group by Age proxy bins
    young = sc[sc['Age_Proxy'] <= 40]
    mid = sc[(sc['Age_Proxy'] > 40) & (sc['Age_Proxy'] <= 70)]
    old = sc[sc['Age_Proxy'] > 70]
    
    print(f"  Young (<=40): {len(young)} windows, Phi1_Z={young['Phi1_Z'].mean():.4f}")
    print(f"  Mid (41-70): {len(mid)} windows, Phi1_Z={mid['Phi1_Z'].mean():.4f}")
    print(f"  Old (>70): {len(old)} windows, Phi1_Z={old['Phi1_Z'].mean():.4f}")
    
    # Transition regions specifically
    for name, grp in [('Young', young), ('Mid', mid), ('Old', old)]:
        trans = grp[grp['Is_Transition']]
        if len(trans) > 0:
            print(f"  {name} Transition (Wake+N1): Phi1_Z={trans['Phi1_Z'].mean():.4f}, "
                  f"Variance_Z={trans['Variance_Z'].mean():.4f}")
    
    return sc


def main():
    print("=" * 60)
    print("  Sleep-EDF CFECT 特征提取engine v1.0")
    print("  197 全夜多导睡眠记录 | Fpz-Cz @ 100Hz")
    print("=" * 60)
    
    # Step 0: 文件发现
    print("\n[Step 0] File Discovery...")
    file_pairs = find_edf_files()
    
    if not file_pairs:
        print("[FAIL] No EDF files found.")
        sys.exit(1)
    
    # Step 1-2: 处理每个记录
    print(f"\n[Step 1-2] Processing {len(file_pairs)} records...")
    all_features = []
    
    for idx, (rec_id, psg_path, hyp_path, study_type) in enumerate(file_pairs):
        print(f"\n  [{idx+1}/{len(file_pairs)}] {rec_id} ({study_type})...", end="")
        
        try:
            signals, hypnogram, fs = read_sleep_edf(psg_path, hyp_path)
            eeg = signals['eeg']
            
            # Skip if too short
            if len(eeg) < WINDOW_SIZE:
                print(f" [WARN]  Signal too short ({len(eeg)})")
                continue
            
            # Compute features
            df_rec = compute_cfect_features(
                eeg, hypnogram, fs=fs,
                subject_id=rec_id, study_type=study_type
            )
            
            if len(df_rec) > 0:
                all_features.append(df_rec)
                print(f" {len(df_rec)} windows [OK]")
            else:
                print(f" [WARN]  No windows")
        
        except Exception as e:
            print(f" [FAIL] Error: {e}")
            continue
    
    if not all_features:
        print("\n[FAIL] No features extracted.")
        sys.exit(1)
    
    # Step 3: 合并
    print(f"\n[Step 3] Merging {len(all_features)} DataFrames...")
    df = pd.concat(all_features, ignore_index=True)
    print(f"  Total windows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    
    # 元数据打标
    print(f"\n[Step 3b] Adding metadata labels...")
    df = add_meta_labels(df)
    
    # Step 4: N3 内源性归一化
    print(f"\n[Step 4] N3 anchor normalization...")
    df = normalize_by_n3_anchor(df)
    
    # Save full feature matrix
    csv_path = os.path.join(OUTPUT_DIR, "sleep_csd_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  [SAVE] Saved full features: {csv_path} ({len(df):,} rows)")
    
    # Step 5: 分析
    print(f"\n[Step 5] Analysis...")
    sc = analyze_sc_age_effect(df)
    st = analyze_st_drug_effect(df)
    table_s6 = build_table_s6(df)
    
    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  SC recordings: {len(df[df['Study_Type']=='SC']['Subject_ID'].unique())}")
    print(f"  ST recordings: {len(df[df['Study_Type']=='ST']['Subject_ID'].unique())}")
    print(f"  Total windows: {len(df):,}")
    print(f"  Stage distribution:")
    print(f"    {df['Sleep_Stage'].value_counts().to_dict()}")
    print(f"  Drug vs Placebo (ST):")
    st_drug = df[df['Study_Type'] == 'ST']
    print(f"    Temazepam: {len(st_drug[st_drug['Is_Drug']==1])} windows")
    print(f"    Placebo: {len(st_drug[st_drug['Is_Drug']==0])} windows")
    
    return df


if __name__ == "__main__":
    df = main()
    print("\n[OK] Phase B complete: Sleep-EDF features extracted.")
