#!/usr/bin/env python3
"""
Sleep-EDF CFECT Phase B — MNE 解析engine v1.0
=============================================
197 全夜多导睡眠记录 (153 SC + 44 ST)
使用 MNE 状态机刚性熔断 EDF+ 注释与信号轨

流程:
  1. mne.io.read_raw_edf -> 延迟加载 Fpz-Cz (100Hz)
  2. mne.read_annotations -> 提取 Hypnogram 30s epoch 标签
  3. set_annotations + events_from_annotations -> 刚性地空对齐
  4. 3000-sample 滑动窗口 (75% 重叠) -> CFECT 特征提取
  5. N3 内源性归一化 + SC/ST 亚组对比 -> Table S6

输出: E:\MEM\paper\real\output2\main\sleep_csd_features.csv
"""

import os, sys, glob, json, logging
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import mne

# -- 日志 --
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('PhaseB')

# -- 路径 --
DATA_ROOT = r"E:\MEM\paper\real\data\sleep-edf-database-expanded-1.0.0"
OUTPUT_DIR = r"E:\MEM\paper\real\output2\main"
SC_DIR = os.path.join(DATA_ROOT, "sleep-cassette")
ST_DIR = os.path.join(DATA_ROOT, "sleep-telemetry")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 已知损坏记录
BROKEN_RECORDS = {'SC4361', 'SC4521', 'SC4132'}

# CFECT 参数
WINDOW_SIZE = 3000      # 30s @ 100Hz
STEP_SIZE = 750         # 7.5s (75% 重叠)
SUB_WINDOWS = 4         # 每个 epoch 4 个子窗

# R&K 阶段映射 (Sleep-EDF 注释命名)
STAGE_MAPPING = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 4,
    'Sleep stage R': 5,
}
STAGE_LABELS = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N3', 5: 'REM'}


def find_pairs():
    """返回 (rec_id, psg_path, hyp_path, study_type) 列表"""
    pairs = []
    
    # SC
    sc_psg = sorted(glob.glob(os.path.join(SC_DIR, "*-PSG.edf")))
    sc_hyp = sorted(glob.glob(os.path.join(SC_DIR, "*-Hypnogram.edf")))
    hyp_map = {os.path.basename(h).replace('-Hypnogram.edf', '')[:6]: h for h in sc_hyp}
    
    for psg in sc_psg:
        base = os.path.basename(psg).replace('-PSG.edf', '')
        rec_id = base[:6]
        if rec_id in BROKEN_RECORDS:
            continue
        if rec_id in hyp_map:
            pairs.append((rec_id, psg, hyp_map[rec_id], 'SC'))
        else:
            logger.warning(f"No hypnogram for {rec_id}")
    
    # ST
    st_psg = sorted(glob.glob(os.path.join(ST_DIR, "*-PSG.edf")))
    st_hyp = sorted(glob.glob(os.path.join(ST_DIR, "*-Hypnogram.edf")))
    hyp_map_st = {os.path.basename(h).replace('-Hypnogram.edf', '')[:6]: h for h in st_hyp}
    
    for psg in st_psg:
        base = os.path.basename(psg).replace('-PSG.edf', '')
        rec_id = base[:6]
        if rec_id in hyp_map_st:
            pairs.append((rec_id, psg, hyp_map_st[rec_id], 'ST'))
        else:
            logger.warning(f"No hypnogram for {rec_id}")
    
    return pairs


def extract_cfect_features(psg_path, hyp_path, rec_id, study_type):
    """
    MNE 刚性解析 -> CFECT 滑动窗口特征。
    
    Returns: pd.DataFrame | None
    """
    try:
        # 1. MNE 延迟加载 Fpz-Cz
        raw = mne.io.read_raw_edf(
            psg_path, include=['EEG Fpz-Cz'],
            preload=False, verbose=False
        )
        raw.load_data(verbose=False)
        sfreq = int(raw.info['sfreq'])
        
        # 2. 解析 EDF+ Hypnogram 注释
        annotations = mne.read_annotations(hyp_path)
        raw.set_annotations(annotations, emit_warning=False)
        
        # 3. 事件提取
        events, event_id = mne.events_from_annotations(
            raw, event_id=STAGE_MAPPING, verbose=False
        )
        
        if len(events) == 0:
            logger.warning(f"  {rec_id}: 0 events extracted")
            return None
        
        # 4. 信号提取
        data, _ = raw[:, :]
        signal = data[0]
        n_epochs = len(events)
        n_windows = n_epochs * SUB_WINDOWS
        
        # 预分配
        records = []
        total_samples = len(signal)
        
        for i, ev in enumerate(events):
            onset = ev[0]  # sample index
            stage_code = ev[2]
            
            for sub in range(SUB_WINDOWS):
                start = onset + sub * STEP_SIZE
                end = start + WINDOW_SIZE
                
                if end > total_samples:
                    continue
                
                win = signal[start:end]
                center_sec = (start + end) / 2.0 / sfreq
                
                # -- CFECT 特征 --
                # 1. Variance (使用 muV² 更好读)
                variance = np.var(win)
                
                # 2. Phi1 (lag-1 autocorrelation)
                demeaned = win - np.mean(win)
                num = np.sum(demeaned[:-1] * demeaned[1:])
                den = np.sum(demeaned ** 2)
                phi1 = num / den if den > 1e-12 else 0.0
                
                # 3. DNB_Std (去趋势标准差)
                trend = np.polyval(
                    np.polyfit(np.arange(len(win)), win, 1),
                    np.arange(len(win))
                )
                dnb_std = np.std(win - trend)
                
                # 4. External_Corr (lag-50 ≈ 500ms)
                ext_corr = 0.0
                if len(win) > 50:
                    ec = np.corrcoef(win[:-50], win[50:])[0, 1]
                    ext_corr = ec if not np.isnan(ec) else 0.0
                
                records.append({
                    'Subject_ID': rec_id,
                    'Study_Type': study_type,
                    'Window': i * SUB_WINDOWS + sub,
                    'Time_Sec': center_sec,
                    'Epoch_Index': i,
                    'Sleep_Stage_Code': stage_code,
                    'Sleep_Stage': STAGE_LABELS.get(stage_code, 'Unknown'),
                    'Variance': variance,
                    'Phi1': phi1,
                    'DNB_Std': dnb_std,
                    'External_Corr': ext_corr,
                })
        
        return pd.DataFrame(records)
    
    except Exception as e:
        logger.error(f"  {rec_id}: CRITICAL ERROR: {e}")
        return None


def normalize_by_n3(df):
    """
    N3 内源性锚点归一化（受试者级）。
    使用 Night 1 的 N3 窗口计算 mu, sigma。
    """
    logger.info("Normalizing by N3 anchor...")
    df = df.copy()
    
    for subj in df['Subject_ID'].unique():
        mask = df['Subject_ID'] == subj
        subj_data = df[mask]
        
        # Night 1 N3: 假设前 1/3 的记录为 Night 1
        night1_cut = len(subj_data) // 3
        n3_mask = subj_data['Sleep_Stage'] == 'N3'
        night1_n3 = subj_data.iloc[:night1_cut][n3_mask]
        
        if len(night1_n3) < 10:
            night1_n3 = subj_data[n3_mask]  # fallback: 所有 N3
        
        if len(night1_n3) < 10:
            logger.warning(f"  {subj}: insufficient N3 ({len(night1_n3)}), skip norm")
            df.loc[mask, 'Variance_Z'] = subj_data['Variance']
            df.loc[mask, 'Phi1_Z'] = subj_data['Phi1']
            df.loc[mask, 'DNB_Z'] = subj_data['DNB_Std']
            continue
        
        mu_var, sigma_var = night1_n3['Variance'].mean(), night1_n3['Variance'].std()
        mu_phi, sigma_phi = night1_n3['Phi1'].mean(), night1_n3['Phi1'].std()
        mu_dnb, sigma_dnb = night1_n3['DNB_Std'].mean(), night1_n3['DNB_Std'].std()
        
        df.loc[mask, 'Variance_Z'] = (subj_data['Variance'].values - mu_var) / (sigma_var + 1e-12)
        df.loc[mask, 'Phi1_Z'] = (subj_data['Phi1'].values - mu_phi) / (sigma_phi + 1e-12)
        df.loc[mask, 'DNB_Z'] = (subj_data['DNB_Std'].values - mu_dnb) / (sigma_dnb + 1e-12)
        
        logger.info(f"  {subj}: N3 n={len(night1_n3)}, mu_var={mu_var:.4e}, mu_phi={mu_phi:.4f}")
    
    return df


def add_meta_labels(df):
    """添加受试者编号、药物类型、夜次等元信息。"""
    df['Subject_Num'] = df['Subject_ID'].apply(
        lambda x: int(x[2:6].lstrip('0') or '0'))
    
    # Night detection
    df['Night'] = df['Subject_ID'].apply(
        lambda x: 1 if 'E0' in x or 'J0' in x else 2)
    
    # Drug (ST only)
    df['Is_Drug'] = df.apply(
        lambda r: 0 if r['Study_Type'] == 'SC' or 'J0' in r['Subject_ID']
                  else (1 if 'JP' in r['Subject_ID'] else 0), axis=1)
    df['Drug_Type'] = df['Is_Drug'].map({0: 'Placebo/None', 1: 'Temazepam'})
    
    return df


def build_table_s6(df):
    """Table S6: 四行亚组对比。"""
    subgroups = {
        'SC_Young': df[(df['Study_Type'] == 'SC') & (df['Subject_Num'] <= 40)],
        'SC_Old': df[(df['Study_Type'] == 'SC') & (df['Subject_Num'] >= 80)],
        'ST_Placebo': df[(df['Study_Type'] == 'ST') & (df['Drug_Type'] == 'Placebo/None')],
        'ST_Temazepam': df[(df['Study_Type'] == 'ST') & (df['Drug_Type'] == 'Temazepam')],
    }
    
    rows = []
    for name, grp in subgroups.items():
        if len(grp) == 0:
            continue
        rows.append({
            'Subgroup': name, 'N': len(grp),
            'Variance_Z': f"{grp['Variance_Z'].mean():.4f} +/- {grp['Variance_Z'].std():.4f}",
            'Phi1_Z': f"{grp['Phi1_Z'].mean():.4f} +/- {grp['Phi1_Z'].std():.4f}",
            'DNB_Z': f"{grp['DNB_Z'].mean():.4f} +/- {grp['DNB_Z'].std():.4f}",
        })
    
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUTPUT_DIR, 'table_s6_subgroups.csv'), index=False)
    logger.info(f"\nTable S6:\n{table.to_string(index=False)}")
    return table


def main():
    logger.info("=" * 60)
    logger.info("  Sleep-EDF CFECT Phase B — MNE 解析engine v1.0")
    logger.info("  197 全夜多导睡眠记录 (153 SC + 44 ST)")
    logger.info("=" * 60)
    
    # Step 0: 发现文件对
    logger.info("\n[0] Discovering file pairs...")
    pairs = find_pairs()
    logger.info(f"  Found {len(pairs)} valid pairs")
    
    # Step 1-2: 处理每个记录
    logger.info(f"\n[1-2] Processing {len(pairs)} records...")
    all_features = []
    errors = []
    
    for idx, (rec_id, psg_path, hyp_path, study_type) in enumerate(pairs):
        logger.info(f"  [{idx+1}/{len(pairs)}] {rec_id} ({study_type})...")
        
        df_rec = extract_cfect_features(psg_path, hyp_path, rec_id, study_type)
        
        if df_rec is not None and len(df_rec) > 0:
            all_features.append(df_rec)
            n_epochs = df_rec['Epoch_Index'].nunique()
            logger.info(f"    [OK] {len(df_rec)} windows, {n_epochs} epochs")
        else:
            errors.append(rec_id)
            logger.warning(f"    [FAIL] FAILED")
    
    if not all_features:
        logger.error("No features extracted!")
        sys.exit(1)
    
    # Step 3: 合并
    logger.info(f"\n[3] Merging {len(all_features)} DataFrames...")
    df = pd.concat(all_features, ignore_index=True)
    logger.info(f"  Total: {len(df):,} windows, {df['Subject_ID'].nunique()} subjects")
    
    # 元数据
    df = add_meta_labels(df)
    
    # Step 4: N3 归一化
    logger.info(f"\n[4] N3 normalization...")
    df = normalize_by_n3(df)
    
    # 保存全量特征
    csv_path = os.path.join(OUTPUT_DIR, 'sleep_csd_features.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")
    
    # Step 5: 分析
    logger.info(f"\n[5] Analysis...")
    table_s6 = build_table_s6(df)
    
    # 阶段分布
    logger.info(f"\nStage distribution:")
    stage_counts = df['Sleep_Stage'].value_counts()
    for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
        if stage in stage_counts.index:
            logger.info(f"  {stage}: {stage_counts[stage]:,}")
    
    # 错误报告
    if errors:
        logger.warning(f"\nFailed records ({len(errors)}): {errors}")
    
    # 摘要
    logger.info(f"\n{'='*60}")
    logger.info("  PHASE B SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  SC: {len(df[df['Study_Type']=='SC']['Subject_ID'].unique())} subjects")
    logger.info(f"  ST: {len(df[df['Study_Type']=='ST']['Subject_ID'].unique())} subjects")
    logger.info(f"  Total windows: {len(df):,}")
    logger.info(f"  Temazepam: {len(df[df['Drug_Type']=='Temazepam']):,}")
    logger.info(f"  Placebo/None: {len(df[df['Drug_Type']=='Placebo/None']):,}")
    
    return df


if __name__ == "__main__":
    df = main()
    logger.info("\n[OK] Phase B complete.")
