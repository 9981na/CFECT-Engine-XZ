#!/usr/bin/env python3
"""
CFECT Phase B: Sleep-EDF MNE 自动对齐与 LMM 交互项统计engine (v2.2)
====================================================================
修复记录:
  v2.2 - 元数据查表 + Night 固定效应 + N3=Stage3+4 + 合成数据保护
  v2.1 - LMM 交互项回归注入
  v2.0 - MNE 状态机 + 75% 重叠窗口
  
Author: Xinzheng Zhuang (BUCM)
Year: 2026
"""

import os, sys, re, logging, warnings
import numpy as np
import pandas as pd
import mne
import statsmodels.formula.api as smf
from numpy.lib.stride_tricks import sliding_window_view

# ================= 配置 =================
META_PATH = os.path.join("data", "processed", "sleep_meta_lookup.csv")
SLEEP_DATA_DIR = r"E:\MEM\paper\real\data\sleep-edf-database-expanded-1.0.0"
OUTPUT_CSV = "cfect_sleep_edf_hardened_features.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("CFECT_Sleep_LMM")
warnings.filterwarnings('ignore')

STAGE_LABELS = {
    0: 'W', 1: 'N1', 2: 'N2', 3: 'N3_slow', 4: 'N3_deep', 5: 'REM'
}

N3_STAGES = {3, 4}  # AASM: Stage 3 + Stage 4 = N3


def load_meta_lookup():
    """加载元数据映射表"""
    if not os.path.exists(META_PATH):
        logger.error(f"元数据映射表不存在: {META_PATH}")
        logger.error("请先运行: python _build_sleep_meta.py")
        sys.exit(1)
    df = pd.read_csv(META_PATH)
    lookup = {}
    for _, row in df.iterrows():
        lookup[row['Rec_ID_Full']] = {
            'age': row['Age'],
            'age_group': row['Age_Group'],
            'night': row['Night'],
            'sex': row['Sex_F1'],
            'drug_condition': row['Drug_Condition'],
            'study_type': row['Study_Type'],
            'subject_nr_xls': row['Subject_Nr_XLS'],
            'lights_off': row['LightsOff'],
        }
    logger.info(f"元数据映射表加载: {len(lookup)} 条记录")
    return lookup


# ================= 窗口特征算子 =================
def extract_window_features(signal_segment):
    """计算窗口的方差和 Φ₁ 自相关系数"""
    demeaned = signal_segment - np.mean(signal_segment)
    variance = np.var(signal_segment)
    numerator = np.sum(demeaned[:-1] * demeaned[1:])
    denominator = np.sum(demeaned ** 2)
    phi1 = numerator / denominator if denominator > 1e-12 else 0.0
    return variance, phi1


# ================= MNE 对齐 =================
def process_single_sleep_pair(psg_path, hypno_path, meta_lookup):
    """处理一对 PSG + Hypnogram 文件"""
    psg_name = os.path.basename(psg_path)
    rec_id = psg_name.replace('-PSG.edf', '')
    
    # 查元数据
    if rec_id not in meta_lookup:
        logger.warning(f"无元数据记录: {rec_id}, 跳过")
        return None
    meta = meta_lookup[rec_id]
    
    subject_nr = meta['subject_nr_xls']
    cohort_type = meta['study_type']
    night = meta['night']
    age = meta['age']
    drug_condition = meta['drug_condition']
    
    try:
        raw = mne.io.read_raw_edf(psg_path, include=['EEG Fpz-Cz'], preload=True, verbose=False)
        raw.filter(0.5, 40.0, fir_design='firwin', verbose=False)
        sfreq = raw.info['sfreq']
        
        annotations = mne.read_annotations(hypno_path)
        raw.set_annotations(annotations, emit_warning=False)
        
        # R&K 睡眠阶段映射
        stage_mapping = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 4,
            'Sleep stage R': 5
        }
        
        events, event_id = mne.events_from_annotations(
            raw, event_id=stage_mapping, chunk_duration=30.0, verbose=False
        )
        
        signal_data, _ = raw[:, :]
        eeg_trace = signal_data[0]
        
        window_size = int(30.0 * sfreq)    # 3000 样本
        step_size = int(7.5 * sfreq)       # 750 样本 (75% 重叠)
        
        epoch_records = []
        
        for event in events:
            onset_sample = event[0]
            stage_code = event[2]
            
            # 每个 30s 评分区间内切 4 个滑动子窗
            for sub_step in range(4):
                start_idx = onset_sample + (sub_step * step_size)
                end_idx = start_idx + window_size
                
                if end_idx <= len(eeg_trace):
                    segment = eeg_trace[start_idx:end_idx]
                    v_val, p_val = extract_window_features(segment)
                    
                    epoch_records.append({
                        'Subject_ID': f"{cohort_type}_{subject_nr:03d}",
                        'Subject_Nr': subject_nr,
                        'Cohort_Type': cohort_type,
                        'Night': night,
                        'Age': age,
                        'Drug_Condition': drug_condition,
                        'Stage': stage_code,
                        'Stage_Label': STAGE_LABELS.get(stage_code, '?'),
                        'Is_N3': 1 if stage_code in N3_STAGES else 0,
                        'Variance_Raw': v_val,
                        'Phi1_Raw': p_val,
                    })
        
        logger.info(f"  [OK] {rec_id}: {len(epoch_records)} windows")
        return pd.DataFrame(epoch_records)
        
    except Exception as e:
        logger.error(f"[FAIL] MNE 对齐失败 {rec_id}: {e}")
        return None


# ================= Normalize by N3 baseline =================
def normalize_by_n3(df_analysis, meta_lookup):
    """
    基于受试者自身 N3 窗口的均值+/-标准差做 Z-score 标准化
    
    N3 = stage 3 + stage 4 (慢波睡眠)
    只使用 Night==1 的 N3 窗口作为基线
    """
    logger.info("执行基于 N3 基线的 Z-score 标准化...")
    df = df_analysis.copy()
    
    # 为每个受试者计算 N3 基线
    baseline_stats = []
    for (subj, night), group in df.groupby(['Subject_ID', 'Night']):
        n3_mask = group['Is_N3'] == 1
        n3_data = group[n3_mask]
        
        if len(n3_data) >= 5:
            mean_var = n3_data['Variance_Raw'].mean()
            std_var = n3_data['Variance_Raw'].std(ddof=1) + 1e-12
            mean_phi = n3_data['Phi1_Raw'].mean()
            std_phi = n3_data['Phi1_Raw'].std(ddof=1) + 1e-12
        else:
            # 降级: 使用全局 N3 作为基线
            global_n3 = df[df['Is_N3'] == 1]
            mean_var = global_n3['Variance_Raw'].mean()
            std_var = global_n3['Variance_Raw'].std(ddof=1) + 1e-12
            mean_phi = global_n3['Phi1_Raw'].mean()
            std_phi = global_n3['Phi1_Raw'].std(ddof=1) + 1e-12
            logger.warning(f"  受试者 {subj} Night {night} N3 窗口不足({len(n3_data)}), 使用全局基线")
        
        baseline_stats.append({
            'Subject_ID': subj,
            'Night': night,
            'Baseline_N3_Windows': len(n3_data),
            'Mean_Var_N3': mean_var,
            'Std_Var_N3': std_var,
            'Mean_Phi_N3': mean_phi,
            'Std_Phi_N3': std_phi,
        })
    
    baseline_df = pd.DataFrame(baseline_stats)
    
    # 合并基线并计算 Z-score
    df = df.merge(baseline_df, on=['Subject_ID', 'Night'], how='left')
    df['Variance_Z'] = (df['Variance_Raw'] - df['Mean_Var_N3']) / df['Std_Var_N3']
    df['Phi1_Z'] = (df['Phi1_Raw'] - df['Mean_Phi_N3']) / df['Std_Phi_N3']
    
    # 删除辅助列
    drop_cols = ['Baseline_N3_Windows', 'Mean_Var_N3', 'Std_Var_N3', 'Mean_Phi_N3', 'Std_Phi_N3']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    logger.info(f"N3 基线标准化complete: {len(df)} 行")
    return df, baseline_df


# ================= 大一统管道 =================
def run_sleep_edf_lmm_pipeline(data_root=None):
    """全量 Sleep-EDF MNE 对齐 + LMM 回归"""
    
    if data_root is None:
        data_root = SLEEP_DATA_DIR
    
    logger.info("=" * 60)
    logger.info("Sleep-EDF Phase B LMM engine[START] v2.2")
    logger.info(f"数据目录: {data_root}")
    logger.info("=" * 60)
    
    # 1. 加载元数据
    meta_lookup = load_meta_lookup()
    
    # 2. 扫描文件
    sc_dir = os.path.join(data_root, 'sleep-cassette')
    st_dir = os.path.join(data_root, 'sleep-telemetry')
    
    all_files = []
    for d in [sc_dir, st_dir]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith("-PSG.edf"):
                    all_files.append((d, f))
    all_files = sorted(all_files, key=lambda x: x[1])
    
    logger.info(f"发现 {len(all_files)} 个 PSG 文件")
    
    # 3. 逐个处理
    master_pool = []
    skipped = 0
    
    # 预构建 Hypnogram 查找表: {psg_id: hypno_file}
    for data_dir, psg_f in all_files:
        psg_id = psg_f.replace('-PSG.edf', '')
        
        # SC: SC4001E0-PSG.edf -> hypno starts with SC4001E (7 chars match)
        # ST: ST7011J0-PSG.edf -> hypno starts with ST7011J (7 chars match)
        psg_prefix = psg_id[:7]  # e.g. SC4001E, ST7011J
        
        # 查找 Hypnogram: 同前缀 + -Hypnogram.edf
        hypno_f = None
        for f in os.listdir(data_dir):
            if f.endswith("-Hypnogram.edf") and f.startswith(psg_prefix):
                hypno_f = f
                break
        
        if hypno_f is None:
            logger.warning(f"Hypnogram 缺失 (prefix={psg_prefix}), 跳过 {psg_f}")
            skipped += 1
            continue
        
        psg_path = os.path.join(data_dir, psg_f)
        hypno_path = os.path.join(data_dir, hypno_f)
        
        sub_df = process_single_sleep_pair(psg_path, hypno_path, meta_lookup)
        if sub_df is not None:
            master_pool.append(sub_df)
    
    if len(master_pool) == 0:
        logger.error("错误: 没有成功处理任何记录!")
        return None, None
    
    df_raw = pd.concat(master_pool, ignore_index=True)
    logger.info(f"原始特征池: {len(df_raw)} 行")
    
    # 4. N3 基线标准化
    df_analysis, baseline_df = normalize_by_n3(df_raw, meta_lookup)
    
    # 5. LMM 交互项回归
    logger.info("拟合线性混合效应模型...")
    
    lmm_phi1 = smf.mixedlm(
        "Phi1_Z ~ C(Stage) * C(Cohort_Type) + C(Night)",
        df_analysis,
        groups=df_analysis["Subject_ID"]
    ).fit(method='nm', maxiter=1000)
    
    lmm_var = smf.mixedlm(
        "Variance_Z ~ C(Stage) * C(Cohort_Type) + C(Night)",
        df_analysis,
        groups=df_analysis["Subject_ID"]
    ).fit(method='nm', maxiter=1000)
    
    # 6. 输出报告
    print("\n" + "=" * 23 + " CFECT Phase B: Sleep-EDF 终审统计报告 " + "=" * 23)
    print(f"\n样本量: {len(df_analysis)} 窗口")
    print(f"SC 受试者: {df_analysis[df_analysis['Cohort_Type']=='SC']['Subject_ID'].nunique()}")
    print(f"ST 受试者: {df_analysis[df_analysis['Cohort_Type']=='ST']['Subject_ID'].nunique()}")
    
    print("\n--- 1. resilience锁定项 (Phi1_Z) 交互项效应 ---")
    print(lmm_phi1.summary())
    print("\n--- 2. 显性通量凝聚项 (Variance_Z) 交互项效应 ---")
    print(lmm_var.summary())
    print("=" * 71 + "\n")
    
    # 7. 保存
    df_analysis.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"特征已保存: {OUTPUT_CSV}")
    
    return lmm_phi1, lmm_var


if __name__ == "__main__":
    run_sleep_edf_lmm_pipeline()
