"""
CHB-MIT Spatial Focus Routing Engine
======================================
Patient-specific focal channel selection + CFECT on focal channels only.

Protocol:
  1. Parse all summary.txt files -> extract seizure timestamps
  2. For each seizure file, scan 5s pre-onset window across ALL 23 channels
  3. Identify Top-2 channels with maximum variance surge
  4. These = "Patient-Specific Focal Channels" (PSFC)
  5. Extract 30min pre-ictal CFECT features ONLY on PSFC
  6. Compare with inter-ictal baseline

Output:
  results/chbmit_spatial_csd.csv

Reference:
  - Focal epilepsy: EEG channel selection via pre-ictal variance surge
  - Schindler et al. (2007), Brain
"""

import numpy as np
import pandas as pd
import os, sys, re, logging, warnings
from collections import defaultdict
from scipy import stats, signal as scipy_signal
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("Spatial_Routing")

# ================= CONFIG =================
CHB_ROOT = r"E:\MEM\paper\real\data\CHB-MIT Scalp EEG Database"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
WINDOW_SIZE = 3000       # 30s @ 256Hz (rounded)
STEP_SIZE = 750          # 7.5s (75% overlap)
PRE_ICTAL_WINDOW = 1800  # 30 min in seconds
SURGE_WINDOW = 5         # 5 seconds pre-onset for channel selection
TOP_N_CHANNELS = 2
N_BINS_TIME = 10         # Time bins for trajectory


def parse_summary(summary_path: str) -> list:
    """
    Parse CHB-MIT summary.txt -> list of (edf_file, seizure_start, seizure_end)
    """
    seizures = []
    current_file = None
    try:
        with open(summary_path, 'r') as f:
            for line in f:
                m = re.match(r'File Name:\s*(\S+)', line)
                if m:
                    current_file = m.group(1)
                m = re.match(r'Seizure Start Time:\s*(\d+)\s*seconds', line)
                if m and current_file:
                    start = int(m.group(1))
                m = re.match(r'Seizure End Time:\s*(\d+)\s*seconds', line)
                if m and current_file:
                    end = int(m.group(1))
                    seizures.append((current_file, start, end))
    except Exception as e:
        logger.warning(f"  Could not parse {summary_path}: {e}")
    return seizures


def find_surge_channels(edf_path: str, pre_ictal_onset: int, n_channels: int = TOP_N_CHANNELS):
    """
    Read EDF, compute variance in 5s pre-onset window for each channel,
    return Top-2 channels with max variance surge.
    """
    try:
        import pyedflib
        edf = pyedflib.EdfReader(edf_path)
        n_sig = edf.signals_in_file
        labels = edf.getSignalLabels()
        fs = int(edf.getSampleFrequencies()[0])  # All 256 Hz
        
        # 5s window: [onset-5, onset]
        start_sample = max(0, (pre_ictal_onset - SURGE_WINDOW) * fs)
        end_sample = pre_ictal_onset * fs
        
        variances = []
        for i in range(n_sig):
            sig = edf.readSignal(i, start_sample, end_sample - start_sample)
            variances.append(np.var(sig))
        
        edf.close()
        
        # Top-N channels with highest variance
        top_idx = np.argsort(variances)[-n_channels:][::-1]
        return [(labels[i], variances[i]) for i in top_idx]
        
    except Exception as e:
        logger.error(f"  Error reading {edf_path}: {e}")
        return None


def lag1_autocorrelation(x: np.ndarray) -> float:
    """AR(1) coefficient via lag-1 autocorrelation."""
    if len(x) < 10 or np.std(x) < 1e-10:
        return 0.0
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def local_variance(x: np.ndarray) -> float:
    """Window variance."""
    return np.var(x) if len(x) > 1 else 0.0


def extract_cfect_on_channel(eeg_signal: np.ndarray, fs: int = 256) -> pd.DataFrame:
    """
    Sliding window CFECT: AR(1) + variance on a single channel.
    Returns DataFrame with Time, Phi1, Variance.
    """
    window = int(WINDOW_SIZE)
    step = int(STEP_SIZE)
    
    rows = []
    for start in range(0, len(eeg_signal) - window, step):
        chunk = eeg_signal[start:start + window]
        phi1 = lag1_autocorrelation(chunk)
        var = local_variance(chunk)
        time_sec = start / fs
        rows.append({'Time_Sec': time_sec, 'Phi1': phi1, 'Variance': var})
    
    return pd.DataFrame(rows)


def process_patient(patient_id: str) -> pd.DataFrame:
    """
    Full pipeline for one patient.
    
    Returns: DataFrame with CFECT features on focal channels, pre and inter ictal.
    """
    patient_dir = os.path.join(CHB_ROOT, patient_id)
    summary_path = os.path.join(patient_dir, f'{patient_id}-summary.txt')
    
    if not os.path.exists(summary_path):
        logger.warning(f"No summary for {patient_id}, skipping.")
        return pd.DataFrame()
    
    seizures = parse_summary(summary_path)
    logger.info(f"  {patient_id}: {len(seizures)} seizures detected")
    
    if not seizures:
        return pd.DataFrame()
    
    all_rows = []
    
    for edf_file, sz_start, sz_end in seizures[:10]:  # Limit to 10 seizures/patient
        edf_path = os.path.join(patient_dir, edf_file)
        if not os.path.exists(edf_path):
            logger.warning(f"  EDF not found: {edf_path}")
            continue
        
        # Step 1: Find focal channels
        focal_channels = find_surge_channels(edf_path, sz_start)
        if not focal_channels:
            continue
        
        fc_labels = [ch[0] for ch in focal_channels]
        logger.info(f"  {edf_file}: focal channels = {fc_labels}, variances = {[f'{ch[1]:.2e}' for ch in focal_channels]}")
        
        # Step 2: Read EDF and extract CFECT on focal channels
        try:
            import pyedflib
            edf = pyedflib.EdfReader(edf_path)
            n_sig = edf.signals_in_file
            fs = int(edf.getSampleFrequencies()[0])
            
            # Map channel labels to indices
            labels = edf.getSignalLabels()
            fc_indices = [i for i, lbl in enumerate(labels) if lbl in fc_labels]
            
            # Pre-ictal: [sz_start - 30min, sz_start]
            pre_start_sec = max(0, sz_start - PRE_ICTAL_WINDOW)
            pre_start_sample = pre_start_sec * fs
            pre_end_sample = sz_start * fs
            
            # Inter-ictal: [sz_start + 30min, sz_end + 60min] (after seizure ends)
            post_start_sample = (sz_end + 600) * fs  # 10 min after seizure
            post_end_sample = post_start_sample + PRE_ICTAL_WINDOW * fs
            
            # Limit to EDF duration
            total_samples = int(edf.getNSamples()[0])
            post_end_sample = min(post_end_sample, total_samples)
            
            for ch_idx in fc_indices:
                ch_label = labels[ch_idx]
                
                # Pre-ictal CFECT
                sig_pre = edf.readSignal(ch_idx, pre_start_sample, pre_end_sample - pre_start_sample)
                df_pre = extract_cfect_on_channel(sig_pre, fs)
                df_pre['Condition'] = 'Pre-ictal'
                df_pre['Time_to_Onset'] = df_pre['Time_Sec'] - (sz_start - pre_start_sec)
                
                # Inter-ictal CFECT (if enough data)
                if post_end_sample > post_start_sample + fs * 300:  # At least 5 min
                    sig_post = edf.readSignal(ch_idx, post_start_sample, post_end_sample - post_start_sample)
                    df_inter = extract_cfect_on_channel(sig_post, fs)
                    df_inter['Condition'] = 'Inter-ictal'
                    df_inter['Time_to_Onset'] = df_inter['Time_Sec'] + (sz_end + 600)
                else:
                    df_inter = pd.DataFrame()
                
                for df_ in [df_pre, df_inter]:
                    if len(df_) > 0:
                        df_['Patient_ID'] = int(patient_id.replace('chb', ''))
                        df_['EDF_File'] = edf_file
                        df_['Seizure_Start'] = sz_start
                        df_['Seizure_End'] = sz_end
                        df_['Focal_Channel'] = ch_label
                        df_['Focal_Rank'] = fc_labels.index(ch_label) + 1
                        all_rows.append(df_)
            
            edf.close()
            
        except Exception as e:
            logger.error(f"  Error processing {edf_path}: {e}")
            continue
    
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


def process_all_patients(limit: int = None):
    """Process all CHB-MIT patients."""
    patients = sorted([d for d in os.listdir(CHB_ROOT) 
                       if d.startswith('chb') and os.path.isdir(os.path.join(CHB_ROOT, d))])
    
    if limit:
        patients = patients[:limit]
    
    logger.info(f"Processing {len(patients)} patients: {patients}")
    
    all_data = []
    for pid in patients:
        logger.info(f"=== {pid} ===")
        df = process_patient(pid)
        if len(df) > 0:
            all_data.append(df)
            logger.info(f"  -> {len(df)} CFECT windows extracted")
        else:
            logger.info(f"  -> No data extracted")
    
    if not all_data:
        logger.warning("No data collected from any patient!")
        return None
    
    result = pd.concat(all_data, ignore_index=True)
    
    # Compute z-scores per patient
    for pid in result['Patient_ID'].unique():
        mask = result['Patient_ID'] == pid
        pre = result[mask & (result['Condition'] == 'Pre-ictal')]
        inter = result[mask & (result['Condition'] == 'Inter-ictal')]
        
        if len(inter) > 0:
            mu_phi = inter['Phi1'].mean()
            sigma_phi = inter['Phi1'].std()
            mu_var = inter['Variance'].mean()
            sigma_var = inter['Variance'].std()
        else:
            # Fallback: use global stats
            all_inter = result[result['Condition'] == 'Inter-ictal']
            mu_phi = all_inter['Phi1'].mean()
            sigma_phi = all_inter['Phi1'].std()
            mu_var = all_inter['Variance'].mean()
            sigma_var = all_inter['Variance'].std()
        
        if sigma_phi > 0:
            result.loc[mask, 'Phi1_Z'] = (result.loc[mask, 'Phi1'] - mu_phi) / sigma_phi
        if sigma_var > 0:
            result.loc[mask, 'Variance_Z'] = (result.loc[mask, 'Variance'] - mu_var) / sigma_var
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'chbmit_spatial_csd.csv')
    result.to_csv(out_path, index=False)
    logger.info(f"Saved {len(result)} rows to {out_path}")
    
    # Summary stats
    pre_count = (result['Condition'] == 'Pre-ictal').sum()
    inter_count = (result['Condition'] == 'Inter-ictal').sum()
    logger.info(f"  Pre-ictal rows: {pre_count}")
    logger.info(f"  Inter-ictal rows: {inter_count}")
    logger.info(f"  Unique patients: {result['Patient_ID'].nunique()}")
    logger.info(f"  Unique focal channels: {result['Focal_Channel'].unique()}")
    
    return result


if __name__ == "__main__":
    # Test with chb01 first
    logger.info("=" * 60)
    logger.info("CHB-MIT SPATIAL FOCUS ROUTING ENGINE v1.0")
    logger.info("=" * 60)
    result = process_all_patients(limit=3)  # First 3 patients
    if result is not None:
        print("\n=== Spatial Routing Complete ===")
        print(f"Pre-ictal Phi1_Z mean: {result[result['Condition']=='Pre-ictal']['Phi1_Z'].mean():.3f}")
        print(f"Inter-ictal Phi1_Z mean: {result[result['Condition']=='Inter-ictal']['Phi1_Z'].mean():.3f}")
