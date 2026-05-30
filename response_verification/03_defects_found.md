# Defects Found During Verification Audit
## Not Corrected — Documented as Known Limitations

---

## Defect #1: ST Temazepam Subject_ID Suffix Truncation

**Severity**: High  
**Status**: ✅ FIXED (see changelog below)  
**Pipeline**: `pipelines/run_sleep_edf_csd.py`

### Root Cause
In `find_edf_files()`, line 84:
```python
rec_id = base[:6]   # "ST7011J0" → "ST7011" — J0/JP suffix lost!
```

The Sleep Telemetry (ST) records use the following naming convention:
- `ST7011J0-PSG.edf` → Placebo night
- `ST7011JP-PSG.edf` → Temazepam night

By truncating to `rec_id = base[:6]`, both nights collapse to `"ST7011"`, making it impossible for `add_meta_labels()` to detect `'J0'` vs `'JP'` in the Subject_ID. All 24,104 ST windows were labeled `Is_Drug=0` (Placebo).

### Fix Applied
Line 84 changed to:
```python
rec_id = base  # Preserve J0/JP suffix for drug labeling
```

And line 87 hypnogram fallback updated from:
```python
hyp = os.path.join(ST_DIR, f"{rec_id}JP-Hypnogram.edf")
```
to:
```python
hyp = os.path.join(ST_DIR, f"{base}JP-Hypnogram.edf")
```

### Expected Outcome After Re-run
- ST Subject_ID count: 22 → 44 (each subject has 2 nights)
- `Is_Drug=1` windows: 0 → ~12,000 (Temazepam)
- `Is_Drug=0` windows: 24,104 → ~12,000 (Placebo)
- `Φ₁_Z(ST Temazepam Wake)` will be measurable

---

## Defect #2: `reproduce_all.py` Synthetic Data Contamination

**Severity**: 🔴 CRITICAL  
**Status**: NOT FIXED — requires full pipeline re-run  
**Pipeline**: `reproduce_all.py`

### Root Cause
Lines 411-451 generate synthetic EEG-like data using random seed 42:
```python
np.random.seed(42)
# All "verified" coefficients (+0.436, -0.107, etc.) are seed-42 artifacts
```

### Impact
All numerical claims in `reproduce_all.py` are not based on real clinical data but on synthetic random data. This includes:
- CFECT coefficient estimates
- Slope values
- "Verified" sensitivity analysis results

### Recommendation
Replace with real data from:
- `E:\MEM\paper\real\output2\main\sleep_csd_features.csv`
- `E:\MEM\paper\real\output2\chb_mit_labeled.csv`

---

## Defect #3: `run_eeg_sleep_staging.py` Random Noise in Final Accuracy

**Severity**: 🔴 CRITICAL  
**Status**: NOT FIXED — requires real scoring pipeline  
**Pipeline**: `pipelines/run_eeg_sleep_staging.py`

### Root Cause
Line 184:
```python
mean_hmm_acc = 93.3 + np.random.normal(0, 1.5)
```

The reported "accuracy" of 93.3% is a constant plus random noise, not an actual computed metric.

### Impact
Any accuracy claim derived from this pipeline is invalid.

---

## Defect #4: `features.py` MSE τ=5 Violates Stam Stationarity

**Severity**: 🟡 MODERATE  
**Status**: NOT FIXED  
**Pipeline**: `src/features.py`

### Root Cause
Line 69 computes multiscale entropy with τ=5 coarse-graining, which may violate the stationarity assumptions required for reliable sample entropy estimation (per Stam 2005).

### Recommendation
Implement adaptive τ selection based on data autocorrelation decay.

---

## Defect #5: Manuscript "BUT PDB" Reference

**Severity**: 🔴 CRITICAL — **BUT PDB is NOT a typo**  
**Status**: ✅ CORRECT AS WRITTEN  

The manuscript references "BUT PDB" (Brno University of Technology — Physionet Database for AFib), not "SDDB" (Sudden Cardiac Death Database). The RedTeam auditor's assumption that it was a typo for "SDDB" is incorrect.

---

## Defect Tracking Summary

| ID | Description | Severity | Status |
|---|---|---|---|
| #1 | ST Temazepam J0/JP suffix truncation | High | ✅ FIXED |
| #2 | `reproduce_all.py` synthetic data | Critical | 🔴 NOT FIXED |
| #3 | `run_eeg_sleep_staging.py` noise floor | Critical | 🔴 NOT FIXED |
| #4 | `features.py` τ=5 stationarity violation | Moderate | 🟡 NOT FIXED |
| #5 | Manuscript "BUT PDB" — not a typo | — | ✅ CORRECT |
