
# PhysioNet Database Loading Guide

## CHB-MIT Scalp EEG Database

### Access Instructions
1. Register at https://physionet.org/
2. Install WFDB package: `pip install wfdb`
3. Download using:
   ```python
   import wfdb
   wfdb.dl_database('chbmit', dl_dir='data/raw/chbmit')
   ```

### File Structure
```
chbmit/
├── chb01/
│   ├── chb01_01.edf
│   ├── chb01_02.edf
│   └── chb01-summary.txt
├── chb02/
└── ...
```

### Annotation Files
- `*-summary.txt`: Contains seizure onset/offset times
- EDF files contain 23 EEG channels sampled at 256 Hz

## SDDB Sudden Death Database

### Access Instructions
1. Register at https://physionet.org/content/sddb/1.0.0/
2. Download using:
   ```python
   wfdb.dl_database('sddb', dl_dir='data/raw/sddb')
   ```

### File Structure
```
sddb/
├── records/
│   ├── 100.atr
│   ├── 100.dat
│   ├── 100.hea
│   └── ...
└── README.md
```

### Data Format
- MIT-BIH format (.dat, .hea, .atr)
- ECG signals sampled at 250 Hz
- Annotation files contain beat labels and rhythm annotations

## Lazy Loading Configuration

For large datasets, use lazy loading to avoid memory issues:

```python
import mne

# Lazy load EDF file
raw = mne.io.read_raw_edf(
    'chb01_01.edf',
    preload=False,  # Critical for memory efficiency
    verbose=False
)

# Load specific channels only
raw.pick_channels(['T7-P7', 'T8-P8'])

# Load data in chunks
for i, epoch in enumerate(raw.iter_evoked()):
    process_chunk(epoch)
```

## Data License

- CHB-MIT: Creative Commons Attribution 4.0 International
- SDDB: PhysioNet Research Data License 1.0.0

## Reference Citations

1. Shoeb A. (2013). CHB-MIT Scalp EEG Database. PhysioNet.
2. Moody GB, Mark RG. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol.
