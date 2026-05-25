# Data Download Guide

This guide explains how to acquire and structure the datasets used in the CFECT-Quantum-Engine.

## Required Dependencies

```bash
pip install mne scipy numpy scikit-learn numba matplotlib seaborn
```

## Dataset 1: Cortical Matrix (Sleep-EDF Database)

### Source
**URL:** [https://physionet.org/content/sleep-edf-database-expanded/1.0.0/](https://physionet.org/content/sleep-edf-database-expanded/1.0.0/)

### Download Instructions
1. Navigate to the PhysioNet website
2. Download the `sleep-cassette` subset (approximately 1.8 GB)
3. Extract contents to `./data/sleep-edf/`

### Required Files
- `*PSG.edf`: Polysomnography recordings
- `*Hypnogram.edf`: Expert annotations

### Expected Structure
```
./data/
└── sleep-edf/
    ├── SC4001E0-PSG.edf
    ├── SC4001EC-Hypnogram.edf
    ├── SC4002E0-PSG.edf
    ├── SC4002EC-Hypnogram.edf
    └── ...
```

### Replication Note
For reproducing the results in Chapter 5, download files for the first **20 subjects** only.

## Dataset 2: Cardiac Matrix (BUT PDB Database)

### Source
**URL:** [https://physionet.org/content/butpdb/1.0.0/](https://physionet.org/content/butpdb/1.0.0/)

### Download Instructions
1. Navigate to the PhysioNet website
2. Download the complete dataset
3. Extract contents to `./data/but_pdb/`

### Required Files
- `*.dat`: ECG signal files
- `*.hea`: Header files

### Expected Structure
```
./data/
└── but_pdb/
    ├── p00001.dat
    ├── p00001.hea
    ├── p00002.dat
    ├── p00002.hea
    └── ...
```

## Dataset 3: Pre-computed Features (Fast-Pass Mode)

For the Reviewer Fast-Pass mode, pre-computed features can be generated using:

```bash
python pipelines/run_eeg_sleep_staging.py --mode fast
```

This will automatically generate synthetic features if no pre-computed file exists.

## Directory Structure

```
CFECT-Quantum-Engine/
├── data/
│   ├── sleep-edf/          # Sleep-EDF dataset (optional)
│   ├── but_pdb/            # BUT PDB dataset (optional)
│   ├── precomputed_features.npz  # Generated features
│   └── DATA_DOWNLOAD_GUIDE.md
├── src/                    # Core algorithms
├── pipelines/              # End-to-end pipelines
└── results/                # Output directory (auto-generated)
```

## Data License

All datasets are provided under the PhysioNet Terms of Use.
Please refer to the respective dataset pages for licensing information.
