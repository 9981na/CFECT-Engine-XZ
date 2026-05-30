"""
CFECT Heterogeneous State-Space Advanced Visualization Engine
=============================================================
Production-grade, zero-heuristic plotting framework.
Fixes applied: Drug_Condition fillna, BUT-PDB illustrative circles,
SDDB path correction, auto-detect paths, N3 unification, auto-scale Panel D.
"""

import os, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
from pathlib import Path

# ================= GLOBAL CONFIG =================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 8.0
rcParams['axes.linewidth'] = 0.75
rcParams['xtick.major.width'] = 0.75
rcParams['ytick.major.width'] = 0.75
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# Color palette
C_YIN    = '#2c5282'  # Variance / DNB axis
C_YANG   = '#9b2c2c'  # Phi1 / Corr axis
C_INTER  = '#718096'  # Inter-ictal / SC-Natural
C_PLACE  = '#dd6b20'  # ST-Placebo (amber)
C_TEMA   = '#319795'  # ST-Temazepam (teal)
C_CARD   = '#6b46c1'  # Cardiac centroids (purple)
C_NULL   = '#cbd5e0'  # Null reference (grey)

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("CFECT_Viz")


def _resolve_candidate(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _try_load_but_pdb(project_root):
    """Try loading BUT-PDB real features; return None if unavailable."""
    but_candidates = [
        str(Path(project_root) / '..' / '..' / 'MEM' / 'paper' / 'real' / 'output2' / 'but_pdb' / 'but_pdb_features.csv'),
        str(Path(r"E:\MEM\paper\real\output2\but_pdb\but_pdb_features.csv")),
    ]
    for p in but_candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            logger.info(f"BUT-PDB -> {p} ({len(df)} records)")
            return df
    logger.warning("BUT-PDB features not found; Panel D will use illustrative markers")
    return None


def load_strict_empirical_data(project_root):
    root = Path(project_root)
    candidates = {
        'chb': [str(root / 'data' / 'processed' / 'chb_mit_csd_master.csv'),
                str(root / 'chb_mit_csd_master.csv')],
        'sddb': [str(root / 'data' / 'processed' / 'sddb_terminal_master.csv'),
                 str(root / 'sddb_terminal_master.csv')],
        'sleep': [str(root / 'cfect_sleep_edf_hardened_features.csv'),
                  str(root / 'data' / 'processed' / 'cfect_sleep_edf_hardened_features.csv')],
    }
    paths = {}
    for key, cands in candidates.items():
        p = _resolve_candidate(cands)
        if p is None:
            raise FileNotFoundError(f"Cannot find {key} data.")
        paths[key] = p
    logger.info(f"CHB-MIT  -> {paths['chb']}")
    logger.info(f"SDDB     -> {paths['sddb']}")
    logger.info(f"Sleep-EDF-> {paths['sleep']}")

    df_chb = pd.read_csv(paths['chb'])
    df_sddb = pd.read_csv(paths['sddb'])
    df_sleep = pd.read_csv(paths['sleep'], low_memory=False)

    if 'Drug_Condition' not in df_sleep.columns:
        df_sleep['Drug_Condition'] = 'Natural_None'
    else:
        df_sleep['Drug_Condition'] = df_sleep['Drug_Condition'].fillna('Natural_None')
    df_sleep['Stage_Unified'] = df_sleep['Stage'].replace({4: 3})
    df_sleep['Stage_Label_Unified'] = df_sleep['Stage_Label'].replace(
        {'N3_slow': 'N3', 'N3_deep': 'N3'})
    df_chb['Time_Min'] = df_chb['Time_to_Onset'] / 60.0

    # Load BUT-PDB real centroids
    df_but = _try_load_but_pdb(project_root)

    logger.info(f"  Sleep-EDF: {len(df_sleep):,} rows, Drug_Condition: "
                f"{df_sleep['Drug_Condition'].value_counts().to_dict()}")
    logger.info(f"  CHB-MIT: {len(df_chb):,} rows, "
                f"Pre-ictal: {(df_chb['Condition']=='Pre-ictal').sum()}, "
                f"Inter-ictal: {(df_chb['Condition']=='Inter-ictal').sum()}")
    logger.info(f"  SDDB: {len(df_sddb):,} rows, records: {df_sddb['Record'].nunique()}")
    return df_chb, df_sddb, df_sleep, df_but


def generate_heterogeneous_publication_plots(project_root):
    df_chb, df_sddb, df_sleep, df_but = load_strict_empirical_data(project_root)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5))

    # ========================= PANEL A: CHB-MIT =========================
    ax_a = axes[0, 0]
    box_data = [
        df_chb[df_chb['Condition'] == 'Inter-ictal']['Phi1_Z'].dropna().values,
        df_chb[df_chb['Condition'] == 'Pre-ictal']['Phi1_Z'].dropna().values,
    ]
    bp = ax_a.boxplot(box_data, tick_labels=['Inter-ictal\n(n=3,680)', 'Pre-ictal\n(n=56,312)'],
                      patch_artist=True, widths=0.5,
                      boxprops=dict(facecolor='white', edgecolor=C_YANG, linewidth=1.2),
                      medianprops=dict(color=C_YANG, linewidth=1.5),
                      whiskerprops=dict(color='#4a5568', linewidth=0.8),
                      capprops=dict(color='#4a5568', linewidth=0.8),
                      flierprops=dict(marker='o', markersize=2, alpha=0.15,
                                     markerfacecolor=C_YANG, markeredgecolor='none'))

    # Add effect size annotation
    inter_mean = box_data[0].mean()
    pre_mean = box_data[1].mean()
    from scipy import stats as scipy_stats
    t_stat, p_val = scipy_stats.ttest_ind(box_data[1], box_data[0], equal_var=False)
    d = (pre_mean - inter_mean) / np.sqrt((np.var(box_data[1],ddof=1)+np.var(box_data[0],ddof=1))/2)
    ax_a.text(0.5, 0.95, f'Cohen\'s d = {d:.3f}',
              transform=ax_a.transAxes, ha='center', va='top',
              fontsize=7, fontstyle='italic', color='#2d3748')
    ax_a.text(0.5, 0.88, f't = {t_stat:.1f}, p = {p_val:.1e}',
              transform=ax_a.transAxes, ha='center', va='top',
              fontsize=6.5, fontstyle='italic', color='#718096')

    ax_a.set_title('a | CHB-MIT Seizure Pre-ictal CSD', fontsize=8, fontweight='bold')
    ax_a.set_ylabel('$\\Phi_{1\\,Z}$ (Structural Memory)')
    ax_a.set_ylim(-3.5, 3.5)

    # ========================= PANEL B: SDDB =========================
    ax_b = axes[0, 1]
    if df_sddb is not None and len(df_sddb) > 0:
        df_sddb['Time_Min'] = df_sddb['Time_to_Event'] / 60.0
        for col in ['DNB_Std', 'External_Correlation']:
            df_sddb[col] = df_sddb[col].replace([np.inf, -np.inf], np.nan)
        sddb_clean = df_sddb.dropna(subset=['DNB_Std', 'External_Correlation']).sort_values('Time_Min')
        ax_b.plot(sddb_clean['Time_Min'], sddb_clean['DNB_Std'],
                  color=C_YIN, linewidth=1.2, label='DNB Volatility (Yin)')
        ax_b.plot(sddb_clean['Time_Min'], sddb_clean['External_Correlation'],
                  color=C_YANG, linewidth=1.2, label='External Corr (Yang)')
        ax_b.axhline(0, color=C_NULL, linestyle='--', linewidth=0.8, alpha=0.6,
                     label='Null (Uncoupled)')
        ax_b.set_title('b | SDDB Terminal Decompensation', fontsize=8, fontweight='bold')
        ax_b.set_xlabel('Time to Event (min)')
        ax_b.set_ylabel('Thermodynamic Metric')
        ax_b.set_xlim(-350, 50)
    else:
        ax_b.text(0.5, 0.5, 'SDDB data not found', ha='center', va='center', color='gray', fontsize=7.5)
        ax_b.set_title('b | SDDB (Pending)', fontsize=8, fontweight='bold', color='gray')
    ax_b.legend(frameon=False, loc='upper left', fontsize=6.2)

    # ========================= PANEL C: Sleep-EDF =========================
    ax_c = axes[1, 0]
    cohort_configs = [
        ('Natural_None', C_INTER, 'SC-Natural (Drug-Free)'),
        ('Placebo',      C_PLACE, 'ST-Placebo (Insomnia)'),
        ('Temazepam',    C_TEMA,  'ST-Temazepam (Damped)'),
    ]
    for cond, color, label in cohort_configs:
        sub = df_sleep[df_sleep['Drug_Condition'] == cond]
        if sub.empty:
            continue
        profile_c = sub.groupby('Stage_Unified').agg(
            Phi1_Z=('Phi1_Z', 'mean'), Variance_Z=('Variance_Z', 'mean'))
        ax_c.plot(profile_c.index, profile_c['Phi1_Z'],
                  color=color, marker='s', linestyle='-',
                  linewidth=1.2, markersize=3, label=label + ' $\\Phi_{1\\,Z}$')

    ax_c.set_xticks([0, 1, 2, 3, 5])
    ax_c.set_xticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'], fontsize=6.8)
    ax_c.set_title('c | Sleep-EDF Attractor Rigidification', fontsize=8, fontweight='bold')
    ax_c.set_xlabel('Sleep Macro-State (Ordinal)')
    ax_c.set_ylabel('$\\Phi_{1\\,Z}$ (Structural Memory)')
    ax_c.legend(frameon=False, loc='upper right', fontsize=6, handletextpad=0.3)

    # ========================= PANEL D: Cross-Cohort Phase-Space =========================
    ax_d = axes[1, 1]

    # Collect ALL point coordinates to auto-scale
    all_var = []
    all_phi = []

    # CHB-MIT pre-ictal scatter
    df_pre = df_chb[df_chb['Condition'] == 'Pre-ictal']
    df_pre_sample = df_pre.sample(min(1000, len(df_pre)), random_state=GLOBAL_SEED)
    all_var.extend(df_pre_sample['Variance_Z'].values.tolist())
    all_phi.extend(df_pre_sample['Phi1_Z'].values.tolist())
    import seaborn as sns
    sns.regplot(data=df_pre_sample, x='Variance_Z', y='Phi1_Z',
                ax=ax_d, color='#cbd5e0',
                scatter_kws={'alpha': 0.12, 's': 3.5, 'edgecolor': 'none'},
                line_kws={'color': '#718096', 'linewidth': 0.8, 'linestyle': ':'})

    # Sleep-EDF centroids
    centroid_data = []
    for cond, color, marker, size, label in [
        ('Natural_None', C_INTER, 'X', 7, 'Centroid: SC-Natural (Healthy)'),
        ('Placebo',      C_PLACE, 'D', 6, 'Centroid: ST-Placebo (Insomnia)'),
        ('Temazepam',    C_TEMA,  '*', 9, 'Centroid: ST-Temazepam (Damped)'),
    ]:
        sub = df_sleep[df_sleep['Drug_Condition'] == cond]
        if sub.empty:
            continue
        cx, cy = sub['Variance_Z'].mean(), sub['Phi1_Z'].mean()
        centroid_data.append((cond, cx, cy))
        all_var.append(cx)
        all_phi.append(cy)
        ax_d.plot(cx, cy, color=color, marker=marker, markersize=size,
                  markeredgecolor='black', markeredgewidth=0.6,
                  label=label, zorder=5)

    # BUT-PDB real centroids (from WFDB annotation parser)
    if df_but is not None and not df_but.empty:
        but_configs = [
            ('AFIB', 'P', 'Centroid: BUT-AFIB (Real)'),
            ('Bigeminy', 'v', 'Centroid: BUT-Bigeminy (Real)'),
        ]
        for pathology, marker, label in but_configs:
            sub = df_but[df_but['Diagnosis'].str.contains(pathology, case=False, na=False)]
            if not sub.empty:
                cx, cy = sub['Variance_Z'].mean(), sub['Phi1_Z'].mean()
                all_var.append(cx)
                all_phi.append(cy)
                ax_d.plot(cx, cy, color=C_CARD, marker=marker, markersize=7,
                          markeredgecolor='black', markeredgewidth=0.6,
                          label=label, zorder=5)
                logger.info(f"  BUT-PDB {pathology}: VarZ={cx:.4f}, PhiZ={cy:.4f} (from {len(sub)} windows)")
    else:
        # Fallback illustrative markers (only if real data unavailable)
        logger.warning("BUT-PDB data not available; using illustrative markers in Panel D")
        for cx, cy, label in [
            (0.25, 0.45, 'BUT-PDB AFIB (Illustrative)'),
            (-0.45, -0.62, 'BUT-PDB Bigeminy (Illustrative)'),
        ]:
            all_var.append(cx)
            all_phi.append(cy)
            circle = patches.Circle((cx, cy), 0.08, fill=False, linestyle='--',
                                    linewidth=1.0, edgecolor=C_CARD, alpha=0.7)
            ax_d.add_patch(circle)
            ax_d.plot(cx, cy, 'o', color=C_CARD, markersize=4, alpha=0.5, zorder=4)
            ax_d.text(cx + 0.08, cy + 0.02, label, fontsize=5.5, color=C_CARD,
                      fontstyle='italic', alpha=0.8)

    # Temazepam damping directional arrow
    if len(centroid_data) >= 3:
        p_xy = centroid_data[1]
        t_xy = centroid_data[2]
        ax_d.annotate('', xy=(t_xy[1], t_xy[2]), xytext=(p_xy[1], p_xy[2]),
                      arrowprops=dict(arrowstyle="->", color='black', lw=0.8, ls='--'))

    # DEFAULT: auto-scale to include all data with 15% padding
    var_min, var_max = min(all_var), max(all_var)
    phi_min, phi_max = min(all_phi), max(all_phi)
    var_pad = 0.15 * (var_max - var_min) if var_max != var_min else 1.0
    phi_pad = 0.15 * (phi_max - phi_min) if phi_max != phi_min else 1.0
    ax_d.set_xlim(var_min - var_pad, var_max + var_pad)
    ax_d.set_ylim(phi_min - phi_pad, phi_max + phi_pad)

    ax_d.set_title('d | Cross-Cohort Phase-Space Topology', fontsize=8, fontweight='bold')
    ax_d.set_xlabel('$Variance_Z$ (Yin / Observable Flux)')
    ax_d.set_ylabel('$\\Phi_{1\\,Z}$ (Yang / Structural Memory)')
    ax_d.legend(frameon=False, loc='upper left', fontsize=5.8, handletextpad=0.2, ncol=1)

    # ========================= UNIFIED FORMATTING =========================
    for row in axes:
        for axis in row:
            axis.tick_params(direction='in', labelsize=7)
            axis.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    out_dir = Path(project_root) / 'results'
    out_dir.mkdir(exist_ok=True)
    png_path = out_dir / 'CFECT_Heterogeneous_Panels.png'
    pdf_path = out_dir / 'CFECT_Heterogeneous_Panels.pdf'
    fig.savefig(str(png_path), dpi=600, bbox_inches='tight')
    fig.savefig(str(pdf_path), bbox_inches='tight')
    plt.close()

    logger.info(f"====== Output generated ======")
    logger.info(f"  PNG: {png_path} ({png_path.stat().st_size / 1024:.0f} KB)")
    logger.info(f"  PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")
    return str(png_path)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent if script_dir.name == 'visualization' else script_dir
    print(f"Project root: {root}")
    generate_heterogeneous_publication_plots(str(root))
