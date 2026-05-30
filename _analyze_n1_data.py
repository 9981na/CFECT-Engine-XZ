"""Analyze real Sleep-EDF N1 data for HSMM duration modeling"""
import pandas as pd
import numpy as np

# Load real Sleep-EDF features
df = pd.read_csv(r'E:\MEM\paper\real\output2\main\sleep_csd_features.csv')
print(f'Total windows: {len(df):,}')
print(f'Stage distribution:')
print(df['Sleep_Stage'].value_counts())

print(f'\nSC subjects: {df[df["Study_Type"]=="SC"]["Subject_ID"].nunique()}')
print(f'ST subjects: {df[df["Study_Type"]=="ST"]["Subject_ID"].nunique()}')

print(f'\nVariance_Z by stage:')
print(df.groupby('Sleep_Stage')['Variance_Z'].describe().round(3))
print(f'\nPhi1_Z by stage:')
print(df.groupby('Sleep_Stage')['Phi1_Z'].describe().round(3))

# N1 stats
n1 = df[df['Sleep_Stage'] == 'N1']
wake = df[df['Sleep_Stage'] == 'W']
n2 = df[df['Sleep_Stage'] == 'N2']
print(f'\n*** Wake vs N1 vs N2 feature overlap ***')
print(f'Wake:  Phi1_Z={wake["Phi1_Z"].mean():.3f}, Var_Z={wake["Variance_Z"].mean():.3f}  (n={len(wake)})')
print(f'N1:    Phi1_Z={n1["Phi1_Z"].mean():.3f}, Var_Z={n1["Variance_Z"].mean():.3f}  (n={len(n1)})')
print(f'N2:    Phi1_Z={n2["Phi1_Z"].mean():.3f}, Var_Z={n2["Variance_Z"].mean():.3f}  (n={len(n2)})')

# Euclidean distances in feature space
for pair, a, b in [('Wake-N1', wake, n1), ('N1-N2', n1, n2), ('Wake-N2', wake, n2)]:
    a_feat = a[['Variance_Z', 'Phi1_Z']].mean().values
    b_feat = b[['Variance_Z', 'Phi1_Z']].mean().values
    dist = np.linalg.norm(a_feat - b_feat)
    print(f'  {pair} centroid distance: {dist:.4f}')

# N1 dwell time analysis
print(f'\n*** N1 Dwell Time Analysis (all subjects) ***')
all_n1_runs = []
for subj in df['Subject_ID'].unique():
    subj_df = df[df['Subject_ID']==subj].sort_values('Time_Sec')
    stages = subj_df['Sleep_Stage'].values
    run_len = 0
    for s in stages:
        if s == 'N1':
            run_len += 1
        else:
            if run_len > 0:
                all_n1_runs.append(run_len)
            run_len = 0
    if run_len > 0:
        all_n1_runs.append(run_len)

all_n1_runs = np.array(all_n1_runs)
if len(all_n1_runs) > 0:
    print(f'Total N1 runs: {len(all_n1_runs)}')
    print(f'Mean dwell: {all_n1_runs.mean():.2f} epochs (x7.5s = {all_n1_runs.mean()*7.5:.1f}s)')
    print(f'Median dwell: {np.median(all_n1_runs):.1f} epochs')
    print(f'Std dwell: {all_n1_runs.std():.2f} epochs')
    print(f'Min-Max: {all_n1_runs.min()}-{all_n1_runs.max()} epochs')
    print(f'Dwell distribution (epochs):')
    for d in range(1, 11):
        pct = (all_n1_runs == d).mean() * 100
        print(f'  d={d}: {pct:.1f}%')

# Check if geometric distribution fits
print(f'\n*** Test: Does N1 dwell follow geometric distribution? ***')
# Geometric: P(d) = (1-p)^(d-1) * p
# MLE for p: p_hat = 1 / mean_dwell
if len(all_n1_runs) > 0:
    p_mle = 1.0 / all_n1_runs.mean()
    print(f'HMM geometric implied p = 1/mean = {p_mle:.4f}')
    # Chi-square goodness of fit
    from scipy import stats
    observed = np.array([(all_n1_runs == d).sum() for d in range(1, 11)])
    geo_probs = [(1-p_mle)**(d-1) * p_mle for d in range(1, 11)]
    geo_probs[-1] = 1 - sum(geo_probs[:-1])  # tail
    expected = geo_probs * len(all_n1_runs)
    from scipy.stats import poisson
    # Poisson fit (HSMM alternative)
    lambda_mle = all_n1_runs.mean()
    pois_probs = [poisson.pmf(d, lambda_mle) for d in range(1, 11)]
    pois_probs[-1] = 1 - sum(pois_probs[:-1])
    pois_expected = pois_probs * len(all_n1_runs)
    
    print(f'Geometric vs Poisson fit comparison:')
    print(f'  d   observed  geometric_exp  poisson_exp')
    for i, d in enumerate(range(1, 11)):
        print(f'  {d:2d}  {observed[i]:6d}   {expected[i]:6.1f}       {pois_expected[i]:6.1f}')
    
    # Effective F1 upper bound from feature overlap
    print(f'\n*** Estimated upper bound F1 for N1 (from feature separability) ***')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import f1_score
    
    # Balance the dataset
    stages_of_interest = ['W', 'N1', 'N2']
    df_sel = df[df['Sleep_Stage'].isin(stages_of_interest)].copy()
    
    # Filter subjects with enough N1
    n1_per_subj = df_sel[df_sel['Sleep_Stage']=='N1'].groupby('Subject_ID').size()
    valid_subjects = n1_per_subj[n1_per_subj >= 20].index
    df_sel = df_sel[df_sel['Subject_ID'].isin(valid_subjects)]
    print(f'Subjects with >=20 N1: {len(valid_subjects)}')
    print(f'Windows after filter: {len(df_sel)}')
    
    # RF on raw features
    X = df_sel[['Variance_Z', 'Phi1_Z']].values
    y = df_sel['Sleep_Stage'].map({'W':0, 'N1':1, 'N2':2}).values
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    n1_f1 = f1_score(y_test == 1, y_pred == 1)
    print(f'\nRF Binary N1 F1: {n1_f1:.4f}')
    print(f'RF Overall Accuracy: {(y_pred == y_test).mean()*100:.2f}%')
    print(f'\nConfusion matrix:')
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f'         Pred: Wake  N1    N2')
    for i, name in enumerate(['Wake', 'N1', 'N2']):
        print(f'  True {name:4s}:  {cm[i,0]:4d}  {cm[i,1]:4d}  {cm[i,2]:4d}')
