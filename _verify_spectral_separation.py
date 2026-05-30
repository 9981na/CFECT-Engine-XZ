#!/usr/bin/env python3
"""
Verify REM-Wake Spectral Separation: Theta/Alpha Ratio as Symmetry-Breaking Operator
=============================================================================
Uses MNE for correct EDF+ annotation parsing (Sleep-EDF hypnograms are annotation-only).
Tests whether spectral features (esp. theta/alpha ratio) significantly distinguish REM from Wake.

Theory:
  REM  -> Hippocampal theta dominance -> high theta/alpha ratio
  Wake -> Occipital alpha dominance   -> low theta/alpha ratio
  N1   -> Transitional theta/alpha mixing (intermediate)
"""
import numpy as np, pandas as pd, os, sys, glob, warnings
from scipy import stats
warnings.filterwarnings('ignore')
import mne
DATA_ROOT = r"E:\MEM\paper\real\data\sleep-edf-database-expanded-1.0.0"
SC_DIR = os.path.join(DATA_ROOT, "sleep-cassette")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import extract_spectral_powers
WINDOW_SIZE=3000; STEP_SIZE=750; SUB_WINDOWS=4
STAGE_MAPPING={'Sleep stage W':0,'Sleep stage 1':1,'Sleep stage 2':2,'Sleep stage 3':3,'Sleep stage 4':4,'Sleep stage R':5}
STAGE_LABELS={0:'W',1:'N1',2:'N2',3:'N3',4:'N3',5:'REM'}
BROKEN={'SC4361','SC4521','SC4132'}
def find_pairs(limit=10):
    sc_psg=sorted(glob.glob(os.path.join(SC_DIR,"*-PSG.edf")))
    sc_hyp=sorted(glob.glob(os.path.join(SC_DIR,"*-Hypnogram.edf")))
    hyp_map={os.path.basename(h).replace('-Hypnogram.edf','')[:6]:h for h in sc_hyp}
    pairs=[]
    for psg in sc_psg:
        if len(pairs)>=limit: break
        base=os.path.basename(psg).replace('-PSG.edf','')
        rid=base[:6]
        if rid in BROKEN: continue
        if rid in hyp_map: pairs.append((rid,psg,hyp_map[rid],'SC'))
    return pairs
def extract_features(psg_path,hyp_path,rec_id):
    raw=mne.io.read_raw_edf(psg_path,include=['EEG Fpz-Cz'],preload=False,verbose=False)
    raw.load_data(verbose=False); sfreq=int(raw.info['sfreq']); n_total=len(raw)
    ann=mne.read_annotations(hyp_path)
    raw.set_annotations(ann,emit_warning=False)
    events,_=mne.events_from_annotations(raw,event_id=STAGE_MAPPING,verbose=False)
    if len(events)==0: return None
    data,_=raw[:,:]; signal=data[0]
    records=[]
    for ii,ev in enumerate(events):
        onset=ev[0]; scode=ev[2]; slab=STAGE_LABELS.get(scode,'Unknown')
        for sub in range(SUB_WINDOWS):
            start=onset+sub*STEP_SIZE; end=start+WINDOW_SIZE
            if end>n_total: continue
            win=signal[start:end]; csec=(start+end)/2.0/sfreq
            var=np.var(win)
            phi1=np.corrcoef(win[:-1],win[1:])[0,1]; phi1=0.0 if np.isnan(phi1) else phi1
            trend=np.polyval(np.polyfit(np.arange(len(win)),win,1),np.arange(len(win)))
            dnb=np.std(win-trend)
            sp=extract_spectral_powers(win,sampling_freq=sfreq)
            d,t,a,s,b=sp; tar=t/(a+1e-10); dtr=d/(t+1e-10)
            records.append(dict(Subject_ID=rec_id,Study_Type='SC',Window=ii*SUB_WINDOWS+sub,
                Time_Sec=csec,Epoch_Index=ii,Sleep_Stage=slab,Stage_Code=scode,
                Variance=var,Autocorrelation=phi1,DNB_Std=dnb,
                Delta=d,Theta=t,Alpha=a,Sigma=s,Beta=b,
                Theta_Alpha_Ratio=tar,Delta_Theta_Ratio=dtr))
    return pd.DataFrame(records)
def sep(group_a,group_b,metric):
    a=group_a[metric].dropna().values; b=group_b[metric].dropna().values
    if len(a)<3 or len(b)<3: return dict(metric=metric,cohens_d=0.0,mannwhitney_p=1.0,mean_a=0.0,mean_b=0.0,n_a=len(a),n_b=len(b))
    n1,n2=len(a),len(b); s1,s2=np.std(a,ddof=1),np.std(b,ddof=1)
    pooled=np.sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2))
    d=(np.mean(a)-np.mean(b))/(pooled+1e-10)
    _,p=stats.mannwhitneyu(a,b,alternative='two-sided')
    return dict(metric=metric,mean_a=float(np.mean(a)),mean_b=float(np.mean(b)),
        cohens_d=float(d),mannwhitney_p=float(p),n_a=int(n1),n_b=int(n2))
def main():
    print("="*65); print("  REM-Wake Spectral Separation Verification  (MNE)"); print("="*65)
    pairs=find_pairs(limit=10)
    print(f"\n[DATA] {len(pairs)} pairs found")
    all_dfs=[]
    for idx,(rid,psg,hy,_) in enumerate(pairs):
        print(f"  [{idx+1}/{len(pairs)}] {rid}...",end=" ",flush=True)
        df=extract_features(psg,hy,rid)
        if df is not None and len(df)>0: all_dfs.append(df); print(f"{len(df)} windows")
        else: print("FAIL")
    if not all_dfs: print("\n[FAIL] No data"); return
    df=pd.concat(all_dfs,ignore_index=True)
    vs=['W','N1','N2','N3','REM']; df=df[df['Sleep_Stage'].isin(vs)]
    print(f"\n[DATA] {len(df):,} windows, {df['Subject_ID'].nunique()} subjects")
    for s in vs:
        c=(df['Sleep_Stage']==s).sum(); print(f"    {s}: {c:>7,} ({c/len(df)*100:5.1f}%)")
    print(f"\n{'='*65}\n  TEST 1: REM vs Wake\n{'='*65}")
    rem=df[df['Sleep_Stage']=='REM']; wake=df[df['Sleep_Stage']=='W']
    metrics=['Theta_Alpha_Ratio','Delta_Theta_Ratio','Delta','Theta','Alpha','Beta']
    scores=[]
    for m in metrics:
        sc=sep(rem,wake,m); scores.append(sc); p=sc['mannwhitney_p']
        sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"  {m:25s}: REM={sc['mean_a']:.4f}, Wake={sc['mean_b']:.4f}, d={sc['cohens_d']:.3f}, p={p:.6f} {sig}")
    print(f"\n{'='*65}\n  TEST 2: Stage Profile\n{'='*65}")
    for ss in vs:
        g=df[df['Sleep_Stage']==ss]
        if len(g)>0: print(f"  {ss:4s}: theta/alpha={g['Theta_Alpha_Ratio'].mean():.3f}+-{g['Theta_Alpha_Ratio'].std():.3f},  n={len(g):,}")
    print(f"\n{'='*65}\n  TEST 3: N1 Intermediate Check\n{'='*65}")
    n1=df[df['Sleep_Stage']=='N1']
    if len(n1)>0 and len(wake)>0 and len(rem)>0:
        nt=n1['Theta_Alpha_Ratio'].mean(); wt=wake['Theta_Alpha_Ratio'].mean(); rt=rem['Theta_Alpha_Ratio'].mean()
        print(f"  theta/alpha: Wake={wt:.3f}, N1={nt:.3f}, REM={rt:.3f}")
        if wt<nt<rt or rt<nt<wt: print("  -> [OK] N1 intermediate")
        else: print("  -> [WARN] N1 not clearly intermediate")
    best=max(scores,key=lambda r:abs(r['cohens_d']) if r['mannwhitney_p']<0.05 else 0)
    print(f"\n{'='*65}\n  CONCLUSION\n{'='*65}")
    print(f"\n  Best separator: {best['metric']}, d={best['cohens_d']:.3f}, p={best['mannwhitney_p']:.6f}")
    if best['mannwhitney_p']<0.05 and abs(best['cohens_d'])>0.5: print("\n  ** SPECTRAL SEPARATION CONFIRMED **")
    elif best['mannwhitney_p']<0.05: print("\n  ** WEAK SEPARATION **")
    else: print("\n  ** NO SIGNIFICANT SEPARATION **")
    csvp=os.path.join(os.path.dirname(os.path.abspath(__file__)),"spectral_separation_verify.csv")
    df.to_csv(csvp,index=False); print(f"\n  Saved: {csvp}")
if __name__=="__main__": main()
