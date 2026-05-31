#!/usr/bin/env python3
"""
Query BUT-PDB (Brno University of Technology ECG) features from SQLite archive.
=============================================================================
Brno University of Technology ECG Signal Database with P-wave annotations.

Usage:
  python response_verification/queries/query_but_pdb.py              # Full report
  python response_verification/queries/query_but_pdb.py --afib-only  # AFIB focused
  python response_verification/queries/query_but_pdb.py --record 07  # Single record
"""
import sqlite3, sys, os

DB_PATH = r"E:\MEM\paper\real\cfect_archive.db"


def get_conn():
    if not os.path.exists(DB_PATH):
        print(f"[FAIL] Database not found: {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def report_all(conn, detail=False):
    cur = conn.cursor()

    # ===== 1. Overview =====
    cur.execute("SELECT COUNT(*) FROM but_pdb_features")
    total = cur.fetchone()[0]
    print(f"\n{'='*65}")
    print(f"  BUT-PDB (Brno University) ECG Database — {total} records")
    print(f"{'='*65}")

    # ===== 2. Diagnosis distribution with CFECT coordinates =====
    cur.execute("""
        SELECT Diagnosis,
               COUNT(*) as n,
               ROUND(AVG(Variance_Z),4) as avgVarZ,
               ROUND(AVG(Phi1_Z),4) as avgPhiZ,
               ROUND(AVG(Mean_RR_Sec),3) as avgRR,
               ROUND(AVG(CV_RR),4) as avgCV,
               ROUND(AVG(P_Presence_Ratio),3) as avgPwave,
               ROUND(AVG(Mean_PR_ms),1) as avgPR
        FROM but_pdb_features
        GROUP BY Diagnosis
        ORDER BY n DESC
    """)
    rows = cur.fetchall()
    print(f"\n  {'Diagnosis':50s} {'n':>3s} {'VarZ':>7s} {'PhiZ':>7s} {'RR(s)':>6s}")
    print(f"  {'-'*50} {'-'*3} {'-'*7} {'-'*7} {'-'*6}")
    for r in rows:
        print(f"  {r['Diagnosis']:50s} {r['n']:>3d} {r['avgVarZ']:>7.4f} {r['avgPhiZ']:>7.4f} {r['avgRR']:>6.3f}")

    # ===== 3. Drug distribution =====
    cur.execute("""
        SELECT Drug, COUNT(*) as n,
               ROUND(AVG(Variance_Z),4) as avgVarZ,
               ROUND(AVG(Phi1_Z),4) as avgPhiZ
        FROM but_pdb_features
        WHERE Drug != 'None'
        GROUP BY Drug
        ORDER BY n DESC
    """)
    drugs = cur.fetchall()
    if drugs:
        print(f"\n  --- Drugs Administered ---")
        for d in drugs:
            print(f"  {d['Drug']:20s} n={d['n']:>2d}  VarZ={d['avgVarZ']:>+.4f}  PhiZ={d['avgPhiZ']:>+.4f}")

    # ===== 4. CFECT phase-space summary =====
    print(f"\n  --- CFECT Phase-Space Summary ---")
    cur.execute("SELECT MIN(Variance_Z), MAX(Variance_Z), MIN(Phi1_Z), MAX(Phi1_Z) FROM but_pdb_features")
    r = cur.fetchone()
    print(f"  Variance_Z: [{r[0]:+.4f}, {r[1]:+.4f}]")
    print(f"  Phi1_Z:     [{r[2]:+.4f}, {r[3]:+.4f}]")

    # ===== 5. AFIB cluster =====
    print(f"\n  --- Atrial Fibrillation Records ---")
    cur.execute("""
        SELECT Record_ID, Variance_Z, Phi1_Z, CV_RR, P_Presence_Ratio
        FROM but_pdb_features
        WHERE Diagnosis LIKE '%AFIB%' OR Diagnosis LIKE '%Atrial Fibrillation%'
        ORDER BY Record_ID
    """)
    afibs = cur.fetchall()
    for a in afibs:
        print(f"  Record {a['Record_ID']:>2s}: VarZ={a['Variance_Z']:+.4f}  "
              f"PhiZ={a['Phi1_Z']:+.4f}  CV_RR={a['CV_RR']:.4f}  "
              f"P-presence={a['P_Presence_Ratio']:.2f}")

    # ===== 6. Detailed records if requested =====
    if detail:
        print(f"\n  --- All Records (detailed) ---")
        cur.execute("SELECT * FROM but_pdb_features ORDER BY Record_ID")
        for row in cur.fetchall():
            print(f"\n  [{row['Record_ID']}] {row['Diagnosis']}")
            print(f"       Drug={row['Drug']:15s} RR={row['Mean_RR_Sec']:.3f}s "
                  f"SDNN={row['SDNN']:.3f}s CV={row['CV_RR']:.4f}")
            print(f"       QRS={row['N_QRS']:>4d} Pwaves={row['N_Pwaves']:>4d} "
                  f"P-ratio={row['P_Presence_Ratio']:.2f}")
            print(f"       PR={row['Mean_PR_ms']:.1f}ms PW_dur={row['Mean_PW_Duration_ms']:.1f}ms")
            print(f"       Variance_Z={row['Variance_Z']:+.4f}  Phi1_Z={row['Phi1_Z']:+.4f}")

    conn.close()


def query_record(conn, record_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM but_pdb_features WHERE Record_ID = ?", (record_id,))
    row = cur.fetchone()
    if not row:
        print(f"[FAIL] Record {record_id} not found")
        return
    print(f"\n{'='*60}")
    print(f"  BUT-PDB Record {record_id}")
    print(f"{'='*60}")
    print(f"  Diagnosis:      {row['Diagnosis']}")
    print(f"  Drug:           {row['Drug']}")
    print(f"  Mean RR:        {row['Mean_RR_Sec']:.3f} s")
    print(f"  SDNN:           {row['SDNN']:.3f} s")
    print(f"  CV_RR:          {row['CV_RR']:.4f}")
    print(f"  N QRS beats:    {row['N_QRS']}")
    print(f"  N P-waves:      {row['N_Pwaves']}")
    print(f"  P presence:     {row['P_Presence_Ratio']:.2f}")
    print(f"  Mean PR:        {row['Mean_PR_ms']:.1f} ms")
    print(f"  P-wave duration:{row['Mean_PW_Duration_ms']:.1f} ms")
    print(f"  Variance_Z:     {row['Variance_Z']:+.4f}")
    print(f"  Phi1_Z:         {row['Phi1_Z']:+.4f}")
    conn.close()


if __name__ == "__main__":
    conn = get_conn()

    if '--record' in sys.argv:
        idx = sys.argv.index('--record')
        rid = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if rid:
            query_record(conn, rid)
        else:
            print("Usage: --record <ID>  (e.g. --record 07)")
    elif '--afib-only' in sys.argv:
        cur = conn.cursor()
        cur.execute("""
            SELECT Record_ID, Diagnosis, Variance_Z, Phi1_Z, CV_RR, P_Presence_Ratio
            FROM but_pdb_features
            WHERE Diagnosis LIKE '%AFIB%' OR Diagnosis LIKE '%Atrial Fibrillation%'
            ORDER BY Record_ID
        """)
        rows = cur.fetchall()
        print(f"\n  --- AFIB Records (n={len(rows)}) ---")
        for r in rows:
            print(f"  [{r['Record_ID']}] {r['Diagnosis']:45s} VarZ={r['Variance_Z']:+.4f} PhiZ={r['Phi1_Z']:+.4f}")
        conn.close()
    else:
        detail = '--detail' in sys.argv
        report_all(conn, detail=detail)
