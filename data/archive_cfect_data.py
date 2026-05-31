#!/usr/bin/env python3
"""
CFECT Data Archiving Pipeline
=============================
Reads all CSV feature files from E:\MEM\paper\real and sample data from
CFECT-Engine-XZ, archives them into a single SQLite database with
proper schema, indexes, and data lineage tracking.

Usage:  python data/archive_cfect_data.py
Output: E:\MEM\paper\real\cfect_archive.db
"""

import sqlite3
import csv
import os
import time
import json
from datetime import datetime

# ─── Configuration ──────────────────────────────────────────
DB_PATH = r"E:\MEM\paper\real\cfect_archive.db"
REAL_DIR = r"E:\MEM\paper\real"
DATA_DIR = r"E:\MEM\paper\real\data"
OUTPUT2_DIR = r"E:\MEM\paper\real\output2"
SAMPLE_DIR = r"C:\Users\Jonhs\Documents\GitHub\CFECT-Engine-XZ\data"

# ─── Data source definitions ────────────────────────────────
# (table_name, file_path, description)
SOURCES = [
    ("chb_mit_records", os.path.join(DATA_DIR, "chb_mit_csd_master.csv"),
     "CHB-MIT CSD features (59,992 windows across 24 patients)"),
    ("chb_mit_labels", os.path.join(REAL_DIR, "chb_mit_labeled.csv"),
     "CHB-MIT with clinical labels (Is_FollowUp, Is_VNS_On, Drug_Withdrawal, Clinical_State)"),
    ("sddb_records", os.path.join(DATA_DIR, "sddb_terminal_master.csv"),
     "SDDB terminal CSD features (2,073 windows from 23 SCD patients)"),
    ("sddb_labels", os.path.join(REAL_DIR, "sddb_labeled.csv"),
     "SDDB with clinical labels (Gender, Age, Phenotype, medication flags)"),
    ("sddb_afib", os.path.join(OUTPUT2_DIR, "sddb_afib_extracted.csv"),
     "SDDB AFib-extracted subset (1,073 windows)"),
    ("sleep_edf_features", os.path.join(OUTPUT2_DIR, "main", "sleep_csd_features.csv"),
     "Sleep-EDF CSD features (112,633 windows, 2 nights, drug/control)"),
    ("but_pdb_features", os.path.join(OUTPUT2_DIR, "but_pdb", "but_pdb_features.csv"),
     "Brno University ECG database features (50 records with diagnosis/drug)"),
    ("sample_ecg", os.path.join(SAMPLE_DIR, "sample_ecg_data.csv"),
     "Sample ECG for development testing (101 rows)"),
    ("sample_eeg", os.path.join(SAMPLE_DIR, "sample_eeg_data.csv"),
     "Sample EEG for development testing (101 rows)"),
]


def detect_types(headers: list[str], rows: list[list[str]]) -> dict:
    """Detect column types from sample rows."""
    types = {}
    for i, h in enumerate(headers):
        col_vals = [r[i] for r in rows[:100] if i < len(r) and r[i].strip()]
        is_float = False
        is_int = False
        for v in col_vals:
            try:
                float(v)
                is_float = True
                if '.' in v or 'e' in v.lower():
                    is_int = False
                else:
                    is_int = True
            except (ValueError, IndexError):
                is_float = False
                is_int = False
                break
        if is_float and not is_int:
            types[h] = "REAL"
        elif is_int:
            types[h] = "INTEGER"
        else:
            types[h] = "TEXT"
        # Override for known ID/string columns
        low_h = h.lower()
        if any(kw in low_h for kw in ['id', 'name', 'record', 'patient', 'subject',
                                       'diagnosis', 'drug', 'phenotype', 'state',
                                       'gender', 'condition', 'stage']):
            types[h] = "TEXT"
        if low_h.endswith('_z'):
            types[h] = "REAL"
    return types


def create_table_from_csv(cursor: sqlite3.Cursor, table_name: str,
                           csv_path: str, primary_key: str = None):
    """Create a table dynamically from CSV headers and types."""
    if not os.path.exists(csv_path):
        print(f"  ⚠ File not found: {csv_path}")
        return False

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        sample_rows = [row for _, row in zip(range(100), reader)]

    col_types = detect_types(headers, sample_rows)
    col_defs = [f'"{h}" {col_types[h]}' for h in headers]
    if primary_key and primary_key in headers:
        col_defs.append(f'PRIMARY KEY ("{primary_key}")')

    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(col_defs)})'
    cursor.execute(create_sql)
    print(f"  ✓ Table '{table_name}' created ({len(headers)} columns)")
    return True


def import_csv_to_table(cursor: sqlite3.Cursor, conn: sqlite3.Connection,
                         table_name: str, csv_path: str) -> int:
    """Import CSV data into table using bulk insert."""
    if not os.path.exists(csv_path):
        return 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        rows = list(reader)

    if not rows:
        return 0

    placeholders = ", ".join(["?" for _ in headers])
    cols = ", ".join([f'"{h}"' for h in headers])
    sql = f'INSERT OR IGNORE INTO "{table_name}" ({cols}) VALUES ({placeholders})'

    batch_size = 1000
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        cursor.executemany(sql, batch)
        total += len(batch)

    conn.commit()
    return total


def create_indexes(cursor: sqlite3.Cursor):
    """Create indexes for common query patterns."""
    indexes = [
        ("idx_chbmit_patient", "chb_mit_records", "Patient_ID"),
        ("idx_chbmit_record", "chb_mit_records", "Record_ID"),
        ("idx_chbmit_condition", "chb_mit_records", "Condition"),
        ("idx_chbmit_seizure", "chb_mit_records", "Seizure_Index"),
        ("idx_chbmit_labels_patient", "chb_mit_labels", "Patient_ID"),
        ("idx_chbmit_labels_record", "chb_mit_labels", "Record_ID"),
        ("idx_sddb_record", "sddb_records", "Record"),
        ("idx_sddb_labels_record", "sddb_labels", "Record"),
        ("idx_sddb_afib_record", "sddb_afib", "Record"),
        ("idx_sleep_edf_subject", "sleep_edf_features", "Subject_ID"),
        ("idx_sleep_edf_stage", "sleep_edf_features", "Sleep_Stage"),
        ("idx_sleep_edf_window", "sleep_edf_features", "Window"),
        ("idx_but_pdb_record", "but_pdb_features", "Record_ID"),
    ]
    for name, table, col in indexes:
        try:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" ("{col}")')
        except sqlite3.OperationalError as e:
            print(f"  ⚠ Index {name}: {e}")


def create_manifest(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    """Create and populate the dataset manifest table."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_manifest (
            table_name TEXT PRIMARY KEY,
            source_file TEXT,
            row_count INTEGER,
            columns INTEGER,
            file_size_bytes INTEGER,
            imported_at TEXT,
            description TEXT
        )
    """)

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name!='dataset_manifest'")
    tables = [r[0] for r in cursor.fetchall()]

    for table_name, csv_path, desc in SOURCES:
        table_key = table_name
        if table_key not in tables:
            continue
        cursor.execute(f'SELECT COUNT(*) FROM "{table_key}"')
        count = cursor.fetchone()[0]
        fsize = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0

        # Get column count
        cursor.execute(f'PRAGMA table_info("{table_key}")')
        ncols = len(cursor.fetchall())

        cursor.execute("""
            INSERT OR REPLACE INTO dataset_manifest
            (table_name, source_file, row_count, columns, file_size_bytes, imported_at, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (table_key, csv_path, count, ncols, fsize,
              datetime.now().isoformat(), desc))

    conn.commit()


def verify_import(cursor: sqlite3.Cursor):
    """Verify row counts match expectations."""
    print("\n" + "=" * 70)
    print("VERIFICATION REPORT")
    print("=" * 70)

    expected = {
        "chb_mit_records": 59992,
        "chb_mit_labels": 59992,
        "sddb_records": 2073,
        "sddb_labels": 1000,
        "sddb_afib": 1073,
        "sleep_edf_features": 112633,
        "but_pdb_features": 50,
        "sample_ecg": 101,
        "sample_eeg": 101,
    }

    all_ok = True
    for table, exp in expected.items():
        try:
            cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            actual = cursor.fetchone()[0]
            status = "✓" if actual == exp else "✗"
            if actual != exp:
                all_ok = False
            print(f"  {status} {table:25s} expected={exp:>6d}  actual={actual:>6d}")
        except sqlite3.OperationalError:
            print(f"  ? {table:25s} TABLE NOT FOUND")

    # Show manifest summary
    print("\n" + "-" * 70)
    print("DATASET MANIFEST")
    print("-" * 70)
    cursor.execute("SELECT table_name, row_count, imported_at FROM dataset_manifest ORDER BY row_count DESC")
    for row in cursor.fetchall():
        print(f"  {row[0]:25s} {row[1]:>8d} rows  [{row[2][:19]}]")

    return all_ok


def main():
    t0 = time.time()
    print("=" * 70)
    print("CFECT DATA ARCHIVING PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Remove existing DB for clean import
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"\n🗑 Removed existing DB: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Phase 1: Create tables
    print("\n── Phase 1: Creating Tables ────────────────────────")
    for table_name, csv_path, _ in SOURCES:
        create_table_from_csv(cursor, table_name, csv_path)

    # Phase 2: Import data
    print("\n── Phase 2: Importing Data ─────────────────────────")
    import_counts = {}
    for table_name, csv_path, _ in SOURCES:
        n = import_csv_to_table(cursor, conn, table_name, csv_path)
        import_counts[table_name] = n
        print(f"  → {table_name:25s} imported {n:>6d} rows")

    # Phase 3: Create indexes
    print("\n── Phase 3: Creating Indexes ───────────────────────")
    create_indexes(cursor)
    conn.commit()
    print("  ✓ All indexes created")

    # Phase 4: Build manifest
    print("\n── Phase 4: Building Dataset Manifest ──────────────")
    create_manifest(cursor, conn)

    # Phase 5: Verify
    print("\n── Phase 5: Verification ───────────────────────────")
    ok = verify_import(cursor)

    # Summary
    elapsed = time.time() - t0
    total_rows = sum(import_counts.values())
    db_size = os.path.getsize(DB_PATH)

    print("\n" + "=" * 70)
    print("ARCHIVE COMPLETE")
    print("=" * 70)
    print(f"  Database:  {DB_PATH}")
    print(f"  Size:      {db_size / 1024:.1f} KB ({db_size / 1024 / 1024:.2f} MB)")
    print(f"  Tables:    {len(SOURCES)}")
    print(f"  Total rows:{total_rows:>8d}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Status:    {'✓ ALL OK' if ok else '✗ MISMATCH DETECTED'}")

    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
