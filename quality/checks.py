import os
import logging
import psycopg2
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB", "healthdb"),
        user=os.getenv("POSTGRES_USER", "healthuser"),
        password=os.getenv("POSTGRES_PASSWORD", "healthpass"),
    )


def log_result(table, check_name, status, checked, failed):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO quality.validation_log
        (table_name, check_name, status, records_checked, records_failed, run_at)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (table, check_name, status, checked, failed, datetime.now()))
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"  [{status}] {table}.{check_name} -- {failed}/{checked} failed")


def check_null_ids(cur, schema, table, id_col="id"):
    cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
    total = cur.fetchone()[0]
    cur.execute(f"SELECT COUNT(*) FROM {schema}.{table} WHERE {id_col} IS NULL")
    nulls = cur.fetchone()[0]
    status = "PASS" if nulls == 0 else "FAIL"
    log_result(table, f"null_{id_col}", status, total, nulls)
    return nulls == 0, total, nulls


def check_duplicates(cur, schema, table, id_col="id"):
    cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
    total = cur.fetchone()[0]
    cur.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT {id_col}, COUNT(*) as cnt
            FROM {schema}.{table}
            GROUP BY {id_col}
            HAVING COUNT(*) > 1
        ) dups
    """)
    dup_keys = cur.fetchone()[0]
    status = "PASS" if dup_keys == 0 else "WARN"
    log_result(table, f"duplicate_{id_col}", status, total, dup_keys)
    return dup_keys == 0, total, dup_keys


def check_row_count(cur, schema, table, min_rows=100):
    cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
    total = cur.fetchone()[0]
    status = "PASS" if total >= min_rows else "FAIL"
    log_result(table, "min_row_count", status, total, 0 if total >= min_rows else 1)
    return total >= min_rows, total


def check_null_rate(cur, schema, table, col, max_null_pct=5.0):
    cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
    total = cur.fetchone()[0]
    cur.execute(f"SELECT COUNT(*) FROM {schema}.{table} WHERE {col} IS NULL")
    nulls = cur.fetchone()[0]
    pct = (nulls / total * 100) if total > 0 else 0
    status = "PASS" if pct <= max_null_pct else "FAIL"
    log_result(table, f"null_rate_{col}", status, total, nulls)
    return pct <= max_null_pct, total, nulls


def check_age_range(cur):
    cur.execute("SELECT COUNT(*) FROM processed.patients")
    total = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM processed.patients
        WHERE age_years < 0 OR age_years > 120
    """)
    bad = cur.fetchone()[0]
    status = "PASS" if bad == 0 else "FAIL"
    log_result("patients", "age_range_0_120", status, total, bad)
    return bad == 0, total, bad


def check_gender_values(cur):
    cur.execute("SELECT COUNT(*) FROM processed.patients")
    total = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM processed.patients
        WHERE gender NOT IN ('M', 'F')
        AND gender IS NOT NULL
    """)
    bad = cur.fetchone()[0]
    status = "PASS" if bad == 0 else "WARN"
    log_result("patients", "gender_valid_values", status, total, bad)
    return bad == 0, total, bad


def check_encounter_cost(cur):
    cur.execute("SELECT COUNT(*) FROM processed.encounters")
    total = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM processed.encounters
        WHERE total_claim_cost < 0
    """)
    bad = cur.fetchone()[0]
    status = "PASS" if bad == 0 else "FAIL"
    log_result("encounters", "negative_claim_cost", status, total, bad)
    return bad == 0, total, bad


def check_duration(cur):
    cur.execute("SELECT COUNT(*) FROM processed.encounters")
    total = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM processed.encounters
        WHERE duration_minutes < 0
    """)
    bad = cur.fetchone()[0]
    status = "PASS" if bad == 0 else "FAIL"
    log_result("encounters", "negative_duration", status, total, bad)
    return bad == 0, total, bad


def quarantine_null_patients(cur, conn):
    cur.execute("""
        INSERT INTO quarantine.patients_bad
        (id, first, last, birthdate, gender, state,
         source_system, quarantine_reason, quarantined_at)
        SELECT id, first, last, birthdate, gender, state,
               source_system, 'null_id', NOW()
        FROM processed.patients
        WHERE id IS NULL
    """)
    rows = cur.rowcount
    if rows > 0:
        cur.execute("DELETE FROM processed.patients WHERE id IS NULL")
        log.warning(f"Quarantined {rows} patients with null IDs")
    conn.commit()
    return rows


def quarantine_null_encounters(cur, conn):
    cur.execute("""
        INSERT INTO quarantine.encounters_bad
        (id, patient, provider, start_ts, encounterclass,
         total_claim_cost, source_system, quarantine_reason, quarantined_at)
        SELECT id, patient, provider, start_ts, encounterclass,
               total_claim_cost, source_system, 'null_patient_id', NOW()
        FROM processed.encounters
        WHERE patient IS NULL
    """)
    rows = cur.rowcount
    if rows > 0:
        cur.execute("DELETE FROM processed.encounters WHERE patient IS NULL")
        log.warning(f"Quarantined {rows} encounters with null patient IDs")
    conn.commit()
    return rows


def run_quality_suite(**context):
    log.info("=" * 55)
    log.info("DAY 3 -- Running data quality checks")
    log.info("=" * 55)

    conn = get_conn()
    cur = conn.cursor()

    failures = []
    warnings = []

    log.info("-- PATIENTS --")
    ok, total, bad = check_null_ids(cur, "processed", "patients")
    if not ok:
        failures.append(f"patients: {bad} null IDs")
        quarantine_null_patients(cur, conn)

    ok, total, dups = check_duplicates(cur, "processed", "patients")
    if not ok:
        warnings.append(f"patients: {dups} duplicate IDs")

    ok, total = check_row_count(cur, "processed", "patients", min_rows=100)
    if not ok:
        failures.append("patients: below minimum row count")

    check_null_rate(cur, "processed", "patients", "birthdate", max_null_pct=0)
    check_age_range(cur)
    check_gender_values(cur)

    log.info("-- PROVIDERS --")
    ok, total, bad = check_null_ids(cur, "processed", "providers")
    if not ok:
        failures.append(f"providers: {bad} null IDs")

    check_duplicates(cur, "processed", "providers")
    check_row_count(cur, "processed", "providers", min_rows=50)
    check_null_rate(cur, "processed", "providers", "name", max_null_pct=0)

    log.info("-- ENCOUNTERS --")
    ok, total, bad = check_null_ids(cur, "processed", "encounters")
    if not ok:
        failures.append(f"encounters: {bad} null IDs")

    ok, total, bad = check_null_rate(
        cur, "processed", "encounters", "patient", max_null_pct=0
    )
    if not ok:
        failures.append(f"encounters: {bad} null patient IDs")
        quarantine_null_encounters(cur, conn)

    check_row_count(cur, "processed", "encounters", min_rows=1000)
    check_encounter_cost(cur)
    check_duration(cur)

    log.info("-- CLAIMS --")
    check_null_ids(cur, "processed", "claims", id_col="id")
    check_row_count(cur, "processed", "claims", min_rows=1000)
    check_null_rate(cur, "processed", "claims", "patientid", max_null_pct=0)

    cur.close()
    conn.close()

    log.info("=" * 55)
    log.info(f"Quality check complete")
    log.info(f"  Failures : {len(failures)}")
    log.info(f"  Warnings : {len(warnings)}")
    log.info("=" * 55)

    if failures:
        raise ValueError(
            f"Data quality FAILED -- {len(failures)} critical issues:\n"
            + "\n".join(failures)
        )

    return {"failures": len(failures), "warnings": len(warnings)}


if __name__ == "__main__":
    run_quality_suite()
