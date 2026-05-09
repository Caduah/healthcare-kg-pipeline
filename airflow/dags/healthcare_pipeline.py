from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import logging
import os
import sys

log = logging.getLogger(__name__)

default_args = {
    "owner": "caleb",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}


def check_raw_data(**context):
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "healthdb"),
        user=os.getenv("POSTGRES_USER", "healthuser"),
        password=os.getenv("POSTGRES_PASSWORD", "healthpass"),
    )
    cur = conn.cursor()
    counts = {}
    for table in ["patients", "providers", "encounters", "claims"]:
        cur.execute(f"SELECT COUNT(*) FROM raw.{table}")
        counts[table] = cur.fetchone()[0]
    cur.close()
    conn.close()
    log.info(f"Raw counts: {counts}")
    for table, count in counts.items():
        if count == 0:
            raise ValueError(f"raw.{table} is empty — aborting")
    context["ti"].xcom_push(key="raw_counts", value=counts)
    return counts


def run_quality_checks(**context):
    sys.path.insert(0, "/opt/airflow")
    os.environ["POSTGRES_HOST"] = "postgres"
    os.environ["POSTGRES_PORT"] = "5432"
    os.environ["POSTGRES_USER"] = "healthuser"
    os.environ["POSTGRES_PASSWORD"] = "healthpass"
    os.environ["POSTGRES_DB"] = "healthdb"
    from quality.checks import run_quality_suite
    return run_quality_suite(**context)


def run_patient_er(**context):
    sys.path.insert(0, "/opt/airflow")
    os.environ["POSTGRES_HOST"] = "postgres"
    os.environ["POSTGRES_PORT"] = "5432"
    os.environ["POSTGRES_USER"] = "healthuser"
    os.environ["POSTGRES_PASSWORD"] = "healthpass"
    os.environ["POSTGRES_DB"] = "healthdb"
    from entity_resolution.patient_matcher import run_patient_matching
    return run_patient_matching(**context)


def run_provider_er(**context):
    sys.path.insert(0, "/opt/airflow")
    os.environ["POSTGRES_HOST"] = "postgres"
    os.environ["POSTGRES_PORT"] = "5432"
    os.environ["POSTGRES_USER"] = "healthuser"
    os.environ["POSTGRES_PASSWORD"] = "healthpass"
    os.environ["POSTGRES_DB"] = "healthdb"
    from entity_resolution.provider_matcher import run_provider_matching
    return run_provider_matching(**context)


def verify_processed(**context):
    import psycopg2
    raw_counts = context["ti"].xcom_pull(
        task_ids="check_raw_data", key="raw_counts"
    )
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "healthdb"),
        user=os.getenv("POSTGRES_USER", "healthuser"),
        password=os.getenv("POSTGRES_PASSWORD", "healthpass"),
    )
    cur = conn.cursor()
    processed = {}
    for table in ["patients", "providers", "encounters", "claims"]:
        cur.execute(f"SELECT COUNT(*) FROM processed.{table}")
        processed[table] = cur.fetchone()[0]
    cur.close()
    conn.close()
    for table, count in processed.items():
        if count == 0:
            raise ValueError(f"processed.{table} is empty after ETL")
        if raw_counts:
            raw = raw_counts.get(table, 0)
            pct = (count / raw * 100) if raw > 0 else 0
            log.info(f"  {table}: {count:,} / {raw:,} ({pct:.1f}% retained)")
    return processed


with DAG(
    dag_id="healthcare_kg_pipeline",
    default_args=default_args,
    description="Healthcare KG pipeline — Day 4: Entity resolution",
    schedule_interval="@daily",
    catchup=False,
    tags=["healthcare", "spark", "etl", "quality", "entity-resolution"],
) as dag:

    t1_check_raw = PythonOperator(
        task_id="check_raw_data",
        python_callable=check_raw_data,
    )

    t2_spark_etl = BashOperator(
        task_id="spark_etl",
        bash_command=(
            "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64 && "
            "export PATH=$JAVA_HOME/bin:$PATH && "
            "cd /opt/airflow && "
            "pip install pyspark==3.5.1 -q && "
            "python spark/jobs/healthcare_etl.py"
        ),
        env={
            "POSTGRES_HOST": "postgres",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "healthuser",
            "POSTGRES_PASSWORD": "healthpass",
            "POSTGRES_DB": "healthdb",
            "PYTHONPATH": "/opt/airflow",
            "JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-arm64",
        },
    )

    t3_quality = PythonOperator(
        task_id="data_quality_checks",
        python_callable=run_quality_checks,
    )

    t4_patient_er = PythonOperator(
        task_id="patient_entity_resolution",
        python_callable=run_patient_er,
    )

    t5_provider_er = PythonOperator(
        task_id="provider_entity_resolution",
        python_callable=run_provider_er,
    )

    t6_verify = PythonOperator(
        task_id="verify_processed",
        python_callable=verify_processed,
    )

    # patient and provider ER run in parallel after quality checks
    t1_check_raw >> t2_spark_etl >> t3_quality
    t3_quality >> [t4_patient_er, t5_provider_er]
    [t4_patient_er, t5_provider_er] >> t6_verify