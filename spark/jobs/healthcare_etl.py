import sys
import os
import logging
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from spark.jobs.utils import get_spark_session, read_table, write_table

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def clean_patients(spark):
    log.info("Cleaning patients...")
    df = read_table(spark, "raw", "patients")
    log.info(f"Raw patients: {df.count()}")

    df = df \
        .filter(F.col("id").isNotNull()) \
        .filter(F.col("first").isNotNull()) \
        .filter(F.col("last").isNotNull()) \
        .withColumn("birthdate", F.to_date(F.col("birthdate"), "yyyy-MM-dd")) \
        .withColumn("deathdate", F.to_date(F.col("deathdate"), "yyyy-MM-dd")) \
        .withColumn("first_norm", F.lower(F.trim(F.col("first")))) \
        .withColumn("last_norm",  F.lower(F.trim(F.col("last")))) \
        .withColumn("full_name",
            F.concat_ws(" ",
                F.initcap(F.trim(F.col("first"))),
                F.initcap(F.trim(F.col("last"))))) \
        .withColumn("full_name_norm",
            F.lower(F.concat_ws(" ",
                F.trim(F.col("first")),
                F.trim(F.col("last"))))) \
        .withColumn("age_years",
            F.floor(F.datediff(
                F.current_date(), F.col("birthdate")) / 365
            ).cast(LongType())) \
        .withColumn("zip_clean",
            F.regexp_replace(F.col("zip"), "[^0-9]", "")) \
        .withColumn("state_upper", F.upper(F.trim(F.col("state")))) \
        .withColumn("is_deceased", F.col("deathdate").isNotNull()) \
        .withColumn("processed_at", F.current_timestamp()) \
        .withColumn("source_system", F.lit("synthea"))

    cols = [
        "id","birthdate","deathdate","first","last",
        "full_name","full_name_norm","first_norm","last_norm",
        "gender","race","ethnicity","address","city","state",
        "state_upper","zip","zip_clean","lat","lon",
        "age_years","is_deceased","source_system","processed_at"
    ]
    df = df.select([c for c in cols if c in df.columns])
    count = df.count()
    write_table(df, "patients")
    log.info(f"Processed patients: {count}")
    return count


def clean_providers(spark):
    log.info("Cleaning providers...")
    df = read_table(spark, "raw", "providers")
    log.info(f"Raw providers: {df.count()}")

    df = df \
        .filter(F.col("id").isNotNull()) \
        .filter(F.col("name").isNotNull()) \
        .withColumn("name_norm",       F.lower(F.trim(F.col("name")))) \
        .withColumn("name_clean",      F.initcap(F.trim(F.col("name")))) \
        .withColumn("speciality_norm", F.lower(F.trim(F.col("speciality")))) \
        .withColumn("speciality_clean",F.initcap(F.trim(F.col("speciality")))) \
        .withColumn("city_norm",       F.lower(F.trim(F.col("city")))) \
        .withColumn("state_upper",     F.upper(F.trim(F.col("state")))) \
        .withColumn("zip_clean",
            F.regexp_replace(F.col("zip"), "[^0-9]", "")) \
        .withColumn("has_npi",
            F.col("npi").isNotNull() & (F.length(F.col("npi")) > 0)) \
        .withColumn("encounters",  F.col("encounters").cast(LongType())) \
        .withColumn("procedures",  F.col("procedures").cast(LongType())) \
        .withColumn("processed_at", F.current_timestamp()) \
        .withColumn("source_system", F.lit("synthea"))

    cols = [
        "id","organization","name","name_norm","name_clean",
        "gender","speciality","speciality_norm","speciality_clean",
        "address","city","city_norm","state","state_upper",
        "zip","zip_clean","lat","lon","npi","has_npi",
        "encounters","procedures","source_system","processed_at"
    ]
    df = df.select([c for c in cols if c in df.columns])
    count = df.count()
    write_table(df, "providers")
    log.info(f"Processed providers: {count}")
    return count


def clean_encounters(spark):
    log.info("Cleaning encounters...")
    df = read_table(spark, "raw", "encounters")
    log.info(f"Raw encounters: {df.count()}")

    df = df \
        .filter(F.col("id").isNotNull()) \
        .filter(F.col("patient").isNotNull()) \
        .filter(F.col("provider").isNotNull()) \
        .withColumn("start_ts",       F.to_timestamp(F.col("start"))) \
        .withColumn("stop_ts",        F.to_timestamp(F.col("stop"))) \
        .withColumn("encounter_date", F.to_date(F.col("start"))) \
        .withColumn("encounter_year", F.year(F.to_timestamp(F.col("start")))) \
        .withColumn("encounter_month",F.month(F.to_timestamp(F.col("start")))) \
        .withColumn("duration_minutes",
            F.round((
                F.unix_timestamp(F.col("stop_ts")) -
                F.unix_timestamp(F.col("start_ts"))
            ) / 60, 2)) \
        .withColumn("encounterclass_norm",
            F.lower(F.trim(F.col("encounterclass")))) \
        .withColumn("total_claim_cost",
            F.col("total_claim_cost").cast(DoubleType())) \
        .withColumn("base_encounter_cost",
            F.col("base_encounter_cost").cast(DoubleType())) \
        .withColumn("payer_coverage",
            F.col("payer_coverage").cast(DoubleType())) \
        .withColumn("patient_cost",
            F.round(F.col("total_claim_cost") - F.col("payer_coverage"), 2)) \
        .withColumn("has_reason", F.col("reasoncode").isNotNull()) \
        .withColumn("description_norm",
            F.lower(F.trim(F.col("description")))) \
        .withColumn("processed_at", F.current_timestamp()) \
        .withColumn("source_system", F.lit("synthea"))

    cols = [
        "id","patient","provider","organization","payer",
        "start_ts","stop_ts","encounter_date","encounter_year","encounter_month",
        "duration_minutes","encounterclass","encounterclass_norm",
        "code","description","description_norm",
        "base_encounter_cost","total_claim_cost","payer_coverage","patient_cost",
        "reasoncode","reasondescription","has_reason",
        "source_system","processed_at"
    ]
    df = df.select([c for c in cols if c in df.columns])
    count = df.count()
    write_table(df, "encounters")
    log.info(f"Processed encounters: {count}")
    return count


def clean_claims(spark):
    log.info("Cleaning claims...")
    df = read_table(spark, "raw", "claims")
    log.info(f"Raw claims: {df.count()}")

    df = df \
        .filter(F.col("id").isNotNull()) \
        .filter(F.col("patientid").isNotNull()) \
        .withColumn("outstanding1",
            F.col("outstanding1").cast(DoubleType())) \
        .withColumn("outstanding2",
            F.col("outstanding2").cast(DoubleType())) \
        .withColumn("outstandingp",
            F.col("outstandingp").cast(DoubleType())) \
        .withColumn("total_outstanding",
            F.round(
                F.coalesce(F.col("outstanding1"), F.lit(0.0)) +
                F.coalesce(F.col("outstanding2"), F.lit(0.0)) +
                F.coalesce(F.col("outstandingp"), F.lit(0.0)), 2)) \
        .withColumn("has_primary_diagnosis",
            F.col("diagnosis1").isNotNull()) \
        .withColumn("diagnosis_count",
            F.size(F.array_remove(
                F.array(
                    F.col("diagnosis1"), F.col("diagnosis2"),
                    F.col("diagnosis3"), F.col("diagnosis4")
                ), None))) \
        .withColumn("status_clean",
            F.coalesce(F.col("status1"), F.col("status2"), F.lit("unknown"))) \
        .withColumn("processed_at", F.current_timestamp()) \
        .withColumn("source_system", F.lit("synthea"))

    cols = [
        "id","patientid","providerid","servicedate","currentillnessdate",
        "diagnosis1","diagnosis2","diagnosis3","diagnosis4",
        "outstanding1","outstanding2","outstandingp","total_outstanding",
        "has_primary_diagnosis","diagnosis_count","status_clean",
        "status1","status2","source_system","processed_at"
    ]
    df = df.select([c for c in cols if c in df.columns])
    count = df.count()
    write_table(df, "claims")
    log.info(f"Processed claims: {count}")
    return count


def run_etl():
    spark = get_spark_session("healthcare-etl")
    log.info("Spark session started")
    results = {}
    try:
        results["patients"]   = clean_patients(spark)
        results["providers"]  = clean_providers(spark)
        results["encounters"] = clean_encounters(spark)
        results["claims"]     = clean_claims(spark)
        log.info("=" * 50)
        log.info("ETL COMPLETE")
        for t, c in results.items():
            log.info(f"  {t}: {c:,} records")
        log.info("=" * 50)
    except Exception as e:
        log.error(f"ETL failed: {e}")
        raise
    finally:
        spark.stop()
    return results


if __name__ == "__main__":
    run_etl()
