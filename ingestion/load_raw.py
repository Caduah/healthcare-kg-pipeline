import os
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_engine():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    db   = os.getenv("POSTGRES_DB", "healthdb")
    user = os.getenv("POSTGRES_USER", "healthuser")
    pw   = os.getenv("POSTGRES_PASSWORD", "healthpass")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


def load_patients():
    engine = get_engine()
    df = pd.read_csv("data/raw/patients/patients.csv", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["source_system"] = "synthea"
    keep = [
        "id","birthdate","deathdate","ssn","drivers","passport",
        "prefix","first","last","suffix","maiden","marital",
        "race","ethnicity","gender","birthplace","address",
        "city","state","county","fips","zip","lat","lon",
        "healthcare_expenses","healthcare_coverage","income","source_system"
    ]
    df = df[[c for c in keep if c in df.columns]]
    df.to_sql("patients", engine, schema="raw",
              if_exists="append", index=False, method="multi", chunksize=500)
    log.info(f"Loaded {len(df)} patients")
    return len(df)


def load_providers():
    engine = get_engine()
    df = pd.read_csv("data/raw/providers/providers.csv", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["source_system"] = "synthea"
    keep = [
        "id","organization","name","gender","speciality",
        "address","city","state","zip","lat","lon",
        "encounters","procedures","source_system"
    ]
    df = df[[c for c in keep if c in df.columns]]
    df.to_sql("providers", engine, schema="raw",
              if_exists="append", index=False, method="multi", chunksize=500)
    log.info(f"Loaded {len(df)} providers")
    return len(df)


def load_encounters():
    engine = get_engine()
    df = pd.read_csv("data/raw/encounters/encounters.csv", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["source_system"] = "synthea"
    keep = [
        "id","start","stop","patient","organization","provider",
        "payer","encounterclass","code","description",
        "base_encounter_cost","total_claim_cost","payer_coverage",
        "reasoncode","reasondescription","source_system"
    ]
    df = df[[c for c in keep if c in df.columns]]
    df.to_sql("encounters", engine, schema="raw",
              if_exists="append", index=False, method="multi", chunksize=500)
    log.info(f"Loaded {len(df)} encounters")
    return len(df)


def load_claims():
    path = "data/raw/claims/claims.csv"
    if not os.path.exists(path):
        log.warning("No claims file — skipping")
        return 0
    engine = get_engine()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["source_system"] = "synthea"
    df.to_sql("claims", engine, schema="raw",
              if_exists="append", index=False, method="multi", chunksize=500)
    log.info(f"Loaded {len(df)} claims")
    return len(df)


if __name__ == "__main__":
    log.info("Starting ingestion...")
    p  = load_patients()
    pr = load_providers()
    e  = load_encounters()
    c  = load_claims()
    log.info(f"Done — patients:{p} providers:{pr} encounters:{e} claims:{c}")
