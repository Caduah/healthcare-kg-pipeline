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

def load_claims():
    path = "data/raw/claims/claims.csv"
    engine = get_engine()
    df = pd.read_csv(path, low_memory=False)

    # lowercase all column names to match postgres table
    df.columns = [c.lower() for c in df.columns]
    df["source_system"] = "synthea"

    log.info(f"Claims columns: {list(df.columns)}")
    log.info(f"Loading {len(df)} claims...")

    df.to_sql("claims", engine, schema="raw",
              if_exists="append", index=False,
              method="multi", chunksize=200)

    log.info(f"Loaded {len(df)} claims into raw.claims")
    return len(df)

if __name__ == "__main__":
    count = load_claims()
    log.info(f"Done — claims: {count}")
