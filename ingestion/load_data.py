import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://healthuser:healthpass@localhost:5433/healthdb"
)

CSV_PATH = "output/csv/"

tables = [
    ("patients",     ["id","birthdate","deathdate","ssn","first","last",
                      "gender","race","ethnicity","city","state","zip"]),
    ("encounters",   ["id","start","stop","patient","organization","provider",
                      "encounterclass","code","description","base_encounter_cost",
                      "total_claim_cost","payer","reasoncode","reasondescription"]),
    ("conditions",   ["start","stop","patient","encounter","code","description"]),
    ("medications",  ["start","stop","patient","encounter","code","description",
                      "base_cost","payer_coverage","dispenses","totalcost",
                      "reasoncode","reasondescription"]),
    ("procedures",   ["start","stop","patient","encounter","code","description",
                      "base_cost","reasoncode","reasondescription"]),
    ("observations", ["date","patient","encounter","category","code",
                      "description","value","units","type"]),
]

for table, columns in tables:
    print(f"Loading {table}...")
    df = pd.read_csv(f"{CSV_PATH}{table}.csv", low_memory=False)
    df.columns = df.columns.str.lower()
    df = df[[c for c in columns if c in df.columns]]
    df.to_sql(table, engine, if_exists="append", index=False)
    print(f"  OK - {len(df)} rows loaded into {table}")

print("\nAll data loaded successfully!")
