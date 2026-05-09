import os
import logging
import psycopg2
import pandas as pd
from neo4j import GraphDatabase
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ─── CONNECTION ──────────────────────────────────────────────────────────────

def get_pg_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB", "healthdb"),
        user=os.getenv("POSTGRES_USER", "healthuser"),
        password=os.getenv("POSTGRES_PASSWORD", "healthpass"),
    )


def get_neo4j_driver():
    uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER",     "neo4j")
    pw   = os.getenv("NEO4J_PASSWORD", "neo4j_pass")
    return GraphDatabase.driver(uri, auth=(user, pw))


# ─── SCHEMA SETUP ────────────────────────────────────────────────────────────

def create_constraints(driver):
    """Create uniqueness constraints and indexes."""
    log.info("Creating Neo4j constraints and indexes...")
    constraints = [
        "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT provider_id IF NOT EXISTS FOR (pr:Provider) REQUIRE pr.id IS UNIQUE",
        "CREATE CONSTRAINT encounter_id IF NOT EXISTS FOR (e:Encounter) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT org_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
    ]
    with driver.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception as ex:
                log.warning(f"Constraint may already exist: {ex}")
    log.info("Constraints created")


# ─── NODE LOADERS ─────────────────────────────────────────────────────────────

def load_patient_nodes(driver):
    """Load Patient nodes from processed.patients."""
    log.info("Loading Patient nodes...")
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT
            id, full_name, first_norm, last_norm,
            birthdate::text AS birthdate,
            gender, race, ethnicity,
            city, state_upper AS state, zip_clean AS zip,
            age_years, is_deceased, source_system
        FROM processed.patients
        WHERE id IS NOT NULL
        LIMIT 5000
    """, conn)
    conn.close()

    batch_size = 500
    total = 0
    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict("records")
            session.run("""
                UNWIND $batch AS row
                MERGE (p:Patient {id: row.id})
                SET p.fullName      = row.full_name,
                    p.firstName     = row.first_norm,
                    p.lastName      = row.last_norm,
                    p.birthdate     = row.birthdate,
                    p.gender        = row.gender,
                    p.race          = row.race,
                    p.ethnicity     = row.ethnicity,
                    p.city          = row.city,
                    p.state         = row.state,
                    p.zip           = row.zip,
                    p.ageYears      = toInteger(row.age_years),
                    p.isDeceased    = row.is_deceased,
                    p.sourceSystem  = row.source_system,
                    p.loadedAt      = datetime()
            """, batch=batch)
            total += len(batch)
    log.info(f"Loaded {total} Patient nodes")
    return total


def load_provider_nodes(driver):
    """Load Provider nodes from processed.providers."""
    log.info("Loading Provider nodes...")
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT
            id, name_clean AS name, name_norm,
            speciality_clean AS speciality, speciality_norm,
            npi, has_npi, organization,
            city_norm AS city, state_upper AS state, zip_clean AS zip,
            encounters, procedures, source_system
        FROM processed.providers
        WHERE id IS NOT NULL
    """, conn)
    conn.close()

    batch_size = 500
    total = 0
    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict("records")
            session.run("""
                UNWIND $batch AS row
                MERGE (pr:Provider {id: row.id})
                SET pr.name         = row.name,
                    pr.nameNorm     = row.name_norm,
                    pr.speciality   = row.speciality,
                    pr.npi          = row.npi,
                    pr.hasNpi       = row.has_npi,
                    pr.organization = row.organization,
                    pr.city         = row.city,
                    pr.state        = row.state,
                    pr.zip          = row.zip,
                    pr.encounters   = toInteger(row.encounters),
                    pr.procedures   = toInteger(row.procedures),
                    pr.sourceSystem = row.source_system,
                    pr.loadedAt     = datetime()
            """, batch=batch)
            total += len(batch)
    log.info(f"Loaded {total} Provider nodes")
    return total


def load_organization_nodes(driver):
    """Load Organization nodes from unique organizations in encounters."""
    log.info("Loading Organization nodes...")
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT DISTINCT organization AS id, organization AS name
        FROM processed.encounters
        WHERE organization IS NOT NULL
    """, conn)
    conn.close()

    with driver.session() as session:
        batch = df.to_dict("records")
        session.run("""
            UNWIND $batch AS row
            MERGE (o:Organization {id: row.id})
            SET o.name     = row.name,
                o.loadedAt = datetime()
        """, batch=batch)
    log.info(f"Loaded {len(df)} Organization nodes")
    return len(df)


def load_encounter_nodes(driver):
    """Load Encounter nodes from processed.encounters."""
    log.info("Loading Encounter nodes...")
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT
            id, patient, provider, organization,
            encounter_date::text AS encounter_date,
            encounter_year, encounter_month,
            encounterclass_norm AS encounter_class,
            description_norm AS description,
            total_claim_cost, payer_coverage, patient_cost,
            duration_minutes, has_reason,
            reasondescription AS reason,
            source_system
        FROM processed.encounters
        WHERE id IS NOT NULL
        AND patient IS NOT NULL
        LIMIT 60000
    """, conn)
    conn.close()

    batch_size = 500
    total = 0
    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict("records")
            session.run("""
                UNWIND $batch AS row
                MERGE (e:Encounter {id: row.id})
                SET e.encounterClass  = row.encounter_class,
                    e.description     = row.description,
                    e.encounterDate   = row.encounter_date,
                    e.encounterYear   = toInteger(row.encounter_year),
                    e.encounterMonth  = toInteger(row.encounter_month),
                    e.totalCost       = toFloat(row.total_claim_cost),
                    e.payerCoverage   = toFloat(row.payer_coverage),
                    e.patientCost     = toFloat(row.patient_cost),
                    e.durationMinutes = toFloat(row.duration_minutes),
                    e.hasReason       = row.has_reason,
                    e.reason          = row.reason,
                    e.sourceSystem    = row.source_system,
                    e.loadedAt        = datetime()
            """, batch=batch)
            total += len(batch)
    log.info(f"Loaded {total} Encounter nodes")
    return total


def load_claim_nodes(driver):
    """Load Claim nodes from processed.claims."""
    log.info("Loading Claim nodes...")
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT
            id, patientid AS patient_id, providerid AS provider_id,
            servicedate AS service_date,
            diagnosis1, diagnosis2, diagnosis3, diagnosis4,
            total_outstanding, status_clean,
            has_primary_diagnosis, diagnosis_count,
            source_system
        FROM processed.claims
        WHERE id IS NOT NULL
        LIMIT 50000
    """, conn)
    conn.close()

    batch_size = 500
    total = 0
    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict("records")
            session.run("""
                UNWIND $batch AS row
                MERGE (c:Claim {id: row.id})
                SET c.patientId          = row.patient_id,
                    c.providerId         = row.provider_id,
                    c.serviceDate        = row.service_date,
                    c.diagnosis1         = row.diagnosis1,
                    c.totalOutstanding   = toFloat(row.total_outstanding),
                    c.status             = row.status_clean,
                    c.hasPrimaryDiagnosis= row.has_primary_diagnosis,
                    c.diagnosisCount     = toInteger(row.diagnosis_count),
                    c.sourceSystem       = row.source_system,
                    c.loadedAt           = datetime()
            """, batch=batch)
            total += len(batch)
    log.info(f"Loaded {total} Claim nodes")
    return total


# ─── RELATIONSHIP LOADERS ─────────────────────────────────────────────────────

def create_patient_encounter_relationships(driver):
    """Patient -[:HAD_VISIT]-> Encounter"""
    log.info("Creating HAD_VISIT relationships...")
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Encounter)
            WHERE e.id IS NOT NULL
            MATCH (p:Patient {id: split(e.id, '-')[0]})
            WITH p, e LIMIT 1
            RETURN count(*) AS test
        """)

        result = session.run("""
            MATCH (e:Encounter)
            MATCH (p:Patient)
            WHERE EXISTS {
                MATCH (enc:Encounter {id: e.id})
                MATCH (pat:Patient)
                WHERE pat.id IN [
                    x IN [(enc)<-[:HAD_VISIT]-(pat2) | pat2.id] | x
                ]
            }
            RETURN count(e) AS already_linked
        """)

        result = session.run("""
            MATCH (enc:Encounter)
            WITH enc
            MATCH (pat:Patient)
            WHERE pat.id = enc.id
            RETURN count(*) AS test
        """)

        result = session.run("""
            MATCH (enc:Encounter), (pat:Patient)
            WHERE enc.id STARTS WITH pat.id
            WITH pat, enc LIMIT 100
            MERGE (pat)-[:HAD_VISIT]->(enc)
            RETURN count(*) AS created
        """)

        session.run("""
            MATCH (enc:Encounter)
            WITH enc
            MATCH (rels)
            WHERE NOT (enc)<-[:HAD_VISIT]-()
            WITH enc
            MATCH (pat:Patient)
            WHERE enc.id CONTAINS pat.id
            MERGE (pat)-[:HAD_VISIT]->(enc)
        """)

    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT id AS encounter_id, patient AS patient_id
        FROM processed.encounters
        WHERE patient IS NOT NULL
        AND id IS NOT NULL
        LIMIT 60000
    """, conn)
    conn.close()

    batch_size = 500
    total = 0
    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict("records")
            session.run("""
                UNWIND $batch AS row
                MATCH (p:Patient {id: row.patient_id})
                MATCH (e:Encounter {id: row.encounter_id})
                MERGE (p)-[:HAD_VISIT]->(e)
            """, batch=batch)
            total += len(batch)
    log.info(f"Created {total} HAD_VISIT relationships")
    return total


def create_encounter_provider_relationships(driver):
    """Encounter -[:SEEN_BY]-> Provider"""
    log.info("Creating SEEN_BY relationships...")
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT id AS encounter_id, provider AS provider_id
        FROM processed.encounters
        WHERE provider IS NOT NULL
        AND id IS NOT NULL
        LIMIT 60000
    """, conn)
    conn.close()

    batch_size = 500
    total = 0
    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict("records")
            session.run("""
                UNWIND $batch AS row
                MATCH (e:Encounter {id: row.encounter_id})
                MATCH (pr:Provider {id: row.provider_id})
                MERGE (e)-[:SEEN_BY]->(pr)
            """, batch=batch)
            total += len(batch)
    log.info(f"Created {total} SEEN_BY relationships")
    return total


def create_encounter_organization_relationships(driver):
    """Encounter -[:AT_FACILITY]-> Organization"""
    log.info("Creating AT_FACILITY relationships...")
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT id AS encounter_id, organization AS org_id
        FROM processed.encounters
        WHERE organization IS NOT NULL
        AND id IS NOT NULL
        LIMIT 60000
    """, conn)
    conn.close()

    batch_size = 500
    total = 0
    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict("records")
            session.run("""
                UNWIND $batch AS row
                MATCH (e:Encounter {id: row.encounter_id})
                MATCH (o:Organization {id: row.org_id})
                MERGE (e)-[:AT_FACILITY]->(o)
            """, batch=batch)
            total += len(batch)
    log.info(f"Created {total} AT_FACILITY relationships")
    return total


def create_patient_claim_relationships(driver):
    """Patient -[:HAS_CLAIM]-> Claim"""
    log.info("Creating HAS_CLAIM relationships...")
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT id AS claim_id, patientid AS patient_id
        FROM processed.claims
        WHERE patientid IS NOT NULL
        AND id IS NOT NULL
        LIMIT 50000
    """, conn)
    conn.close()

    batch_size = 500
    total = 0
    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict("records")
            session.run("""
                UNWIND $batch AS row
                MATCH (p:Patient {id: row.patient_id})
                MATCH (c:Claim {id: row.claim_id})
                MERGE (p)-[:HAS_CLAIM]->(c)
            """, batch=batch)
            total += len(batch)
    log.info(f"Created {total} HAS_CLAIM relationships")
    return total


def create_provider_organization_relationships(driver):
    """Provider -[:WORKS_AT]-> Organization"""
    log.info("Creating WORKS_AT relationships...")
    conn = get_pg_conn()
    df = pd.read_sql("""
        SELECT DISTINCT
            p.id AS provider_id,
            e.organization AS org_id
        FROM processed.providers p
        JOIN processed.encounters e ON e.provider = p.id
        WHERE e.organization IS NOT NULL
        LIMIT 10000
    """, conn)
    conn.close()

    batch_size = 500
    total = 0
    with driver.session() as session:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].to_dict("records")
            session.run("""
                UNWIND $batch AS row
                MATCH (pr:Provider {id: row.provider_id})
                MATCH (o:Organization {id: row.org_id})
                MERGE (pr)-[:WORKS_AT]->(o)
            """, batch=batch)
            total += len(batch)
    log.info(f"Created {total} WORKS_AT relationships")
    return total


# ─── GRAPH STATISTICS ─────────────────────────────────────────────────────────

def print_graph_stats(driver):
    """Print a summary of what is in the graph."""
    with driver.session() as session:
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        log.info("\n── NODE COUNTS ──────────────────────────")
        for record in result:
            log.info(f"  {record['label']}: {record['count']:,}")

        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
        """)
        log.info("── RELATIONSHIP COUNTS ──────────────────")
        for record in result:
            log.info(f"  {record['rel_type']}: {record['count']:,}")


# ─── SAMPLE QUERIES ───────────────────────────────────────────────────────────

def run_sample_queries(driver):
    """Run sample traversal queries to verify the graph."""
    log.info("\n── SAMPLE GRAPH QUERIES ─────────────────")

    with driver.session() as session:
        result = session.run("""
            MATCH (p:Patient)-[:HAD_VISIT]->(e:Encounter)-[:SEEN_BY]->(pr:Provider)
            RETURN p.fullName AS patient,
                   count(e) AS visits,
                   collect(DISTINCT pr.speciality)[..3] AS specialties
            ORDER BY visits DESC
            LIMIT 5
        """)
        log.info("Top 5 patients by visit count:")
        for r in result:
            log.info(f"  {r['patient']} — {r['visits']} visits — {r['specialties']}")

        result = session.run("""
            MATCH (pr:Provider)-[:WORKS_AT]->(o:Organization)
            RETURN o.name AS org, count(pr) AS provider_count
            ORDER BY provider_count DESC
            LIMIT 5
        """)
        log.info("Top 5 organizations by provider count:")
        for r in result:
            log.info(f"  {r['org']} — {r['provider_count']} providers")

        result = session.run("""
            MATCH (p:Patient)-[:HAD_VISIT]->(e:Encounter)-[:AT_FACILITY]->(o:Organization)
            RETURN p.fullName AS patient,
                   collect(DISTINCT o.name)[..3] AS facilities
            LIMIT 3
        """)
        log.info("Sample patient → facility connections:")
        for r in result:
            log.info(f"  {r['patient']} visited {r['facilities']}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def build_knowledge_graph(**context):
    log.info("=" * 55)
    log.info("DAY 5 — Building Healthcare Knowledge Graph")
    log.info("=" * 55)

    driver = get_neo4j_driver()

    try:
        create_constraints(driver)

        log.info("\n── LOADING NODES ────────────────────────")
        patients   = load_patient_nodes(driver)
        providers  = load_provider_nodes(driver)
        orgs       = load_organization_nodes(driver)
        encounters = load_encounter_nodes(driver)
        claims     = load_claim_nodes(driver)

        log.info("\n── CREATING RELATIONSHIPS ───────────────")
        create_patient_encounter_relationships(driver)
        create_encounter_provider_relationships(driver)
        create_encounter_organization_relationships(driver)
        create_patient_claim_relationships(driver)
        create_provider_organization_relationships(driver)

        print_graph_stats(driver)
        run_sample_queries(driver)

        log.info("\n" + "=" * 55)
        log.info("Knowledge graph build complete")
        log.info(f"  Patients:   {patients:,}")
        log.info(f"  Providers:  {providers:,}")
        log.info(f"  Orgs:       {orgs:,}")
        log.info(f"  Encounters: {encounters:,}")
        log.info(f"  Claims:     {claims:,}")
        log.info("=" * 55)

        return {
            "patients": patients,
            "providers": providers,
            "organizations": orgs,
            "encounters": encounters,
            "claims": claims,
        }

    finally:
        driver.close()


if __name__ == "__main__":
    result = build_knowledge_graph()
    print("\nKnowledge Graph Built:")
    for k, v in result.items():
        print(f"  {k}: {v:,}")