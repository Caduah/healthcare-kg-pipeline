import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import psycopg2
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare Knowledge Graph API",
    description="Query the healthcare knowledge graph built from Synthea data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── CONNECTIONS ─────────────────────────────────────────────────────────────

def get_pg_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),
        dbname=os.getenv("POSTGRES_DB", "healthdb"),
        user=os.getenv("POSTGRES_USER", "healthuser"),
        password=os.getenv("POSTGRES_PASSWORD", "healthpass"),
    )


def get_neo4j_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "neo4j_pass"),
        )
    )


# ─── MODELS ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 10


class PatientResponse(BaseModel):
    id: str
    full_name: str
    age_years: Optional[int]
    gender: Optional[str]
    state: Optional[str]
    is_deceased: Optional[bool]
    total_encounters: Optional[int]
    total_claims: Optional[int]


class ProviderResponse(BaseModel):
    id: str
    name: str
    speciality: Optional[str]
    organization: Optional[str]
    city: Optional[str]
    state: Optional[str]
    total_encounters: Optional[int]


# ─── HEALTH CHECK ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Healthcare Knowledge Graph API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": [
            "/health",
            "/patients/{patient_id}",
            "/patients/{patient_id}/providers",
            "/patients/{patient_id}/encounters",
            "/providers/{provider_id}",
            "/graph/stats",
            "/query",
        ]
    }


@app.get("/health")
def health_check():
    """Check connectivity to all backend services."""
    status = {"api": "healthy"}

    try:
        conn = get_pg_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM processed.patients")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        status["postgres"] = f"healthy — {count:,} patients"
    except Exception as e:
        status["postgres"] = f"unhealthy: {e}"

    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            count = result.single()["count"]
        driver.close()
        status["neo4j"] = f"healthy — {count:,} nodes"
    except Exception as e:
        status["neo4j"] = f"unhealthy: {e}"

    return status


# ─── PATIENT ENDPOINTS ────────────────────────────────────────────────────────

@app.get("/patients/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str):
    """Get a patient by ID with encounter and claim counts."""
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            p.id, p.full_name, p.age_years, p.gender,
            p.state_upper, p.is_deceased,
            COUNT(DISTINCT e.id) AS total_encounters,
            COUNT(DISTINCT c.id) AS total_claims
        FROM processed.patients p
        LEFT JOIN processed.encounters e ON e.patient = p.id
        LEFT JOIN processed.claims c ON c.patientid = p.id
        WHERE p.id = %s
        GROUP BY p.id, p.full_name, p.age_years,
                 p.gender, p.state_upper, p.is_deceased
    """, (patient_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    return PatientResponse(
        id=row[0],
        full_name=row[1],
        age_years=row[2],
        gender=row[3],
        state=row[4],
        is_deceased=row[5],
        total_encounters=row[6],
        total_claims=row[7],
    )


@app.get("/patients/{patient_id}/providers")
def get_patient_providers(patient_id: str):
    """Get all providers who treated a patient — multi-hop graph traversal."""
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (p:Patient {id: $patient_id})-[:HAD_VISIT]->(e:Encounter)
                      -[:SEEN_BY]->(pr:Provider)
                RETURN
                    pr.id AS provider_id,
                    pr.name AS name,
                    pr.speciality AS speciality,
                    pr.organization AS organization,
                    count(e) AS visit_count
                ORDER BY visit_count DESC
            """, patient_id=patient_id)

            providers = []
            for record in result:
                providers.append({
                    "provider_id": record["provider_id"],
                    "name": record["name"],
                    "speciality": record["speciality"],
                    "organization": record["organization"],
                    "visit_count": record["visit_count"],
                })

        if not providers:
            raise HTTPException(
                status_code=404,
                detail=f"No providers found for patient {patient_id}"
            )

        return {
            "patient_id": patient_id,
            "provider_count": len(providers),
            "providers": providers,
        }
    finally:
        driver.close()


@app.get("/patients/{patient_id}/encounters")
def get_patient_encounters(patient_id: str, limit: int = 20):
    """Get recent encounters for a patient."""
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            e.id, e.encounterclass_norm, e.description_norm,
            e.encounter_date, e.duration_minutes,
            e.total_claim_cost, e.patient_cost, e.has_reason,
            e.reasondescription
        FROM processed.encounters e
        WHERE e.patient = %s
        ORDER BY e.encounter_date DESC
        LIMIT %s
    """, (patient_id, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    encounters = []
    for row in rows:
        encounters.append({
            "encounter_id": row[0],
            "encounter_class": row[1],
            "description": row[2],
            "encounter_date": str(row[3]) if row[3] else None,
            "duration_minutes": row[4],
            "total_cost": row[5],
            "patient_cost": row[6],
            "has_reason": row[7],
            "reason": row[8],
        })

    return {
        "patient_id": patient_id,
        "encounter_count": len(encounters),
        "encounters": encounters,
    }


# ─── PROVIDER ENDPOINTS ───────────────────────────────────────────────────────

@app.get("/providers/{provider_id}", response_model=ProviderResponse)
def get_provider(provider_id: str):
    """Get a provider by ID."""
    conn = get_pg_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            p.id, p.name_clean, p.speciality_clean,
            p.organization, p.city_norm, p.state_upper,
            p.encounters
        FROM processed.providers p
        WHERE p.id = %s
    """, (provider_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")

    return ProviderResponse(
        id=row[0],
        name=row[1],
        speciality=row[2],
        organization=row[3],
        city=row[4],
        state=row[5],
        total_encounters=row[6],
    )


# ─── GRAPH STATS ──────────────────────────────────────────────────────────────

@app.get("/graph/stats")
def get_graph_stats():
    """Get knowledge graph statistics."""
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            nodes = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
            """)
            node_counts = {r["label"]: r["count"] for r in nodes}

            rels = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(r) AS count
                ORDER BY count DESC
            """)
            rel_counts = {r["rel_type"]: r["count"] for r in rels}

        return {
            "nodes": node_counts,
            "relationships": rel_counts,
            "total_nodes": sum(node_counts.values()),
            "total_relationships": sum(rel_counts.values()),
        }
    finally:
        driver.close()


# ─── NATURAL LANGUAGE QUERY ───────────────────────────────────────────────────

@app.post("/query")
def natural_language_query(request: QueryRequest):
    """
    Answer natural language questions about the healthcare graph.
    Uses keyword matching to route to appropriate Cypher queries.
    In production this would use LangChain + ChromaDB for full RAG.
    """
    question = request.question.lower()
    driver = get_neo4j_driver()

    try:
        with driver.session() as session:

            if "how many patients" in question:
                result = session.run("MATCH (p:Patient) RETURN count(p) AS count")
                count = result.single()["count"]
                return {"question": request.question, "answer": f"There are {count:,} patients in the healthcare graph."}

            elif "most visits" in question or "most encounters" in question:
                result = session.run("""
                    MATCH (p:Patient)-[:HAD_VISIT]->(e:Encounter)
                    RETURN p.fullName AS patient, count(e) AS visits
                    ORDER BY visits DESC LIMIT 5
                """)
                patients = [{"patient": r["patient"], "visits": r["visits"]} for r in result]
                return {"question": request.question, "answer": patients}

            elif "specialit" in question or "specialty" in question:
                result = session.run("""
                    MATCH (pr:Provider)
                    RETURN pr.speciality AS speciality, count(pr) AS count
                    ORDER BY count DESC LIMIT 10
                """)
                specialties = [{"speciality": r["speciality"], "count": r["count"]} for r in result]
                return {"question": request.question, "answer": specialties}

            elif "organization" in question or "hospital" in question or "facility" in question:
                result = session.run("""
                    MATCH (o:Organization)<-[:AT_FACILITY]-(e:Encounter)
                    RETURN o.name AS organization, count(e) AS encounters
                    ORDER BY encounters DESC LIMIT 10
                """)
                orgs = [{"organization": r["organization"], "encounters": r["encounters"]} for r in result]
                return {"question": request.question, "answer": orgs}

            else:
                return {
                    "question": request.question,
                    "answer": "I can answer questions about patient counts, most visited patients, provider specialties, and organization encounter volumes. Try: 'How many patients are there?' or 'Which patients have the most visits?'"
                }
    finally:
        driver.close()