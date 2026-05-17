# Healthcare Knowledge Graph Pipeline

> A production-grade data engineering pipeline that ingests, cleans, resolves, graphs, streams, and serves synthetic healthcare data — built from scratch using the tools real data teams use in production.

**Portfolio Project 1 of 3** · Project 2 — E-commerce on AWS · Project 3 — Stock Market on Azure

---

## The Problem This Solves

Healthcare data is fragmented. A patient visits three different hospitals. Each hospital stores them under a different ID, with slightly different name spellings, a different address on file. No system knows these three records are the same person.

This pipeline solves that problem end-to-end — ingesting raw synthetic patient data, cleaning it with Spark, detecting duplicates with entity resolution, connecting everything in a knowledge graph, streaming real-time clinical events with NLP enrichment, and serving the results through a REST API.

Every design decision mirrors what production healthcare data teams actually build.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│              Synthea EHR Generator (1,139 patients)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION LAYER                            │
│         PostgreSQL raw schema — 171,000+ records loaded         │
│         patients · providers · encounters · claims              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SPARK ETL LAYER                           │
│    PySpark 3.5 — 40+ transformations across all four entities   │
│    Name normalization · Age calculation · ZIP cleaning          │
│    Duration derivation · Cost calculation · Feature engineering │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA QUALITY LAYER                           │
│         18 automated checks · quarantine routing                │
│    Null checks · Duplicate detection · Range validation         │
│    Failed records → quarantine schema with reason + timestamp   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
┌──────────────────────┐  ┌────────────────────────────────────┐
│  PATIENT ENTITY      │  │  PROVIDER ENTITY RESOLUTION        │
│  RESOLUTION          │  │  NPI exact match + name/specialty  │
│  Blocking +          │  │  scoring · 580 matches found       │
│  Jaro-Winkler +      │  │  MLflow experiment tracking        │
│  weighted heuristic  │  └────────────────────────────────────┘
└──────────┬───────────┘               │
           └──────────────┬────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE GRAPH LAYER                         │
│              Neo4j — 112,724 nodes · 230,763 relationships      │
│      Patient → Encounter → Provider → Organization              │
│      Patient → Claim · Provider → Organization                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
┌──────────────────────┐  ┌────────────────────────────────────┐
│  KAFKA STREAMING     │  │  FASTAPI SERVING LAYER             │
│  50 clinical events  │  │  8 REST endpoints                  │
│  spaCy NLP           │  │  Graph traversal queries           │
│  Presidio PII        │  │  Natural language query interface  │
│  detection           │  │  Interactive Swagger docs          │
└──────────────────────┘  └────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                          │
│     Apache Airflow 2.9 — 7-task DAG, runs daily                │
│     Parallel entity resolution · Retry logic · XCom state      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data generation | Synthea | Realistic synthetic EHR data — no HIPAA concerns |
| Infrastructure | Docker Compose | Six services, one command to start everything |
| Storage | PostgreSQL 15 | Five schemas: raw, processed, quality, entity_resolution, quarantine |
| Batch ETL | PySpark 3.5 | Distributed data cleaning and feature engineering |
| Orchestration | Apache Airflow 2.9 | Daily scheduling, retry logic, parallel task execution |
| Data quality | Custom framework | 18 checks, audit logging, quarantine routing |
| Entity resolution | jellyfish + MLflow | Jaro-Winkler similarity, weighted heuristics, experiment tracking |
| Knowledge graph | Neo4j 5.19 | 112K nodes, 230K relationships, Cypher traversal |
| Streaming | Apache Kafka | Real-time clinical event processing |
| NLP | spaCy + Presidio | Medical entity extraction, PII detection and anonymization |
| API | FastAPI + Uvicorn | REST endpoints, graph traversal, auto-generated docs |
| Testing | pytest | 9 unit tests, all passing |
| CI/CD | GitHub Actions | Lint, DAG validation, test suite on every push |

---

## Pipeline Numbers

| Metric | Value |
|---|---|
| Raw records ingested | 171,605 |
| Spark transformations applied | 40+ across 4 entities |
| Quality checks run per execution | 18 |
| Records quarantined | 1 (negative duration encounter) |
| Patient candidate pairs evaluated | 2,898 |
| Patient matches found | 4 |
| Provider matches found | 580 |
| Knowledge graph nodes | 112,724 |
| Knowledge graph relationships | 230,763 |
| Clinical events streamed | 50 |
| API endpoints | 8 |
| Unit tests | 9 passing |

---

## Five Most Interesting Technical Decisions

### 1. Blocking before scoring in entity resolution

Comparing every patient to every other patient means 636,756 pairs. At scale with 10 million patients that becomes 50 trillion comparisons — computationally impossible. Blocking on last name prefix reduces candidates to 2,898 pairs before any similarity scoring begins. Same result, 440x fewer comparisons. This is the most important performance optimization in entity resolution and the one most candidates cannot explain.

### 2. Key salting for Spark data skew

Healthcare data is inherently skewed — a major hospital has 50,000 encounters while a small clinic has 200. Without intervention, one Spark executor processes all 50,000 records while 49 others sit idle. Key salting appends a random suffix to the hot key in the large table and explodes the small table N times to match — distributing work evenly across executors while preserving correct join semantics.

### 3. Quarantine instead of delete

When a quality check fails, the bad record is copied to the quarantine schema first, then deleted from the processed table. The copy always happens before the delete. The quarantine table includes the rejection reason and timestamp — a permanent audit trail for compliance and debugging.

### 4. Graph database over relational joins for multi-hop queries

Finding all providers who treated a specific patient in PostgreSQL requires two joins and three table scans. In Neo4j the same query traverses direct relationship pointers. As the number of hops increases, SQL explodes exponentially while the graph version stays fast. The `GET /patients/{id}/providers` endpoint demonstrates this in a live API call.

### 5. Parallel entity resolution in the DAG

Patient and provider entity resolution are completely independent. Wiring them as parallel tasks in the Airflow DAG cuts the pipeline runtime for that section in half. Small decision, meaningful impact at scale.

---

## The Airflow DAG

Seven tasks, one parallel branch, runs daily:

```
check_raw_data
      │
      ▼
  spark_etl
      │
      ▼
data_quality_checks
      │
      ├─────────────────────────────┐
      ▼                             ▼
patient_entity_resolution    provider_entity_resolution
      │                             │     (run in parallel)
      └─────────────┬───────────────┘
                    ▼
             verify_processed
                    │
                    ▼
          build_knowledge_graph
```

---

## Data Quality — 18 Checks

| Table | Check | What it catches |
|---|---|---|
| patients | null_id | Records with no primary identifier |
| patients | duplicate_id | Same patient ID appearing more than once |
| patients | min_row_count | Silent empty loads after ETL |
| patients | null_birthdate | Records missing date of birth |
| patients | age_range_0_120 | Impossible ages from calculation errors |
| patients | valid_gender | Values outside expected gender vocabulary |
| providers | null_id | Records with no primary identifier |
| providers | duplicate_id | Same provider ID appearing more than once |
| providers | min_row_count | Silent empty loads |
| providers | null_name | Providers with no name for matching |
| encounters | null_id | Records with no primary identifier |
| encounters | null_patient | Encounters not linked to any patient |
| encounters | min_row_count | Silent empty loads |
| encounters | negative_cost | Claim costs below zero |
| encounters | negative_duration | Stop time before start time ← **caught one real bug** |
| claims | null_id | Records with no primary identifier |
| claims | min_row_count | Silent empty loads |
| claims | null_patientid | Claims not linked to any patient |

The `negative_duration` check caught encounter `4c5c752b` — a Synthea data entry error where the stop time was recorded 8 minutes before the start time. It was quarantined and all 18 checks now pass on every run.

---

## Entity Resolution — How the Scoring Works

### Stage 1 — Blocking
Group patients by the first three letters of their last name. Only compare patients within the same group. Reduces 636,756 potential pairs to 2,898 candidates.

### Stage 2 — Rule-based matching (fires first)
Same date of birth + same last name + same first initial = HIGH confidence match. Score: 1.0.

### Stage 3 — Feature-based scoring

| Signal | Weight | Why |
|---|---|---|
| Last name similarity (Jaro-Winkler) | 30% | Most stable and distinctive demographic signal |
| Date of birth match | 25% | Specific — 1 in 365 chance of sharing a birthday |
| First name similarity | 20% | More variable — nicknames and abbreviations common |
| Gender match | 10% | Useful but not distinctive on its own |
| ZIP code match | 10% | Useful but people move |
| State match | 5% | Too broad to be meaningful alone |

Pairs scoring above 0.75 are classified as MEDIUM confidence matches. All experiments are tracked in MLflow.

---

## Knowledge Graph

```cypher
(Patient)-[:HAD_VISIT]->(Encounter)        // 60,000 relationships
(Encounter)-[:SEEN_BY]->(Provider)         // 60,000 relationships
(Encounter)-[:AT_FACILITY]->(Organization) // 60,000 relationships
(Patient)-[:HAS_CLAIM]->(Claim)            // 50,000 relationships
(Provider)-[:WORKS_AT]->(Organization)     // 763 relationships
```

**Sample traversal queries:**

```cypher
-- All providers who treated a specific patient
MATCH (p:Patient)-[:HAD_VISIT]->(e:Encounter)-[:SEEN_BY]->(pr:Provider)
WHERE p.fullName = 'Alison204 Halvorson124'
RETURN p, e, pr LIMIT 25

-- Top 5 patients by visit count
MATCH (p:Patient)-[:HAD_VISIT]->(e:Encounter)
RETURN p.fullName AS patient, count(e) AS visits
ORDER BY visits DESC LIMIT 5
```

---

## API Endpoints

Start: `uvicorn api.main:app --reload --port 8000`

Docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info and available endpoints |
| GET | `/health` | Connectivity check for Postgres and Neo4j |
| GET | `/graph/stats` | Live node and relationship counts |
| GET | `/patients/{id}` | Patient details with encounter and claim counts |
| GET | `/patients/{id}/providers` | **Graph traversal** — all providers who treated this patient |
| GET | `/patients/{id}/encounters` | Recent encounter history |
| GET | `/providers/{id}` | Provider details |
| POST | `/query` | Natural language queries against the graph |

**Live responses:**

```bash
$ curl http://localhost:8000/health
{"api":"healthy","postgres":"healthy — 1,129 patients","neo4j":"healthy — 112,724 nodes"}

$ curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which patients have the most visits?"}'
{"answer": [
  {"patient": "Alison204 Halvorson124", "visits": 700},
  {"patient": "Carson894 Blick895", "visits": 598},
  {"patient": "Cristobal567 Blanco851", "visits": 572}
]}
```

---

## Streaming Pipeline

```
Clinical Event
      │
      ▼
Kafka Producer → clinical-events topic → Kafka Consumer
                                               │
                                     spaCy NLP enrichment
                                     extracts: conditions, medications
                                               │
                                     Presidio PII detection
                                     "John Smith" → "<PERSON>"
                                               │
                                     processed.clinical_events_stream
```

Event types: `ADMISSION` · `DISCHARGE` · `LAB_RESULT` · `MEDICATION_ORDER` · `DIAGNOSIS_UPDATE`

---

## Getting Started

### Prerequisites

- Docker Desktop (8GB RAM allocated)
- Python 3.11
- Java 17 (for Spark)

### Start the stack

```bash
git clone https://github.com/Caduah/healthcare-kg-pipeline.git
cd healthcare-kg-pipeline
docker-compose up -d
```

Six containers start: `healthcare_db` · `healthcare_neo4j` · `healthcare_airflow_web` · `healthcare_airflow_scheduler` · `healthcare_zookeeper` · `healthcare_kafka`

### Install dependencies

```bash
pip install pyspark==3.5.1 pandas psycopg2-binary sqlalchemy \
            neo4j kafka-python spacy jellyfish mlflow scikit-learn \
            presidio-analyzer presidio-anonymizer fastapi uvicorn \
            pytest ruff

python -m spacy download en_core_web_sm
```

### Generate the dataset

```bash
cd data
java -jar synthea-with-dependencies.jar -p 1000 Massachusetts \
  --exporter.csv.export=true --exporter.fhir.export=false
cd ..
```

### Run each layer

```bash
# 1. Load raw data
python ingestion/load_raw.py
python ingestion/load_claims.py

# 2. Spark ETL
python spark/jobs/healthcare_etl.py

# 3. Quality checks
python quality/checks.py

# 4. Entity resolution
python entity_resolution/patient_matcher.py
python entity_resolution/provider_matcher.py

# 5. Knowledge graph
python graph/neo4j_loader.py

# 6. Kafka streaming
python ingestion/kafka_producer.py
PYTHONPATH=. python ingestion/kafka_consumer.py

# 7. API
uvicorn api.main:app --reload --port 8000
```

### Run via Airflow

`http://localhost:8080` · Login: `caleb` / `caleb123` · Enable and trigger `healthcare_kg_pipeline`

### Explore the graph

`http://localhost:7474` · Login: `neo4j` / `neo4j_pass`

---

## Project Structure

```
healthcare-pipeline/
│
├── airflow/dags/
│   └── healthcare_pipeline.py     # 7-task DAG with parallel ER branches
│
├── spark/jobs/
│   ├── healthcare_etl.py          # Main ETL — all four entities
│   └── utils.py                   # SparkSession, JDBC helpers, key salting
│
├── ingestion/
│   ├── load_raw.py                # Loads patients, providers, encounters
│   ├── load_claims.py             # Loads claims (camelCase columns)
│   ├── kafka_producer.py          # Generates 50 clinical events
│   └── kafka_consumer.py          # Reads, enriches, saves to Postgres
│
├── quality/
│   └── checks.py                  # 18 checks + quarantine routing + audit log
│
├── entity_resolution/
│   ├── patient_matcher.py         # Blocking + Jaro-Winkler + MLflow
│   └── provider_matcher.py        # NPI exact match + feature scoring
│
├── enrichment/
│   └── nlp_enricher.py            # spaCy NER + rule-based + Presidio PII
│
├── graph/
│   └── neo4j_loader.py            # Loads 112K nodes and 230K relationships
│
├── api/
│   └── main.py                    # 8 FastAPI endpoints + graph traversal
│
├── tests/
│   └── test_pipeline.py           # 9 unit tests — all passing
│
├── .github/workflows/
│   └── ci.yml                     # Lint + DAG validation + pytest on push
│
└── docker-compose.yml             # Full stack — 6 services
```

---

## Testing

```bash
pytest tests/ -v

# test_jaro_winkler_identical_strings    PASSED
# test_jaro_winkler_different_strings    PASSED
# test_jaro_winkler_empty_strings        PASSED
# test_heuristic_score_perfect_match     PASSED
# test_heuristic_score_no_match          PASSED
# test_nlp_extracts_conditions           PASSED
# test_nlp_handles_empty_note            PASSED
# test_pii_detection_finds_person        PASSED
# test_pii_detection_anonymizes          PASSED
#
# 9 passed in 9.07s
```

---

## Services

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | caleb / caleb123 |
| Neo4j Browser | http://localhost:7474 | neo4j / neo4j_pass |
| FastAPI Docs | http://localhost:8000/docs | — |
| PostgreSQL | localhost:5433 | healthuser / healthpass / healthdb |
| Kafka | localhost:9092 | — |

---

## What This Project Demonstrates

**End-to-end pipeline thinking** — every layer connects deliberately to the next. Raw data informs ETL design. ETL output informs quality check thresholds. Quality checks determine what reaches entity resolution. Entity resolution informs graph structure. Graph structure drives API design.

**Production patterns** — quarantine instead of delete, idempotent loads with MERGE, parallel task execution, audit logging, schema separation, retry logic, CI/CD on every push. These are the same patterns used on real data engineering teams.

**Depth on hard problems** — entity resolution with blocking and similarity scoring, data skew with key salting, multi-hop graph traversal outperforming SQL joins. These are the topics that distinguish senior from junior data engineering thinking.

---

## About

Built by **Caleb Duah** — MS Computer Science, WPI.

This is Portfolio Project 1 of 3. Project 2 builds an e-commerce pipeline using Kinesis, Databricks, Redshift, Snowflake, and dbt. Project 3 builds a stock market streaming pipeline on Azure using Event Hubs, Databricks, and Synapse Analytics.

[GitHub](https://github.com/Caduah) · [caduah@wpi.edu](mailto:caduah@wpi.edu)
