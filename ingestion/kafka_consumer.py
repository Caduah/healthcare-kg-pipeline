import json
import logging
import os
import psycopg2
from datetime import datetime
from kafka import KafkaConsumer

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


def create_streaming_table():
    """Create table to store processed streaming events."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed.clinical_events_stream (
            id SERIAL PRIMARY KEY,
            event_id VARCHAR UNIQUE,
            event_type VARCHAR,
            patient_id VARCHAR,
            patient_name VARCHAR,
            provider_id VARCHAR,
            provider_name VARCHAR,
            facility_id VARCHAR,
            department VARCHAR,
            severity VARCHAR,
            clinical_note TEXT,
            anonymized_note TEXT,
            conditions_detected TEXT[],
            medications_detected TEXT[],
            pii_detected BOOLEAN,
            pii_types TEXT[],
            event_timestamp TIMESTAMP,
            processed_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    log.info("Streaming table ready")


def save_event_to_postgres(event, enriched_data):
    """Save a processed event to PostgreSQL."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        nlp = enriched_data.get("nlp_enrichment", {})
        pii = nlp.get("pii_detection", {})

        cur.execute("""
            INSERT INTO processed.clinical_events_stream
            (event_id, event_type, patient_id, patient_name,
             provider_id, provider_name, facility_id, department,
             severity, clinical_note, anonymized_note,
             conditions_detected, medications_detected,
             pii_detected, pii_types, event_timestamp)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (event_id) DO NOTHING
        """, (
            event.get("event_id"),
            event.get("event_type"),
            event.get("patient_id"),
            event.get("patient_name"),
            event.get("provider_id"),
            event.get("provider_name"),
            event.get("facility_id"),
            event.get("department"),
            event.get("severity"),
            event.get("clinical_note"),
            nlp.get("anonymized_note"),
            nlp.get("conditions_detected", []),
            nlp.get("medications_detected", []),
            pii.get("pii_detected", False),
            pii.get("pii_types", []),
            datetime.fromisoformat(event.get("timestamp", datetime.now().isoformat())),
        ))
        conn.commit()
    except Exception as e:
        log.error(f"Failed to save event: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def run_consumer(max_messages=50):
    """
    Consume clinical events from Kafka, enrich with NLP, save to Postgres.
    In production this runs continuously.
    """
    from enrichment.nlp_enricher import enrich_clinical_event

    create_streaming_table()

    log.info("Starting Kafka consumer...")
    consumer = KafkaConsumer(
        "clinical-events",
        bootstrap_servers=["localhost:9092"],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="healthcare-consumer-group",
        consumer_timeout_ms=10000,
    )

    processed = 0
    pii_detected = 0

    for message in consumer:
        event = message.value
        log.info(f"Received: {event['event_type']} for {event['patient_name']}")

        enriched = enrich_clinical_event(event)
        nlp = enriched.get("nlp_enrichment", {})
        pii = nlp.get("pii_detection", {})

        if pii.get("pii_detected"):
            pii_detected += 1
            log.warning(f"  PII detected: {pii.get('pii_types')}")

        conditions = nlp.get("conditions_detected", [])
        if conditions:
            log.info(f"  Conditions found: {conditions}")

        save_event_to_postgres(event, enriched)
        processed += 1

        if processed >= max_messages:
            break

    consumer.close()
    log.info(f"Consumer complete — processed {processed} events, PII detected in {pii_detected}")
    return processed


if __name__ == "__main__":
    run_consumer(max_messages=50)