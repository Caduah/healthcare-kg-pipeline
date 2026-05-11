import json
import random
import time
import uuid
import logging
import psycopg2
import os
from datetime import datetime
from kafka import KafkaProducer

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


def load_real_patients():
    """Load real patient IDs from the database."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, full_name FROM processed.patients LIMIT 100")
    patients = cur.fetchall()
    cur.close()
    conn.close()
    return patients


def load_real_providers():
    """Load real provider IDs from the database."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM processed.providers LIMIT 50")
    providers = cur.fetchall()
    cur.close()
    conn.close()
    return providers


def generate_clinical_event(patients, providers):
    """
    Generate a realistic synthetic clinical event.
    In production these would come from an EHR system in real time.
    """
    patient = random.choice(patients)
    provider = random.choice(providers)

    event_types = [
        "ADMISSION", "DISCHARGE", "LAB_RESULT",
        "MEDICATION_ORDER", "DIAGNOSIS_UPDATE"
    ]

    event_type = random.choice(event_types)

    clinical_notes = {
        "ADMISSION": [
            "Patient presents with chest pain and shortness of breath. History of hypertension.",
            "65-year-old male admitted with acute onset confusion and fever of 101.2F.",
            "Patient admitted following fall at home. No loss of consciousness reported.",
        ],
        "DISCHARGE": [
            "Patient discharged in stable condition. Follow up with PCP in 2 weeks.",
            "Discharge summary: treated for pneumonia, responding well to antibiotics.",
            "Patient discharged after successful knee replacement surgery.",
        ],
        "LAB_RESULT": [
            "HbA1c 8.2 - elevated, patient counseled on diabetes management.",
            "CBC shows WBC 12.4 - mild leukocytosis, monitoring for infection.",
            "Creatinine 1.8 mg/dL - mild renal impairment noted.",
        ],
        "MEDICATION_ORDER": [
            "Metformin 500mg twice daily prescribed for Type 2 diabetes management.",
            "Lisinopril 10mg daily for hypertension management.",
            "Atorvastatin 40mg at bedtime for hyperlipidemia.",
        ],
        "DIAGNOSIS_UPDATE": [
            "New diagnosis: Type 2 diabetes mellitus confirmed via HbA1c.",
            "Hypertension diagnosis added based on consistent elevated readings.",
            "Chronic kidney disease stage 2 diagnosed based on eGFR.",
        ],
    }

    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "patient_id": patient[0],
        "patient_name": patient[1],
        "provider_id": provider[0],
        "provider_name": provider[1],
        "facility_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "clinical_note": random.choice(clinical_notes[event_type]),
        "severity": random.choice(["LOW", "MEDIUM", "HIGH"]),
        "department": random.choice([
            "Emergency", "Cardiology", "Internal Medicine",
            "Orthopedics", "Neurology", "Oncology"
        ]),
    }
    return event


def run_producer(num_events=50, delay_seconds=0.5):
    """
    Produce clinical events to Kafka.
    In production this runs continuously from an EHR system.
    """
    log.info("Starting Kafka producer...")

    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )

    patients = load_real_patients()
    providers = load_real_providers()
    log.info(f"Loaded {len(patients)} patients and {len(providers)} providers")

    log.info(f"Producing {num_events} clinical events to Kafka...")
    for i in range(num_events):
        event = generate_clinical_event(patients, providers)

        producer.send(
            topic="clinical-events",
            key=event["event_type"],
            value=event,
        )

        if i % 10 == 0:
            log.info(f"  Sent {i+1}/{num_events} events — latest: {event['event_type']} for {event['patient_name']}")

        time.sleep(delay_seconds)

    producer.flush()
    producer.close()
    log.info(f"Producer complete — {num_events} events sent to clinical-events topic")


if __name__ == "__main__":
    run_producer(num_events=50, delay_seconds=0.2)