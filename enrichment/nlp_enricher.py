import json
import logging
import os
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load spaCy model once at module level
nlp = spacy.load("en_core_web_sm")

# Initialize Presidio engines for PII detection
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


def extract_medical_entities(text):
    """
    Extract medical entities from clinical notes using spaCy.
    Identifies persons, organizations, dates, and medical terms.
    """
    if not text:
        return {}

    doc = nlp(text)

    entities = {
        "persons": [],
        "organizations": [],
        "dates": [],
        "conditions": [],
        "medications": [],
        "raw_entities": [],
    }

    for ent in doc.ents:
        entities["raw_entities"].append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })

        if ent.label_ == "PERSON":
            entities["persons"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["dates"].append(ent.text)

    # Rule-based medical term extraction
    medical_keywords = {
        "conditions": [
            "diabetes", "hypertension", "pneumonia", "infection",
            "confusion", "pain", "fever", "fracture", "cancer",
            "failure", "disease", "disorder", "syndrome"
        ],
        "medications": [
            "metformin", "lisinopril", "atorvastatin", "aspirin",
            "insulin", "warfarin", "antibiotics", "medication",
            "prescribed", "dosage"
        ],
    }

    text_lower = text.lower()
    for category, keywords in medical_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                entities[category].append(keyword)

    entities["conditions"] = list(set(entities["conditions"]))
    entities["medications"] = list(set(entities["medications"]))

    return entities


def detect_pii(text):
    """
    Detect PII in clinical notes using Microsoft Presidio.
    Returns detected PII types and anonymized text.
    """
    if not text:
        return {"pii_detected": False, "pii_types": [], "anonymized_text": text}

    results = analyzer.analyze(
        text=text,
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS",
                  "MEDICAL_LICENSE", "US_SSN", "LOCATION"],
        language="en"
    )

    if not results:
        return {
            "pii_detected": False,
            "pii_types": [],
            "anonymized_text": text
        }

    pii_types = list(set([r.entity_type for r in results]))

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)

    return {
        "pii_detected": True,
        "pii_types": pii_types,
        "pii_count": len(results),
        "anonymized_text": anonymized.text
    }


def enrich_clinical_event(event):
    """
    Enrich a clinical event with NLP analysis.
    Extracts entities, detects PII, and adds structured fields.
    """
    note = event.get("clinical_note", "")

    # Extract medical entities
    entities = extract_medical_entities(note)

    # Detect and anonymize PII
    pii_result = detect_pii(note)

    enriched = {
        **event,
        "nlp_enrichment": {
            "extracted_entities": entities,
            "conditions_detected": entities.get("conditions", []),
            "medications_detected": entities.get("medications", []),
            "entity_count": len(entities.get("raw_entities", [])),
            "pii_detection": pii_result,
            "anonymized_note": pii_result.get("anonymized_text", note),
            "enriched_at": __import__("datetime").datetime.now().isoformat(),
        }
    }

    return enriched


def process_event_batch(events):
    """Process a batch of clinical events through the NLP pipeline."""
    enriched_events = []
    for event in events:
        try:
            enriched = enrich_clinical_event(event)
            enriched_events.append(enriched)
        except Exception as e:
            log.error(f"Failed to enrich event {event.get('event_id')}: {e}")
            enriched_events.append(event)
    return enriched_events


if __name__ == "__main__":
    test_note = """
    Patient John Smith, SSN 123-45-6789, presents with chest pain.
    History of hypertension and Type 2 diabetes.
    Prescribed Metformin 500mg twice daily.
    Follow up appointment scheduled for next week.
    """

    log.info("Testing NLP enricher...")
    log.info(f"Input: {test_note.strip()}")

    entities = extract_medical_entities(test_note)
    log.info(f"Extracted entities: {json.dumps(entities, indent=2)}")

    pii = detect_pii(test_note)
    log.info(f"PII detected: {pii['pii_types']}")
    log.info(f"Anonymized: {pii['anonymized_text']}")