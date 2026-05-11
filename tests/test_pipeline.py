"""
Basic unit tests for the healthcare pipeline.
These test pure logic functions without requiring database connections.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_jaro_winkler_identical_strings():
    from entity_resolution.patient_matcher import jaro_winkler_similarity
    score = jaro_winkler_similarity("john smith", "john smith")
    assert score == 1.0


def test_jaro_winkler_different_strings():
    from entity_resolution.patient_matcher import jaro_winkler_similarity
    score = jaro_winkler_similarity("john smith", "jane doe")
    assert score < 0.8


def test_jaro_winkler_empty_strings():
    from entity_resolution.patient_matcher import jaro_winkler_similarity
    score = jaro_winkler_similarity("", "john smith")
    assert score == 0.0


def test_heuristic_score_perfect_match():
    from entity_resolution.patient_matcher import heuristic_score
    features = {
        "last_name_sim": 1.0,
        "first_name_sim": 1.0,
        "dob_match": 1.0,
        "gender_match": 1.0,
        "zip_match": 1.0,
        "state_match": 1.0,
    }
    score = heuristic_score(features)
    assert score == 1.0


def test_heuristic_score_no_match():
    from entity_resolution.patient_matcher import heuristic_score
    features = {
        "last_name_sim": 0.0,
        "first_name_sim": 0.0,
        "dob_match": 0.0,
        "gender_match": 0.0,
        "zip_match": 0.0,
        "state_match": 0.0,
    }
    score = heuristic_score(features)
    assert score == 0.0


def test_nlp_extracts_conditions():
    from enrichment.nlp_enricher import extract_medical_entities
    note = "Patient has hypertension and diabetes. Prescribed metformin."
    entities = extract_medical_entities(note)
    assert "hypertension" in entities["conditions"]
    assert "diabetes" in entities["conditions"]
    assert "metformin" in entities["medications"]


def test_nlp_handles_empty_note():
    from enrichment.nlp_enricher import extract_medical_entities
    entities = extract_medical_entities("")
    assert entities == {}


def test_pii_detection_finds_person():
    from enrichment.nlp_enricher import detect_pii
    result = detect_pii("Patient John Smith needs follow up.")
    assert result["pii_detected"] is True
    assert "PERSON" in result["pii_types"]


def test_pii_detection_anonymizes():
    from enrichment.nlp_enricher import detect_pii
    result = detect_pii("Call John Smith at his appointment.")
    assert "John Smith" not in result["anonymized_text"]