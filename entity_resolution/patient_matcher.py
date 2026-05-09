import os
import uuid
import logging
import psycopg2
import pandas as pd
import numpy as np
import mlflow
import jellyfish
from datetime import datetime

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


def load_patients():
    conn = get_conn()
    df = pd.read_sql("""
        SELECT
            id,
            first_norm,
            last_norm,
            full_name_norm,
            birthdate,
            gender,
            zip_clean,
            state_upper,
            source_system
        FROM processed.patients
        WHERE id IS NOT NULL
        AND last_norm IS NOT NULL
    """, conn)
    conn.close()
    log.info(f"Loaded {len(df)} patients for matching")
    return df


def generate_blocks(df):
    """
    Blocking reduces candidate pairs using last name prefix.
    Without blocking: 1129 x 1129 = 1.27 million pairs.
    With blocking on last_name[:3]: only a few thousand pairs.
    This is the most critical performance optimization in ER.
    """
    log.info("Generating candidate blocks on last name prefix...")
    df = df.copy()
    df["block_key"] = df["last_norm"].str[:3].fillna("unk")

    candidates = []
    blocks = df.groupby("block_key")

    for block_key, group in blocks:
        if len(group) < 2:
            continue
        ids = group["id"].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                candidates.append((ids[i], ids[j], block_key))

    log.info(f"Generated {len(candidates)} candidate pairs from {len(blocks)} blocks")
    return candidates, df


def jaro_winkler_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    try:
        return jellyfish.jaro_winkler_similarity(str(s1), str(s2))
    except Exception:
        return 0.0


def compute_features(row1, row2):
    features = {}
    features["first_name_sim"] = jaro_winkler_similarity(
        row1.get("first_norm"), row2.get("first_norm")
    )
    features["last_name_sim"] = jaro_winkler_similarity(
        row1.get("last_norm"), row2.get("last_norm")
    )
    features["full_name_sim"] = jaro_winkler_similarity(
        row1.get("full_name_norm"), row2.get("full_name_norm")
    )
    b1 = row1.get("birthdate")
    b2 = row2.get("birthdate")
    features["dob_match"] = float(
        b1 is not None and b2 is not None and b1 == b2
    )
    g1 = row1.get("gender")
    g2 = row2.get("gender")
    features["gender_match"] = float(
        g1 is not None and g2 is not None and g1 == g2
    )
    z1 = str(row1.get("zip_clean") or "")[:5]
    z2 = str(row2.get("zip_clean") or "")[:5]
    features["zip_match"] = float(z1 == z2 and z1 != "")
    s1 = row1.get("state_upper")
    s2 = row2.get("state_upper")
    features["state_match"] = float(
        s1 is not None and s2 is not None and s1 == s2
    )
    return features


def heuristic_score(features):
    """
    Weighted heuristic score.
    DOB + last name is the strongest signal in healthcare ER.
    Weights based on healthcare entity resolution research.
    """
    score = (
        features["last_name_sim"]  * 0.30 +
        features["first_name_sim"] * 0.20 +
        features["dob_match"]      * 0.25 +
        features["gender_match"]   * 0.10 +
        features["zip_match"]      * 0.10 +
        features["state_match"]    * 0.05
    )
    return round(score, 4)


def rule_based_match(row1, row2):
    """
    Exact rule-based matching — highest precision.
    If these rules fire it is almost certainly a match.
    """
    b1 = row1.get("birthdate")
    b2 = row2.get("birthdate")

    if (b1 is not None and b2 is not None
            and b1 == b2
            and row1.get("last_norm") == row2.get("last_norm")
            and str(row1.get("first_norm", ""))[:1] == str(row2.get("first_norm", ""))[:1]):
        return True, "dob_lastname_initial", 1.0

    name_sim = jaro_winkler_similarity(
        row1.get("full_name_norm"), row2.get("full_name_norm")
    )
    if (name_sim > 0.95
            and b1 is not None and b2 is not None and b1 == b2):
        return True, "high_name_sim_dob", 0.98

    return False, None, 0.0


def run_patient_matching(**context):
    run_id = str(uuid.uuid4())[:8]
    log.info(f"Starting patient matching run: {run_id}")

    mlflow.set_experiment("healthcare-patient-er")

    with mlflow.start_run(run_name=f"patient_er_{run_id}"):

        df = load_patients()
        mlflow.log_param("total_patients", len(df))

        candidates, df_indexed = generate_blocks(df)
        mlflow.log_param("candidate_pairs", len(candidates))
        mlflow.log_param("blocking_strategy", "last_name_prefix_3")

        patient_index = df.set_index("id").to_dict("index")

        results = []
        rule_matches = 0
        feature_matches = 0

        for pid1, pid2, block_key in candidates:
            if pid1 not in patient_index or pid2 not in patient_index:
                continue

            row1 = patient_index[pid1]
            row2 = patient_index[pid2]

            is_rule_match, match_type, rule_score = rule_based_match(row1, row2)
            features = compute_features(row1, row2)
            h_score = heuristic_score(features)

            final_score = max(rule_score, h_score)
            is_match = is_rule_match or final_score >= 0.75

            if is_rule_match:
                rule_matches += 1
                confidence = "HIGH"
            elif final_score >= 0.75:
                feature_matches += 1
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            if is_match:
                results.append({
                    "patient_id_1": pid1,
                    "patient_id_2": pid2,
                    "match_type": match_type or "feature_based",
                    "rule_score": rule_score,
                    "name_similarity": features["full_name_sim"],
                    "dob_match": bool(features["dob_match"]),
                    "gender_match": bool(features["gender_match"]),
                    "zip_similarity": features["zip_match"],
                    "final_score": final_score,
                    "is_match": True,
                    "confidence": confidence,
                    "run_id": run_id,
                })

        log.info(f"Found {len(results)} matches")
        log.info(f"  Rule-based: {rule_matches}")
        log.info(f"  Feature-based: {feature_matches}")

        mlflow.log_metric("total_candidates", len(candidates))
        mlflow.log_metric("total_matches", len(results))
        mlflow.log_metric("rule_matches", rule_matches)
        mlflow.log_metric("feature_matches", feature_matches)
        mlflow.log_param("heuristic_threshold", 0.75)

        if results:
            conn = get_conn()
            cur = conn.cursor()
            for r in results:
                cur.execute("""
                    INSERT INTO entity_resolution.patient_matches
                    (patient_id_1, patient_id_2, match_type, rule_score,
                     name_similarity, dob_match, gender_match, zip_similarity,
                     final_score, is_match, confidence, run_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    r["patient_id_1"], r["patient_id_2"], r["match_type"],
                    r["rule_score"], r["name_similarity"], r["dob_match"],
                    r["gender_match"], r["zip_similarity"], r["final_score"],
                    r["is_match"], r["confidence"], r["run_id"]
                ))
            conn.commit()
            cur.close()
            conn.close()
            log.info(f"Saved {len(results)} matches to entity_resolution.patient_matches")

        return {
            "run_id": run_id,
            "total_candidates": len(candidates),
            "total_matches": len(results),
            "rule_matches": rule_matches,
            "feature_matches": feature_matches,
        }


if __name__ == "__main__":
    result = run_patient_matching()
    print("\nEntity Resolution Complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")