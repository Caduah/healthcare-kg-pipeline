import os
import uuid
import logging
import psycopg2
import pandas as pd
import mlflow
import jellyfish

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


def load_providers():
    conn = get_conn()
    df = pd.read_sql("""
        SELECT
            id,
            name_norm,
            name_clean,
            speciality_norm,
            npi,
            has_npi,
            city_norm,
            state_upper,
            zip_clean,
            organization,
            source_system
        FROM processed.providers
        WHERE id IS NOT NULL
        AND name_norm IS NOT NULL
    """, conn)
    conn.close()
    log.info(f"Loaded {len(df)} providers for matching")
    return df


def jaro_winkler_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    try:
        return jellyfish.jaro_winkler_similarity(str(s1), str(s2))
    except Exception:
        return 0.0


def generate_provider_blocks(df):
    """
    Block on first 3 letters of provider name.
    Providers with similar names are candidates.
    NPI exact match is handled separately as a rule.
    """
    log.info("Generating provider candidate blocks...")
    df = df.copy()
    df["block_key"] = df["name_norm"].str[:3].fillna("unk")

    candidates = []
    blocks = df.groupby("block_key")

    for block_key, group in blocks:
        if len(group) < 2:
            continue
        ids = group["id"].tolist()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                candidates.append((ids[i], ids[j], block_key))

    log.info(f"Generated {len(candidates)} provider candidate pairs")
    return candidates


def provider_rule_match(row1, row2):
    """
    Rule 1: Exact NPI match — definitive match.
    NPI is a unique 10-digit identifier issued by CMS.
    Two providers sharing an NPI are the same provider.
    """
    npi1 = row1.get("npi")
    npi2 = row2.get("npi")
    if (npi1 and npi2
            and str(npi1).strip() == str(npi2).strip()
            and len(str(npi1).strip()) >= 10):
        return True, "exact_npi", 1.0

    return False, None, 0.0


def provider_heuristic_score(row1, row2):
    """
    Feature-based scoring for providers without NPI match.
    Name similarity is the primary signal.
    Specialty and location are secondary signals.
    """
    name_sim = jaro_winkler_similarity(
        row1.get("name_norm"), row2.get("name_norm")
    )
    spec_match = float(
        row1.get("speciality_norm") is not None
        and row2.get("speciality_norm") is not None
        and row1.get("speciality_norm") == row2.get("speciality_norm")
    )
    state_match = float(
        row1.get("state_upper") is not None
        and row2.get("state_upper") is not None
        and row1.get("state_upper") == row2.get("state_upper")
    )
    city_sim = jaro_winkler_similarity(
        row1.get("city_norm"), row2.get("city_norm")
    )

    score = (
        name_sim    * 0.50 +
        spec_match  * 0.25 +
        state_match * 0.15 +
        city_sim    * 0.10
    )
    return round(score, 4)


def run_provider_matching(**context):
    run_id = str(uuid.uuid4())[:8]
    log.info(f"Starting provider matching run: {run_id}")

    mlflow.set_experiment("healthcare-provider-er")

    with mlflow.start_run(run_name=f"provider_er_{run_id}"):

        df = load_providers()
        mlflow.log_param("total_providers", len(df))

        candidates = generate_provider_blocks(df)
        mlflow.log_param("candidate_pairs", len(candidates))

        provider_index = df.set_index("id").to_dict("index")

        results = []
        rule_matches = 0
        feature_matches = 0

        for pid1, pid2, block_key in candidates:
            if pid1 not in provider_index or pid2 not in provider_index:
                continue

            row1 = provider_index[pid1]
            row2 = provider_index[pid2]

            is_rule_match, match_type, rule_score = provider_rule_match(
                row1, row2
            )
            h_score = provider_heuristic_score(row1, row2)
            final_score = max(rule_score, h_score)
            is_match = is_rule_match or final_score >= 0.80

            if is_rule_match:
                rule_matches += 1
                confidence = "HIGH"
            elif final_score >= 0.80:
                feature_matches += 1
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            if is_match:
                results.append({
                    "provider_id_1": pid1,
                    "provider_id_2": pid2,
                    "match_type": match_type or "feature_based",
                    "rule_score": rule_score,
                    "name_similarity": jaro_winkler_similarity(
                        row1.get("name_norm"), row2.get("name_norm")
                    ),
                    "speciality_match": bool(
                        row1.get("speciality_norm") == row2.get("speciality_norm")
                    ),
                    "npi_match": is_rule_match,
                    "final_score": final_score,
                    "is_match": True,
                    "confidence": confidence,
                    "run_id": run_id,
                })

        log.info(f"Found {len(results)} provider matches")
        log.info(f"  Rule-based (NPI): {rule_matches}")
        log.info(f"  Feature-based: {feature_matches}")

        mlflow.log_metric("total_candidates", len(candidates))
        mlflow.log_metric("total_matches", len(results))
        mlflow.log_metric("rule_matches", rule_matches)
        mlflow.log_metric("feature_matches", feature_matches)

        if results:
            conn = get_conn()
            cur = conn.cursor()
            for r in results:
                cur.execute("""
                    INSERT INTO entity_resolution.provider_matches
                    (provider_id_1, provider_id_2, match_type, rule_score,
                     name_similarity, speciality_match, npi_match,
                     final_score, is_match, confidence, run_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    r["provider_id_1"], r["provider_id_2"], r["match_type"],
                    r["rule_score"], r["name_similarity"],
                    r["speciality_match"], r["npi_match"],
                    r["final_score"], r["is_match"],
                    r["confidence"], r["run_id"]
                ))
            conn.commit()
            cur.close()
            conn.close()
            log.info(f"Saved {len(results)} matches to entity_resolution.provider_matches")
        else:
            log.info("No provider matches found above threshold")

        return {
            "run_id": run_id,
            "total_candidates": len(candidates),
            "total_matches": len(results),
            "rule_matches": rule_matches,
            "feature_matches": feature_matches,
        }


if __name__ == "__main__":
    result = run_provider_matching()
    print("\nProvider Entity Resolution Complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")