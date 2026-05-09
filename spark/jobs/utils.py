import os
from pyspark.sql import SparkSession
from dotenv import load_dotenv

load_dotenv()


def get_spark_session(app_name="healthcare-pipeline"):
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.driver.memory", "2g") \
        .config("spark.jars", os.path.abspath("spark/jars/postgresql-42.7.3.jar")) \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()


def get_jdbc_url():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5433")
    db   = os.getenv("POSTGRES_DB", "healthdb")
    return f"jdbc:postgresql://{host}:{port}/{db}"


def get_jdbc_props():
    return {
        "user":     os.getenv("POSTGRES_USER", "healthuser"),
        "password": os.getenv("POSTGRES_PASSWORD", "healthpass"),
        "driver":   "org.postgresql.Driver"
    }


def read_table(spark, schema, table):
    return spark.read \
        .format("jdbc") \
        .option("url", get_jdbc_url()) \
        .option("dbtable", f"{schema}.{table}") \
        .options(**get_jdbc_props()) \
        .load()


def write_table(df, table, mode="overwrite"):
    df.write \
        .format("jdbc") \
        .option("url", get_jdbc_url()) \
        .option("dbtable", f"processed.{table}") \
        .options(**get_jdbc_props()) \
        .mode(mode) \
        .save()


def salt_join(large_df, small_df, large_key, small_key,
              salt_factor=10, join_type="left"):
    """
    Skew-safe join using key salting.
    Use when one key in large_df has far more records than others.

    Args:
        large_df    : the large DataFrame with the hot key
        small_df    : the small lookup DataFrame
        large_key   : join column name on large_df
        small_key   : join column name on small_df
        salt_factor : number of salt buckets (default 10)
        join_type   : left, inner, etc.

    Example:
        result = salt_join(
            encounters_df, providers_df,
            large_key="provider",
            small_key="id",
            salt_factor=10
        )
    """
    from pyspark.sql import functions as F

    salt_col = "_salt_key"

    large_salted = large_df.withColumn(
        salt_col,
        F.concat(
            F.col(large_key).cast("string"),
            F.lit("_"),
            (F.rand() * salt_factor).cast("int").cast("string")
        )
    )

    small_exploded = small_df.withColumn(
        "_salt_val",
        F.explode(
            F.array([F.lit(i) for i in range(salt_factor)])
        )
    ).withColumn(
        salt_col,
        F.concat(
            F.col(small_key).cast("string"),
            F.lit("_"),
            F.col("_salt_val").cast("string")
        )
    ).drop("_salt_val")

    result = large_salted.join(
        F.broadcast(small_exploded),
        salt_col,
        join_type
    ).drop(salt_col)

    return result