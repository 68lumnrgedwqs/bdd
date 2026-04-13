"""
fraud_streaming.py
==================
PySpark Structured Streaming job - the real-time fraud detection engine.

Architecture
------------

    [JSON files / Kafka topic]
           |
           v
    [readStream  (FileSource / KafkaSource)]
           |
           v
    [Feature Engineering  - column transforms, no shuffle]
           |
           v
    [Watermark + Time Window Aggregation  - velocity features]
           |
           v
    [Broadcast Join vs. Blacklist  - zero shuffle on large side]
           |
           v
    [MLlib PipelineModel.transform()  - distributed inference]
           |
           v
    [Fraud Enrichment + Alert Routing]
           |
           +--> [Console sink  - demo / debug]
           +--> [File sink (JSON)  - persistent alert store]

Scalability (10x data growth)
------------------------------
1. Spark auto-partitions the stream based on available cores.
2. The blacklist broadcast never needs to change - it stays in each executor's
   memory regardless of stream throughput.
3. Watermark-based state management bounds memory usage even under high load.
4. Switching from FileSource to KafkaSource is a one-line change (see comment
   in read_stream()).
5. Horizontal scale: add more executors on YARN / k8s - no code change.

Usage
-----
    python fraud_streaming.py

    # On a cluster:
    spark-submit \\
        --master yarn \\
        --executor-memory 8g \\
        --executor-cores 4 \\
        --num-executors 10 \\
        fraud_streaming.py
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fraud_streaming")

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

from config import (
    STREAM_INPUT, ALERT_OUTPUT, CHECKPOINT_DIR, MODEL_DIR,
    WATERMARK_DELAY, WINDOW_DURATION, WINDOW_SLIDE,
    STREAM_TRIGGER_SECONDS, MAX_FILES_PER_TRIGGER,
    LABEL_COL, PREDICTION_COL, PROBABILITY_COL, FRAUD_SCORE_THRESHOLD,
    ALL_FEATURE_COLS, FEATURES_COL
)
from spark_session import get_spark_session
from fraud_engine import (
    paysim_schema, load_blacklist, load_model,
    feature_engineering, join_with_blacklist
)

os.makedirs(ALERT_OUTPUT,  exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Stream Source
# -----------------------------------------------------------------------------

def read_stream(spark):
    """
    Read from simulated file-based stream (interchangeable with Kafka).

    To switch to Kafka, replace with:
    ------------------------------------------------------------
    stream_df = (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "broker:9092")
        .option("subscribe", "transactions")
        .option("startingOffsets", "latest")
        .load()
        .select(F.from_json(
            F.col("value").cast("string"),
            paysim_schema()
        ).alias("data"))
        .select("data.*"))
    ------------------------------------------------------------
    """
    logger.info("Opening stream from: %s", STREAM_INPUT)

    stream_df = (
        spark.readStream
             .format("json")
             .schema(paysim_schema())
             .option("maxFilesPerTrigger", MAX_FILES_PER_TRIGGER)   # rate control
             .option("cleanSource", "off")                           # keep files for replay
             .load(STREAM_INPUT)
    )
    return stream_df


# -----------------------------------------------------------------------------
# 2. Streaming Feature Engineering
# -----------------------------------------------------------------------------

def apply_streaming_features(df):
    """
    Apply stateless column transforms (safe in streaming context).

    Window aggregates that rely on ordering (add_window_features) can NOT be
    used directly in Structured Streaming because Window.orderBy is not
    supported in streaming mode.  Instead, we use the built-in
    window() function with watermark to compute velocity features over
    event-time windows - this IS supported and state is bounded by the watermark.
    """
    # -- Stateless transforms ------------------------------------------------
    df = feature_engineering(df)

    # -- Event-time watermark  (drop late data > 5 min) ----------------------
    df = df.withWatermark("event_time", WATERMARK_DELAY)

    return df


def compute_velocity_features(df):
    """
    Compute per-account velocity in a sliding event-time window.

    Spark Structured Streaming supports groupBy(window(...)) natively.
    State is stored in RocksDB (or memory) and cleaned up automatically
    by the watermark -> bounded memory even for long-running streams.

    Returns a separate DataFrame that must be joined back to the main stream.
    NOTE: stream-stream joins require matching watermarks; here we use a
    static join since the window aggregates are written to a sink and read
    back.  In a production system you would use foreachBatch to perform the
    enrichment.
    """
    velocity_df = (
        df.groupBy(
            F.window("event_time", WINDOW_DURATION, WINDOW_SLIDE),
            F.col("nameOrig")
        )
        .agg(
            F.count("amount").alias("txn_count_window"),
            F.sum("amount").alias("total_amount_window"),
            F.max("amount").alias("max_amount_window"),
            F.countDistinct("nameDest").alias("distinct_dest_window")
        )
    )
    return velocity_df


# -----------------------------------------------------------------------------
# 3. foreachBatch - the micro-batch processing function
# -----------------------------------------------------------------------------

def make_batch_processor(blacklist_df, model):
    """
    Returns a foreachBatch function with blacklist and model captured in closure.

    Why foreachBatch?
    -----------------
    * Allows us to perform batch-mode operations (Window orderBy, complex joins,
      ML inference) that are not 100% supported in streaming-native APIs.
    * Each micro-batch is a regular Spark DataFrame - full API available.
    * The model.transform() call stays distributed - no .collect() / .toPandas().
    * Output sinks can vary per micro-batch (e.g., write alerts to DB, normal
      transactions to cold storage).
    """

    def process_batch(batch_df, batch_id):
        logger.info("Processing batch_id=%d ...", batch_id)

        if batch_df.isEmpty():
            logger.info("Batch %d is empty. Skipping.", batch_id)
            return

        # -- 2a. Feature engineering (stateless) -----------------------------
        df = feature_engineering(batch_df)

        # -- 2b. Velocity features (Window on batch -> safe in foreachBatch) --
        from pyspark.sql.window import Window
        w = (Window.partitionBy("nameOrig")
                   .orderBy("step")
                   .rowsBetween(-10, Window.currentRow))

        df = (df
            .withColumn("txn_count_window",    F.count("amount").over(w))
            .withColumn("total_amount_window",  F.sum("amount").over(w))
        )

        # -- 2c. Broadcast join with blacklist -------------------------------
        df = join_with_blacklist(df, blacklist_df)

        # -- 2d. ML inference (distributed, no toPandas) ---------------------
        predictions = model.transform(df)

        # Extract fraud probability from probability vector (index 1 = fraud class)
        get_fraud_prob = F.udf(lambda v: float(v[1]) if v is not None else 0.0,
                               returnType=__import__("pyspark.sql.types", fromlist=["DoubleType"]).DoubleType())

        predictions = (predictions
            .withColumn("fraud_probability", get_fraud_prob(F.col("probability")))
            .withColumn("is_fraud_predicted",
                        F.col("fraud_probability") >= FRAUD_SCORE_THRESHOLD)
            # -- Rule-based override: blacklisted destination -> always flag --
            .withColumn("is_fraud_final",
                        F.col("is_fraud_predicted") | F.col("blacklisted"))
        )

        # -- 2e. Alert enrichment ---------------------------------------------
        fraud_alerts = (
            predictions
            .filter(F.col("is_fraud_final"))
            .select(
                F.current_timestamp().alias("alert_time"),
                F.lit(batch_id).alias("batch_id"),
                F.col("nameOrig").alias("sender_account"),
                F.col("nameDest").alias("receiver_account"),
                F.col("type").alias("transaction_type"),
                F.col("amount"),
                F.col("fraud_probability"),
                F.col("blacklisted"),
                F.col("bl_reason").alias("blacklist_reason"),
                F.col("is_fraud_predicted").alias("ml_flag"),
                F.col("isFraud").alias("ground_truth"),          # available during dev only
                F.col("event_time")
            )
        )

        alert_count = fraud_alerts.count()
        total_count = predictions.count()
        logger.info("Batch %d: %d/%d transactions flagged as fraud (%.1f%%)",
                    batch_id, alert_count, total_count,
                    100 * alert_count / max(total_count, 1))

        if alert_count > 0:
            # -- Write alerts to persistent JSON sink ------------------------
            alert_path = os.path.join(ALERT_OUTPUT, f"batch_{batch_id:06d}")
            (fraud_alerts
                .coalesce(1)
                .write
                .mode("overwrite")
                .json(alert_path))
            logger.info("Alerts written to: %s", alert_path)

            # -- Print sample to console --------------------------------------
            print(f"\n{'='*55}")
            print(f"  [ALERT] FRAUD ALERTS - Batch {batch_id}  ({alert_count} alerts)")
            print(f"{'='*55}")
            (fraud_alerts
                .select("sender_account", "receiver_account",
                        "transaction_type", "amount",
                        "fraud_probability", "blacklisted", "ml_flag")
                .show(20, truncate=False))

        # -- Execution plan logged at DEBUG level -----------------------------
        if logger.isEnabledFor(logging.DEBUG):
            predictions.explain(mode="formatted")

    return process_batch


# -----------------------------------------------------------------------------
# 4. Streaming Query
# -----------------------------------------------------------------------------

def start_streaming_query(spark, stream_df, blacklist_df, model):
    """
    Wire up the Structured Streaming query with foreachBatch sink.

    Checkpoint location is mandatory for exactly-once semantics and allows
    the query to resume after a failure without reprocessing old data.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "fraud_stream_v1")
    os.makedirs(checkpoint_path, exist_ok=True)

    batch_processor = make_batch_processor(blacklist_df, model)

    query = (
        stream_df
        .writeStream
        .outputMode("append")
        .foreachBatch(batch_processor)
        .option("checkpointLocation", checkpoint_path)
        .trigger(processingTime=f"{STREAM_TRIGGER_SECONDS} seconds")
        .start()
    )

    logger.info("Streaming query started. Query ID: %s", query.id)
    return query


# -----------------------------------------------------------------------------
# 5. Supplementary: Windowed Alert Stream (native streaming API)
# -----------------------------------------------------------------------------

def start_velocity_monitor(spark, stream_df):
    """
    A separate streaming query that monitors per-account velocity in real time.

    Writes aggregated velocity stats to console every trigger interval.
    This demonstrates native Structured Streaming window + watermark support.
    """
    df = stream_df.withWatermark("event_time", WATERMARK_DELAY)

    velocity_query = (
        df.groupBy(
            F.window("event_time", WINDOW_DURATION, WINDOW_SLIDE),
            F.col("nameOrig")
        )
        .agg(
            F.count("*").alias("txn_count"),
            F.sum("amount").alias("total_sent"),
            F.countDistinct("nameDest").alias("unique_destinations")
        )
        # Flag accounts with high velocity
        .filter(F.col("txn_count") >= 3)
        .writeStream
        .outputMode("update")
        .format("console")
        .option("truncate", "false")
        .option("numRows", 10)
        .option("checkpointLocation",
                os.path.join(CHECKPOINT_DIR, "velocity_monitor"))
        .trigger(processingTime=f"{STREAM_TRIGGER_SECONDS} seconds")
        .start()
    )

    logger.info("Velocity monitor started. Query ID: %s", velocity_query.id)
    return velocity_query


# -----------------------------------------------------------------------------
# 6. Main
# -----------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  FRAUD DETECTION SYSTEM - STREAMING JOB")
    print("="*60 + "\n")

    spark = get_spark_session("FraudDetection-Streaming")

    # -- Load static assets (broadcast-ready) --------------------------------
    model_path = os.path.join(MODEL_DIR, "gbt_fraud_model")
    logger.info("Loading model from: %s", model_path)
    model = load_model(model_path)

    logger.info("Loading blacklist ...")
    blacklist_df = load_blacklist(spark)

    # -- Verify broadcast join plan -------------------------------------------
    from pyspark.sql import functions as F
    sample_txn = spark.range(1).select(
        F.lit("C123").alias("nameDest"),
        F.lit(100.0).alias("amount")
    )
    sample_joined = join_with_blacklist(sample_txn, blacklist_df)
    logger.info("Verifying broadcast join plan ↓")
    sample_joined.explain(mode="simple")

    # -- Open stream source ---------------------------------------------------
    stream_df = read_stream(spark)
    stream_df = apply_streaming_features(stream_df)

    # -- Start queries --------------------------------------------------------
    fraud_query    = start_streaming_query(spark, stream_df, blacklist_df, model)
    velocity_query = start_velocity_monitor(spark, stream_df)

    logger.info("All queries running. Waiting for termination ...")
    logger.info("Press Ctrl+C to stop.")

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Stopping streaming queries ...")
        fraud_query.stop()
        velocity_query.stop()
        logger.info("Streaming job terminated cleanly.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
