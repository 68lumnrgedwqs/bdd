"""
config.py
=========
Central configuration module for the Real-time Fraud Detection System.
All tunable parameters, paths, and Spark settings live here – change once,
affect everywhere.

Windows note
------------
* HADOOP_HOME is set programmatically in spark_session.py.
* spark.local.dir points to a subdirectory inside the project to avoid
  Windows %TEMP% permission issues.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "Data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
LOG_DIR       = os.path.join(BASE_DIR, "logs")

PAYSIM_CSV    = os.path.join(DATA_DIR, "paysim.csv")
BLACKLIST_PATH = os.path.join(DATA_DIR, "blacklist.csv")          # generated at runtime
ALERT_OUTPUT  = os.path.join(OUTPUT_DIR, "fraud_alerts")
STREAM_INPUT  = os.path.join(DATA_DIR, "stream_input")            # simulated Kafka landing zone

# ---------------------------------------------------------------------------
# SparkSession tuning
# ---------------------------------------------------------------------------
SPARK_CONFIG = {
    # Executor memory - tune to your machine RAM
    "spark.driver.memory":                       "4g",
    "spark.executor.memory":                     "4g",
    "spark.executor.cores":                      "4",

    # Shuffle optimisation ------------------------------------------------------
    # Reduce default 200 partitions -> match your data size (PaySim ~500 MB)
    "spark.sql.shuffle.partitions":              "4",
    # Adaptive Query Execution: let Spark auto-coalesce small partitions
    "spark.sql.adaptive.enabled":               "false",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
    "spark.sql.adaptive.skewJoin.enabled":       "true",

    # Broadcast join threshold - blacklist is tiny, force broadcast
    "spark.sql.autoBroadcastJoinThreshold":      "20971520",  # 20 MB

    # Kryo serialiser - faster than Java default
    "spark.serializer":                          "org.apache.spark.serializer.KryoSerializer",
    "spark.kryoserializer.buffer.max":           "512m",

    # Windows: use a project-local temp dir to avoid permission issues with %TEMP%
    "spark.local.dir":                           os.path.join(BASE_DIR, "spark_tmp"),

    # Streaming micro-batch interval (seconds)
    "spark.sql.streaming.statefulOperator.checkCorrectness.enabled": "false",
}

# ---------------------------------------------------------------------------
# Streaming parameters
# ---------------------------------------------------------------------------
STREAM_TRIGGER_SECONDS  = 10        # micro-batch trigger interval
WATERMARK_DELAY         = "5 minutes"
WINDOW_DURATION         = "10 minutes"
WINDOW_SLIDE            = "2 minutes"

# Max files processed per micro-batch (rate control)
MAX_FILES_PER_TRIGGER   = 50

# ---------------------------------------------------------------------------
# ML parameters
# ---------------------------------------------------------------------------
LABEL_COL               = "isFraud"
FEATURES_COL            = "features"
PREDICTION_COL          = "prediction"
PROBABILITY_COL         = "probability"
FRAUD_SCORE_THRESHOLD   = 0.5       # probability cutoff for flagging

# GBT / RandomForest
ML_NUM_TREES            = 100
ML_MAX_DEPTH            = 8
ML_MAX_BINS             = 64
ML_SEED                 = 42

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# Columns used as numeric features after preprocessing
NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balanceOrigDiff",     # engineered
    "balanceDestDiff",     # engineered
    "errorBalanceOrig",    # engineered
    "errorBalanceDest",    # engineered
    "txn_count_window",   # windowed aggregate
    "total_amount_window", # windowed aggregate
]

CATEGORICAL_FEATURES = ["type"]     # StringIndexed + OneHot

ALL_FEATURE_COLS = NUMERIC_FEATURES + ["type_indexed"]
