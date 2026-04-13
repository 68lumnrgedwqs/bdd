"""
fraud_engine.py
===============
Core business-logic layer:

  1. schema()           - canonical PaySim schema (avoids CSV header inference).
  2. build_blacklist()  - generates a synthetic blacklist + caches it for broadcast.
  3. feature_engineering() - pure Spark Column-level transformations (no Pandas).
  4. build_ml_pipeline()   - Spark MLlib pipeline (GBTClassifier).
  5. load_model()           - helper to load a pre-trained PipelineModel.

Why no .toPandas() anywhere?
------------------------------
All transformations stay in the distributed DAG.  We never pull data to the
driver.  Even the blacklist is read as a Spark DataFrame and broadcast via the
Spark SQL planner (autoBroadcastJoinThreshold) rather than Python dicts.
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, LongType
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier

from config import (
    BLACKLIST_PATH, LABEL_COL, FEATURES_COL, PREDICTION_COL,
    PROBABILITY_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    ALL_FEATURE_COLS, ML_NUM_TREES, ML_MAX_DEPTH, ML_MAX_BINS, ML_SEED,
    WATERMARK_DELAY, WINDOW_DURATION, WINDOW_SLIDE
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Schema
# -----------------------------------------------------------------------------

def paysim_schema() -> StructType:
    """
    Explicit schema for PaySim CSV.

    Providing a schema instead of inferSchema=True avoids a costly full-scan
    of the file just to guess types - critical when data is 10x larger.
    """
    return StructType([
        StructField("step",            IntegerType(), True),  # 1 hour unit
        StructField("type",            StringType(),  True),  # TRANSFER, CASH_OUT, ...
        StructField("amount",          DoubleType(),  True),
        StructField("nameOrig",        StringType(),  True),  # sender account
        StructField("oldbalanceOrg",   DoubleType(),  True),
        StructField("newbalanceOrig",  DoubleType(),  True),
        StructField("nameDest",        StringType(),  True),  # receiver account
        StructField("oldbalanceDest",  DoubleType(),  True),
        StructField("newbalanceDest",  DoubleType(),  True),
        StructField("isFraud",         IntegerType(), True),  # ground-truth label
        StructField("isFlaggedFraud",  IntegerType(), True),
    ])


# -----------------------------------------------------------------------------
# 2. Blacklist
# -----------------------------------------------------------------------------

def build_blacklist(spark: SparkSession, df: DataFrame) -> DataFrame:
    """
    Derive a blacklist from known fraudulent destinations.

    In production this would be a lookup table from a risk database or an
    external API.  Here we synthesise it from the training labels so the demo
    is self-contained.

    The result is cached with MEMORY_AND_DISK so it survives eviction, and
    Spark's autoBroadcast will automatically pick it up for hash-broadcast
    joins (threshold set in config.py).

    Returns
    -------
    blacklist_df : DataFrame with columns [account_id, risk_score, reason]
    """
    blacklist_df = (
        df.filter(F.col("isFraud") == 1)
          .select(
              F.col("nameDest").alias("account_id"),
              F.lit(1.0).alias("risk_score"),
              F.lit("confirmed_fraud_destination").alias("reason")
          )
          .distinct()
    )

    # Persist so it is not recomputed for every streaming micro-batch
    blacklist_df.cache()
    count = blacklist_df.count()   # materialise the cache
    logger.info("Blacklist built: %d accounts", count)

    # Also write to disk so the streaming job can load it without recomputing
    (blacklist_df
        .coalesce(1)                   # single file -> easy to read back
        .write.mode("overwrite")
        .option("header", "true")
        .csv(BLACKLIST_PATH))

    return blacklist_df


def load_blacklist(spark: SparkSession) -> DataFrame:
    """
    Load the blacklist CSV from disk and cache it.
    This is used by the streaming job so it does not depend on the training data.
    """
    blacklist_schema = StructType([
        StructField("account_id", StringType(), True),
        StructField("risk_score",  DoubleType(),  True),
        StructField("reason",      StringType(),  True),
    ])
    bl = (spark.read
               .option("header", "true")
               .schema(blacklist_schema)
               .csv(BLACKLIST_PATH))
    bl.cache()
    bl.count()   # materialise
    logger.info("Blacklist loaded from disk and cached.")
    return bl


# -----------------------------------------------------------------------------
# 3. Feature Engineering
# -----------------------------------------------------------------------------

def feature_engineering(df: DataFrame) -> DataFrame:
    """
    Pure Spark column-level transformations - runs in the distributed DAG.

    New columns created
    -------------------
    balanceOrigDiff  : how much the sender's balance changed
    balanceDestDiff  : how much the receiver's balance changed
    errorBalanceOrig : discrepancy in sender's expected balance after tx
    errorBalanceDest : discrepancy in receiver's expected balance after tx
    event_time       : synthetic timestamp derived from `step` (1 step = 1 hour)
    """
    df = (df
        # -- Balance delta features ------------------------------------------
        .withColumn("balanceOrigDiff",
                    F.col("newbalanceOrig") - F.col("oldbalanceOrg"))
        .withColumn("balanceDestDiff",
                    F.col("newbalanceDest") - F.col("oldbalanceDest"))
        # -- Balance integrity check (non-zero = possible manipulation) ------
        .withColumn("errorBalanceOrig",
                    F.col("oldbalanceOrg") - F.col("amount") - F.col("newbalanceOrig"))
        .withColumn("errorBalanceDest",
                    F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"))
        # -- Synthetic event timestamp for windowed aggregation ---------------
        .withColumn("event_time",
                    (F.col("step") * 3600).cast("timestamp"))  # step -> seconds -> timestamp
        # -- High-risk transaction type flag ---------------------------------
        .withColumn("is_high_risk_type",
                    F.when(F.col("type").isin("TRANSFER", "CASH_OUT"), 1).otherwise(0))
    )
    return df


def add_window_features(df: DataFrame) -> DataFrame:
    """
    Compute per-account windowed aggregation features using Spark Window functions.

    These features capture velocity signals:
      - txn_count_window    : # transactions by sender in the last 10 steps
      - total_amount_window : total amount sent by sender in the last 10 steps

    Note: Uses Window.partitionBy (not groupBy + join) to avoid an extra shuffle.
    """
    from pyspark.sql.window import Window

    # Partition by sender account, order by step; rolling last 10 steps
    w = (Window
         .partitionBy("nameOrig")
         .orderBy("step")
         .rowsBetween(-10, Window.currentRow))

    df = (df
        .withColumn("txn_count_window",
                    F.count("amount").over(w))
        .withColumn("total_amount_window",
                    F.sum("amount").over(w))
    )
    return df


# -----------------------------------------------------------------------------
# 4. ML Pipeline
# -----------------------------------------------------------------------------

def build_ml_pipeline() -> Pipeline:
    """
    Build a Spark MLlib pipeline:

        StringIndexer -> VectorAssembler -> StandardScaler -> GBTClassifier

    Why GBT?
    ---------
    Gradient-Boosted Trees handle class imbalance (fraud << normal) better than
    plain logistic regression and are natively supported by Spark MLlib, so the
    whole pipeline remains distributed.

    Scalability
    -----------
    The pipeline is serialisable -> save once, load in every executor without
    re-training.  On a cluster, the model artefact lives on HDFS/S3 so every
    node can access it.
    """
    # Encode categorical column (transaction type) numerically
    type_indexer = StringIndexer(
        inputCol="type",
        outputCol="type_indexed",
        handleInvalid="keep"   # unknown types -> last index, not an error
    )

    # Assemble all feature columns into a single dense vector
    assembler = VectorAssembler(
        inputCols=ALL_FEATURE_COLS,
        outputCol="raw_features",
        handleInvalid="keep"
    )

    # Scale features to zero-mean / unit variance (helps GBT convergence)
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol=FEATURES_COL,
        withMean=True,
        withStd=True
    )

    # GBT Classifier - binary classification
    gbt = GBTClassifier(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        predictionCol=PREDICTION_COL,
        maxIter=ML_NUM_TREES,
        maxDepth=ML_MAX_DEPTH,
        maxBins=ML_MAX_BINS,
        seed=ML_SEED,
        # Subsample 80% of data per tree -> reduces variance + speeds training
        subsamplingRate=0.8,
    )

    return Pipeline(stages=[type_indexer, assembler, scaler, gbt])


def build_random_forest_pipeline() -> Pipeline:
    """Alternative pipeline using RandomForest - useful as a baseline comparison."""
    type_indexer = StringIndexer(
        inputCol="type", outputCol="type_indexed", handleInvalid="keep"
    )
    assembler = VectorAssembler(
        inputCols=ALL_FEATURE_COLS, outputCol="raw_features", handleInvalid="keep"
    )
    scaler = StandardScaler(
        inputCol="raw_features", outputCol=FEATURES_COL, withMean=True, withStd=True
    )
    rf = RandomForestClassifier(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        predictionCol=PREDICTION_COL,
        numTrees=ML_NUM_TREES,
        maxDepth=ML_MAX_DEPTH,
        seed=ML_SEED,
    )
    return Pipeline(stages=[type_indexer, assembler, scaler, rf])


def load_model(model_path: str) -> PipelineModel:
    """Load a previously saved PipelineModel from disk / HDFS / S3."""
    model = PipelineModel.load(model_path)
    logger.info("Model loaded from: %s", model_path)
    return model


# -----------------------------------------------------------------------------
# 5. Blacklist Join (Broadcast)
# -----------------------------------------------------------------------------

def join_with_blacklist(txn_df: DataFrame, blacklist_df: DataFrame) -> DataFrame:
    """
    Left-join transactions against the blacklist using an explicit broadcast hint.

    Why broadcast?
    --------------
    The blacklist is tiny (thousands of rows) vs. the transaction stream
    (millions of rows).  Broadcasting the small side avoids a shuffle of the
    large side - the most expensive operation in distributed joins.

    The F.broadcast() hint instructs the Spark planner to send the blacklist
    to every executor's memory, enabling a local hash-lookup on each partition
    of the transaction stream.

    Execution plan note
    -------------------
    Run .explain(True) on the result to verify the plan shows
    'BroadcastHashJoin' and NOT 'SortMergeJoin'.

    Returns
    -------
    DataFrame with extra columns:
        blacklisted     : boolean - True if destination is in blacklist
        bl_risk_score   : risk score from blacklist (null if not in list)
        bl_reason       : reason string (null if not in list)
    """
    joined = (
        txn_df
        .join(
            F.broadcast(blacklist_df.withColumnRenamed("account_id", "nameDest_bl")),
            on=txn_df["nameDest"] == F.col("nameDest_bl"),
            how="left"
        )
        .withColumn("blacklisted",   F.col("bl_risk_score").isNotNull())
        .withColumn("bl_risk_score", F.coalesce(F.col("risk_score"), F.lit(0.0)))
        .withColumn("bl_reason",     F.coalesce(F.col("reason"),     F.lit("none")))
        .drop("nameDest_bl", "risk_score", "reason")
    )
    return joined
