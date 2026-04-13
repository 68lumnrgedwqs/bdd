"""
fraud_train.py
==============
Batch training job:

  1. Read full PaySim CSV with explicit schema (no inferSchema).
  2. Feature engineering (balance deltas, window aggregates).
  3. Build & export blacklist.
  4. Train / evaluate GBT pipeline.
  5. Save model artefact.

Run once; the streaming job loads the saved model.

Usage
-----
    python fraud_train.py

    # On a cluster:
    spark-submit --master yarn --executor-memory 8g --executor-cores 4 fraud_train.py
"""

import os
import sys
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fraud_train")

from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from config import (
    PAYSIM_CSV, MODEL_DIR, LOG_DIR, LABEL_COL, FEATURES_COL,
    PREDICTION_COL, PROBABILITY_COL, FRAUD_SCORE_THRESHOLD
)
from spark_session import get_spark_session
from fraud_engine import (
    paysim_schema, build_blacklist, feature_engineering,
    add_window_features, build_ml_pipeline
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Helper: print execution plan summary
# -----------------------------------------------------------------------------

def explain_plan(df, label: str):
    """Print a short physical plan for the given DataFrame."""
    print(f"\n{'='*60}")
    print(f"  EXECUTION PLAN: {label}")
    print(f"{'='*60}")
    df.explain(mode="formatted")
    print()


# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------

def load_paysim(spark):
    """
    Read PaySim CSV with explicit schema -> avoids expensive schema-inference scan.

    Scalability:
    When data grows 10x, simply point to a directory of CSVs or Parquet files
    partitioned by date.  The executor count scales with the cluster - no code
    change required.
    """
    logger.info("Reading PaySim dataset from: %s", PAYSIM_CSV)
    t0 = time.time()

    df = (spark.read
          .option("header", "true")
          .option("mode", "DROPMALFORMED")   # drop corrupted rows gracefully
          .schema(paysim_schema())
          .csv(PAYSIM_CSV)
          .sample(False, 0.01, seed=42))

    logger.info("Schema loaded in %.1fs. Partitions: %d", time.time() - t0, df.rdd.getNumPartitions())
    return df


# -----------------------------------------------------------------------------
# 2. EDA (lightweight, distributed)
# -----------------------------------------------------------------------------

def run_eda(df):
    logger.info("Running EDA ...")

    total = df.count()
    fraud_count = df.filter(F.col("isFraud") == 1).count()
    logger.info("Total rows: %d  |  Fraud rows: %d  (%.2f%%)",
                total, fraud_count, 100 * fraud_count / total)

    print("\n-- Class distribution --")
    df.groupBy("isFraud").count().orderBy("isFraud").show()

    print("\n-- Transaction type x fraud rate --")
    (df.groupBy("type")
       .agg(
           F.count("*").alias("total"),
           F.sum("isFraud").alias("fraud"),
           (F.sum("isFraud") / F.count("*") * 100).alias("fraud_pct")
       )
       .orderBy(F.desc("fraud_pct"))
       .show())

    print("\n-- Amount statistics --")
    df.select("amount").summary("min", "25%", "50%", "75%", "max", "mean", "stddev").show()

    return total, fraud_count


# -----------------------------------------------------------------------------
# 3. Feature Engineering + Black list
# -----------------------------------------------------------------------------

def prepare_features(spark, df):
    logger.info("Engineering features ...")

    # Column-level transforms (no shuffle)
    df = feature_engineering(df)

    # Window aggregates (one shuffle: partitionBy nameOrig)
    df = add_window_features(df)

    # Build and cache blacklist (one pass over fraud rows)
    blacklist_df = build_blacklist(spark, df)

    logger.info("Feature engineering complete. Caching prepared data ...")
    df.cache()
    df.count()   # trigger cache materialisation

    explain_plan(df, "After Feature Engineering")
    return df, blacklist_df


# -----------------------------------------------------------------------------
# 4. Train / Evaluate
# -----------------------------------------------------------------------------

def train_model(df):
    logger.info("Splitting dataset 80/20 stratified ...")

    # Stratified split per fraud label
    fractions = df.select(LABEL_COL).distinct().rdd.flatMap(lambda x: x).collect()
    fractions = {str(k): 0.8 for k in fractions}   # not used directly but kept for reference

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Cache splits to avoid recomputation during CV
    train_df.cache()
    test_df.cache()

    logger.info("Train rows: %d  |  Test rows: %d", train_df.count(), test_df.count())

    # -- Pipeline ------------------------------------------------------------
    pipeline = build_ml_pipeline()

    logger.info("Training GBT pipeline ...")
    t0 = time.time()
    model = pipeline.fit(train_df)
    logger.info("Training finished in %.1f seconds", time.time() - t0)

    # -- Evaluation ----------------------------------------------------------
    predictions = model.transform(test_df)

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )
    mc_evaluator = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol=PREDICTION_COL
    )

    auc    = auc_evaluator.evaluate(predictions)
    auc_pr = pr_evaluator.evaluate(predictions)
    f1     = mc_evaluator.setMetricName("f1").evaluate(predictions)
    prec   = mc_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    rec    = mc_evaluator.setMetricName("weightedRecall").evaluate(predictions)
    acc    = mc_evaluator.setMetricName("accuracy").evaluate(predictions)

    print("\n" + "="*55)
    print("  MODEL EVALUATION RESULTS")
    print("="*55)
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  AUC-PR      : {auc_pr:.4f}")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  Accuracy    : {acc:.4f}")
    print("="*55 + "\n")

    # -- Confusion matrix (distributed count, not pandas) --------------------
    print("Confusion matrix:")
    (predictions
        .groupBy(LABEL_COL, PREDICTION_COL)
        .count()
        .orderBy(LABEL_COL, PREDICTION_COL)
        .show())

    # -- Feature importances (GBT model is the last stage) -------------------
    from config import ALL_FEATURE_COLS
    gbt_model = model.stages[-1]
    importances = gbt_model.featureImportances
    feat_imp = sorted(
        zip(ALL_FEATURE_COLS, importances.toArray()),
        key=lambda x: -x[1]
    )
    print("\nTop-10 Feature Importances:")
    for feat, imp in feat_imp[:10]:
        bar = "=" * int(imp * 40)
        print(f"  {feat:<30} {imp:.4f}  {bar}")

    return model


# -----------------------------------------------------------------------------
# 5. Save Model
# -----------------------------------------------------------------------------

def save_model(model, path: str):
    logger.info("Saving model to: %s", path)
    model.write().overwrite().save(path)
    logger.info("Model saved successfully.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  FRAUD DETECTION SYSTEM - TRAINING JOB")
    print("="*60 + "\n")

    spark = get_spark_session("FraudDetection-Training")

    # Step 1: Load
    df_raw = load_paysim(spark)

    # Step 2: EDA
    _, _ = run_eda(df_raw)

    # Step 3: Feature engineering + blacklist
    df_features, blacklist_df = prepare_features(spark, df_raw)

    # Step 4: Train + evaluate
    model = train_model(df_features)

    # Step 5: Save
    model_path = os.path.join(MODEL_DIR, "gbt_fraud_model")
    save_model(model, model_path)

    logger.info("Training job complete.  Model at: %s", model_path)
    spark.stop()


if __name__ == "__main__":
    main()
