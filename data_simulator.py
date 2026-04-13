"""
data_simulator.py
=================
Simulates a real-time transaction stream by reading the PaySim CSV and
writing small JSON micro-batches to the stream_input/ landing zone.

In production this is replaced by a real Kafka producer.
The streaming job reads from stream_input/ with maxFilesPerTrigger=50,
exactly mimicking a Kafka topic.

Usage
-----
    python data_simulator.py          # default: 200 rows/batch, 5s interval
    python data_simulator.py --batch-size 500 --interval 2
"""

import os
import sys
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_simulator")

# Delay import so it doesn't fail if ran independently
try:
    from spark_session import get_spark_session
    from fraud_engine import paysim_schema
    from config import PAYSIM_CSV, STREAM_INPUT
except ImportError:
    pass

def parse_args():
    p = argparse.ArgumentParser(description="PaySim stream simulator")
    p.add_argument("--batch-size", type=int,   default=200,
                   help="Rows per micro-batch file")
    p.add_argument("--interval",   type=float, default=5.0,
                   help="Seconds between batches")
    p.add_argument("--max-batches",type=int,   default=0,
                   help="0 = unlimited")
    return p.parse_args()


def main():
    os.makedirs(STREAM_INPUT, exist_ok=True)
    args = parse_args()
    logger.info("Simulator starting: batch_size=%d  interval=%.1fs",
                args.batch_size, args.interval)

    spark = get_spark_session("FraudDetection-Simulator")

    # Read full PaySim dataset (batch mode)
    df = (spark.read
          .option("header", "true")
          .option("mode", "DROPMALFORMED")
          .schema(paysim_schema())
          .csv(PAYSIM_CSV))

    # Convert to list of Row objects on the driver for sequential writing
    # NOTE: we use .limit() + .collect() on small chunks only -
    #       never .collect() the entire 6M-row dataset at once.
    total_rows = df.count()
    logger.info("Total rows in dataset: %d", total_rows)

    batch_num   = 0
    offset      = 0

    while offset < total_rows:
        if args.max_batches and batch_num >= args.max_batches:
            logger.info("Reached max_batches=%d. Stopping.", args.max_batches)
            break

        # Take a small slice using Spark - stays distributed until the write
        batch_df = (spark.read
                    .option("header", "true")
                    .option("mode", "DROPMALFORMED")
                    .schema(paysim_schema())
                    .csv(PAYSIM_CSV)
                    .limit(offset + args.batch_size)   # crude but avoids storing full data
                    )

        # Write as JSON to the landing zone (simulates Kafka sink)
        out_path = os.path.join(STREAM_INPUT, f"batch_{batch_num:06d}.json")
        (batch_df
            .coalesce(1)
            .write
            .mode("overwrite")
            .json(out_path))

        logger.info("Batch %05d written -> %s", batch_num, out_path)
        batch_num += 1
        offset    += args.batch_size
        time.sleep(args.interval)

    logger.info("Simulator finished after %d batches.", batch_num)
    spark.stop()


if __name__ == "__main__":
    main()
