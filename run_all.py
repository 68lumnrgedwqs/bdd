"""
run_all.py
==========
One-click orchestrator for the full Fraud Detection pipeline:

    Step 1 : Generate blacklist (training run - batch mode)
    Step 2 : Train GBT model and save artefact
    Step 3 : Launch data simulator (background thread)
    Step 4 : Start Structured Streaming inference

Usage
-----
    python run_all.py               # full pipeline
    python run_all.py --skip-train  # skip training (use saved model)
    python run_all.py --train-only  # only run training, no streaming
"""

import os
import sys
import argparse
import logging
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_all")

from config import MODEL_DIR, STREAM_INPUT, DATA_DIR


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fraud Detection Pipeline Orchestrator")
    p.add_argument("--skip-train",  action="store_true",
                   help="Skip training; load existing model")
    p.add_argument("--train-only",  action="store_true",
                   help="Only run training step, then exit")
    p.add_argument("--sim-delay",   type=float, default=10.0,
                   help="Seconds to wait after starting simulator before streaming")
    p.add_argument("--sim-batches", type=int,   default=0,
                   help="Max simulator batches (0=unlimited)")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Step 1+2 : Training
# -----------------------------------------------------------------------------

def run_training():
    """Import and execute the training job in-process."""
    logger.info("="*55)
    logger.info("STEP 1-2: Training GBT Fraud Model")
    logger.info("="*55)

    # Import inside function to avoid circular SparkSession conflicts
    import fraud_train
    fraud_train.main()
    logger.info("Training complete.")


# -----------------------------------------------------------------------------
# Step 3 : Data Simulator
# -----------------------------------------------------------------------------

def run_simulator_background(max_batches: int = 0):
    """Launch the data simulator in a background thread."""

    def _sim():
        logger.info("Simulator thread started.")
        # Patch sys.argv to pass arguments to the simulator's argparse
        sys_argv_backup = sys.argv
        sys.argv = ["data_simulator.py",
                    "--batch-size", "300",
                    "--interval",   "5",
                    "--max-batches", str(max_batches)]
        try:
            import data_simulator
            data_simulator.main()
        except Exception as e:
            logger.error("Simulator thread error: %s", e)
        finally:
            sys.argv = sys_argv_backup
            logger.info("Simulator thread finished.")

    t = threading.Thread(target=_sim, daemon=True, name="SimulatorThread")
    t.start()
    logger.info("Simulator running in background thread.")
    return t


# -----------------------------------------------------------------------------
# Step 4 : Streaming
# -----------------------------------------------------------------------------

def run_streaming():
    logger.info("="*55)
    logger.info("STEP 4: Starting Structured Streaming Inference")
    logger.info("="*55)

    import fraud_streaming
    fraud_streaming.main()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    banner = """
+============================================================+
|   REAL-TIME FRAUD DETECTION SYSTEM - BIG DATA PROJECT      |
|   Stack: PySpark Structured Streaming + MLlib GBT          |
|   Dataset: PaySim (Kaggle) - 6M transactions               |
+============================================================+
"""
    print(banner)

    args = parse_args()

    # -- Validate prerequisites -----------------------------------------------
    from config import PAYSIM_CSV, BLACKLIST_PATH
    if not os.path.exists(PAYSIM_CSV):
        logger.error("PaySim CSV not found at: %s", PAYSIM_CSV)
        sys.exit(1)

    model_path = os.path.join(MODEL_DIR, "gbt_fraud_model")

    # -- Training -------------------------------------------------------------
    if not args.skip_train:
        run_training()
    else:
        if not os.path.exists(model_path):
            logger.error("--skip-train specified but model not found at: %s", model_path)
            sys.exit(1)
        logger.info("Skipping training. Using saved model at: %s", model_path)

    if args.train_only:
        logger.info("--train-only flag set. Exiting after training.")
        return

    # -- Ensure stream input directory has data -------------------------------
    os.makedirs(STREAM_INPUT, exist_ok=True)

    # -- Start simulator -------------------------------------------------------
    sim_thread = run_simulator_background(args.sim_batches)

    # Wait for simulator to write at least one batch
    logger.info("Waiting %.1f seconds for simulator to produce initial data ...",
                args.sim_delay)
    time.sleep(args.sim_delay)

    # -- Start streaming -------------------------------------------------------
    run_streaming()


if __name__ == "__main__":
    main()
