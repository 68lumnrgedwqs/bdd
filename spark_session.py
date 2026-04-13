"""
spark_session.py
================
Factory that builds a production-grade SparkSession.

Design decisions
----------------
* Single entry-point (get_spark_session) - prevents accidental duplicate contexts.
* All config is pulled from config.py -> easy to override per-environment.
* Kryo serialiser + AQE + broadcast threshold are pre-wired.
* Log level set to WARN to suppress verbose Spark INFO spam in notebooks.
* HADOOP_HOME is auto-configured for Windows (winutils.exe requirement).
"""

import os
import sys
import logging
from pyspark.sql import SparkSession
from config import SPARK_CONFIG

# -----------------------------------------------------------------------------
# Windows: configure HADOOP_HOME so Spark can find winutils.exe
# winutils.exe must be at: C:\hadoop\bin\winutils.exe
# Download from: https://github.com/cdarlint/winutils
# -----------------------------------------------------------------------------
def _configure_windows_hadoop():
    """Set HADOOP_HOME and PATH so PySpark finds winutils.exe on Windows."""
    if sys.platform != "win32":
        return  # Linux / macOS: not needed

    hadoop_home = r"C:\hadoop"
    winutils   = os.path.join(hadoop_home, "bin", "winutils.exe")

    if not os.path.exists(winutils):
        raise FileNotFoundError(
            f"winutils.exe not found at: {winutils}\n"
            "Please download it from: "
            "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin/winutils.exe\n"
            f"and place it at: {winutils}"
        )

    os.environ["HADOOP_HOME"]   = hadoop_home
    os.environ["hadoop.home.dir"] = hadoop_home

    # Prepend to PATH so hadoop.dll is also discoverable
    bin_path = os.path.join(hadoop_home, "bin")
    if bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

    # Force Spark workers to use the exact same Python executable.
    # On Windows, PySpark crashes if the Python path has spaces (e.g. Program Files).
    # We use ctypes to get the 8.3 short path name (e.g. C:\\Progra~1\\...)
    import ctypes
    buf = ctypes.create_unicode_buffer(512)
    ctypes.windll.kernel32.GetShortPathNameW(sys.executable, buf, 512)
    short_python_path = buf.value if buf.value else sys.executable

    os.environ["PYSPARK_PYTHON"] = short_python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = short_python_path


_configure_windows_hadoop()

logger = logging.getLogger(__name__)


def get_spark_session(app_name: str = "FraudDetectionSystem") -> SparkSession:
    """
    Return (or re-use) a configured SparkSession.

    Scalability note
    ----------------
    When deployed on a YARN / Kubernetes cluster, `spark.master` is passed
    via spark-submit --master, so we deliberately do NOT hard-code it here.
    On a laptop, Spark defaults to local[*] (all cores).
    """
    builder = SparkSession.builder.appName(app_name)

    for key, value in SPARK_CONFIG.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()

    # Suppress noisy INFO logs - keep WARNING and above
    spark.sparkContext.setLogLevel("WARN")

    logger.info("SparkSession created: %s  |  master=%s",
                app_name, spark.sparkContext.master)
    return spark
