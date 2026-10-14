import argparse
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from common import session, inputs, FEATURE_PATH, MODEL_PATH, F, load_features, score_model
parser = argparse.ArgumentParser()
parser.add_argument("--station", required=True)
parser.add_argument("--time", required=True, help="Singapore local time YYYY-MM-DD HH:MM:SS")
parser.add_argument("--threshold", required=True, type=float)
parser.add_argument("--model", default=None, help="Path to trained JSON model file (defaults to MODEL_PATH)")
args = parser.parse_args()
if not 0 <= args.threshold <= 25:
    parser.error("Supported threshold range is 0 to 25 mm")
spark = session()
base = load_features(spark).filter(
    (F.col("station_id") == args.station) & (F.col("prediction_time") == F.to_timestamp(F.lit(args.time))))
# Discard future target before transforming: prediction does not use it.
base = base.drop("future_rain_30m").withColumn("threshold_mm", F.lit(args.threshold))
if not base.take(1):
    raise ValueError("No eligible historical example for that station/time. Use an exact five-minute timestamp with complete windows.")
import json
from pathlib import Path

model_path = Path(args.model) if args.model else Path(MODEL_PATH)
if not model_path.is_file():
    raise FileNotFoundError(
        f"Model file not found at '{model_path}'. "
        "Please train the model first with 02_train_with_zero.py or pass --model <path>."
    )

result = score_model(inputs(base), json.loads(model_path.read_text()))
result.select("station_id", "prediction_time", "threshold_mm",
              F.col("p").alias("probability_exceeding_threshold")).show(truncate=False)
spark.stop()
