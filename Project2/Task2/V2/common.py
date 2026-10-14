import os
import sys
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression

THRESHOLDS = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
from pathlib import Path

def _resolve_path(rel_str, alt_rel_str=None):
    p = Path(rel_str)
    if p.exists():
        return str(p)
    if alt_rel_str and Path(alt_rel_str).exists():
        return str(Path(alt_rel_str))
    p_up = Path("../../") / rel_str
    if p_up.exists():
        return str(p_up)
    return rel_str

FEATURE_PATH = _resolve_path("output/task2/features.csv")
MODEL_PATH = _resolve_path("output/task2/comparison/lr_model.json", "output/task2/improved_lr_model.json")
NUMERIC = ["rain_5m", "rain_15m", "rain_30m", "rain_60m", "rain_trend",
           "wet_intervals_30m", "log_rain_5m", "log_rain_15m", "log_rain_30m",
           "log_rain_60m", "is_raining", "threshold_mm", "log_threshold",
           "threshold_x_rain", "threshold_x_wet"]
CATEGORIES = ["station_id", "hour_category", "month_category", "month_hour_category"]

def session():
    spark = (SparkSession.builder.appName("Rainfall Task 2 improved")
             .master("local[4]").config("spark.driver.memory", "6g")
             .config("spark.sql.shuffle.partitions", "64")
             .config("spark.sql.session.timeZone", "Asia/Singapore").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark

def inputs(base):
    # This function needs no future rainfall: also used at prediction time.
    base = (base.withColumn("hour_category", F.hour("prediction_time").cast("string"))
            .withColumn("month_category", F.month("prediction_time").cast("string"))
            .withColumn("month_hour_category", F.concat_ws("_", "month_category", "hour_category"))
            .withColumn("is_raining", (F.col("rain_5m") > 0).cast("double")))
    for name in ["rain_5m", "rain_15m", "rain_30m", "rain_60m"]:
        base = base.withColumn("log_" + name, F.log1p(name))
    return (base.withColumn("log_threshold", F.log1p("threshold_mm"))
            .withColumn("threshold_x_rain", F.col("log_threshold") * F.col("log_rain_30m"))
            .withColumn("threshold_x_wet", F.col("log_threshold") * F.col("is_raining")))

def dataset(spark, years, fraction):
    # Deterministic station-time sample; shared by every model/threshold.
    base = load_features(spark).filter(F.col("year").isin(years))
    end_year = F.year(F.from_unixtime(F.col("t") + 1800).cast("timestamp"))
    base = base.filter(end_year.isin(years))
    base = base.filter(F.pmod(F.xxhash64("station_id", "t"), F.lit(1000000)) < int(fraction * 1000000))
    thresholds = spark.createDataFrame([(float(a),) for a in THRESHOLDS], ["threshold_mm"])
    return inputs(base.crossJoin(F.broadcast(thresholds))).withColumn(
        "label", (F.col("future_rain_30m") > F.col("threshold_mm")).cast("double"))

def pipeline(reg):
    stages = [StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid="keep") for c in CATEGORIES]
    stages += [OneHotEncoder(inputCols=[c+"_index" for c in CATEGORIES],
                             outputCols=[c+"_vector" for c in CATEGORIES], handleInvalid="keep"),
               VectorAssembler(inputCols=NUMERIC+[c+"_vector" for c in CATEGORIES], outputCol="raw_features"),
               StandardScaler(inputCol="raw_features", outputCol="features", withMean=False, withStd=True),
               LogisticRegression(featuresCol="features", labelCol="label", regParam=reg,
                                  elasticNetParam=0.0, maxIter=80, standardization=False)]
    return Pipeline(stages=stages)


def load_features(spark):
    # Explicit schema avoids a full extra scan for type inference.
    schema = ("station_id string, prediction_time timestamp, t long, year int, "
              "rain_5m double, rain_15m double, rain_30m double, rain_60m double, "
              "rain_trend double, wet_intervals_30m long, future_rain_30m double")
    return spark.read.option("header", True).schema(schema).csv(FEATURE_PATH)

def export_model(fitted, path):
    import json
    from pathlib import Path
    indexers = fitted.stages[:len(CATEGORIES)]
    encoder = fitted.stages[len(CATEGORIES)]
    scaler = fitted.stages[-2]
    lr = fitted.stages[-1]
    # Folding scaling into coefficients exactly preserves fitted probabilities.
    weights = [float(c/s) if s != 0 else 0.0 for c,s in zip(lr.coefficients, scaler.std)]
    offset = len(NUMERIC)
    category_weights = {}
    for name, indexer, width in zip(CATEGORIES, indexers, encoder.categorySizes):
        # With handleInvalid=keep and dropLast=True, encoded width=categorySize.
        category_weights[name] = {"labels": list(indexer.labels), "weights": weights[offset:offset+width]}
        offset += width
    if offset != len(weights):
        raise ValueError("Unexpected encoder dimensions; refusing incomplete export")
    payload = {"intercept": float(lr.intercept), "numeric_weights": dict(zip(NUMERIC, weights[:len(NUMERIC)])),
               "categorical_weights": category_weights, "format_version": 1}
    # Verify exported scoring against the native Spark pipeline before saving.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
    return payload

def score_model(data, payload):
    z = F.lit(payload["intercept"])
    for name, weight in payload["numeric_weights"].items():
        z = z + F.col(name)*F.lit(weight)
    for name, item in payload["categorical_weights"].items():
        labels, weights = item["labels"], item["weights"]
        unknown = weights[len(labels)] if len(weights) > len(labels) else 0.0
        contribution = F.lit(unknown)
        for label, weight in zip(labels, weights):
            contribution = F.when(F.col(name) == label, F.lit(weight)).otherwise(contribution)
        z = z + contribution
    data = data.withColumn("score", z)
    data = data.withColumn("p", F.when(F.col("score") >= 0, 1/(1+F.exp(-F.col("score"))))
                           .otherwise(F.exp("score")/(1+F.exp("score"))))
    return data
