import os
import sys

# Use this same installed Python for Spark workers on Windows.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    DateType,
    TimestampType,
    StringType,
    DoubleType
)


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("Singapore Rainfall Analysis")
    .master("local[4]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.sql.session.timeZone", "Asia/Singapore")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. DEFINE DATA SCHEMA
# ============================================================

rainfall_schema = StructType([
    StructField("date", DateType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("update_timestamp", TimestampType(), True),
    StructField("station_id", StringType(), True),
    StructField("station_name", StringType(), True),
    StructField("station_device_id", StringType(), True),
    StructField("location_longitude", DoubleType(), True),
    StructField("location_latitude", DoubleType(), True),
    StructField("reading_update_timestamp", TimestampType(), True),
    StructField("reading_value", DoubleType(), True),
    StructField("reading_type", StringType(), True),
    StructField("reading_unit", StringType(), True)
])


# ============================================================
# 3. DEFINE INPUT FILES
# ============================================================

files = [
    f"data/HistoricalRainfallacrossSingapore{year}.csv"
    for year in range(2017, 2025)
]


# ============================================================
# 4. LOAD ALL 8 YEARS INTO ONE SPARK DATAFRAME
# ============================================================

df = (
    spark.read
    .option("header", True)
    .schema(rainfall_schema)
    .csv(files)
)


# ============================================================
# 5. CHECK SCHEMA
# ============================================================

print("\n========== DATA SCHEMA ==========")

df.printSchema()


# ============================================================


from pyspark.sql.window import Window
import math
# 7. CLEAN AND ALIGN FIVE-MINUTE TIMESTAMPS
# Permit at most one second of timing jitter (e.g. 08:04:59 -> 08:05:00).
# Ambiguous slots with multiple readings are excluded, not summed or imputed.
valid = df.select("station_id", "timestamp", "reading_value").filter(
    F.col("station_id").isNotNull()
    & F.col("timestamp").isNotNull()
    & F.col("reading_value").isNotNull()
    & ~F.isnan("reading_value")
    & (F.col("reading_value") >= 0)
)
valid = valid.withColumn("raw_seconds", F.col("timestamp").cast("long"))
valid = valid.withColumn("t", (F.round(F.col("raw_seconds") / 300) * 300).cast("long"))
valid = valid.filter(F.abs(F.col("raw_seconds") - F.col("t")) <= 1)
series = (
    valid.groupBy("station_id", "t")
    .agg(F.count("*").alias("slot_count"), F.first("reading_value").alias("rain_5m"))
    .filter(F.col("slot_count") == 1)
    .withColumn("prediction_time", F.from_unixtime("t").cast("timestamp"))
)

# 8. FEATURES AND FUTURE TARGET
# At t, rain_5m is the latest observed five-minute total.
# Past 60 minutes: t-55, ..., t (12 readings).
# Next 30 minutes: t+5, ..., t+30 (6 readings).
w = Window.partitionBy("station_id").orderBy("t")
past15 = w.rangeBetween(-600, 0)
preceding15 = w.rangeBetween(-1500, -900)
past30 = w.rangeBetween(-1500, 0)
past60 = w.rangeBetween(-3300, 0)
future30 = w.rangeBetween(300, 1800)
examples = (
    series
    .withColumn("rain_15m", F.sum("rain_5m").over(past15))
    .withColumn("rain_30m", F.sum("rain_5m").over(past30))
    .withColumn("rain_60m", F.sum("rain_5m").over(past60))
    .withColumn("rain_trend", F.col("rain_15m") - F.sum("rain_5m").over(preceding15))
    .withColumn("wet_intervals_30m", F.sum(F.when(F.col("rain_5m") > 0, 1).otherwise(0)).over(past30))
    .withColumn("past_count", F.count("*").over(past60))
    .withColumn("future_count", F.count("*").over(future30))
    .withColumn("future_rain_30m", F.sum("rain_5m").over(future30))
    .filter((F.col("past_count") == 12) & (F.col("future_count") == 6))
    .withColumn("year", F.year("prediction_time"))
    .withColumn("minute_of_day", F.hour("prediction_time") * 60 + F.minute("prediction_time"))
    .withColumn("month_index", F.month("prediction_time") - 1)
    .withColumn("hour_sin", F.sin(F.col("minute_of_day") * (2 * math.pi / 1440)))
    .withColumn("hour_cos", F.cos(F.col("minute_of_day") * (2 * math.pi / 1440)))
    .withColumn("month_sin", F.sin(F.col("month_index") * (2 * math.pi / 12)))
    .withColumn("month_cos", F.cos(F.col("month_index") * (2 * math.pi / 12)))
)


# Save once; fail rather than overwrite an existing prepared dataset.
columns = ["station_id", "prediction_time", "t", "year", "rain_5m", "rain_15m",
           "rain_30m", "rain_60m", "rain_trend", "wet_intervals_30m", "future_rain_30m"]
import csv
from pathlib import Path
output = Path("output/task2/features.csv")
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise FileExistsError("features.csv already exists; rename it before preparing again")
temporary = output.with_suffix(".csv.partial")
# Stream one Spark partition at a time, never collect the whole dataset.
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(columns)
    
    # Smaller output partitions reduce memory needed during CSV streaming.
    export_rows = examples.select(*columns).repartition(512)

    for count, row in enumerate(
        export_rows.toLocalIterator(prefetchPartitions=False), 1):
        writer.writerow(list(row))
        if count % 1000000 == 0:
            print(f"Saved {count:,} feature rows", flush=True)
temporary.replace(output)
print("Prepared features saved to", output)
spark.stop()
