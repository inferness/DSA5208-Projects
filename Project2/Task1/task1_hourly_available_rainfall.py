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
# 6. HOURLY PATTERN: ALL AVAILABLE COMPLETE STATION-HOURS
# ============================================================

from pathlib import Path
import matplotlib.pyplot as plt

# Group by the recorded timestamp hour in Singapore local time.
# No daily, monthly, or yearly completeness exclusions are applied.
# Require complete hours: 12 readings covering 12 five-minute slots.

readings = (
    df.select(
        "station_id",
        F.to_timestamp("timestamp").alias("ts"),
        F.col("reading_value").cast("double").alias("rain_mm"),
    )
    .filter(
        F.col("station_id").isNotNull()
        & F.col("ts").isNotNull()
        & F.col("rain_mm").isNotNull()
        & (~F.isnan("rain_mm"))
        & (F.col("rain_mm") >= 0)
        & F.year("ts").between(2017, 2024)
    )
    .withColumn("hour_start", F.date_trunc("hour", "ts"))
    .withColumn("five_minute_slot", F.floor(F.minute("ts") / 5))
)

# Require 12 readings AND 12 distinct five-minute slots per station-hour.
# This prevents a duplicate reading from compensating for a missing slot.
station_hours = (
    readings.groupBy("station_id", "hour_start")
    .agg(
        F.count("*").alias("n_readings"),
        F.countDistinct("five_minute_slot").alias("n_slots"),
        F.sum("rain_mm").alias("hourly_rain_mm"),
    )
    .filter((F.col("n_readings") == 12) & (F.col("n_slots") == 12))
    .withColumn("hour", F.hour("hour_start"))
)

# Include complete dry hours (zero rainfall) in the average.
# Each complete station-date-hour has equal weight.
hourly_pattern = (
    station_hours.groupBy("hour")
    .agg(
        F.avg("hourly_rain_mm").alias("avg_hourly_rain_mm"),
        F.count("*").alias("n_complete_station_hours"),
    )
    .orderBy("hour")
)

# Collect only the 24 aggregated rows, not the full rainfall dataset.
rows = hourly_pattern.collect()
if not rows:
    raise ValueError("No complete station-hours found. Check timestamps and input data.")

print("Hour | Average rainfall (mm) | Complete station-hours")
for row in rows:
    print(f"{row['hour']:02d}:00 | {row['avg_hourly_rain_mm']:.4f} | "
          f"{row['n_complete_station_hours']:,}")

values = {row['hour']: row['avg_hourly_rain_mm'] for row in rows}
hours = list(range(24))
means = [values.get(h, float('nan')) for h in hours]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(hours, means, marker="o", linewidth=2, color="steelblue")
ax.set_title("Average Rainfall by Hour of Day in Singapore (2017–2024)")
ax.set_xlabel("Hour of day (Singapore time)")
ax.set_ylabel("Average hourly rainfall per station (mm)")
ax.set_xticks(hours)
ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
image_path = output_dir / "hourly_rainfall_pattern.png"
fig.savefig(image_path, dpi=300, bbox_inches="tight")
print(f"Graph saved to: {image_path.resolve()}")

spark.stop()
plt.show()
