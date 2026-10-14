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
from pyspark.sql.window import Window


# ============================================================
# 1. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("Singapore Rainfall Analysis")
    .master("local[4]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "64")
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
# 6. FIRST-PASS YEARLY COMPARISON: ALL AVAILABLE STATIONS
from pyspark import StorageLevel
import matplotlib.pyplot as plt

MIN_DAILY_READINGS = 274
USE_SCALED_DAILY_TOTAL = False  # True applies observed_total * 288 / readings.

valid = df.filter(
    F.col("station_id").isNotNull()
    & F.col("timestamp").isNotNull()
    & F.col("reading_value").isNotNull()
    & ~F.isnan("reading_value")
    & (F.col("reading_value") >= 0)
).withColumn("observation_date", F.to_date("timestamp"))
daily = (
    valid.groupBy("station_id", "observation_date")
    .agg(F.sum("reading_value").alias("observed_daily_mm"),
         F.count("*").alias("n_readings"),
         F.countDistinct("timestamp").alias("n_unique_timestamps"))
    .filter((F.col("n_unique_timestamps") >= MIN_DAILY_READINGS)
            & (F.col("n_unique_timestamps") <= 288)
            & (F.col("n_readings") == F.col("n_unique_timestamps")))
    .withColumn("year", F.year("observation_date"))
    .withColumn("month", F.month("observation_date"))
    .filter(F.col("year").between(2017, 2024))
    .withColumn("scaled_daily_mm",
                F.col("observed_daily_mm") * 288.0 / F.col("n_unique_timestamps"))
    .persist(StorageLevel.MEMORY_AND_DISK)
)
# No monthly or yearly completeness exclusions.
# Equal weight per eligible station-day, matching the monthly analysis.
value_col = "scaled_daily_mm" if USE_SCALED_DAILY_TOTAL else "observed_daily_mm"
annual = daily.groupBy("year").agg(
    F.avg(value_col).alias("mean_daily_rainfall_mm"),
    F.count("*").alias("eligible_station_days"),
    F.countDistinct("station_id").alias("n_stations"),
    F.countDistinct("observation_date").alias("dates_with_eligible_station"),
    F.countDistinct("month").alias("months_represented")
).orderBy("year")
print("\n========== YEARLY MEAN: AVAILABLE ELIGIBLE STATION-DAYS ==========")
annual.show(8, False)

# Coverage remains visible even though it is no longer an exclusion rule.
calendar_grid = spark.range(2017, 2025).select(F.col("id").cast("int").alias("year")).crossJoin(
    spark.range(1, 13).select(F.col("id").cast("int").alias("month")))
monthly_counts = daily.groupBy("year", "month").agg(
    F.count("*").alias("eligible_station_days"),
    F.countDistinct("station_id").alias("n_stations"),
    F.countDistinct("observation_date").alias("dates_with_eligible_station"))
print("\n========== MONTHLY CONTRIBUTIONS TO YEARLY MEANS ==========")
calendar_grid.join(monthly_counts, ["year", "month"], "left").fillna(0).orderBy(
    "year", "month").show(96, False)

annual_pd = annual.toPandas().set_index("year").reindex(range(2017, 2025))
if annual_pd["mean_daily_rainfall_mm"].notna().any():
    plt.figure(figsize=(11, 5))
    plt.plot(annual_pd.index, annual_pd["mean_daily_rainfall_mm"], marker="o", color="steelblue")
    plt.xticks(range(2017, 2025))
    plt.xlabel("Year")
    measure = "Estimated" if USE_SCALED_DAILY_TOTAL else "Recorded"
    plt.ylabel(f"{measure} rainfall (mm/day/station)")
    plt.title("Yearly mean rainfall among available station-days\n"
              "274–288 readings/day; station and seasonal coverage vary")
    plt.grid(alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
else:
    print("No eligible station-days found.")
daily.unpersist()
spark.stop()
plt.show()
