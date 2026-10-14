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


# Additional imports and settings for analysis
from pyspark import StorageLevel
import calendar
import matplotlib.pyplot as plt

EXPECTED_READINGS_PER_DAY = 288

# 6. VALID OBSERVATIONS; preserve raw timestamps.
rainfall = df.select("station_id", "timestamp",
                     F.col("reading_value").alias("rainfall_mm"))
invalid = (
    F.col("station_id").isNull() | F.col("timestamp").isNull()
    | F.col("rainfall_mm").isNull() | F.isnan("rainfall_mm")
    | (F.col("rainfall_mm") < 0)
)
print("\n========== INVALID ROWS ==========")
rainfall.agg(F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid_rows")).show()
valid = rainfall.filter(~invalid).withColumn("date", F.to_date("timestamp"))

# 7. DAILY TOTALS AND COMPLETENESS
# Count-based completeness is provisional: 288 timestamps alone does not
# prove regularly spaced observations. Entirely absent days do not appear here.
daily = (
    valid.groupBy("station_id", "date")
    .agg(F.sum("rainfall_mm").alias("daily_rainfall_mm"),
         F.count("*").alias("n_readings"),
         F.countDistinct("timestamp").alias("n_unique_timestamps"))
    .withColumn("year", F.year("date"))
    .withColumn("month", F.month("date"))
    .withColumn("daily_coverage_pct", 100.0 * F.col("n_unique_timestamps") / 288)
    .withColumn("coverage_group",
        F.when(F.col("n_unique_timestamps") > 288, "More than 288")
         .when(F.col("n_unique_timestamps") == 288, "Exactly 288")
         .when(F.col("daily_coverage_pct") >= 95, "95% to below 100%")
         .otherwise("Below 95%"))
    .persist(StorageLevel.MEMORY_AND_DISK)
)
print("\n========== DAILY COMPLETENESS ==========")
daily.groupBy("coverage_group").count().orderBy("coverage_group").show(truncate=False)
print("\n========== DAILY COMPLETENESS BY YEAR ==========")
daily.groupBy("year").agg(
    F.count("*").alias("observed_station_days"),
    F.sum(F.when(F.col("n_unique_timestamps") == 288, 1).otherwise(0)).alias("days_with_288_readings"),
    F.sum(F.when(F.col("n_unique_timestamps") < 288, 1).otherwise(0)).alias("days_below_288"),
    F.sum(F.when(F.col("n_unique_timestamps") > 288, 1).otherwise(0)).alias("days_above_288")
).orderBy("year").show(truncate=False)
print("\n========== EXAMPLES OF INCOMPLETE DAYS ==========")
daily.filter(F.col("n_unique_timestamps") < 288).orderBy(
    "n_unique_timestamps", "station_id", "date").show(20, truncate=False)

print("\n========== MOST COMMON DAILY READING COUNTS BY YEAR ==========")

reading_counts = (
    daily.groupBy("year", "n_unique_timestamps")
    .count()
    .withColumnRenamed("count", "station_days")
)

ranking_window = Window.partitionBy("year").orderBy(
    F.desc("station_days"),
    F.desc("n_unique_timestamps")
)

(
    reading_counts
    .withColumn("rank", F.row_number().over(ranking_window))
    .filter(F.col("rank") <= 10)
    .orderBy("year", "rank")
    .show(80, truncate=False)
)


# 8. PROVISIONAL RULE: exactly 288 distinct timestamps and 288 rows.
# Incomplete days remain in daily for diagnostics; no scaling or filling.
# Main exploratory analysis: at least 95% daily coverage.
# No missing rainfall is filled or scaled.
MIN_READINGS_PER_DAY = 274

complete_daily = daily.filter(
    (F.col("n_unique_timestamps") >= MIN_READINGS_PER_DAY)
    & (F.col("n_unique_timestamps") <= EXPECTED_READINGS_PER_DAY)
    & (F.col("n_readings") == F.col("n_unique_timestamps"))
)

# 9. STATION-MONTH COVERAGE (calendar denominator includes absent days)
observed = daily.groupBy("station_id", "year", "month").agg(
    F.count("*").alias("observed_days"))
eligible = complete_daily.groupBy("station_id", "year", "month").agg(
    F.count("*").alias("complete_days"),
    F.avg("daily_rainfall_mm").alias("mean_daily_rainfall_mm"))
station_monthly = (
    observed.join(eligible, ["station_id", "year", "month"], "left")
    .fillna({"complete_days": 0})  # Fill the COUNT only, never rainfall.
    .withColumn("days_in_month", F.dayofmonth(F.last_day(
        F.make_date(F.col("year"), F.col("month"), F.lit(1)))))
    .withColumn("complete_day_coverage_pct",
                100.0 * F.col("complete_days") / F.col("days_in_month"))
)
print("\n========== STATION-MONTH COVERAGE ==========")
station_monthly.orderBy("complete_day_coverage_pct", "year", "month", "station_id").show(30, truncate=False)

# 10. MONTHLY MEANS: each eligible station-day has equal weight.
# A station-month coverage cutoff has not yet been finalized.
monthly_by_year = complete_daily.groupBy("year", "month").agg(
    F.avg("daily_rainfall_mm").alias("mean_daily_rainfall_mm"),
    F.count("*").alias("complete_station_days"),
    F.countDistinct("station_id").alias("n_stations")
).orderBy("year", "month")
print("\n========== MONTHLY RAINFALL BY YEAR ==========")
monthly_by_year.show(100, truncate=False)
# Equal weight per available year in the overall seasonal pattern.
monthly_pattern = monthly_by_year.groupBy("month").agg(
    F.avg("mean_daily_rainfall_mm").alias("mean_daily_rainfall_mm"),
    F.countDistinct("year").alias("n_years"),
    F.sum("complete_station_days").alias("complete_station_days")
).orderBy("month")
print("\n========== AVERAGE MONTHLY PATTERN, 2017–2024 ==========")
monthly_pattern.show(12, truncate=False)

# 11. PLOTS: collect only small monthly summaries to the driver.
pattern_pd = monthly_pattern.toPandas()
year_month_pd = monthly_by_year.toPandas()
if pattern_pd.empty:
    daily.unpersist()
    spark.stop()
    raise RuntimeError("No station-days passed the provisional completeness rule.")

plt.figure(figsize=(11, 5))
plt.bar(pattern_pd["month"], pattern_pd["mean_daily_rainfall_mm"], color="steelblue")
plt.xticks(range(1, 13), [calendar.month_abbr[m] for m in range(1, 13)])
plt.xlabel("Month")
plt.ylabel("Mean rainfall (mm/day/station)")
plt.title("Monthly rainfall pattern, 2017–2024\nStation-days with at least 95 percent reading coverage")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.figure(figsize=(12, 6))
for year, group in year_month_pd.groupby("year"):
    # Reindex so absent months show gaps rather than connecting across them.
    group = group.set_index("month").reindex(range(1, 13))
    plt.plot(group.index, group["mean_daily_rainfall_mm"], marker="o", label=str(int(year)))
plt.xticks(range(1, 13), [calendar.month_abbr[m] for m in range(1, 13)])
plt.xlabel("Month")
plt.ylabel("Mean rainfall (mm/day/station)")
plt.title("Monthly rainfall patterns by year\nProvisional: only station-days with at least 95 percent reading coverage")
plt.legend(title="Year", ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()

daily.unpersist()
spark.stop()
plt.show()
