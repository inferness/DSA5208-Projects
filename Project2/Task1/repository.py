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
# 6. CHECK SAMPLE DATA
# ============================================================

print("\n========== SAMPLE DATA ==========")

df.show(10, truncate=False)


# ============================================================
# 7. PROFILE: DATE RANGE
# ============================================================

print("\n========== DATE RANGE ==========")

df.select(
    F.min("timestamp").alias("earliest_timestamp"),
    F.max("timestamp").alias("latest_timestamp")
).show()


# ============================================================
# 8. PROFILE: RECORDS AND STATIONS PER YEAR
# ============================================================

print("\n========== RECORDS AND STATIONS PER YEAR ==========")

year_summary = (
    df
    .withColumn("year", F.year("timestamp"))
    .groupBy("year")
    .agg(
        F.count("*").alias("number_of_records"),
        F.countDistinct("station_id").alias("number_of_stations")
    )
    .orderBy("year")
)

year_summary.show()


# ============================================================
# 9. PROFILE: BASIC RAINFALL STATISTICS
# ============================================================

print("\n========== RAINFALL STATISTICS ==========")

df.select("reading_value").describe().show()


# ============================================================
# 10. PROFILE: MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])

missing_values.show(truncate=False)


# ============================================================
# 11. PROFILE: DUPLICATE STATION-TIMESTAMP RECORDS
# ============================================================

print("\n========== DUPLICATE STATION-TIMESTAMP RECORDS ==========")

duplicate_station_timestamp = (
    df
    .groupBy("station_id", "timestamp")
    .count()
    .filter(F.col("count") > 1)
)

print(
    "Number of duplicated station-timestamp combinations:",
    duplicate_station_timestamp.count()
)

duplicate_station_timestamp.show(20, truncate=False)


# ============================================================
# 12. PROFILE: READING TYPES AND UNITS
# ============================================================

print("\n========== READING TYPES ==========")

df.select("reading_type").distinct().show(truncate=False)

print("\n========== READING UNITS ==========")

df.select("reading_unit").distinct().show(truncate=False)


# ============================================================
# 14. PROFILE: STATION METADATA CONSISTENCY
# ============================================================

print("\n========== STATION METADATA CONSISTENCY ==========")

station_consistency = (
    df
    .groupBy("station_id")
    .agg(
        F.countDistinct("station_name").alias("number_of_names"),
        F.countDistinct("station_device_id").alias("number_of_device_ids"),
        F.countDistinct("location_longitude").alias("number_of_longitudes"),
        F.countDistinct("location_latitude").alias("number_of_latitudes"),
        F.min("timestamp").alias("first_observation"),
        F.max("timestamp").alias("last_observation")
    )
    .orderBy("station_id")
)

station_consistency.show(100, truncate=False)


# Show only stations whose metadata changed
print("\n========== STATIONS WITH CHANGING METADATA ==========")

station_changes = (
    station_consistency
    .filter(
        (F.col("number_of_names") > 1) |
        (F.col("number_of_device_ids") > 1) |
        (F.col("number_of_longitudes") > 1) |
        (F.col("number_of_latitudes") > 1)
    )
)

station_changes.show(100, truncate=False)


# ============================================================
# 15. STATION COVERAGE BY YEAR
# ============================================================

print("\n========== STATION COVERAGE BY YEAR ==========")

station_year = (
    df
    .withColumn("year", F.year("timestamp"))
    .groupBy("station_id")
    .agg(
        F.sort_array(F.collect_set("year")).alias("years_available")
    )
    .orderBy("station_id")
)

station_year.show(100, truncate=False)


# ============================================================
# 17. PROFILE: TIME INTERVALS BETWEEN READINGS
# ============================================================

print("\n========== TIME INTERVALS BETWEEN READINGS ==========")

# For each station, order observations chronologically
station_window = (
    Window
    .partitionBy("station_id")
    .orderBy("timestamp")
)

# Find the previous timestamp for each observation
# and calculate the gap in seconds
time_gaps = (
    df
    .select("station_id", "timestamp")
    .withColumn(
        "previous_timestamp",
        F.lag("timestamp").over(station_window)
    )
    .withColumn(
        "gap_seconds",
        F.unix_timestamp("timestamp")
        - F.unix_timestamp("previous_timestamp")
    )
    .filter(F.col("previous_timestamp").isNotNull())
)


# ============================================================
# 18. MOST COMMON TIME GAPS
# ============================================================

print("\n========== MOST COMMON TIME GAPS ==========")

(
    time_gaps
    .groupBy("gap_seconds")
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)


# ============================================================
# 19. SUMMARY OF TIME GAPS
# ============================================================

print("\n========== TIME GAP SUMMARY ==========")

time_gaps.select("gap_seconds").describe().show()


# ============================================================
# 20. COUNT GAPS LONGER THAN 5 MINUTES
# ============================================================

print("\n========== GAPS LONGER THAN 5 MINUTES ==========")

large_gaps = time_gaps.filter(F.col("gap_seconds") > 300)

print(
    "Number of gaps longer than 5 minutes:",
    large_gaps.count()
)


# ============================================================
# 21. LARGEST GAPS
# ============================================================

print("\n========== LARGEST TIME GAPS ==========")

(
    large_gaps
    .orderBy(F.desc("gap_seconds"))
    .show(20, truncate=False)
)


# ============================================================
# 22. INVESTIGATE UNUSUAL SHORT GAPS
# ============================================================

print("\n========== GAPS SHORTER THAN 5 MINUTES ==========")

short_gaps = time_gaps.filter(F.col("gap_seconds") < 300)

print(
    "Number of gaps shorter than 5 minutes:",
    short_gaps.count()
)

(
    short_gaps
    .groupBy("gap_seconds")
    .count()
    .orderBy("gap_seconds")
    .show(50, truncate=False)
)


print("\n========== EXAMPLES OF SHORT GAPS ==========")

(
    short_gaps
    .orderBy("timestamp")
    .show(30, truncate=False)
)


# ============================================================
# 23. GAP CATEGORIES
# ============================================================

print("\n========== TIME GAP CATEGORIES ==========")

gap_categories = (
    time_gaps
    .withColumn(
        "gap_category",
        F.when(F.col("gap_seconds") < 300, "Shorter than 5 min")
         .when(F.col("gap_seconds") == 300, "Exactly 5 min")
         .when(F.col("gap_seconds") <= 1800, "5-30 min gap")
         .when(F.col("gap_seconds") <= 86400, "30 min-1 day gap")
         .otherwise("More than 1 day")
    )
    .groupBy("gap_category")
    .count()
    .orderBy(F.desc("count"))
)

gap_categories.show(truncate=False)


# ============================================================
# 24. STOP SPARK
# ============================================================

spark.stop()