# Databricks notebook source
from pyspark.sql.functions import (
    col,
    count,
    avg,
    median,
    min,
    max,
    round,
    when,
    sum
)

df = spark.table("gold_construction_permits")

print("Rows:", df.count())
print("Columns:", len(df.columns))

# COMMAND ----------

df.select("processing_days").summary().show()

# COMMAND ----------

df.groupBy("delayed").agg(
    count("*").alias("permits"),
    round(avg("processing_days"), 2).alias("avg_processing_days")
).orderBy("delayed").show()

# COMMAND ----------

df.groupBy("application_year").agg(
    count("*").alias("permits"),
    round(avg("processing_days"), 2).alias("avg_processing_days"),
    round(avg("delayed") * 100, 2).alias("delay_rate_pct")
).orderBy("application_year").show()

# COMMAND ----------

df.groupBy("application_year").agg(
    count("*").alias("permits"),
    round(avg("processing_days"), 2).alias("avg_processing_days"),
    round(median("processing_days"), 2).alias("median_processing_days"),
    round(avg("delayed") * 100, 2).alias("delay_rate_pct")
).orderBy("application_year").show()

# COMMAND ----------

df.groupBy(
    "application_year",
    "application_month"
).agg(
    count("*").alias("permits"),
    round(median("processing_days"), 2).alias("median_processing_days"),
    round(avg("processing_days"), 2).alias("avg_processing_days"),
    round(avg("delayed") * 100, 2).alias("delay_rate_pct")
).orderBy(
    "application_year",
    "application_month"
).show(50)

# COMMAND ----------

df.select(
    "applicationdate",
    "issuedate",
    "processing_days"
).orderBy(
    col("applicationdate").desc()
).show(30, truncate=False)

# COMMAND ----------

df_mature = df.filter(
    col("applicationdate") <= "2025-12-31"
)

print("Mature cohort:", df_mature.count())

df_mature.groupBy("application_year").agg(
    count("*").alias("permits"),
    round(median("processing_days"), 2).alias("median_processing_days"),
    round(avg("processing_days"), 2).alias("avg_processing_days"),
    round(avg("delayed") * 100, 2).alias("delay_rate_pct")
).orderBy("application_year").show()

# COMMAND ----------

df_mature.groupBy("application_year", "application_month").agg(
    count("*").alias("permits"),
    round(median("processing_days"), 2).alias("median_processing_days"),
    round(avg("processing_days"), 2).alias("avg_processing_days"),
    round(avg("delayed") * 100, 2).alias("delay_rate_pct")
).orderBy(
    "application_year",
    "application_month"
).show(50)

# COMMAND ----------

df_mature.groupBy("delayed").count().show()