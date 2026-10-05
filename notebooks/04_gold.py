# Databricks notebook source
from pyspark.sql.functions import (
    col,
    when,
    year,
    month,
    dayofweek,
    dayofmonth,
    quarter
)

df_silver = spark.table("silver_construction_permits")

# COMMAND ----------

from pyspark.sql.functions import (
    col,
    when,
    year,
    month,
    dayofweek,
    dayofmonth,
    quarter
)

df_gold = df_silver.filter(
    col("applicationdate").isNotNull() &
    col("issuedate").isNotNull()
)

# COMMAND ----------

df_gold = df_gold.withColumn(
    "delayed",
    when(col("processing_days") > 40, 1).otherwise(0)
)

# COMMAND ----------

df_gold.groupBy("delayed").count().show()

# COMMAND ----------

df_gold = (
    df_gold
    .withColumn("application_year", year("applicationdate"))
    .withColumn("application_month", month("applicationdate"))
    .withColumn("application_day", dayofmonth("applicationdate"))
    .withColumn("application_day_of_week", dayofweek("applicationdate"))
    .withColumn("application_quarter", quarter("applicationdate"))
)

# COMMAND ----------

for c in [
    "isexcavation",
    "isfixture",
    "ispaving",
    "islandscaping",
    "isprojections",
    "ispsrental"
]:
    print(f"\n--- {c} ---")
    df_gold.groupBy(c).count().orderBy(
        col("count").desc()
    ).show(10, truncate=False)

# COMMAND ----------

for c in [
    "status",
    "permitteename",
    "ownername",
    "contractorname",
    "applicantcompanyname"
]:
    print(f"\n--- {c} ---")
    print("Distinct:", df_gold.select(c).distinct().count())

# COMMAND ----------

df_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_construction_permits")

# COMMAND ----------

spark.sql("""
SELECT *
FROM gold_construction_permits
LIMIT 10
""").show()

# COMMAND ----------

from pyspark.sql.functions import col, sum

df_gold.select(
    sum(col("applicationdate").isNull().cast("int")).alias("missing_application_date"),
    sum(col("issuedate").isNull().cast("int")).alias("missing_issue_date"),
    sum(col("processing_days").isNull().cast("int")).alias("missing_processing_days"),
    sum(col("latitude").isNull().cast("int")).alias("missing_latitude"),
    sum(col("longitude").isNull().cast("int")).alias("missing_longitude"),
    sum(col("permitteename").isNull().cast("int")).alias("missing_permittee"),
    sum(col("ownername").isNull().cast("int")).alias("missing_owner"),
    sum(col("contractorname").isNull().cast("int")).alias("missing_contractor"),
    sum(col("applicantcompanyname").isNull().cast("int")).alias("missing_applicant_company"),
    sum(col("workdetail").isNull().cast("int")).alias("missing_workdetail")
).show()

# COMMAND ----------

df_gold.select(
    "permitteename",
    "ownername",
    "contractorname",
    "applicantcompanyname"
).describe().show()

# COMMAND ----------

df_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_construction_permits")

# COMMAND ----------

spark.sql("""
SELECT
    delayed,
    COUNT(*) AS permits,
    ROUND(AVG(processing_days), 2) AS avg_processing_days
FROM gold_construction_permits
GROUP BY delayed
ORDER BY delayed
""").show()