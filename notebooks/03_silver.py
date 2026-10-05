# Databricks notebook source
df_bronze = spark.table("bronze_construction_permits")

# COMMAND ----------

df_bronze.show(5)

# COMMAND ----------

df_bronze.printSchema()

# COMMAND ----------

print(df_bronze.columns)

# COMMAND ----------

df_silver = df_bronze

for column in df_silver.columns:
    df_silver = df_silver.withColumnRenamed(column,column.strip().lower().replace(" ","_"))

# COMMAND ----------

print(df_silver.columns)

# COMMAND ----------

print([ 
c for c in df_silver.columns 
if "date" in c])

# COMMAND ----------

df_silver.printSchema()

# COMMAND ----------

# DBTITLE 1,e
from pyspark.sql.functions import try_to_date

# Raw date strings look like '2026/03/10 14:02:46+00' (plus some malformed values),
# which the default to_date() cast cannot parse — use an explicit, tolerant pattern.
date_fmt = "yyyy/MM/dd HH:mm:ssX"

df_silver = df_silver.withColumn("applicationdate", try_to_date("applicationdate", date_fmt))

# COMMAND ----------



df_silver = df_silver.withColumn("intakedate", try_to_date("intakedate", date_fmt))

# COMMAND ----------


df_silver = df_silver.withColumn("effectivedate", try_to_date("effectivedate", date_fmt))

# COMMAND ----------


df_silver = df_silver.withColumn("applicationdate", try_to_date("applicationdate", date_fmt))

# COMMAND ----------


df_silver = df_silver.withColumn("expirationdate", try_to_date("expirationdate", date_fmt))

# COMMAND ----------



df_silver = df_silver.withColumn("readyforreviewdate", try_to_date("readyforreviewdate", date_fmt))

# COMMAND ----------

df_silver.printSchema()

# COMMAND ----------

from pyspark.sql.functions import datediff, try_to_date


df_silver = df_silver.withColumn("issuedate", try_to_date("issuedate", date_fmt))
df_silver = df_silver.withColumn("processing_days", datediff("issuedate", "applicationdate"))

# COMMAND ----------

# DBTITLE 1,Cell 18
df_silver.select(
    "applicationdate",
    "issuedate",
    "processing_days"
).show(10)

# COMMAND ----------

df_silver.select("processing_days").summary().show()

# COMMAND ----------

df_silver.filter(
    df_silver.processing_days.isNull()
).count()

# COMMAND ----------

df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_construction_permits")

# COMMAND ----------

df_silver.filter(
    df_silver.processing_days.isNull()
).groupBy("status").count().show()

# COMMAND ----------

df_silver.select("processing_days").summary().show()

# COMMAND ----------

from pyspark.sql.functions import col
df_silver.filter(
    col("status").rlike("^[0-9.-]+$")
).select(
    "status"
).show(50, truncate=False)

# COMMAND ----------

df_silver.filter(
    col("processing_days").isNotNull()
).count()

# COMMAND ----------

df_silver.filter(
    col("processing_days").isNull()
).count()

# COMMAND ----------

df_silver.filter(
    col("status").rlike("^[0-9.-]+$")
).show(20, truncate=False)

# COMMAND ----------

df_silver.printSchema()

# COMMAND ----------

print(df_silver.columns)

# COMMAND ----------

df_silver.filter(
    col("status").rlike("^[0-9.-]+$")
).count()

# COMMAND ----------

df_silver.filter(
    col("applicationdate").isNull()
).count()

# COMMAND ----------

df_silver.filter(
    col("applicationdate").isNotNull()
).select(
    "applicationdate"
).show(20, truncate=False)

# COMMAND ----------

df_silver.select(
    "applicationdate"
).orderBy(
    col("applicationdate").desc()
).show(20, truncate=False)

# COMMAND ----------

df_silver = df_silver.withColumn(
    "processing_days",
    datediff("issuedate", "applicationdate")
)

# COMMAND ----------

from pyspark.sql.functions import col

df_silver.filter(
    col("applicationdate").isNull()
).select(
    "x",
    "y",
    "applicationdate",
    "isexcavation",
    "isfixture",
    "ispaving",
    "trackingnumber",
    "permitnumber",
    "intakedate",
    "issuedate",
    "status",
    "wlfulladdress",
    "permitteename",
    "ownername",
    "contractorname"
).show(20, truncate=False)

# COMMAND ----------

df_silver.select(
    "processing_days",
    "applicationdate",
    "issuedate",
    "status",
    "trackingnumber",
    "permitnumber",
    "wlfulladdress",
    "permitteename",
    "contractorname"
).orderBy(
    col("processing_days").desc()
).show(20, truncate=False)

# COMMAND ----------

df_silver.filter(
    col("applicationdate").isNull()
).select(
    "x",
    "y",
    "applicationdate",
    "isexcavation",
    "isfixture",
    "ispaving",
    "trackingnumber",
    "permitnumber",
    "intakedate",
    "issuedate",
    "status",
    "wlfulladdress",
    "permitteename",
    "ownername",
    "contractorname"
).show(20, truncate=False)

# COMMAND ----------

df_silver.select(
    "processing_days",
    "applicationdate",
    "issuedate",
    "status",
    "trackingnumber",
    "permitnumber",
    "wlfulladdress",
    "permitteename",
    "contractorname"
).orderBy(
    col("processing_days").desc()
).show(20, truncate=False)

# COMMAND ----------

from pyspark.sql.functions import col, when, sum

df_silver.select(
    sum(
        when(
            col("applicationdate").isNotNull() &
            col("issuedate").isNotNull(),
            1
        ).otherwise(0)
    ).alias("application_and_issue"),

    sum(
        when(
            col("applicationdate").isNotNull() &
            col("issuedate").isNull(),
            1
        ).otherwise(0)
    ).alias("application_only"),

    sum(
        when(
            col("applicationdate").isNull() &
            col("issuedate").isNotNull(),
            1
        ).otherwise(0)
    ).alias("issue_only"),

    sum(
        when(
            col("applicationdate").isNull() &
            col("issuedate").isNull(),
            1
        ).otherwise(0)
    ).alias("neither")
).show()

# COMMAND ----------

df_silver.filter(
    col("applicationdate").isNotNull()
).groupBy(
    "status"
).count().orderBy(
    col("count").desc()
).show(30, truncate=False)

# COMMAND ----------

from pyspark.sql.functions import col, when, sum

df_silver.select(
    sum(
        when(
            col("applicationdate").isNotNull() &
            col("issuedate").isNotNull(),
            1
        ).otherwise(0)
    ).alias("application_and_issue"),

    sum(
        when(
            col("applicationdate").isNotNull() &
            col("issuedate").isNull(),
            1
        ).otherwise(0)
    ).alias("application_only"),

    sum(
        when(
            col("applicationdate").isNull() &
            col("issuedate").isNotNull(),
            1
        ).otherwise(0)
    ).alias("issue_only"),

    sum(
        when(
            col("applicationdate").isNull() &
            col("issuedate").isNull(),
            1
        ).otherwise(0)
    ).alias("neither")
).show()

# COMMAND ----------

df_silver.filter(
    col("applicationdate").isNotNull()
).groupBy(
    "status"
).count().orderBy(
    col("count").desc()
).show(30, truncate=False)