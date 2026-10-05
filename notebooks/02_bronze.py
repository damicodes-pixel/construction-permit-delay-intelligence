# Databricks notebook source
path = "file:/Workspace/Users/oluwado2@umbc.edu/Construction_Permits_in_2026.csv"

# COMMAND ----------

df_raw = spark.read.csv(path, header=True, inferSchema=True)

# COMMAND ----------

df_raw.show(10)

# COMMAND ----------

df_raw.columns

# COMMAND ----------

df_raw.printSchema()

# COMMAND ----------

df_raw.count()

# COMMAND ----------

from pyspark.sql.functions import col, sum

missing = df_raw.select([sum(col(c).isNull().cast("int")).alias(c) for c in df_raw.columns])

# COMMAND ----------

df_raw.write.format("delta").mode("overwrite").saveAsTable("bronze_construction_permits")

# COMMAND ----------

spark.sql("""SELECT * FROM bronze_construction_permits LIMIT 10 """).show()