# Databricks notebook source
from pyspark.sql.functions import col

df = spark.table("gold_construction_permits")

df_mature = df.filter(
    col("applicationdate") <= "2025-12-31"
)

print("Mature cohort:", df_mature.count())

# COMMAND ----------

feature_columns = [
    "application_year",
    "application_month",
    "application_day",
    "application_day_of_week",
    "application_quarter",
    "isexcavation",
    "isfixture",
    "ispaving",
    "islandscaping",
    "isprojections",
    "latitude",
    "longitude",
    "permitteename",
    "ownername",
    "applicantcompanyname"
]

df_model = df_mature.select(
    *feature_columns,
    "delayed"
)

df_model.printSchema()

# COMMAND ----------

df_model.show(5, truncate=False)

# COMMAND ----------

df_model.groupBy("delayed").count().orderBy("delayed").show()

# COMMAND ----------

df_mature.groupBy(
    "application_year",
    "application_month"
).count().orderBy(
    "application_year",
    "application_month"
).show(50)

# COMMAND ----------

df_2025 = df_model.filter(
    col("application_year") == 2025
)

print("2025 permits:", df_2025.count())

df_2025.groupBy("delayed").count().orderBy("delayed").show()

# COMMAND ----------

from pyspark.sql.functions import count, avg, round, col

# COMMAND ----------

df_2025 = df_model.filter(
    col("application_year") == 2025
)

print("2025 permits:", df_2025.count())

df_2025.groupBy(
    "application_month"
).agg(
    count("*").alias("permits"),
    round(avg("delayed") * 100, 2).alias("delay_rate_pct")
).orderBy("application_month").show()

# COMMAND ----------

train_df, test_df = df_2025.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training rows:", train_df.count())
print("Test rows:", test_df.count())

# COMMAND ----------

print("Training distribution:")

train_df.groupBy("delayed").count().orderBy("delayed").show()

print("Test distribution:")

test_df.groupBy("delayed").count().orderBy("delayed").show()

# COMMAND ----------

from pyspark.sql.functions import col

train_df = (
    train_df
    .withColumn("latitude", col("latitude").cast("double"))
    .withColumn("longitude", col("longitude").cast("double"))
)

test_df = (
    test_df
    .withColumn("latitude", col("latitude").cast("double"))
    .withColumn("longitude", col("longitude").cast("double"))
)

# COMMAND ----------

from pyspark.sql.functions import expr

train_df = train_df.withColumn(
    "latitude",
    expr("try_cast(latitude as double)")
).withColumn(
    "longitude",
    expr("try_cast(longitude as double)")
)

test_df = test_df.withColumn(
    "latitude",
    expr("try_cast(latitude as double)")
).withColumn(
    "longitude",
    expr("try_cast(longitude as double)")
)

# COMMAND ----------

from pyspark.sql.functions import col, expr

df = spark.table("gold_construction_permits")

df_2025 = df.filter(
    col("application_year") == 2025
)

train_df, test_df = df_2025.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training rows:", train_df.count())
print("Test rows:", test_df.count())

# COMMAND ----------

train_df = (
    train_df
    .withColumn("latitude", expr("try_cast(latitude as double)"))
    .withColumn("longitude", expr("try_cast(longitude as double)"))
)

test_df = (
    test_df
    .withColumn("latitude", expr("try_cast(latitude as double)"))
    .withColumn("longitude", expr("try_cast(longitude as double)"))
)

# COMMAND ----------

train_df.select(
    "latitude",
    "longitude"
).describe().show()

# COMMAND ----------

train_df.select(
    expr("sum(case when latitude is null then 1 else 0 end)").alias("missing_latitude"),
    expr("sum(case when longitude is null then 1 else 0 end)").alias("missing_longitude")
).show()

# COMMAND ----------

from pyspark.sql.functions import col, when

train_df = (
    train_df
    .withColumn(
        "latitude",
        when(
            (col("latitude") >= 38) & (col("latitude") <= 39),
            col("latitude")
        ).otherwise(None)
    )
    .withColumn(
        "longitude",
        when(
            (col("longitude") >= -78) & (col("longitude") <= -76),
            col("longitude")
        ).otherwise(None)
    )
)

test_df = (
    test_df
    .withColumn(
        "latitude",
        when(
            (col("latitude") >= 38) & (col("latitude") <= 39),
            col("latitude")
        ).otherwise(None)
    )
    .withColumn(
        "longitude",
        when(
            (col("longitude") >= -78) & (col("longitude") <= -76),
            col("longitude")
        ).otherwise(None)
    )
)

# COMMAND ----------

train_df.select(
    "latitude",
    "longitude"
).describe().show()

# COMMAND ----------

train_df.select(
    expr("sum(case when latitude is null then 1 else 0 end)").alias("missing_latitude"),
    expr("sum(case when longitude is null then 1 else 0 end)").alias("missing_longitude")
).show()

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer
)
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

numeric_cols = [
    "application_month",
    "application_day",
    "application_day_of_week",
    "application_quarter",
    "latitude",
    "longitude"
]

categorical_cols = [
    "isexcavation",
    "isfixture",
    "ispaving",
    "islandscaping",
    "isprojections",
    "permitteename",
    "ownername",
    "applicantcompanyname"
]

# COMMAND ----------

indexers = [
    StringIndexer(
        inputCol=column,
        outputCol=column + "_index",
        handleInvalid="keep"
    )
    for column in categorical_cols
]

# COMMAND ----------

encoder = OneHotEncoder(
    inputCols=[column + "_index" for column in categorical_cols],
    outputCols=[column + "_encoded" for column in categorical_cols]
)

# COMMAND ----------

imputer = Imputer(
    inputCols=["latitude", "longitude"],
    outputCols=["latitude_imputed", "longitude_imputed"],
    strategy="median"
)

# COMMAND ----------

assembler = VectorAssembler(
    inputCols=[
        "application_month",
        "application_day",
        "application_day_of_week",
        "application_quarter",
        "latitude_imputed",
        "longitude_imputed"
    ] + [
        column + "_encoded"
        for column in categorical_cols
    ],
    outputCol="features"
)

# COMMAND ----------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="delayed",
    maxIter=100
)

# COMMAND ----------

pipeline = Pipeline(
    stages=indexers + [
        encoder,
        imputer,
        assembler,
        lr
    ]
)

# COMMAND ----------

lr_model = pipeline.fit(train_df)

# COMMAND ----------

predictions = lr_model.transform(test_df)

predictions.select(
    "delayed",
    "prediction",
    "probability"
).show(10, truncate=False)

# COMMAND ----------

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

# COMMAND ----------

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="delayed",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

roc_auc = roc_evaluator.evaluate(predictions)

print("Logistic Regression ROC-AUC:", roc_auc)

# COMMAND ----------

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="delayed",
    predictionCol="prediction",
    metricName="f1"
)

f1 = f1_evaluator.evaluate(predictions)

print("Logistic Regression F1:", f1)

# COMMAND ----------

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="delayed",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

precision = precision_evaluator.evaluate(predictions)

print("Logistic Regression Precision:", precision)

# COMMAND ----------

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="delayed",
    predictionCol="prediction",
    metricName="weightedRecall"
)

recall = recall_evaluator.evaluate(predictions)

print("Logistic Regression Recall:", recall)

# COMMAND ----------

predictions.groupBy(
    "delayed",
    "prediction"
).count().orderBy(
    "delayed",
    "prediction"
).show()

# COMMAND ----------

df_tableau = spark.table("gold_construction_permits").select(
    "applicationdate",
    "issuedate",
    "processing_days",
    "delayed",
    "application_year",
    "application_month",
    "application_day_of_week",
    "application_quarter",
    "isexcavation",
    "isfixture",
    "ispaving",
    "islandscaping",
    "isprojections",
    "latitude",
    "longitude",
    "permitteename",
    "ownername",
    "applicantcompanyname"
)

# COMMAND ----------

df_tableau.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("tableau_construction_permits")

# COMMAND ----------

spark.table("tableau_construction_permits").count()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM tableau_construction_permits
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM tableau_construction_permits;