# Databricks notebook source
#load_bronze
df_bronze=spark.read.parquet("/Volumes/workspace/default/fraud_data/bronze/transactions_raw/")
print(f"Bronze record count: {df_bronze.count():,}")
df_bronze.printSchema()

# COMMAND ----------

#creating time columns
from pyspark.sql.functions import col, floor, round as spark_round, when, log1p, stddev, avg
df_features = df_bronze \
    .withColumn(
        "hour_of_day",
        floor(col("Time") / 3600) % 24
    ) \
    .withColumn(
        "day_number",
        when(col("Time") < 86400, 1).otherwise(2)
    ) \
    .withColumn(
        "is_night",
        when(
            (floor(col("Time") / 3600) % 24).between(0, 5), 1
        ).otherwise(0)
    )

display(df_features.select("Time", "hour_of_day", "day_number", "is_night").limit(10))


# COMMAND ----------

#amount based features

df_features = df_features \
    .withColumn(
        "amount_log",
        log1p(col("Amount"))
    ) \
    .withColumn(
        "is_small_amount",
        when(col("Amount") < 10, 1).otherwise(0)
    ) \
    .withColumn(
        "is_large_amount",
        when(col("Amount") > 1000, 1).otherwise(0)
    )

display(df_features.select("Amount", "amount_log", "is_small_amount", "is_large_amount").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC - amount_log — log transform smooths out extreme values (a $10,000 transaction doesn't dwarf everything else
# MAGIC - is_small_amount — fraudsters often test cards with tiny transactions first.

# COMMAND ----------

print(f"Total columns now: {len(df_features.columns)}")
print(f"New columns: {['hour_of_day', 'day_number', 'is_night', 'amount_log', 'is_small_amount', 'is_large_amount']}")

# Check fraud vs legit on new features
df_features.groupBy("Class").agg(
    spark_round(avg("hour_of_day"), 2).alias("avg_hour"),
    spark_round(avg("Amount"), 2).alias("avg_amount"),
    spark_round(avg("is_night"), 4).alias("night_rate"),
    spark_round(avg("is_small_amount"), 4).alias("small_amount_rate")
).orderBy("Class").show()

# COMMAND ----------

# Calculate weight for each class
# Formula: total / (num_classes * class_count)

total = df_features.count()
fraud_count = df_features.filter(col("Class") == 1).count()
legit_count = df_features.filter(col("Class") == 0).count()
num_classes = 2

fraud_weight = total / (num_classes * fraud_count)
legit_weight = total / (num_classes * legit_count)

print(f"Total: {total:,}")
print(f"Legit count: {legit_count:,} | Weight: {legit_weight:.4f}")
print(f"Fraud count: {fraud_count:,} | Weight: {fraud_weight:.4f}")

# Add weight column
df_weighted = df_features.withColumn(
    "class_weight",
    when(col("Class") == 1, fraud_weight)
    .otherwise(legit_weight)
)

display(df_weighted.select("Class", "Amount", "class_weight").limit(10))

# COMMAND ----------

# Oversample the minority class (fraud) to balance the dataset
# We'll duplicate fraud rows to get ~10% fraud ratio

fraud_df = df_features.filter(col("Class") == 1)
legit_df = df_features.filter(col("Class") == 0)

# How many times to duplicate fraud rows
oversample_ratio = int(legit_count / fraud_count / 10)
print(f"Oversampling fraud by: {oversample_ratio}x")

from functools import reduce
from pyspark.sql import DataFrame

fraud_oversampled = reduce(
    DataFrame.union,
    [fraud_df] * oversample_ratio
)

df_balanced = legit_df.union(fraud_oversampled)

print(f"Balanced dataset size: {df_balanced.count():,}")
df_balanced.groupBy("Class").count().show()

# COMMAND ----------

# We'll go with the WEIGHTED approach for Silver
# It preserves original data distribution and is cleaner
# We use df_weighted as our Silver DataFrame

df_silver = df_weighted

print(f"Silver dataset: {df_silver.count():,} rows, {len(df_silver.columns)} columns")

# COMMAND ----------

df_silver = df_silver.select(
    "Time",
    "hour_of_day",
    "day_number",
    "is_night",
    "Amount",
    "amount_log",
    "is_small_amount",
    "is_large_amount",
    "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "class_weight",
    "Class"
)

print(f"Final Silver columns: {len(df_silver.columns)}")
display(df_silver.limit(5))

# COMMAND ----------

df_silver.write \
    .mode("overwrite") \
    .partitionBy("Class") \
    .parquet("/Volumes/workspace/default/fraud_data/silver/transactions_clean")

print("✅ Silver layer written!")

# COMMAND ----------

df_verify = spark.read.parquet("/Volumes/workspace/default/fraud_data/silver/transactions_clean")
print(f"Silver record count: {df_verify.count():,}")
df_verify.groupBy("Class").count().orderBy("Class").show()
display(df_verify.limit(5))

# COMMAND ----------

