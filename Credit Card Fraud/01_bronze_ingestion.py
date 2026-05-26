# Databricks notebook source
#reading the CSV 
df_raw=spark.read.csv("/Volumes/workspace/default/fraud_data/creditcard.csv",header=True,inferSchema=True)

# COMMAND ----------

print(f"Total Rows:{df_raw.count():,}")
print("Total columns:",len(df_raw.columns))

# COMMAND ----------

df_raw.printSchema()

# COMMAND ----------

display(df_raw.limit(10))

# COMMAND ----------

#looking for class imbalance
from pyspark.sql.functions import col, count, round as spark_round
display(df_raw.groupBy("Class").agg(count("*").alias("Total_transaction")).withColumn("Percentage_Distribution",spark_round(((col("Total_transaction")/(df_raw.count()))*100),2)))

# COMMAND ----------

from pyspark.sql.functions import avg, min, max, stddev

df_raw.groupBy("Class").agg(
    spark_round(avg("Amount"), 2).alias("avg_amount"),
    spark_round(min("Amount"), 2).alias("min_amount"),
    spark_round(max("Amount"), 2).alias("max_amount"),
    spark_round(stddev("Amount"), 2).alias("stddev_amount")
).orderBy("Class").show()

# COMMAND ----------

from pyspark.sql.functions import floor

# Convert seconds to hour of day (0-47 since it's 2 days)
display(df_raw.withColumn("hour", floor(col("Time") / 3600)).groupBy("hour", "Class").count().orderBy("hour","Class").limit(50))

# COMMAND ----------

from pyspark.sql.functions import col, sum, when
#counting null values
null_counts=df.select(
    [
        sum(when(col(c).isNull(),1).otherwise(0)).alias(c)
        for c in df.columns
    ]
)
display(null_counts)
#no nulls

# COMMAND ----------

#writing bronze layer 
from pyspark.sql.functions import current_timestamp
df_bronze=df_raw.withColumn("ingestion_timestamp", current_timestamp())

df_bronze.write.mode("overwrite").parquet("/Volumes/workspace/default/fraud_data/bronze/transactions_raw")
print("Bronze Layer Written Successfully!")

# COMMAND ----------

#verify check 
df_verify = spark.read.parquet("/Volumes/workspace/default/fraud_data/bronze/transactions_raw")
print(f"Bronze record count: {df_verify.count():,}")
display(df_verify.limit(5))

# COMMAND ----------

