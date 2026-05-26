# Databricks notebook source
df_silver = spark.read.parquet(
    "/Volumes/workspace/default/fraud_data/silver/transactions_clean"
)

print(f"Silver records: {df_silver.count():,}")
df_silver.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# All features we'll use
feature_cols = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26",
    "V27", "V28",
    "amount_log",       # engineered on Day 2
    "hour_of_day",      # engineered on Day 2
    "is_night",         # engineered on Day 2
    "is_small_amount",  # engineered on Day 2
    "is_large_amount"   # engineered on Day 2
]

print(f"Total features: {len(feature_cols)}")

# COMMAND ----------

#vectorizing
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw"
)

df_assembled = assembler.transform(df_silver)
display(df_assembled.select("features_raw", "Class", "class_weight").limit(5))

# COMMAND ----------

#Scale features
# StandardScaler normalizes features to same scale
# Important because V1-V28 and Amount_log have very different ranges

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

print("✅ Features assembled and scaled")
display(df_scaled.select("features", "Class").limit(3))

# COMMAND ----------

#TRAIN / TEST SPLIT
# 80% train, 20% test — stratified by Class
# seed=42 for reproducibility (always use a seed — shows good practice)

df_train, df_test = df_scaled.randomSplit([0.8, 0.2], seed=42)

print(f"Training set: {df_train.count():,}")
print(f"Test set:     {df_test.count():,}")

# Verify fraud distribution is maintained in both splits
from pyspark.sql.functions import col, count, round as spark_round

total_train = df_train.count()
total_test = df_test.count()

print("\nTrain class distribution:")
df_train.groupBy("Class") \
    .agg(spark_round(count("*") * 100.0 / total_train, 4).alias("pct")) \
    .orderBy("Class").show()

print("Test class distribution:")
df_test.groupBy("Class") \
    .agg(spark_round(count("*") * 100.0 / total_test, 4).alias("pct")) \
    .orderBy("Class").show()

# COMMAND ----------

#MODEL 1: LOGISTIC REGRESSION
lr = LogisticRegression(
    featuresCol="features",
    labelCol="Class",
    weightCol="class_weight",   # using our Day 2 weights
    maxIter=20,
    regParam=0.01
)

lr_model = lr.fit(df_train)
print("✅ Logistic Regression trained")

# COMMAND ----------

#Predict
lr_predictions = lr_model.transform(df_test)

display(lr_predictions.select(
    "Class",
    "prediction",
    "probability"
).limit(10))

# COMMAND ----------

#Evaluate LR
# AUC-ROC
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="Class",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

# F1, Precision, Recall
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="Class",
    predictionCol="prediction"
)

lr_auc       = auc_evaluator.evaluate(lr_predictions)
lr_f1        = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "f1"})
lr_precision = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "weightedPrecision"})
lr_recall    = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "weightedRecall"})

print("=== Logistic Regression Results ===")
print(f"AUC-ROC:   {lr_auc:.4f}")
print(f"F1 Score:  {lr_f1:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")

# COMMAND ----------

#MODEL 2: RANDOM FOREST
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="Class",
    weightCol="class_weight",
    numTrees=100,
    maxDepth=10,
    seed=42
)

rf_model = rf.fit(df_train)
print("✅ Random Forest trained")

# COMMAND ----------

# Predict & Evaluate
rf_predictions = rf_model.transform(df_test)

rf_auc       = auc_evaluator.evaluate(rf_predictions)
rf_f1        = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "f1"})
rf_precision = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "weightedPrecision"})
rf_recall    = multi_evaluator.evaluate(rf_predictions, {multi_evaluator.metricName: "weightedRecall"})

print("=== Random Forest Results ===")
print(f"AUC-ROC:   {rf_auc:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")

# COMMAND ----------

# Compare both models
print("=" * 45)
print(f"{'Metric':<15} {'Log Regression':>15} {'Random Forest':>15}")
print("=" * 45)
print(f"{'AUC-ROC':<15} {lr_auc:>15.4f} {rf_auc:>15.4f}")
print(f"{'F1 Score':<15} {lr_f1:>15.4f} {rf_f1:>15.4f}")
print(f"{'Precision':<15} {lr_precision:>15.4f} {rf_precision:>15.4f}")
print(f"{'Recall':<15} {lr_recall:>15.4f} {rf_recall:>15.4f}")
print("=" * 45)

# COMMAND ----------

#Which features matter most?
import pandas as pd

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.featureImportances.toArray()
}).sort_values("importance", ascending=False)

print("Top 15 most important features:")
print(feature_importance.head(15).to_string(index=False))

# COMMAND ----------

#CONFUSION MATRIX
from pyspark.sql.functions import col

# True Positives, False Positives, True Negatives, False Negatives
cm = rf_predictions.groupBy("Class", "prediction").count().orderBy("Class", "prediction")
display(cm)

# Human readable
tp = rf_predictions.filter((col("Class") == 1) & (col("prediction") == 1)).count()
tn = rf_predictions.filter((col("Class") == 0) & (col("prediction") == 0)).count()
fp = rf_predictions.filter((col("Class") == 0) & (col("prediction") == 1)).count()
fn = rf_predictions.filter((col("Class") == 1) & (col("prediction") == 0)).count()

print(f"\n=== Confusion Matrix ===")
print(f"True Positives  (caught frauds):      {tp}")
print(f"True Negatives  (correct legit):       {tn:,}")
print(f"False Positives (legit flagged wrong): {fp}")
print(f"False Negatives (missed frauds):       {fn}  ← most critical")
print(f"\nFrauds missed: {fn} out of {tp + fn} total frauds")

# COMMAND ----------

#SAVE PREDICTIONS TO GOLD
from pyspark.sql.functions import current_timestamp

df_predictions_gold = rf_predictions.select(
    "Time",
    "hour_of_day",
    "day_number",
    "is_night",
    "Amount",
    "amount_log",
    "is_small_amount",
    "Class",
    "prediction",
    "probability"
) \
.withColumn("predicted_fraud", col("prediction").cast("int")) \
.withColumn("scored_at", current_timestamp())

df_predictions_gold.write \
    .mode("overwrite") \
    .parquet("/Volumes/workspace/default/fraud_data/gold/predictions")

print(f"✅ Predictions written: {df_predictions_gold.count():,} rows")

# COMMAND ----------

#Save feature importance
fi_spark = spark.createDataFrame(feature_importance)
fi_spark.write \
    .mode("overwrite") \
    .parquet("/Volumes/workspace/default/fraud_data/gold/feature_importance")

print("✅ Feature importance written")

# COMMAND ----------

