# Databricks notebook source
#creating respected folders
folders = [
    "/Volumes/workspace/default/fraud_data/charts",
    "/Volumes/workspace/default/fraud_data/exports"
]

for folder in folders:
    dbutils.fs.mkdirs(folder)
    print(f"✅ Created: {folder}")

# COMMAND ----------

#loading all gold tables 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# Load gold tables into Pandas
df_summary      = spark.read.parquet("/Volumes/workspace/default/fraud_data/gold/fraud_summary").toPandas()
df_hourly       = spark.read.parquet("/Volumes/workspace/default/fraud_data/gold/hourly_fraud").toPandas()
df_daily        = spark.read.parquet("/Volumes/workspace/default/fraud_data/gold/daily_fraud").toPandas()
df_buckets      = spark.read.parquet("/Volumes/workspace/default/fraud_data/gold/amount_buckets").toPandas()
df_night        = spark.read.parquet("/Volumes/workspace/default/fraud_data/gold/night_vs_day").toPandas()
df_cumulative   = spark.read.parquet("/Volumes/workspace/default/fraud_data/gold/cumulative_fraud").toPandas()
df_predictions  = spark.read.parquet("/Volumes/workspace/default/fraud_data/gold/predictions").toPandas()
df_fi           = spark.read.parquet("/Volumes/workspace/default/fraud_data/gold/feature_importance").toPandas()

print("✅ All Gold tables loaded")

# COMMAND ----------

#class imbalance viz
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Credit Card Fraud — Class Distribution", fontsize=14, fontweight="bold")

labels = ["Legitimate", "Fraud"]
counts = df_summary["total_transactions"].values
colors = ["#42A5F5", "#EF5350"]

# Bar chart
axes[0].bar(labels, counts, color=colors, edgecolor="white", width=0.5)
axes[0].set_title("Transaction Count by Class")
axes[0].set_ylabel("Count")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for i, v in enumerate(counts):
    axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

# Pie chart
axes[1].pie(
    df_summary["pct_of_total"],
    labels=[f"{l}\n({p:.2f}%)" for l, p in zip(labels, df_summary["pct_of_total"])],
    colors=colors,
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
axes[1].set_title("Percentage Split")

plt.tight_layout()
plt.savefig("/Volumes/workspace/default/fraud_data/charts/class_distribution.png", dpi=150)
plt.show()
print("✅ Saved: class_distribution.png")

# COMMAND ----------

#FRAUD RATE BY HOUR
df_hourly_sorted = df_hourly.sort_values("hour_of_day")

fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.bar(df_hourly_sorted["hour_of_day"],
        df_hourly_sorted["total_txns"],
        color="#B0BEC5", alpha=0.6, label="Total Transactions")
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Total Transactions", color="#607D8B")
ax1.set_xticks(range(0, 24))

ax2 = ax1.twinx()
ax2.plot(df_hourly_sorted["hour_of_day"],
         df_hourly_sorted["fraud_rate_pct"],
         color="#EF5350", linewidth=2.5,
         marker="o", markersize=5, label="Fraud Rate %")
ax2.set_ylabel("Fraud Rate (%)", color="#EF5350")

ax1.set_title("Fraud Rate vs Transaction Volume by Hour", fontsize=13, fontweight="bold")
fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.92))
plt.tight_layout()
plt.savefig("/Volumes/workspace/default/fraud_data/charts/fraud_by_hour.png", dpi=150)
plt.show()
print("✅ Saved: fraud_by_hour.png")

# COMMAND ----------

# FRAUD BY AMOUNT BUCKET
df_buckets_sorted = df_buckets.sort_values("amount_bucket")

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["#EF5350" if r == df_buckets_sorted["fraud_rate_pct"].max()
          else "#90CAF9" for r in df_buckets_sorted["fraud_rate_pct"]]

bars = ax.bar(df_buckets_sorted["amount_bucket"],
              df_buckets_sorted["fraud_rate_pct"],
              color=colors, edgecolor="white")

ax.set_title("Fraud Rate by Transaction Amount — Highest Risk Bucket in Red",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Amount Bucket")
ax.set_ylabel("Fraud Rate (%)")
for bar, val in zip(bars, df_buckets_sorted["fraud_rate_pct"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f"{val:.3f}%", ha="center", fontsize=9)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("/Volumes/workspace/default/fraud_data/charts/fraud_by_amount.png", dpi=150)
plt.show()
print("✅ Saved: fraud_by_amount.png")

# COMMAND ----------

#NIGHT VS DAY
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Night vs Day Fraud Analysis", fontsize=13, fontweight="bold")

colors = ["#5C6BC0", "#FFA726"]

axes[0].bar(df_night["time_of_day"],
            df_night["fraud_count"],
            color=colors, edgecolor="white")
axes[0].set_title("Fraud Count")
axes[0].set_ylabel("Fraud Transactions")

axes[1].bar(df_night["time_of_day"],
            df_night["fraud_rate_pct"],
            color=colors, edgecolor="white")
axes[1].set_title("Fraud Rate %")
axes[1].set_ylabel("Fraud Rate (%)")

plt.tight_layout()
plt.savefig("/Volumes/workspace/default/fraud_data/charts/night_vs_day.png", dpi=150)
plt.show()
print("✅ Saved: night_vs_day.png")

# COMMAND ----------

#feature importance 
df_fi_top15 = df_fi.head(15)

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#EF5350" if "V" not in f else "#42A5F5"
          for f in df_fi_top15["feature"]]

ax.barh(df_fi_top15["feature"][::-1],
        df_fi_top15["importance"][::-1],
        color=colors[::-1], edgecolor="white")
ax.set_title("Top 15 Feature Importances — Random Forest\n(Red = Engineered Features)",
             fontsize=13, fontweight="bold")

             
ax.set_xlabel("Importance Score")

red_patch = mpatches.Patch(color="#EF5350", label="Engineered Features")
blue_patch = mpatches.Patch(color="#42A5F5", label="PCA Features (V1-V28)")
ax.legend(handles=[red_patch, blue_patch])

plt.tight_layout()
plt.savefig("/Volumes/workspace/default/fraud_data/charts/feature_importance.png", dpi=150)
plt.show()
print("✅ Saved: feature_importance.png")

# COMMAND ----------

# CUMULATIVE FRAUD OVER TIME
df_cum_sorted = df_cumulative.sort_values("hour_of_day")

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(df_cum_sorted["hour_of_day"],
                df_cum_sorted["cumulative_frauds"],
                alpha=0.3, color="#EF5350")
ax.plot(df_cum_sorted["hour_of_day"],
        df_cum_sorted["cumulative_frauds"],
        color="#EF5350", linewidth=2.5, marker="o")
ax.set_title("Cumulative Fraud Count by Hour of Day",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Cumulative Frauds")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig("/Volumes/workspace/default/fraud_data/charts/cumulative_fraud.png", dpi=150)
plt.show()
print("✅ Saved: cumulative_fraud.png")

# COMMAND ----------

#model prediction results
df_fi_sorted = df_fi.sort_values("importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#EF5350" if "V" not in f else "#42A5F5"
          for f in df_fi_sorted["feature"]]
ax.barh(df_fi_sorted["feature"],
        df_fi_sorted["importance"],
        color=colors)

red_patch   = mpatches.Patch(color="#EF5350", label="Engineered Features")
blue_patch  = mpatches.Patch(color="#42A5F5", label="PCA Features (V1-V28)")
ax.legend(handles=[red_patch, blue_patch])

ax.set_title("Top 15 Feature Importances — Random Forest", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("/Volumes/workspace/default/fraud_data/charts/feature_importance.png", dpi=150)
plt.show()
print("✅ Saved: feature_importance.png")

# COMMAND ----------

#heatmap
from pyspark.sql.functions import col as spark_col
import numpy as np

# Recalculate from predictions
df_pred_spark = spark.read.parquet(
    "/Volumes/workspace/default/fraud_data/gold/predictions"
)

tp = df_pred_spark.filter((spark_col("Class")==1) & (spark_col("predicted_fraud")==1)).count()
tn = df_pred_spark.filter((spark_col("Class")==0) & (spark_col("predicted_fraud")==0)).count()
fp = df_pred_spark.filter((spark_col("Class")==0) & (spark_col("predicted_fraud")==1)).count()
fn = df_pred_spark.filter((spark_col("Class")==1) & (spark_col("predicted_fraud")==0)).count()

cm_data = np.array([[tn, fp], [fn, tp]])
labels  = [["TN", "FP"], ["FN", "TP"]]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_data, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"], fontsize=12)
ax.set_yticklabels(["Actual Legit", "Actual Fraud"], fontsize=12)

for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{labels[i][j]}\n{cm_data[i][j]:,}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if cm_data[i][j] > cm_data.max()/2 else "black")

ax.set_title("Confusion Matrix — Random Forest", fontsize=14, fontweight="bold")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("/Volumes/workspace/default/fraud_data/charts/confusion_matrix.png", dpi=150)
plt.show()
print("✅ Saved: confusion_matrix.png")

# COMMAND ----------

#EXPORT GOLD TABLES TO CSV
export_map = {
    "fraud_summary"    : df_summary,
    "hourly_fraud"     : df_hourly,
    "daily_fraud"      : df_daily,
    "amount_buckets"   : df_buckets,
    "night_vs_day"     : df_night,
    "feature_importance": df_fi,
}

for name, df in export_map.items():
    path = f"/Volumes/workspace/default/fraud_data/exports/{name}.csv"
    df.to_csv(path, index=False)
    print(f"✅ Exported: {name}.csv")

# COMMAND ----------

