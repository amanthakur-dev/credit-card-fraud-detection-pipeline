# Databricks notebook source
df_silver = spark.read.parquet("/Volumes/workspace/default/fraud_data/silver/transactions_clean")
df_silver.createOrReplaceTempView("transactions")

print(f"Silver records loaded: {df_silver.count():,}")

# COMMAND ----------

#Overall Fraud Summary
df_summary = spark.sql("""
    SELECT
        Class,
        COUNT(*)                                AS total_transactions,
        ROUND(COUNT(*) * 100.0 
              / SUM(COUNT(*)) OVER(), 4)        AS pct_of_total,
        ROUND(AVG(Amount), 2)                   AS avg_amount,
        ROUND(MIN(Amount), 2)                   AS min_amount,
        ROUND(MAX(Amount), 2)                   AS max_amount,
        ROUND(SUM(Amount), 2)                   AS total_amount
    FROM transactions
    GROUP BY Class
    ORDER BY Class
""")

display(df_summary)

# COMMAND ----------

#Fraud by Hour of Day
df_hourly_fraud = spark.sql("""
    WITH hourly AS (
        SELECT
            hour_of_day,
            COUNT(*)                            AS total_txns,
            SUM(CASE WHEN Class = 1 THEN 1 
                ELSE 0 END)                     AS fraud_count,
            SUM(CASE WHEN Class = 0 THEN 1 
                ELSE 0 END)                     AS legit_count
        FROM transactions
        GROUP BY hour_of_day
    )
    SELECT
        hour_of_day,
        total_txns,
        fraud_count,
        legit_count,
        ROUND(fraud_count * 100.0 
              / total_txns, 4)                  AS fraud_rate_pct,
        RANK() OVER (
            ORDER BY fraud_count DESC
        )                                       AS fraud_rank
    FROM hourly
    ORDER BY hour_of_day
""")

display(df_hourly_fraud)

# COMMAND ----------

# Fraud by Day
df_daily_fraud = spark.sql("""
    SELECT
        day_number,
        COUNT(*)                                AS total_txns,
        SUM(CASE WHEN Class = 1 THEN 1 
            ELSE 0 END)                         AS fraud_count,
        ROUND(SUM(CASE WHEN Class = 1 THEN 1 
            ELSE 0 END) * 100.0 
            / COUNT(*), 4)                      AS fraud_rate_pct,
        ROUND(AVG(CASE WHEN Class = 1 
            THEN Amount END), 2)                AS avg_fraud_amount,
        ROUND(AVG(CASE WHEN Class = 0 
            THEN Amount END), 2)                AS avg_legit_amount
    FROM transactions
    GROUP BY day_number
    ORDER BY day_number
""")

display(df_daily_fraud)

# COMMAND ----------

#Amount Buckets: Where Does Fraud Hide?
df_amount_buckets = spark.sql("""
    SELECT
        CASE
            WHEN Amount < 10    THEN '1. Under $10'
            WHEN Amount < 50    THEN '2. $10 - $50'
            WHEN Amount < 100   THEN '3. $50 - $100'
            WHEN Amount < 500   THEN '4. $100 - $500'
            WHEN Amount < 1000  THEN '5. $500 - $1000'
            ELSE                     '6. Over $1000'
        END                                     AS amount_bucket,
        COUNT(*)                                AS total_txns,
        SUM(CASE WHEN Class = 1 THEN 1 
            ELSE 0 END)                         AS fraud_count,
        ROUND(SUM(CASE WHEN Class = 1 THEN 1 
            ELSE 0 END) * 100.0 
            / COUNT(*), 4)                      AS fraud_rate_pct,
        ROUND(AVG(Amount), 2)                   AS avg_amount
    FROM transactions
    GROUP BY amount_bucket
    ORDER BY amount_bucket
""")

display(df_amount_buckets)

# COMMAND ----------

# Night vs Day Fraud Rate
df_night_fraud = spark.sql("""
    SELECT
        CASE is_night 
            WHEN 1 THEN 'Night (12AM-5AM)'
            ELSE 'Day (6AM-11PM)'
        END                                     AS time_of_day,
        COUNT(*)                                AS total_txns,
        SUM(CASE WHEN Class = 1 THEN 1 
            ELSE 0 END)                         AS fraud_count,
        ROUND(SUM(CASE WHEN Class = 1 THEN 1 
            ELSE 0 END) * 100.0 
            / COUNT(*), 4)                      AS fraud_rate_pct,
        ROUND(AVG(Amount), 2)                   AS avg_amount
    FROM transactions
    GROUP BY is_night
    ORDER BY is_night DESC
""")

display(df_night_fraud)

# COMMAND ----------

# Running Cumulative Fraud Count Over Time
df_cumulative = spark.sql("""
    WITH hourly_fraud AS (
        SELECT
            hour_of_day,
            SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS hourly_frauds
        FROM transactions
        GROUP BY hour_of_day
    )
    SELECT
        hour_of_day,
        hourly_frauds,
        SUM(hourly_frauds) OVER (
            ORDER BY hour_of_day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                       AS cumulative_frauds
    FROM hourly_fraud
    ORDER BY hour_of_day
""")

display(df_cumulative)

# COMMAND ----------

# Top 10 Highest Value Fraud Transactions
df_top_frauds = spark.sql("""
    SELECT
        Time,
        hour_of_day,
        Amount,
        amount_log,
        is_night,
        V1, V2, V3, V4, V5,
        RANK() OVER (ORDER BY Amount DESC)      AS amount_rank
    FROM transactions
    WHERE Class = 1
    QUALIFY RANK() OVER (ORDER BY Amount DESC) <= 10
""")

display(df_top_frauds)

# COMMAND ----------

gold_tables = {
    "fraud_summary"     : df_summary,
    "hourly_fraud"      : df_hourly_fraud,
    "daily_fraud"       : df_daily_fraud,
    "amount_buckets"    : df_amount_buckets,
    "night_vs_day"      : df_night_fraud,
    "cumulative_fraud"  : df_cumulative,
    "top_frauds"        : df_top_frauds
}

for name, df in gold_tables.items():
    df.write.mode("overwrite") \
      .parquet(f"/Volumes/workspace/default/fraud_data/gold/{name}")
    print(f"✅ Written: {name}")

# COMMAND ----------

for name in gold_tables.keys():
    df_check = spark.read.parquet(f"/Volumes/workspace/default/fraud_data/gold/{name}")
    print(f"{name}: {df_check.count()} rows")

# COMMAND ----------

