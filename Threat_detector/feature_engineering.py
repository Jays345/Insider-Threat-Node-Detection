import mysql.connector
import pandas as pd
import numpy as np
from db import get_db_connection
connection = get_db_connection()

query = """
SELECT 
    user_id,
    login_time,
    log_date,
    files_accessed,
    sensitive_files_accessed,
    failed_login_attempts,
    usb_connected,
    is_anomalous
FROM activity_logs
"""

df = pd.read_sql(query, connection)
connection.close()

print("Total records loaded:", len(df))


# SAFE DATE + TIME PROCESSING
df['login_time'] = pd.to_datetime(df['login_time'], errors='coerce')
df['log_date'] = pd.to_datetime(df['log_date'], errors='coerce')

df['login_hour'] = df['login_time'].dt.hour

# After-hours flag (9 AM – 6 PM)
df['is_after_hours'] = df['login_hour'].apply(
    lambda x: 1 if x < 9 or x >= 18 else 0
)


# USER BASELINE CALCULATION

user_stats = df.groupby('user_id').agg({
    'files_accessed': ['mean', 'std'],
    'sensitive_files_accessed': ['mean', 'std']
}).reset_index()

user_stats.columns = [
    'user_id',
    'files_mean', 'files_std',
    'sensitive_mean', 'sensitive_std'
]

df = df.merge(user_stats, on='user_id', how='left')


df['files_std'] = df['files_std'].replace(0, 1)
df['sensitive_std'] = df['sensitive_std'].replace(0, 1)


# DEVIATION FEATURES

df['files_deviation'] = df['files_accessed'] - df['files_mean']
df['sensitive_deviation'] = df['sensitive_files_accessed'] - df['sensitive_mean']

# Z-SCORE BEHAVIOR DRIFT
df['files_zscore'] = df['files_deviation'] / df['files_std']
df['sensitive_zscore'] = df['sensitive_deviation'] / df['sensitive_std']


# 7-DAY ROLLING AVERAGE FEATURES

df = df.sort_values(['user_id', 'log_date'])

df['rolling_files_7d'] = df.groupby('user_id')['files_accessed'] \
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())

df['rolling_sensitive_7d'] = df.groupby('user_id')['sensitive_files_accessed'] \
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())


# FINAL FEATURE SET

feature_columns = [
    'files_accessed',
    'sensitive_files_accessed',
    'is_after_hours',
    'failed_login_attempts',
    'usb_connected',
    'files_deviation',
    'sensitive_deviation',
    'files_zscore',
    'sensitive_zscore',
    'rolling_files_7d',
    'rolling_sensitive_7d'
]

final_df = df[feature_columns + ['is_anomalous']]

# Save dataset
final_df.to_csv("../dataset/advanced_insider_dataset.csv", index=False)

print(" Advanced feature dataset created.")
print("Total anomalies:", final_df['is_anomalous'].sum())