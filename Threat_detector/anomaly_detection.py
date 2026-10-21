import joblib
import pandas as pd
import numpy as np

from db import get_db_connection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.calibration import CalibratedClassifierCV

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBClassifier
from sqlalchemy import create_engine, text

connection = get_db_connection()




print("Loading data...")

query = """
SELECT
    log_id,
    user_id,
    files_accessed,
    sensitive_files_accessed,
    failed_login_attempts,
    usb_connected,
    login_time,
    log_date,
    is_anomalous,
    CASE
        WHEN HOUR(login_time) < 8 OR HOUR(login_time) > 18
        THEN 1 ELSE 0
    END AS is_after_hours
FROM activity_logs
"""

df = pd.read_sql(query, connection)

if df.empty:
    print("No data found")
    exit()

print("Total records:", len(df))



# CLEANING


df = df.drop_duplicates()
df = df.fillna(0)

df = df[df['files_accessed'] >= 0]
df = df[df['sensitive_files_accessed'] >= 0]



# FEATURE ENGINEERING


df['log_date'] = pd.to_datetime(df['log_date'])
df['login_hour'] = pd.to_timedelta(df['login_time']).dt.seconds // 3600
df['day_of_week'] = df['log_date'].dt.dayofweek
df['weekend_access'] = df['day_of_week'].isin([5,6]).astype(int)

df = df.sort_values(['user_id','log_date'])

# User baseline
user_stats = df.groupby('user_id').agg({
    'files_accessed':['mean','std'],
    'sensitive_files_accessed':['mean','std']
}).reset_index()

user_stats.columns = [
    'user_id',
    'files_mean','files_std',
    'sensitive_mean','sensitive_std'
]

df = df.merge(user_stats,on='user_id')

df['files_std'] = df['files_std'].replace(0,1)
df['sensitive_std'] = df['sensitive_std'].replace(0,1)

df['files_z'] = (df['files_accessed'] - df['files_mean']) / df['files_std']
df['sensitive_z'] = (df['sensitive_files_accessed'] - df['sensitive_mean']) / df['sensitive_std']

# Rolling behavior
df['rolling_files_7d'] = df.groupby('user_id')['files_accessed'].transform(lambda x: x.rolling(7,1).mean())
df['rolling_sensitive_7d'] = df.groupby('user_id')['sensitive_files_accessed'].transform(lambda x: x.rolling(7,1).mean())

# Change features
df['files_change'] = df.groupby('user_id')['files_accessed'].diff().fillna(0)
df['sensitive_change'] = df.groupby('user_id')['sensitive_files_accessed'].diff().fillna(0)
df['login_fail_change'] = df.groupby('user_id')['failed_login_attempts'].diff().fillna(0)

# Interaction features
df['activity_ratio'] = df['sensitive_files_accessed']/(df['files_accessed']+1)
df['fail_per_file'] = df['failed_login_attempts']/(df['files_accessed']+1)

# (boosts all metrics)
df['behavior_score'] = (
    abs(df['files_z']) +
    abs(df['sensitive_z']) +
    abs(df['login_fail_change'])
)

features = [
'files_accessed',
'sensitive_files_accessed',
'failed_login_attempts',
'usb_connected',
'is_after_hours',
'login_hour',
'weekend_access',
'files_z',
'sensitive_z',
'rolling_files_7d',
'rolling_sensitive_7d',
'files_change',
'sensitive_change',
'login_fail_change',
'activity_ratio',
'fail_per_file',
'behavior_score'
]

X = df[features]
y = df['is_anomalous']



# SCALING


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
X_scaled,y,test_size=0.25,random_state=42,stratify=y
)



# CONTAMINATION FIX


real_contamination = y.mean()



# ISOLATION FOREST


if_model = IsolationForest(
n_estimators=400,
contamination=real_contamination,
random_state=42
)

if_model.fit(X_train)
df['if_score'] = -if_model.decision_function(X_scaled)



# LOCAL OUTLIER FACTOR


lof = LocalOutlierFactor(
n_neighbors=25,
contamination=real_contamination
)

lof.fit(X_scaled)
df['lof_score'] = -lof.negative_outlier_factor_



# AUTOENCODER

X_train_normal = X_train[y_train==0]

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(32,activation="relu")(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(16,activation="relu")(encoded)
encoded = Dense(8,activation="relu")(encoded)

decoded = Dense(16,activation="relu")(encoded)
decoded = Dense(32,activation="relu")(decoded)

output_layer = Dense(input_dim,activation="linear")(decoded)

autoencoder = Model(input_layer,output_layer)
autoencoder.compile(optimizer="adam",loss="mse")

early_stop = EarlyStopping(monitor="loss",patience=5,restore_best_weights=True)

autoencoder.fit(
X_train_normal,
X_train_normal,
epochs=60,
batch_size=32,
shuffle=True,
verbose=0,
callbacks=[early_stop]
)

recon = autoencoder.predict(X_scaled,verbose=0)
df['ae_error'] = np.mean(np.square(X_scaled - recon),axis=1)



# XGBOOST (IMPROVED + CALIBRATED)


ratio = len(y_train[y_train==0]) / max(len(y_train[y_train==1]),1)

xgb = XGBClassifier(
scale_pos_weight=ratio,
eval_metric="logloss",
objective="binary:logistic",
tree_method="hist",
reg_alpha=0.5,
reg_lambda=1.5
)

param_grid = {
'n_estimators':[300,400,500],
'max_depth':[4,5,6,7],
'learning_rate':[0.01,0.03,0.05],
'subsample':[0.7,0.8,1],
'colsample_bytree':[0.7,0.8,1],
'gamma':[0,0.1,0.3]
}

search = RandomizedSearchCV(
xgb,
param_grid,
n_iter=20,
scoring="precision",
cv=3,
n_jobs=-1,
verbose=1
)

search.fit(X_train,y_train)

xgb = search.best_estimator_

#  Calibration
xgb = CalibratedClassifierCV(xgb, method='sigmoid')
xgb.fit(X_train, y_train)

df['xgb_prob'] = xgb.predict_proba(X_scaled)[:,1]



# FUSION SCORING (IMPROVED)


scaler_mm = MinMaxScaler()

df[['xgb_scaled']] = scaler_mm.fit_transform(df[['xgb_prob']])
df[['ae_scaled']] = scaler_mm.fit_transform(df[['ae_error']])
df[['if_scaled']] = scaler_mm.fit_transform(df[['if_score']])
df[['lof_scaled']] = scaler_mm.fit_transform(df[['lof_score']])


# INDIVIDUAL MODEL SCORES (0–100 SCALE)


df['xgb_score'] = (df['xgb_scaled'] * 100).clip(0,100)
df['ae_score'] = (df['ae_scaled'] * 100).clip(0,100)
df['if_score_norm'] = (df['if_scaled'] * 100).clip(0,100)
df['lof_score_norm'] = (df['lof_scaled'] * 100).clip(0,100)

df['fusion_score'] = (
df['xgb_scaled']*0.55 +
df['ae_scaled']*0.25 +
df['if_scaled']*0.10 +
df['lof_scaled']*0.05 +
abs(df['sensitive_z'])*0.05
)

df['risk_score'] = (df['fusion_score']*100).clip(0,100)




# THRESHOLD OPTIMIZATION (PRECISION CONTROLLED)


precision,recall,thresholds = precision_recall_curve(y,df['risk_score'])

target_recall = 0.85

valid_idx = np.where(recall[:-1] >= target_recall)[0]

best_threshold = thresholds[valid_idx[np.argmax(precision[:-1][valid_idx])]]

df['ml_anomaly'] = (df['risk_score'] >= best_threshold).astype(int)


# INDIVIDUAL MODEL FLAGS


df['if_anomaly'] = (df['if_scaled'] > 0.6).astype(int)
df['ae_anomaly'] = (df['ae_scaled'] > 0.6).astype(int)



# THREAT REASON GENERATION


def generate_reason(row):
    reasons = []

    if row['ml_anomaly'] == 1:
        reasons.append("ML Risk Spike")

    if row['if_anomaly'] == 1:
        reasons.append("Isolation Forest Outlier")

    if row['ae_anomaly'] == 1:
        reasons.append("Autoencoder Reconstruction Error")

    if row['is_after_hours'] == 1:
        reasons.append("After Hours Login")

    if row['files_z'] > 2:
        reasons.append("Unusual File Access")

    if row['sensitive_z'] > 2:
        reasons.append("Sensitive Data Spike")

    if row['failed_login_attempts'] > 3:
        reasons.append("Multiple Failed Logins")

    return ", ".join(reasons) if reasons else "Normal Activity"


df['threat_reason'] = df.apply(generate_reason, axis=1)


# EVALUATION


print(confusion_matrix(y,df['ml_anomaly']))
print(classification_report(y,df['ml_anomaly']))
print("AUC:",roc_auc_score(y,df['risk_score']))



# RISK LEVEL


def risk_level(score):
    if score >= 60:
        return "Critical"
    elif score >= 40:
        return "Review"
    else:
        return "Normal"

df['risk_level'] = df['risk_score'].apply(risk_level)



# STORE RESULTS


with connection.begin() as conn:

    conn.execute(text("DELETE FROM risk_scores"))

    insert = text("""
    INSERT INTO risk_scores
    (log_id,user_id,risk_score,risk_level,ml_anomaly,
     if_anomaly,ae_anomaly,threat_reason,xgb_score,ae_score,if_score,lof_score)
    VALUES
    (:log_id,:user_id,:risk_score,:risk_level,:ml_anomaly,
     :if_anomaly,:ae_anomaly,:threat_reason,:xgb_score,:ae_score,:if_score,:lof_score)
    """)

    for _,row in df.iterrows():
        conn.execute(insert,{
        "log_id":int(row['log_id']),
        "user_id":int(row['user_id']),
        "risk_score":float(row['risk_score']),
        "risk_level":row['risk_level'],
        "ml_anomaly":int(row['ml_anomaly']),
        "if_anomaly": int(row['if_anomaly']),
        "ae_anomaly": int(row['ae_anomaly']),
        "threat_reason": row['threat_reason'],
        "xgb_score": float(row['xgb_score']),
        "ae_score": float(row['ae_score']),
        "if_score": float(row['if_score_norm']),
        "lof_score": float(row['lof_score_norm'])
        })



# SAVE FILE


df.to_csv("../dataset/insider_activity_with_anomalies.csv",index=False)

print(" Insider Threat Detection Completed Successfully")