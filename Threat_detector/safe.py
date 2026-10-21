import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from xgboost import XGBClassifier

from sqlalchemy import create_engine, text
from sklearn.neighbors import LocalOutlierFactor
from db import get_db_connection
connection = get_db_connection()

print("Loading data from MySQL activity_logs...")

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

df = pd.read_sql(query,connection)

if df.empty:
    print("No logs found")
    exit()

print("Total records:", len(df))



# FEATURE ENGINEERING


df['log_date'] = pd.to_datetime(df['log_date'])

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

df['files_std']=df['files_std'].replace(0,1)
df['sensitive_std']=df['sensitive_std'].replace(0,1)

df['files_deviation']=df['files_accessed']-df['files_mean']
df['sensitive_deviation']=df['sensitive_files_accessed']-df['sensitive_mean']

df['files_zscore']=df['files_deviation']/df['files_std']
df['sensitive_zscore']=df['sensitive_deviation']/df['sensitive_std']

df=df.sort_values(['user_id','log_date'])

df['rolling_files_7d']=df.groupby('user_id')['files_accessed']\
.transform(lambda x:x.rolling(7,1).mean())

df['rolling_sensitive_7d']=df.groupby('user_id')['sensitive_files_accessed']\
.transform(lambda x:x.rolling(7,1).mean())

df['files_change']=df.groupby('user_id')['files_accessed'].diff().fillna(0)

df['sensitive_change']=df.groupby('user_id')['sensitive_files_accessed'].diff().fillna(0)

df['login_fail_change']=df.groupby('user_id')['failed_login_attempts'].diff().fillna(0)


# FEATURES


features=[

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
'rolling_sensitive_7d',

'files_change',
'sensitive_change',
'login_fail_change'

]

X=df[features]
y=df['is_anomalous']


scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)



# ISOLATION FOREST


contamination_rate=0.04

if_model=IsolationForest(
n_estimators=150,
contamination=contamination_rate,
random_state=42
)

df['if_anomaly']=(if_model.fit_predict(X_scaled)==-1).astype(int)

print("IsolationForest:",df['if_anomaly'].sum())



# AUTOENCODER


low_risk_mask=(

(df['is_after_hours']==0)&
(df['files_accessed']<df['files_accessed'].quantile(0.4))&
(df['sensitive_files_accessed']==0)

)

X_train=X_scaled[low_risk_mask]

if len(X_train)<50:
    X_train=X_scaled[df['if_anomaly']==0]


input_dim=X_train.shape[1]

input_layer=Input(shape=(input_dim,))

encoded=Dense(8,activation="relu")(input_layer)
encoded=Dense(4,activation="relu")(encoded)

decoded=Dense(8,activation="relu")(encoded)
output_layer=Dense(input_dim,activation="linear")(decoded)

autoencoder=Model(input_layer,output_layer)

autoencoder.compile(
optimizer="adam",
loss="mse"
)

early_stop=EarlyStopping(
monitor="loss",
patience=5,
restore_best_weights=True
)

autoencoder.fit(
X_train,
X_train,
epochs=40,
batch_size=32,
shuffle=True,
verbose=0,
callbacks=[early_stop]
)


reconstructions=autoencoder.predict(X_scaled,verbose=0)

reconstruction_error=np.mean(
np.square(X_scaled-reconstructions),
axis=1
)

df['ae_error']=reconstruction_error

ae_threshold=np.percentile(
reconstruction_error,
100*(1-contamination_rate)
)

df['ae_anomaly']=(df['ae_error']>ae_threshold).astype(int)

print("Autoencoder:",df['ae_anomaly'].sum())



# XGBOOST MODEL


print("Training XGBoost...")

ratio = len(y[y==0]) / len(y[y==1])

xgb = XGBClassifier(
n_estimators=400,
max_depth=6,
learning_rate=0.03,
subsample=0.8,
colsample_bytree=0.8,
scale_pos_weight=ratio,
eval_metric='logloss'
)

xgb.fit(X_scaled,y)

df['xgb_prob']=xgb.predict_proba(X_scaled)[:,1]


print("XGBoost trained")



# FUSION SCORING


df['if_score']=-if_model.decision_function(X_scaled)

def scale(col):

    low=col.quantile(0.05)
    high=col.quantile(0.95)

    return ((col-low)/(high-low)).clip(0,1)


ae_scaled=scale(df['ae_error'])
if_scaled=scale(df['if_score'])
xgb_scaled=df['xgb_prob']


df['fusion_score']=(
xgb_scaled*0.50+
if_scaled*0.20+
ae_scaled*0.15+
scale(abs(df['sensitive_zscore']))*0.10+
scale(abs(df['files_zscore']))*0.05
)



# RISK SCORE


df['risk_score']=(df['fusion_score']*100)

df['risk_score']=df['risk_score'].clip(0,100)
df['risk_score']=df['risk_score'].round().astype(int)



# OPTIMAL THRESHOLD


thresholds=np.arange(20,80,2)

best_f1=0
best_threshold=50

for t in thresholds:

    preds=(df['risk_score']>=t).astype(int)

    f1=f1_score(df['is_anomalous'],preds)

    if f1>best_f1:

        best_f1=f1
        best_threshold=t


print("Best Threshold:",best_threshold)
print("Best F1:",best_f1)


df['ml_anomaly']=(
df['risk_score']>=best_threshold
).astype(int)

print("Final ML anomalies:",df['ml_anomaly'].sum())



# RISK LEVEL


def level(score):

    if score>=80:
        return "Critical"
    elif score>=60:
        return "High"
    elif score>=30:
        return "Medium"
    else:
        return "Low"


df['risk_level']=df['risk_score'].apply(level)



# THREAT EXPLANATION


high_file=df['files_accessed'].quantile(0.8)

def explain(row):

    r=[]

    if row['is_after_hours']:
        r.append("After-hours")

    if row['files_accessed']>high_file:
        r.append("High file access")

    if row['sensitive_files_accessed']>0:
        r.append("Sensitive access")

    if row['if_anomaly']:
        r.append("IF anomaly")

    if row['ae_anomaly']:
        r.append("AE anomaly")

    if row['xgb_prob']>0.7:
        r.append("XGB anomaly")

    return "Normal" if not r else ", ".join(r)


df['threat_reason']=df.apply(explain,axis=1)



# CLUSTERING


kmeans=KMeans(
n_clusters=3,
random_state=42
)

df['behavior_cluster']=kmeans.fit_predict(X_scaled)




print("Updating database...")


with connection.begin() as conn:

    conn.execute(text("DELETE FROM risk_scores"))

    insert=text("""

INSERT INTO risk_scores

(log_id,user_id,risk_score,risk_level,
threat_reason,if_anomaly,
ae_anomaly,ml_anomaly)

VALUES

(:log_id,:user_id,:risk_score,
:risk_level,:threat_reason,
:if_anomaly,:ae_anomaly,
:ml_anomaly)

""")


    for _,row in df.iterrows():

        conn.execute(insert,{

"log_id":int(row['log_id']),
"user_id":int(row['user_id']),
"risk_score":int(row['risk_score']),
"risk_level":row['risk_level'],
"threat_reason":row['threat_reason'],
"if_anomaly":int(row['if_anomaly']),
"ae_anomaly":int(row['ae_anomaly']),
"ml_anomaly":int(row['ml_anomaly'])

})


df.to_csv(
"../dataset/insider_activity_with_anomalies.csv",
index=False
)

print("\nPipeline Completed Successfully")