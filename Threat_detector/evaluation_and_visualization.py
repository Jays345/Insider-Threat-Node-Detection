import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

# Fix backend issues (important for PyCharm/VS Code)
plt.switch_backend('TkAgg')


# LOAD DATA

df = pd.read_csv('../dataset/insider_activity_with_anomalies.csv')

print("Total records:", len(df))
print("Columns:", df.columns)


# BASIC STATS

print("\n===== BASIC STATS =====")

if 'ml_anomaly' in df.columns:
    print("Final ML anomalies:", df['ml_anomaly'].sum())

if 'risk_level' in df.columns:
    print(df['risk_level'].value_counts())


# SCATTER: Behavior vs Risk

if all(col in df.columns for col in ['files_accessed', 'risk_score']):
    plt.figure()
    plt.scatter(df['files_accessed'], df['risk_score'])
    plt.xlabel("Files Accessed")
    plt.ylabel("Risk Score")
    plt.title("Files Accessed vs Risk Score")
    plt.show()

# RISK DISTRIBUTION

if 'risk_score' in df.columns:
    plt.figure()
    plt.hist(df['risk_score'], bins=50)
    plt.title("Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.show()


# RISK LEVEL DISTRIBUTION

if 'risk_level' in df.columns:
    plt.figure()
    df['risk_level'].value_counts().plot(kind='bar')
    plt.title("Risk Level Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.show()


# AFTER HOURS IMPACT

if all(col in df.columns for col in ['is_after_hours', 'risk_score']):
    plt.figure()
    df.groupby('is_after_hours')['risk_score'].mean().plot(kind='bar')
    plt.title("After Hours vs Risk")
    plt.show()


# TIME TREND

if 'log_date' not in df.columns:
    df['log_date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='h')

df['log_date'] = pd.to_datetime(df['log_date'])

plt.figure()
df.groupby('log_date')['risk_score'].mean().plot()
plt.title("Risk Trend Over Time")
plt.show()


# TOP RISK USERS

if all(col in df.columns for col in ['user_id', 'risk_score']):
    top_users = df.groupby('user_id')['risk_score'].max().sort_values(ascending=False).head(5)
    print("\nTop Risk Users:")
    print(top_users)


# MODEL EVALUATION

print("\n===== MODEL EVALUATION =====")

if 'is_anomalous' not in df.columns:
    print(" Ground truth not available. Add 'is_anomalous' column.")
    exit()

df['is_anomalous'] = df['is_anomalous'].astype(int)

y_true = df['is_anomalous']
scores = df['risk_score']


# CONFUSION MATRIX FUNCTION

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


# CURRENT THRESHOLD

print("\n--- Current Model Performance ---")

y_pred = df['ml_anomaly']

plot_conf_matrix(y_true, y_pred, "Current Model")

print(classification_report(y_true, y_pred))


# ROC CURVE

fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC Curve (AUC={roc_auc:.2f})")
plt.show()

print("AUC:", round(roc_auc, 4))


# PRECISION-RECALL CURVE

precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


# BEST F1 THRESHOLD

f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
best_f1_idx = np.argmax(f1)

best_f1_threshold = pr_thresholds[best_f1_idx]
print("\nBest F1 Threshold:", round(best_f1_threshold, 2))

y_pred_f1 = (scores >= best_f1_threshold).astype(int)

plot_conf_matrix(y_true, y_pred_f1, "F1 Optimized")

print(classification_report(y_true, y_pred_f1))


# ROC-BASED THRESHOLD

optimal_idx = np.argmax(tpr - fpr)
roc_threshold = thresholds[optimal_idx]

print("\nROC Optimal Threshold:", round(roc_threshold, 2))

y_pred_roc = (scores >= roc_threshold).astype(int)

plot_conf_matrix(y_true, y_pred_roc, "ROC Optimized")

print(classification_report(y_true, y_pred_roc))


# MODEL COMPARISON

print("\n MODEL COMPARISON ")

model_scores = {
    "XGBoost": df["xgb_score"] if "xgb_score" in df else None,
    "Autoencoder": df["ae_score"] if "ae_score" in df else None,
    "Isolation Forest": df["if_score"] if "if_score" in df else None,
    "LOF": df["lof_score"] if "lof_score" in df else None,
    "Hybrid": df["risk_score"]
}

# ROC Comparison
plt.figure()

for name, score in model_scores.items():
    if score is not None:
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

print("\n Evaluation Complete")