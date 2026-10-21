import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Enterprise SOC", layout="wide")


#  UI THEME (LIGHT + CLEAN)

st.markdown("""
<style>
body { background-color: #f8fafc; color: #0f172a; }
.block-container { padding-top: 1.5rem; }

section[data-testid="stSidebar"] {
    background: #111827;
    color: #e5e7eb;
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

div[role="radiogroup"] > label {
    background-color: #1f2937;
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 8px;
    border: 1px solid #374151;
}
div[role="radiogroup"] > label:hover {
    background-color: #3b82f6;
    color: white !important;
}
div[role="radiogroup"] > label[data-selected="true"] {
    background-color: #2563eb;
    color: white !important;
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #e5e7eb;
    padding: 18px;
    border-radius: 12px;
    color: #0f172a;
}

h1, h2, h3 { color: #1e40af; }

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


#  LOGIN

if "token" not in st.session_state:
    st.title(" SOC Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        res = requests.post(
            f"{API_URL}/login",
            params={"username": username, "password": password}
        )

        if res.status_code == 200:
            data = res.json()
            st.session_state.token = data["access_token"]
            st.session_state.role = data["role"]
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}
role = st.session_state.role


# 📡 FETCH DATA

risk_df = pd.DataFrame(requests.get(f"{API_URL}/risk-data", headers=headers).json())
critical_df = pd.DataFrame(requests.get(f"{API_URL}/critical-threats", headers=headers).json())

if "created_at" in risk_df.columns:
    risk_df["created_at"] = pd.to_datetime(risk_df["created_at"], errors="coerce")


#  SIDEBAR

st.sidebar.markdown("## ⚙ SOC Controls")
st.sidebar.markdown(f"**Role:** `{role.upper()}`")

page = st.sidebar.radio("Navigation", [" Dashboard", " User Investigation"])
refresh_rate = st.sidebar.slider("Auto Refresh", 5, 60, 10)


#  DASHBOARD

if page == " Dashboard":

    st.title("🛡 Enterprise SOC Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric(" Total Records", len(risk_df))
    c2.metric(" Anomalies", int(risk_df["ml_anomaly"].sum()))
    c3.metric(" Critical", len(critical_df))

    st.markdown("---")


    #  MODEL COMPARISON (GLOBAL)

    st.subheader(" Model Performance Comparison")

    # Check if columns exist (safe)
    required_cols = ["xgb_score", "ae_score", "if_score", "lof_score"]

    if all(col in risk_df.columns for col in required_cols):

        model_scores = pd.DataFrame({
            "Model": ["XGBoost", "Autoencoder", "Isolation Forest", "LOF"],
            "Avg Score": [
                risk_df["xgb_score"].mean(),
                risk_df["ae_score"].mean(),
                risk_df["if_score"].mean(),
                risk_df["lof_score"].mean()
            ]
        })

        fig = px.bar(model_scores, x="Model", y="Avg Score", text_auto=True)

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a")
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Model scores not available. Run ML pipeline first.")

    fig = px.histogram(risk_df, x="risk_score", nbins=40)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(" Top Risk Users")
    top_users = (
        risk_df.groupby("user_id")
        .agg(avg_risk=("risk_score", "mean"))
        .sort_values("avg_risk", ascending=False)
        .head(10)
    )
    st.dataframe(top_users)

    st.subheader(" Critical Threats")
    st.dataframe(critical_df if not critical_df.empty else pd.DataFrame())


# 👤 USER PAGE

if page == "👤 User Investigation":

    st.title("👤 User Investigation")

    selected_user = st.selectbox("Select User", sorted(risk_df["user_id"].unique()))
    user_df = risk_df[risk_df["user_id"] == selected_user]

    # Fetch user info
    user_info = {}
    try:
        res = requests.get(f"{API_URL}/user-info/{selected_user}", headers=headers)
        if res.status_code == 200:
            user_info = res.json()
    except:
        pass

    # ---------------- PROFILE ----------------
    st.subheader(" User Profile")

    col1, col2, col3 = st.columns([1,2,2])

    with col1:
        st.image(user_info.get("image_url","https://i.pravatar.cc/150"), width=80)

    col2.metric("Name", user_info.get("name","Unknown"))
    col3.metric("User ID", selected_user)

    col1, col2, col3 = st.columns(3)
    col1.metric("Department", user_info.get("department","N/A"))
    col2.metric("Role", user_info.get("job_role","N/A"))
    col3.metric("Device", user_info.get("device","N/A"))

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Login", user_info.get("avg_login","N/A"))

    if not user_df.empty:
        avg_risk = round(user_df["risk_score"].mean(),2)
    else:
        avg_risk = 0

    col2.metric("Avg Risk", avg_risk)
    col3.metric("Anomalies", int(user_df["ml_anomaly"].sum()))

    # Risk badge
    if avg_risk > 80:
        color = "red"
    elif avg_risk > 60:
        color = "orange"
    else:
        color = "green"


    # ---------------- GRAPH ----------------
    st.subheader(" Risk Trend")

    fig = px.line(user_df, x="created_at", y="risk_score")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- ANOMALIES ----------------
    st.subheader(" Anomalies")
    st.dataframe(user_df[user_df["ml_anomaly"] == 1])

    # ---------------- CRITICAL ----------------
    st.subheader(" User Critical Threats")
    st.dataframe(critical_df[critical_df["user_id"] == selected_user])


#  AUTO REFRESH

st.caption(f" Refresh every {refresh_rate}s")
time.sleep(refresh_rate)
st.rerun()