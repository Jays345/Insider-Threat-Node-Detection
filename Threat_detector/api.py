from db import get_db_connection
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd
import jwt
import datetime
import os
connection = get_db_connection()

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"



app = FastAPI()
security = HTTPBearer()



def create_token(username: str, role: str):
    payload = {
        "username": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded
    except:
        raise HTTPException(status_code=401, detail="Invalid or expired token")




@app.post("/login")
def login(username: str, password: str):

    users = {
        "admin": {"password": "admin123", "role": "admin"},
        "analyst": {"password": "analyst123", "role": "analyst"}
    }

    if username in users and password == users[username]["password"]:
        role = users[username]["role"]
        token = create_token(username, role)
        return {
            "access_token": token,
            "role": role
        }

    raise HTTPException(status_code=401, detail="Invalid credentials")




@app.get("/risk-data")
def get_risk_data(user=Depends(verify_token)):
    df = pd.read_sql("SELECT * FROM risk_scores", connection)
    return df.to_dict(orient="records")


@app.get("/critical-threats")
def get_critical(user=Depends(verify_token)):
    df = pd.read_sql(
        "SELECT * FROM risk_scores WHERE risk_level='Critical'", connection
    )
    return df.to_dict(orient="records")


@app.get("/alerts")
def get_alerts(user=Depends(verify_token)):
    df = pd.read_sql("SELECT * FROM alerts ORDER BY created_at DESC", connection)
    return df.to_dict(orient="records")




@app.get("/user-info/{user_id}")
def user_info(user_id: int, user=Depends(verify_token)):

    query = text("SELECT * FROM users WHERE user_id = :uid")
    df = pd.read_sql(query, connection, params={"uid": user_id})

    if df.empty:
        raise HTTPException(status_code=404, detail="User not found")

    user_row = df.iloc[0]

    def format_time(t):
        if pd.isna(t):
            return "N/A"
        return str(t).split(" ")[-1][:5]

    start_str = format_time(user_row["baseline_login_start"])
    end_str = format_time(user_row["baseline_login_end"])

    return {
        "user_id": int(user_row["user_id"]),
        "name": user_row["username"],
        "department": user_row["department"],
        "job_role": user_row["role"],
        "device": "Corporate Device",
        "avg_login": f"{start_str} - {end_str}",
        "image_url": f"https://i.pravatar.cc/150?u={user_id}"
    }




@app.get("/user-files/{user_id}")
def user_files(user_id: int, user=Depends(verify_token)):

    try:
        query = text("""
            SELECT file_name, action, created_at
            FROM file_access_logs
            WHERE user_id = :uid
            ORDER BY created_at DESC
            LIMIT 50
        """)
        df = pd.read_sql(query, connection, params={"uid": user_id})

        return df.to_dict(orient="records")

    except:
        return []


@app.get("/user-logins/{user_id}")
def user_logins(user_id: int, user=Depends(verify_token)):

    try:
        query = text("""
            SELECT HOUR(created_at) as login_hour, created_at
            FROM login_logs
            WHERE user_id = :uid
            ORDER BY created_at DESC
            LIMIT 100
        """)
        df = pd.read_sql(query, connection, params={"uid": user_id})

        return df.to_dict(orient="records")

    except:
        return []