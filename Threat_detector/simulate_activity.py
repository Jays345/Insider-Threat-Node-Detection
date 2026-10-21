import random
import mysql.connector
from datetime import datetime, timedelta
import numpy as np
from db import get_db_connection
connection = get_db_connection()

cursor = connection.cursor()


# FETCH USERS

cursor.execute("SELECT user_id, department FROM users")
users = cursor.fetchall()

if not users:
    raise Exception("No users found in database.")

NUM_RECORDS = 6000
START_DATE = datetime.today() - timedelta(days=30)

print("Generating SOC-level enterprise activity logs...")


# CREATE USER BEHAVIOR PROFILES

user_profiles = {}

for user_id, department in users:

    if department == "IT":
        base_files = random.randint(25, 40)
        base_sensitive = random.randint(5, 12)
    elif department == "HR":
        base_files = random.randint(15, 25)
        base_sensitive = random.randint(8, 18)
    else:
        base_files = random.randint(8, 20)
        base_sensitive = random.randint(2, 6)

    work_style = random.choice(["normal", "flexible", "night_shift"])

    risk_tendency = random.choices(
        ["low", "medium", "high"],
        weights=[0.75, 0.20, 0.05]
    )[0]

    user_profiles[user_id] = {
        "department": department,
        "base_files": base_files,
        "base_sensitive": base_sensitive,
        "work_style": work_style,
        "risk_tendency": risk_tendency,
        "escalation_day": random.randint(15, 25) if risk_tendency == "high" else None
    }


# GENERATE LOGS

for _ in range(NUM_RECORDS):

    user_id, department = random.choice(users)
    profile = user_profiles[user_id]

    random_day_offset = random.randint(0, 30)
    random_day = START_DATE + timedelta(days=random_day_offset)

    is_weekend = random_day.weekday() >= 5


    # LOGIN TIME BASED ON WORK STYLE

    if profile["work_style"] == "normal":
        login_hour = int(np.random.normal(9, 1.5))
    elif profile["work_style"] == "flexible":
        login_hour = int(np.random.normal(11, 3))
    else:
        login_hour = int(np.random.normal(22, 2))

    login_hour = max(0, min(23, login_hour))
    login_minute = random.randint(0, 59)

    login = datetime.combine(
        random_day.date(),
        datetime.strptime(f"{login_hour}:{login_minute}", "%H:%M").time()
    )


    # BASELINE BEHAVIOR

    files = int(np.random.normal(profile["base_files"], 5))
    sensitive = int(np.random.normal(profile["base_sensitive"], 3))

    files = max(1, files)
    sensitive = max(0, sensitive)


    # MONTH-END BUSINESS SPIKE

    if random_day.day >= 27:
        files += random.randint(10, 25)


    # GRADUAL ESCALATION FOR HIGH RISK USERS

    is_anomalous = 0

    if profile["risk_tendency"] == "high":
        if random_day_offset >= profile["escalation_day"]:

            escalation_factor = (random_day_offset - profile["escalation_day"]) * 0.5

            files += int(escalation_factor * random.randint(2, 5))
            sensitive += int(escalation_factor * random.randint(1, 3))

            login_hour = min(23, login_hour + random.randint(1, 3))
            is_anomalous = 1


    # LOW-AND-SLOW EXFILTRATION (MEDIUM USERS)

    if profile["risk_tendency"] == "medium":
        if random.random() < 0.4:
            files += random.randint(5, 15)
            sensitive += random.randint(3, 8)
            is_anomalous = 1


    # ACCIDENTAL HEAVY DAY (FALSE POSITIVE REALISM)

    if random.random() < 0.03:
        files *= random.randint(2, 3)
        # NOT malicious
        is_anomalous = 0


    # WEEKEND SUSPICION BOOST

    if is_weekend and sensitive > profile["base_sensitive"] * 1.5:
        is_anomalous = 1


    # ADDITIONAL REALISTIC FEATURES

    failed_logins = random.randint(0, 3)

    if profile["risk_tendency"] == "high":
        failed_logins += random.randint(1, 4)

    usb_connected = random.choices([0, 1], weights=[0.8, 0.2])[0]

    session_duration = random.randint(1800, 20000)
    logout = login + timedelta(seconds=session_duration)


    # THREAT LEVEL ASSIGNMENT

    if is_anomalous == 1:
        if sensitive > profile["base_sensitive"] * 3:
            threat_level = "Severe"
        else:
            threat_level = "Moderate"
    else:
        threat_level = "Normal"


    # INSERT INTO DATABASE

    cursor.execute("""
        INSERT INTO activity_logs
        (user_id, login_time, logout_time, session_duration,
         files_accessed, sensitive_files_accessed,
         log_date, failed_login_attempts,
         usb_connected, is_anomalous, threat_level)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        user_id,
        login.time(),
        logout.time(),
        session_duration,
        files,
        sensitive,
        login.date(),
        failed_logins,
        usb_connected,
        is_anomalous,
        threat_level
    ))

connection.commit()
cursor.close()
connection.close()

print(" SOC-Level realistic enterprise logs generated.")
print("Total records inserted:", NUM_RECORDS)