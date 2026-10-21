
import os
import mysql.connector
from dotenv import load_dotenv


load_dotenv()

def get_db_connection():

    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )

        return connection

    except mysql.connector.Error:

        raise RuntimeError(
            "Database connection failed. Check your database settings."
        ) from None
