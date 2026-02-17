import sqlite3
import datetime
import os

DB_NAME = "gate_automation.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Authorized Plates Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS authorized_plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            owner_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Access Logs Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action TEXT,
            image_path TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def add_authorized_plate(plate_number, owner_name):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO authorized_plates (plate_number, owner_name) VALUES (?, ?)", 
                  (plate_number.upper(), owner_name))
        conn.commit()
        conn.close()
        return True, "Plate added successfully."
    except sqlite3.IntegrityError:
        return False, "Plate already exists."
    except Exception as e:
        return False, str(e)

def remove_authorized_plate(plate_number):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM authorized_plates WHERE plate_number = ?", (plate_number.upper(),))
        conn.commit()
        rows_affected = c.rowcount
        conn.close()
        if rows_affected > 0:
            return True, "Plate removed successfully."
        else:
            return False, "Plate not found."
    except Exception as e:
        return False, str(e)

def get_all_plates():
    conn = get_db_connection()
    plates = conn.execute("SELECT * FROM authorized_plates").fetchall()
    conn.close()
    return [dict(plate) for plate in plates]

def check_access(plate_number):
    conn = get_db_connection()
    plate = conn.execute("SELECT * FROM authorized_plates WHERE plate_number = ?", 
                         (plate_number.upper(),)).fetchone()
    conn.close()
    return plate is not None

def log_access(plate_number, action, image_path=None):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO access_logs (plate_number, action, image_path) VALUES (?, ?, ?)",
              (plate_number.upper() if plate_number else "UNKNOWN", action, image_path))
    conn.commit()
    conn.close()

def get_logs():
    conn = get_db_connection()
    logs = conn.execute("SELECT * FROM access_logs ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(log) for log in logs]
