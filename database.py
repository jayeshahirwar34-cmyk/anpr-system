import mysql.connector
import streamlit as st
import os

def get_conn():
    return mysql.connector.connect(
        host=st.secrets.get("DB_HOST", os.environ.get("DB_HOST")),
        port=st.secrets.get("DB_PORT", 3306),  # <-- ADD THIS LINE
        user=st.secrets.get("DB_USER", os.environ.get("DB_USER")),
        password=st.secrets.get("DB_PASS", os.environ.get("DB_PASS")),
        database=st.secrets.get("DB_NAME", os.environ.get("DB_NAME")),
        ssl_ca="/etc/ssl/certs/ca-certificates.crt"
    )

def setup_tables():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            track_id   INT PRIMARY KEY,
            plate_text VARCHAR(20),
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_id  VARCHAR(20)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sightings (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            track_id   INT,
            frame_num  INT,
            bbox_cx    FLOAT,
            bbox_cy    FLOAT,
            seen_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (track_id) REFERENCES vehicles(track_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS challans (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            track_id    INT,
            plate_text  VARCHAR(20),
            speed_kmph  FLOAT,
            violation   VARCHAR(50),
            issued_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            pdf_path    VARCHAR(200),
            FOREIGN KEY (track_id) REFERENCES vehicles(track_id)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_vehicle(track_id, plate_text, camera_id="CAM_01"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT IGNORE INTO vehicles (track_id, plate_text, camera_id)
        VALUES (%s, %s, %s)
    """, (track_id, plate_text, camera_id))
    conn.commit()
    cur.close()
    conn.close()

def insert_sighting(track_id, frame_num, cx, cy):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO sightings (track_id, frame_num, bbox_cx, bbox_cy)
        VALUES (%s, %s, %s, %s)
    """, (track_id, frame_num, cx, cy))
    conn.commit()
    cur.close()
    conn.close()

def insert_challan(track_id, plate_text, speed, pdf_path):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO challans (track_id, plate_text, speed_kmph, violation, pdf_path)
        VALUES (%s, %s, %s, %s, %s)
    """, (track_id, plate_text, speed, "Overspeeding", pdf_path))
    conn.commit()
    cur.close()
    conn.close()
