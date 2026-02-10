import sqlite3

DB_NAME = "hospital.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            disease TEXT,
            result TEXT
        )
    """)

    conn.commit()
    conn.close()

def insert_record(name, disease, result):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO records (name, disease, result) VALUES (?, ?, ?)",
                   (name, disease, result))

    conn.commit()
    conn.close()

def fetch_all():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM records")
    rows = cursor.fetchall()

    conn.close()
    return rows
