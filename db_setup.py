# db_setup.py
import sqlite3

def initialize_database():
    with sqlite3.connect('parking_lot.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                fare INTEGER,
                UNIQUE (plate_number, entry_time)
            )
        """)
        conn.commit()

if __name__ == "__main__":
    initialize_database()
    print("✅ Database initialized.")
