# dashboard.py
from flask import Flask, jsonify, render_template, render_template_string
import sqlite3


app = Flask(__name__)

def fetch_all_entries():
    with sqlite3.connect('parking_lott.db') as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM entries ORDER BY entry_time DESC")
        return [dict(row) for row in cursor.fetchall()]

def fetch_currently_parked():
    with sqlite3.connect('parking_lott.db') as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT plate_number, entry_time FROM entries WHERE exit_time IS NULL ORDER BY entry_time DESC")
        return [dict(row) for row in cursor.fetchall()]

@app.route('/api/entries', methods=['GET'])
def get_entries():
    data = fetch_all_entries()
    return jsonify(data)

@app.route('/')
def index():
    all_entries = fetch_all_entries()
    parked = fetch_currently_parked()
    return render_template('dashboard.html', all_entries=all_entries, parked=parked)

if __name__ == "__main__":
    app.run(debug=True)
