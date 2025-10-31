import sqlite3
import os

db_path = 'users.db'

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create users table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL
)
''')

# Insert default admin user
try:
    c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
              ('admin', 'admin123', 'admin'))
except sqlite3.IntegrityError:
    print("Admin user already exists.")

conn.commit()
conn.close()
print("Database created successfully.")
