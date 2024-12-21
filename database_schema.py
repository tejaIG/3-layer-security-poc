# database_schema.py
import sqlite3
import pandas as pd
from datetime import datetime
import bcrypt
import os

def init_database():
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            face_encoding TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create login_history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS login_history (
            login_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            login_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN,
            failure_reason TEXT,
            orientation_vectors TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    # Insert dummy data
    dummy_users = [
        ('john.doe@example.com', 'password123', 'John Doe'),
        ('jane.smith@example.com', 'password456', 'Jane Smith'),
        ('bob.wilson@example.com', 'password789', 'Bob Wilson'),
        ('alice.brown@example.com', 'passwordabc', 'Alice Brown'),
        ('charlie.davis@example.com', 'passwordxyz', 'Charlie Davis')
    ]
    
    for user in dummy_users:
        # Hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(user[1].encode('utf-8'), salt)
        
        # Insert user
        try:
            c.execute('''
                INSERT OR IGNORE INTO users (email, password_hash, full_name)
                VALUES (?, ?, ?)
            ''', (user[0], hashed.decode('utf-8'), user[2]))
        except sqlite3.IntegrityError:
            pass
    
    conn.commit()
    conn.close()

# Vector data storage
class VectorDataStorage:
    def __init__(self):
        self.csv_folder = 'user_vectors'
        os.makedirs(self.csv_folder, exist_ok=True)
    
    def save_vectors(self, user_id, vectors_data):
        filename = f"{self.csv_folder}/user_{user_id}_vectors.csv"
        df = pd.DataFrame(vectors_data)
        df['timestamp'] = datetime.now()
        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

# User authentication
class UserAuth:
    def __init__(self):
        self.conn = sqlite3.connect('user_auth.db')
        self.vector_storage = VectorDataStorage()
    
    def verify_user(self, email, password):
        c = self.conn.cursor()
        c.execute('SELECT user_id, password_hash FROM users WHERE email = ?', (email,))
        result = c.fetchone()
        
        if result and bcrypt.checkpw(password.encode('utf-8'), result[1].encode('utf-8')):
            return result[0]
        return None
    
    def log_login_attempt(self, user_id, success, failure_reason=None, vectors=None):
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO login_history 
            (user_id, success, failure_reason, orientation_vectors)
            VALUES (?, ?, ?, ?)
        ''', (user_id, success, failure_reason, str(vectors)))
        self.conn.commit()
        
        if success and vectors:
            self.vector_storage.save_vectors(user_id, vectors)
    
    def close(self):
        self.conn.close()