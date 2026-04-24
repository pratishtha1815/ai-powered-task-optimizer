import sqlite3
import os

db_path = os.path.join("data", "task_optimizer.db")

def migrate():
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print("Adding voice_score column...")
        cursor.execute("ALTER TABLE employee_logs ADD COLUMN voice_score FLOAT")
    except sqlite3.OperationalError as e:
        print(f"voice_score column might already exist: {e}")
        
    try:
        print("Adding voice_transcript column...")
        cursor.execute("ALTER TABLE employee_logs ADD COLUMN voice_transcript TEXT")
    except sqlite3.OperationalError as e:
        print(f"voice_transcript column might already exist: {e}")
    
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
