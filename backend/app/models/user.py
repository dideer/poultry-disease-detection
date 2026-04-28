# app/models/user.py
# User model and DB operations

from app import get_db_connection, bcrypt

class User:
    @staticmethod
    def create_table():
        """
        Create the users table in PostgreSQL if it does not exist.
        Table: users (id, username, email, password, created_at)
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) NOT NULL UNIQUE,
                    email VARCHAR(100) NOT NULL UNIQUE,
                    password VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            print("[INFO] users table created or already exists.")
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Failed to create users table: {e}")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def create_user(username, email, password):
        """
        Insert a new user into the users table with hashed password.
        Returns the new user's id or None on failure.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        try:
            cur.execute("""
                INSERT INTO users (username, email, password)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (username, email, hashed_pw))
            user_id = cur.fetchone()[0]
            conn.commit()
            return user_id
        except Exception as e:
            conn.rollback()
            return None
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def find_by_username(username):
        """
        Find a user by username. Returns user tuple or None.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, password, created_at FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        return user

    @staticmethod
    def find_by_email(email):
        """
        Find a user by email. Returns user tuple or None.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, password, created_at FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        return user
