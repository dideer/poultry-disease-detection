# app/models/detection.py
# Detection model - stores every prediction made

from app import get_db_connection

class Detection:

    @staticmethod
    def create_table():
        """
        Create detections table if it does not exist.
        Stores every disease prediction made by the system.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50),
                    image_name VARCHAR(255),
                    predicted_class VARCHAR(50) NOT NULL,
                    confidence FLOAT NOT NULL,
                    crd_prob FLOAT,
                    fowlpox_prob FLOAT,
                    healthy_prob FLOAT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            print("[INFO] detections table created or already exists.")
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Failed to create detections table: {e}")
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def save_detection(username, image_name, predicted_class,
                      confidence, crd_prob, fowlpox_prob, healthy_prob):
        """
        Save a new detection result to the database.
        Returns the new detection id or None on failure.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO detections 
                (username, image_name, predicted_class, 
                 confidence, crd_prob, fowlpox_prob, healthy_prob)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (username, image_name, predicted_class,
                  confidence, crd_prob, fowlpox_prob, healthy_prob))
            detection_id = cur.fetchone()[0]
            conn.commit()
            return detection_id
        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Failed to save detection: {e}")
            return None
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_all_detections():
        """
        Get all detections ordered by most recent first.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, username, image_name, predicted_class,
                       confidence, crd_prob, fowlpox_prob, 
                       healthy_prob, detected_at
                FROM detections
                ORDER BY detected_at DESC;
            """)
            rows = cur.fetchall()
            return rows
        except Exception as e:
            print(f"[ERROR] Failed to fetch detections: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_detections_by_user(username):
        """
        Get detections for a specific user.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, username, image_name, predicted_class,
                       confidence, crd_prob, fowlpox_prob,
                       healthy_prob, detected_at
                FROM detections
                WHERE username = %s
                ORDER BY detected_at DESC;
            """, (username,))
            rows = cur.fetchall()
            return rows
        except Exception as e:
            print(f"[ERROR] Failed to fetch user detections: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    @staticmethod
    def get_statistics():
        """
        Get total counts per disease for dashboard stats.
        """
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT predicted_class, COUNT(*) as total
                FROM detections
                GROUP BY predicted_class;
            """)
            rows = cur.fetchall()
            stats = {'crd': 0, 'fowlpox': 0, 'healthy': 0}
            for row in rows:
                stats[row[0]] = row[1]
            return stats
        except Exception as e:
            print(f"[ERROR] Failed to fetch statistics: {e}")
            return {'crd': 0, 'fowlpox': 0, 'healthy': 0}
        finally:
            cur.close()
            conn.close()