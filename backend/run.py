# run.py
from app import create_app
from app.models.user import User
from app.models.detection import Detection          # ← NEW

app = create_app()

if __name__ == '__main__':
    try:
        User.create_table()
        Detection.create_table()                    # ← NEW
        print("[INFO] All tables ready!")
    except Exception as exc:
        print(f"[WARN] Skipping database initialization: {exc}")
    app.run(debug=True)