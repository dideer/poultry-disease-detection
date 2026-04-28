# app/__init__.py
from pathlib import Path
from flask import Flask, redirect, send_from_directory
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import psycopg2
from config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

bcrypt = Bcrypt()

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def create_app():
    frontend_dir = Path(__file__).resolve().parents[2] / 'frontend'
    app = Flask(__name__, 
                static_folder=str(frontend_dir), 
                static_url_path='')
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    bcrypt.init_app(app)

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.predict import predict_bp       # ← NEW

    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)              # ← NEW

    @app.route('/')
    def index():
        return redirect('/dashboard.html')

    @app.route('/<path:path>')
    def serve_frontend(path):
        return send_from_directory(app.static_folder, path)

    return app