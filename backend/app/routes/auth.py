# app/routes/auth.py
# Auth API routes: /api/register, /api/login

from flask import Blueprint, request, jsonify
from app.models.user import User
from app.utils.helpers import get_message, hash_password, check_password
from app import bcrypt

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    lang = data.get('lang', 'en')

    if not username or not email or not password:
        return jsonify({
            'success': False,
            'message': get_message('register_error', lang)
        }), 400

    if User.find_by_username(username) or User.find_by_email(email):
        return jsonify({
            'success': False,
            'message': get_message('register_exists', lang)
        }), 409

    user_id = User.create_user(username, email, password)
    if user_id:
        return jsonify({
            'success': True,
            'message': get_message('register_success', lang)
        }), 201
    else:
        return jsonify({
            'success': False,
            'message': get_message('register_error', lang)
        }), 500

@auth_bp.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    lang = data.get('lang', 'en')

    user = User.find_by_username(username)
    if not user:
        return jsonify({
            'success': False,
            'message': get_message('login_invalid', lang)
        }), 401

    hashed = user[3]  # password field
    if check_password(bcrypt, hashed, password):
        return jsonify({
            'success': True,
            'message': get_message('login_success', lang)
        }), 200
    else:
        return jsonify({
            'success': False,
            'message': get_message('login_invalid', lang)
        }), 401

@auth_bp.route('/api/predict', methods=['GET'])
def predict():
    # Placeholder for future AI model integration
    return jsonify({
        'success': False,
        'message': 'Prediction endpoint not implemented yet.'
    }), 501
