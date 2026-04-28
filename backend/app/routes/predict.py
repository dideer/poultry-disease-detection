# app/routes/predict.py
# AI prediction route - handles image upload and disease detection

from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import uuid
import gdown
from app.models.detection import Detection

predict_bp = Blueprint('predict', __name__)

# -----------------------------------------------
# Load model once when app starts
# -----------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.keras')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'frontend', 'uploads')

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None  # ← always define first

# Download model only if not present (works locally AND on Render)
if not os.path.exists(MODEL_PATH):
    print("⬇️  Downloading model from Google Drive...")
    try:
        gdown.download(
            "https://drive.google.com/uc?id=1nFeccdyQ6jwZLaH-H5PvYqRyegXNEY_z",
            MODEL_PATH,
            quiet=False
        )
        print("✅ Model downloaded!")
    except Exception as e:
        print(f"[WARN] Could not download model: {e}")
else:
    print("✅ Model already exists, skipping download.")

# Load the AI model
try:
    model = load_model(MODEL_PATH)
    print("✅ AI Model loaded successfully!")
except Exception as e:
    model = None
    print(f"[ERROR] Failed to load model: {e}")
# Class names and advice
CLASSES = ['crd', 'fowlpox', 'healthy']

ADVICE = {
    'crd': {
        'full_name': 'Chronic Respiratory Disease (CRD)',
        'symptoms': 'Nasal discharge, coughing, swollen face, breathing difficulty',
        'urgency': 'High - Act within 24 hours',
        'treatment': [
            'Isolate infected birds immediately',
            'Administer Tylosin or Oxytetracycline antibiotics',
            'Improve ventilation in the poultry house',
            'Consult a veterinarian for proper dosage',
            'Disinfect the poultry house thoroughly'
        ]
    },
    'fowlpox': {
        'full_name': 'Fowl Pox',
        'symptoms': 'Wart-like lesions on comb, face, and legs',
        'urgency': 'Medium - Act within 48 hours',
        'treatment': [
            'Isolate infected birds from healthy flock',
            'Apply iodine solution to lesions',
            'Vaccinate healthy birds immediately',
            'Provide vitamin A supplements',
            'Keep wounds clean to prevent secondary infections'
        ]
    },
    'healthy': {
        'full_name': 'Healthy Chicken',
        'symptoms': 'No disease symptoms detected',
        'urgency': 'None - Continue routine care',
        'treatment': [
            'Continue regular feeding schedule',
            'Maintain clean water supply',
            'Keep poultry house well ventilated',
            'Monitor flock regularly',
            'Keep vaccination schedule up to date'
        ]
    }
}

def preprocess_image(image_path):
    """Preprocess image for model prediction."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@predict_bp.route('/api/predict', methods=['POST'])
def predict():
    """
    Receive image, run AI prediction, save to DB, return result.
    """
    if model is None:
        return jsonify({
            'success': False,
            'message': 'AI model not available'
        }), 500

    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No image uploaded'
        }), 400

    file = request.files['file']
    username = request.form.get('username', 'unknown')

    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No file selected'
        }), 400

    try:
        # Save uploaded image with unique name
        ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)

        # Preprocess and predict
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = CLASSES[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        # Individual probabilities
        crd_prob     = float(predictions[0][0]) * 100
        fowlpox_prob = float(predictions[0][1]) * 100
        healthy_prob = float(predictions[0][2]) * 100

        # Save to database
        Detection.save_detection(
            username=username,
            image_name=unique_name,
            predicted_class=predicted_class,
            confidence=round(confidence, 2),
            crd_prob=round(crd_prob, 2),
            fowlpox_prob=round(fowlpox_prob, 2),
            healthy_prob=round(healthy_prob, 2)
        )

        # Return result
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'probabilities': {
                'crd': round(crd_prob, 2),
                'fowlpox': round(fowlpox_prob, 2),
                'healthy': round(healthy_prob, 2)
            },
            'disease_info': ADVICE[predicted_class],
            'image_name': unique_name
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Prediction failed: {str(e)}'
        }), 500

@predict_bp.route('/api/history', methods=['GET'])
def history():
    """Get all detection history."""
    username = request.args.get('username', None)
    if username:
        detections = Detection.get_detections_by_user(username)
    else:
        detections = Detection.get_all_detections()

    results = []
    for row in detections:
        results.append({
            'id': row[0],
            'username': row[1],
            'image_name': row[2],
            'predicted_class': row[3],
            'confidence': row[4],
            'crd_prob': row[5],
            'fowlpox_prob': row[6],
            'healthy_prob': row[7],
            'detected_at': str(row[8])
        })

    return jsonify({'success': True, 'detections': results}), 200

@predict_bp.route('/api/statistics', methods=['GET'])
def statistics():
    """Get detection statistics for dashboard."""
    stats = Detection.get_statistics()
    total = sum(stats.values())
    return jsonify({
        'success': True,
        'statistics': stats,
        'total': total
    }), 200