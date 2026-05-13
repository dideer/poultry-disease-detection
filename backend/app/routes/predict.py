import os
import uuid

from flask import Blueprint, jsonify, request
import gdown
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

from app.models.detection import Detection
from app.models.user import User

predict_bp = Blueprint('predict', __name__)

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, 'chicken_cnn_final.tflite')
YOLO_PATH     = os.path.join(BASE_DIR, 'yolov8m.pt')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'frontend', 'uploads')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Download CNN model ────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading CNN model...")
    try:
        gdown.download(
            id="1RhX6Ea_f0xLkjvqThiE_HqEQAbgXZOLO",
            output=MODEL_PATH, quiet=False)
        print("[INFO] CNN model downloaded.")
    except Exception as exc:
        print(f"[WARN] Could not download CNN model: {exc}")
else:
    print("[INFO] CNN model already exists.")

# ── Download YOLO model ───────────────────────────────────────────────────────
if not os.path.exists(YOLO_PATH):
    print("[INFO] Downloading YOLO model...")
    try:
        gdown.download(
            id="1d-xjIR0TWf16NuZYHCYj6SvtGspD13UU",
            output=YOLO_PATH, quiet=False)
        print("[INFO] YOLO model downloaded.")
    except Exception as exc:
        print(f"[WARN] Could not download YOLO model: {exc}")
else:
    print("[INFO] YOLO model already exists.")

# ── Load CNN (TFLite) ─────────────────────────────────────────────────────────
interpreter    = None
input_details  = None
output_details = None

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("[INFO] CNN model loaded successfully.")
except Exception as exc:
    interpreter = None
    print(f"[ERROR] Failed to load CNN model: {exc}")

# ── Load YOLO ─────────────────────────────────────────────────────────────────
yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO(YOLO_PATH)
    print("[INFO] YOLO model loaded successfully.")
except Exception as exc:
    yolo_model = None
    print(f"[ERROR] Failed to load YOLO model: {exc}")

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = ['crd', 'fowlpox', 'healthy']

REJECT_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'laptop', 'tv', 'cell phone', 'keyboard', 'mouse'
}

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


def preprocess_array(img_rgb):
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array   = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def run_cnn(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]


def run_pipeline(image_path):
    img_orig = cv2.imread(image_path)
    img_rgb  = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    # Step 1 — YOLO detection
    results       = yolo_model.predict(image_path, conf=0.15, verbose=False)
    boxes         = results[0].boxes
    all_classes   = [yolo_model.names[int(b.cls)] for b in boxes]
    chicken_boxes = [b for b in boxes if int(b.cls) == 14]
    has_reject    = any(c in REJECT_CLASSES for c in all_classes)

    # Step 2 — Decide mode
    if has_reject and len(chicken_boxes) == 0:
        return None, None, None, "rejected"

    elif len(chicken_boxes) > 0:
        box             = chicken_boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop            = img_rgb[y1:y2, x1:x2]
        img_array       = preprocess_array(crop)
        mode            = "yolo"

    else:
        img_array = preprocess_array(img_rgb)
        mode      = "closeup"

    # Step 3 — CNN classification
    preds      = run_cnn(img_array)
    class_idx  = int(np.argmax(preds))
    disease    = CLASSES[class_idx]
    confidence = float(preds[class_idx]) * 100

    return disease, confidence, preds, mode


@predict_bp.route('/api/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({'success': False, 'message': 'CNN model not available'}), 500
    if yolo_model is None:
        return jsonify({'success': False, 'message': 'YOLO model not available'}), 500
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'}), 400

    file     = request.files['file']
    username = request.form.get('username', 'unknown')
    user     = User.find_by_username(username) if username and username != 'unknown' else None

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400

    try:
        ext         = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        filepath    = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)

        disease, confidence, preds, mode = run_pipeline(filepath)

        if mode == "rejected":
            return jsonify({
                'success': False,
                'message': 'No chicken detected. Please upload a chicken image.'
            }), 400

        crd_prob     = float(preds[0]) * 100
        fowlpox_prob = float(preds[1]) * 100
        healthy_prob = float(preds[2]) * 100

        Detection.save_detection(
            user_id=user['id'] if user else None,
            username=username,
            image_name=unique_name,
            predicted_class=disease,
            confidence=round(confidence, 2),
            crd_prob=round(crd_prob, 2),
            fowlpox_prob=round(fowlpox_prob, 2),
            healthy_prob=round(healthy_prob, 2),
            chicken_count=1
        )

        return jsonify({
            'success': True,
            'predicted_class': disease,
            'confidence': round(confidence, 2),
            'detection_mode': mode,
            'probabilities': {
                'crd':     round(crd_prob, 2),
                'fowlpox': round(fowlpox_prob, 2),
                'healthy': round(healthy_prob, 2)
            },
            'disease_info': ADVICE[disease],
            'image_name': unique_name
        }), 200

    except Exception as exc:
        return jsonify({'success': False, 'message': f'Prediction failed: {exc}'}), 500


@predict_bp.route('/api/history', methods=['GET'])
def history():
    username   = request.args.get('username', None)
    detections = Detection.get_all_detections(username=username)
    return jsonify({
        'success': True,
        'detections': [{
            **item,
            'detected_at': str(item['detected_at'])
        } for item in detections]
    }), 200


@predict_bp.route('/api/statistics', methods=['GET'])
def statistics():
    stats = Detection.get_statistics()
    total = sum(stats.values())
    return jsonify({
        'success': True,
        'statistics': stats,
        'total': total
    }), 200