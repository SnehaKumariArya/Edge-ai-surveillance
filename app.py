import os
import base64
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Load stable CPU model for Render
MODEL_PATH = "yolov8n.pt"
try:
    model = YOLO(MODEL_PATH)
    print("[SUCCESS] YOLOv8 Engine loaded safely.")
except Exception as e:
    print(f"[CRITICAL] AI engine failed to load: {e}")
    model = None

CONFIDENCE_THRESHOLD = 0.25
alert_logs = []
THREAT_CLASSES = ["knife", "scissors", "baseball bat", "backpack", "person"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global CONFIDENCE_THRESHOLD, alert_logs
    if model is None:
        return jsonify({'error': 'AI Engine offline'}), 503

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    try:
        header, encoded_string = data['image'].split(',', 1)
        image_bytes = base64.b64decode(encoded_string)
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Frame decoding failure'}), 400

        # Run inference using the efficient ONNX model structure
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # 🌟 CRITICAL FIX: Force ONNX to explicitly render text labels and boxes
        annotated_frame = results[0].plot(labels=True, conf=True, boxes=True)

        # Check inference outcomes for target threat anomalies
        detected_items = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                if class_name in THREAT_CLASSES:
                    detected_items.append(class_name)

        # Log alerts dynamically if targets are caught
        if detected_items:
            unique_threats = list(set(detected_items))
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] Anomaly Flagged: {', '.join(unique_threats).upper()}"
            if log_entry not in alert_logs:
                alert_logs.insert(0, log_entry)
            if len(alert_logs) > 30:
                alert_logs.pop()

        # Compress and encode the annotated frame back to base64 string bytes
        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'processed_image': f"data:image/jpeg;base64,{processed_base64}"})
    except Exception as e:
        print(f"[ERROR] Inference step crash: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/alerts', methods=['GET'])
def get_alerts():
    return jsonify(alerts=alert_logs[:20])

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global CONFIDENCE_THRESHOLD
    data = request.get_json()
    if data and 'threshold' in data:
        CONFIDENCE_THRESHOLD = float(data['threshold'])
        return jsonify(success=True, threshold=CONFIDENCE_THRESHOLD)
    return jsonify(success=False), 400

if __name__ == "__main__":
    web_port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=web_port, debug=False)