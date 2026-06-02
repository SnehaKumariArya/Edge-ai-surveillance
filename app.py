import os
import base64
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# -------------------------------------------------------------------------
# 1. INITIALIZE AI ENGINE & GLOBAL VARIABLES
# -------------------------------------------------------------------------
# Load and export the model to OpenVINO format upon startup if not already compiled
MODEL_PATH = "yolov8n.pt"
OPENVINO_MODEL_PATH = "yolov8n_openvino_model"

try:
    if not os.path.exists(OPENVINO_MODEL_PATH):
        print("[INFO] Compiling YOLOv8 to hardware-accelerated OpenVINO format...")
        base_model = YOLO(MODEL_PATH)
        base_model.export(format="openvino")
    
    # Load the highly optimized local Intel OpenVINO instance
    model = YOLO(OPENVINO_MODEL_PATH, task="detect")
    print("[SUCCESS] Hardware-accelerated OpenVINO Engine loaded successfully.")
except Exception as e:
    print(f"[WARNING] OpenVINO compilation failed or skipped, reverting to default: {e}")
    model = YOLO(MODEL_PATH)

# Global configuration states
confidence_threshold = 0.25
system_alerts = []

# Target threat watchlist classes (YOLOv8 default COCO mapping strings)
THREAT_CLASSES = ["knife", "scissors", "baseball bat", "backpack", "person"]

# -------------------------------------------------------------------------
# 2. FLASK WEB CORE APPLICATION ROUTES
# -------------------------------------------------------------------------
@app.route('/')
def index():
    """Serves the central dark-mode tracking command dashboard."""
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Receives incoming base64 video frames from client browsers,
    runs them through the AI model, logs threats, and returns annotated images.
    """
    global confidence_threshold, system_alerts
    
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No frame telemetry received'}), 400

    try:
        # Strip the data URI header scheme (e.g., "data:image/jpeg;base64,")
        header, encoded_data = data['image'].split(',', 1)
        
        # Decode base64 bytes straight into an OpenCV usable matrix format
        frame_bytes = base64.b64decode(encoded_data)
        np_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Frame decoding malfunction'}), 400

        # Execute live object detection with the dynamic threshold slider limits
        results = model(frame, conf=confidence_threshold, verbose=False)
        annotated_frame = results[0].plot()

        # Parse detected object indices to search for threats
        boxes = results[0].boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name in THREAT_CLASSES:
                timestamp = datetime.now().strftime("%H:%M:%S")
                alert_msg = f"[{timestamp}] WARNING: Threat Entity '{class_name.upper()}' identified at Node 01."
                
                # Prepend the alert to keep logs descending (newest on top)
                if alert_msg not in system_alerts:
                    system_alerts.insert(0, alert_msg)
                    
                # Cap log cache history length at 30 items to save server memory
                if len(system_alerts) > 30:
                    system_alerts.pop()

        # Re-encode the newly drawn annotated frame to base64 jpeg format
        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': f"data:image/jpeg;base64,{processed_base64}"
        })

    except Exception as e:
        print(f"[ERROR] Frame Processing Exception pipeline: {e}")
        return jsonify({'error': 'Internal server compute failure'}), 500


@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Polled regularly by the client dashboard frontend to refresh logs."""
    return jsonify({'alerts': system_alerts})


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """Updates the internal model filtering confidence instantly from frontend slider data."""
    global confidence_threshold
    data = request.get_json()
    if data and 'threshold' in data:
        confidence_threshold = float(data['threshold'])
        return jsonify({'status': 'success', 'current_threshold': confidence_threshold})
    return jsonify({'status': 'error', 'message': 'Invalid parameter data structure'}), 400

# -------------------------------------------------------------------------
# 3. PRODUCTION PORT ALLOCATION RUNNER
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Render binds the application to an environment variable called PORT.
    # If run locally, it cleanly falls back to manual testing on port 5000.
    web_port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=web_port, debug=False)