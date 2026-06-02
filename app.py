import cv2
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import threading
import time
from datetime import datetime

app = Flask(__name__)

# 1. Load the optimized OpenVINO model folder
model = YOLO("yolov8n_openvino_model/")

# 2. Simplified to ONLY use your live webcam
CAMERA_SOURCES = {
    "front_gate": 0  
}

# Shared memory spaces for the background threads
latest_frames = {}
processed_frames = {}
alert_logs = []

# --- CORE AI LOGIC SETTINGS ---
CONFIDENCE_THRESHOLD = 0.25  # Lowered to 25% so it easily catches scissors/objects in indoor light
THREAT_CLASSES = ["knife", "scissors", "baseball bat", "backpack", "person"]

def camera_capture_worker(cam_id, source):
    """Background loop fetching raw frames directly from your webcam."""
    cap = cv2.VideoCapture(source)
    while True:
        success, frame = cap.read()
        if success:
            latest_frames[cam_id] = frame
        time.sleep(0.01) # Yields to prevent CPU hogging

def ai_inference_worker():
    """Background engine loop running OpenVINO hardware matrix math."""
    global CONFIDENCE_THRESHOLD
    while True:
        for cam_id in list(latest_frames.keys()):
            raw_frame = latest_frames.get(cam_id)
            if raw_frame is None:
                continue
                
            # Create a detached copy to ensure safe thread reads
            frame = raw_frame.copy()
            
            # Execute OpenVINO inference at default 640x640 shape
            results = model(frame, stream=True, verbose=False)
            threat_detected = False
            detected_items = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]

                    if conf > CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # --- DIAGNOSTIC TERMINAL PRINT ---
                        # This will print directly to your terminal window whenever ANY object is found
                        print(f"📡 AI TARGET SPOTTED: {class_name} ({conf*100:.1f}%)")

                        if class_name in THREAT_CLASSES:
                            color = (0, 0, 255) # Warning Red
                            label = f"ALERT: {class_name} ({conf:.2f})"
                            threat_detected = True
                            detected_items.append(class_name)
                        else:
                            color = (0, 255, 0) # Normal Green
                            label = f"{class_name} ({conf:.2f})"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if threat_detected:
                unique_threats = list(set(detected_items))
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                log_entry = f"[{timestamp}] Anomaly Flagged: {', '.join(unique_threats)}"
                if log_entry not in alert_logs:
                    alert_logs.insert(0, log_entry)

            # Push the annotated drawing data into the viewing queue
            processed_frames[cam_id] = frame
        
        # Give the webcam space to pass fresh frames
        time.sleep(0.03)

# --- START THE BACKGROUND THREADS ---
for camera_id, camera_source in CAMERA_SOURCES.items():
    t = threading.Thread(target=camera_capture_worker, args=(camera_id, camera_source), daemon=True)
    t.start()

ai_worker = threading.Thread(target=ai_inference_worker, daemon=True)
ai_worker.start()

def generate_mjpeg_stream(cam_id):
    """Formats processed images to streamable MJPEG byte lines for Flask paths."""
    while True:
        # Check if the AI engine has finished drawing boxes on a frame
        if cam_id in processed_frames and processed_frames[cam_id] is not None:
            # Grab the annotated frame and encode it immediately
            ret, buffer = cv2.imencode('.jpg', processed_frames[cam_id])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # CRITICAL: Clear out the frame slot after serving it.
                # This forces Flask to pause and wait until the AI engine finishes processing the next frame.
                processed_frames[cam_id] = None
        
        # Match standard webcam refresh rate (~30ms)
        time.sleep(0.03)
@app.route('/')
def index():
    return render_template('index.html', cam_ids=CAMERA_SOURCES.keys())

@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    return Response(generate_mjpeg_stream(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def get_alerts():
    return jsonify(alerts=alert_logs[:20])

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global CONFIDENCE_THRESHOLD
    data = request.get_json()
    CONFIDENCE_THRESHOLD = float(data.get('threshold', 0.25))
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)