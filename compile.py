from ultralytics import YOLO

# Load your existing standard PyTorch model
model = YOLO("yolov8n.pt")

# Export it! 
# half=True compresses the weights from FP32 down to FP16 precision,
# which doubles processing speeds on low-power CPUs without losing noticeable accuracy.
model.export(format="openvino", half=True)