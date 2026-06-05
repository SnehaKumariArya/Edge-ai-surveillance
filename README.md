# ⚡ EDGE-AI SURVEILLANCE COMMAND CENTER

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005ced?style=for-the-badge&logo=onnx&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-8A2BE2?style=for-the-badge&logo=ai&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

<p align="center">
  A lightweight, low-latency, cloud-optimized computer vision ecosystem that processes live client-side browser video streams via WebRTC to detect spatial threats instantly using YOLOv8.
</p>

<h4>
  <a href="#-key-architecture">Architecture</a> •
  <a href="#-ui-features">UI Features</a> •
  <a href="#-installation--setup">Quick Start</a> •
  <a href="#-core-endpoints">API Telemetry</a>
</h4>

</div>

---

## ⚙️ Key Architecture

Standard AI web integrations usually suffer from severe lag because video capture and deep learning inference overwhelm a single server thread. This ecosystem completely eliminates hardware bottlenecks by leveraging an asymmetric **Client-Server WebRTC Pipeline**:

* 🌐 **Asynchronous WebRTC Frontend:** Captures the user's camera feed directly inside the web browser using lightweight client-side JavaScript. This offloads all video hardware dependencies entirely from the server.
* 🛡️ **Collision-Locked Ingestion:** Uses a smart concurrency guard (`isProcessingFrame`) to drop overlapping frame transits, ensuring data packets never crowd the network layer.
* 🧠 **Optimized ONNX Inference Engine:** Executes target tracking via a highly compressed ONNX serialization layout. This keeps memory usage well under **250 MB RAM**, rendering bounding boxes instantly even on shared, single-core cloud micro-containers.

---

## 🎨 UI Features

The control room dashboard transforms raw matrices into an interactive cyber-defense layout:
* **Cyberpunk Command Grid:** A high-contrast dark user interface built using custom CSS variables for low eye strain during long-term monitoring.
* **Dynamic Filter Slider:** An integrated JavaScript range slider that dynamically updates AI prediction confidence boundaries on the fly without a system restart.
* **Isolated Log Stream:** An auto-refreshing log timeline panel that scrolls real-time threat detection logs on a detached poll rate.

---

## 🚀 Installation & Setup

### 1. Prerequisite Environment
Clone the repository, navigate to the target directory, and install the optimized system requirements:
```bash
git clone [https://github.com/SnehaKumariArya/Edge-ai-surveillance.git](https://github.com/SnehaKumariArya/Edge-ai-surveillance.git)
cd Edge-ai-surveillance
pip install -r requirements.txt
