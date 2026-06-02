# ⚡ EDGE-AI SURVEILLANCE COMMAND CENTER

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-0071C5?style=for-the-badge&logo=intel&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-8A2BE2?style=for-the-badge&logo=ai&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

<p align="center">
  A high-performance, completely private, hardware-accelerated computer vision ecosystem that processes live video streams at the edge to detect spatial threats instantly.
</p>

[![Render Deployment Status](https://img.shields.io/badge/Render-Live_App-9333EA?style=for-the-badge&logo=render&logoColor=white)](https://edge-ai-surveillance.onrender.com)

<h4>
  <a href="#-key-architecture">Architecture</a> •
  <a href="#-ui-features">UI Features</a> •
  <a href="#-installation--setup">Quick Start</a> •
  <a href="#-under-the-hood">Technical Insights</a>
</h4>

</div>

---

## ⚙️ Key Architecture

Standard AI integrations usually experience severe frame dropping or lagging because the web server interface and the neural network processing try to use the same CPU thread. This project completely eliminates bottlenecks by utilizing a **detached multi-threaded memory channel pipeline**:

* 🎥 **Camera Capture Thread:** Interacts with physical video devices via OpenCV lanes, instantly copying frames into a global cache memory socket.
* 🧠 **OpenVINO Inference Engine:** Runs natively on a separate processing thread, applying hardware-accelerated matrix math to identify target objects without blocking the main runtime.
* 🌐 **Asynchronous Flask Interface:** Streams fully annotated imagery to an interactive client layout via optimized MJPEG byte blocks.

---

## 🎨 UI Features

The modern dashboard transforms raw surveillance coordinates into a premium command console layout:
- **Cyberpunk Dark Mode Grid:** Styled using customized CSS variables for optimal structural contrast.
- **Dynamic Filter Control:** An integrated JavaScript range slider that adjusts AI confidence detection limits on the fly without requiring a server reboot.
- **Isolated Log Stream:** An auto-refreshing log timeline panel that tracks anomaly timestamps dynamically.

---

## 🚀 Installation & Setup

### 1. Initialize Your Environment
Clone the repository, navigate to the project directory, and install the optimized software dependencies:
```bash
pip install -r requirements.txt
