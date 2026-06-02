# Edge-AI Local Surveillance Command Center

A high-performance, private, hardware-accelerated local surveillance hub that uses **YOLOv8** and Intel's **OpenVINO** toolkit to detect threats in real-time from a live webcam feed.

## 🏗️ Architecture Layout

- **Camera Capture Thread:** Continuously pulls raw frames from hardware lanes into a memory buffer.
- **OpenVINO Inference Thread:** Runs localized matrix math independently on the CPU to detect anomalies (persons, scissors, knives, backpacks) without lagging the video stream.
- **Flask Web Server:** Serves a modern, dark-themed cyber-grid dashboard showing live video analytics, alert timelines, and dynamic confidence filters.

## 🚀 Getting Started

### 1. Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt