# 🌀 Gesture-Controlled Thomas Attractor System

> Real-time gesture-driven chaotic system visualization using Computer Vision, Mathematics, and Interactive Rendering.

---

## 🚀 Live Demo

👉 https://huggingface.co/spaces/YOUR-SPACE-LINK

---

## ✨ Overview

This project is an interactive visualization system that combines:

- Nonlinear dynamical systems (Thomas Attractor)
- Computer Vision (hand tracking)
- Gesture-based control
- Real-time rendering pipeline

Users can control chaos in real time using simple hand gestures like pinch.

---

## 🔥 Key Features

- Real-time Thomas Attractor simulation  
- Gesture-based control (pinch detection)  
- Hand tracking using MediaPipe  
- Control of zoom, rotation, and parameter `b`  
- Smooth motion using EMA filtering  
- Split-screen visualization  
- Low-latency streaming using WebRTC  
- UI controls via Streamlit  
- Live performance metrics  

---

## 🧠 Tech Stack

- Python  
- NumPy  
- OpenCV  
- MediaPipe  
- Streamlit  
- streamlit-webrtc  
- PyAV  

---

## 📁 Project Structure


gesture-attractor/
│
├── app.py
├── main.py
├── attractor.py
├── hand_tracker.py
├── renderer.py
├── ui.py
├── requirements.txt
└── README.md


---

## 🧠 Mathematical Model

### Thomas Attractor


dx/dt = sin(y) - b * x
dy/dt = sin(z) - b * y
dz/dt = sin(x) - b * z


Where:
- x, y, z → state variables  
- b → controls chaos  

---

## ⚙️ Numerical Method (Euler)


x(t+1) = x(t) + dt * (sin(y) - b * x)
y(t+1) = y(t) + dt * (sin(z) - b * y)
z(t+1) = z(t) + dt * (sin(x) - b * z)


---

## 🔄 Transformation


R = Ry * Rx

x' = x * scale + cx
y' = y * scale + cy


---

## 🎨 Rendering

- Particle simulation  
- Rotation  
- Projection  
- Intensity accumulation  
- Glow effect  


I' = log(1 + kI) / log(1 + k)


---

## ✋ Gesture Mapping

| Gesture | Action |
|--------|-------|
| Pinch | Activate control |
| Left Hand | Zoom + rotation |
| Right Hand | Control b |
| No gesture | Slider control |

---

## 🏗️ Architecture


Camera → Hand Tracking → Gesture → Parameters → Simulation → Rendering → Output


---

## ▶️ Run


git clone https://github.com/your-username/gesture-attractor.git

cd gesture-attractor
pip install -r requirements.txt
streamlit run app.py


---

## 📊 Performance

- 640×480 resolution  
- ~15 FPS  
- Optimized using NumPy  

---

## 👤 Author

Rishi Raj Chaurasiya  
B.Tech AI & ML  

---

## 📜 License

MIT License
