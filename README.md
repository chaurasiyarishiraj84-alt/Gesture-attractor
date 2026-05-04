---

```markdown
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

Users can control chaos in real time using simple hand gestures like pinch, enabling intuitive human–computer interaction.

---

## 🔥 Key Features

- Real-time Thomas Attractor simulation  
- Gesture-based control (thumb + index pinch)  
- Hand tracking using MediaPipe  
- Dynamic control of:
  - Zoom  
  - Rotation  
  - System parameter `b`  
- Smooth interaction using EMA filtering  
- Split-screen visualization (model + camera)  
- Low-latency streaming (WebRTC)  
- UI controls via Streamlit  
- Live performance metrics (FPS, hands, particles)  

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

```

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

```

---

## 🧠 Mathematical Model

### Thomas Attractor

```

dx/dt = sin(y) - b * x
dy/dt = sin(z) - b * y
dz/dt = sin(x) - b * z

```

Where:
- x, y, z → state variables  
- b → controls system behavior  

---

## ⚙️ Numerical Simulation (Euler Method)

```

x(t+1) = x(t) + Δt * (sin(y(t)) - b * x(t))
y(t+1) = y(t) + Δt * (sin(z(t)) - b * y(t))
z(t+1) = z(t) + Δt * (sin(x(t)) - b * z(t))

```

- Δt → time step  
- Multi-particle system used  

---

## 🔄 3D Transformation

### Rotation

```

R = Ry * Rx

```

### Projection

```

x' = x * scale + center_x
y' = y * scale + center_y

```

---

## 🎨 Rendering Pipeline

1. Simulate particles  
2. Apply rotation  
3. Project to 2D  
4. Accumulate intensity  
5. Apply glow  

### Glow Function

```

I' = log(1 + k * I) / log(1 + k)

```

---

## ✋ Gesture Mapping

| Gesture               | Action                |
|----------------------|---------------------|
| Pinch                | Activate control     |
| Left hand            | Zoom + rotation      |
| Right hand           | Control parameter b  |
| No gesture           | UI slider fallback   |

---

## 🏗️ System Architecture

```

Webcam Input
↓
MediaPipe Hand Tracking
↓
Gesture Processing
↓
Parameter Mapping
↓
Attractor Simulation
↓
3D Transformation
↓
Rendering
↓
Output Stream

```

---

## ▶️ Run Locally

### Clone repo
```

git clone [https://github.com/your-username/gesture-attractor.git](https://github.com/your-username/gesture-attractor.git)
cd gesture-attractor

```

### Install dependencies
```

pip install -r requirements.txt

```

### Run app
```

streamlit run app.py

```

---

## 🎛️ Controls

- b parameter  
- Zoom  
- Particle count  
- Toggle gesture control  
- Show finger tracking  

---

## 📊 Performance

- Resolution: 640×480  
- FPS: ~15  
- Optimizations:
  - NumPy vectorization  
  - EMA smoothing  
  - Efficient rendering  

---

## 🌍 Deployment

- Hugging Face Spaces  
- Streamlit Cloud  

---

## 🚀 Future Enhancements

- GPU acceleration  
- ML gesture classification  
- Multi-user interaction  
- Advanced visuals  

---

## 👤 Author

Rishi Raj Chaurasiya  
B.Tech AI & ML  

---

## 📜 License

MIT License
```
