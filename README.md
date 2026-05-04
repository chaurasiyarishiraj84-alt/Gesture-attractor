# 🌀 Gesture-Controlled Thomas Attractor System

An end-to-end **real-time interactive visualization system** built using **Computer Vision, Mathematics, and AI-based gesture control**, with a **Streamlit + WebRTC interface** for live interaction.

This project demonstrates a complete pipeline — from **hand tracking and gesture recognition** to **chaotic system simulation, rendering, and real-time visualization**.

---

## 🚀 Live Demo

👉 https://huggingface.co/spaces/YOUR-SPACE-LINK

---

## 🚀 Features

* Real-time Thomas Attractor simulation
* Gesture-based control (thumb + index pinch)
* Hand tracking using MediaPipe
* Zoom, rotation, and parameter control via gestures
* Smooth motion using EMA filtering
* Split visualization (model + camera feed)
* Low-latency streaming using WebRTC
* Adjustable parameters via UI sliders
* Real-time performance metrics (FPS, hands, particles)

---

## 🧠 Tech Stack

* **Python**
* **NumPy**
* **OpenCV**
* **MediaPipe**
* **Streamlit**
* **streamlit-webrtc**
* **AV (PyAV)**

---

## 📁 Project Structure

```id="t1m9k2"
gesture-attractor/
│
├── app.py           # Streamlit + WebRTC app (real-time UI + streaming)
├── main.py          # Standalone OpenCV runner (no Streamlit, direct execution)
├── attractor.py     # Thomas attractor simulation (math + particle updates)
├── hand_tracker.py  # Hand tracking & gesture detection (MediaPipe)
├── renderer.py      # Rendering engine (projection, glow, drawing)
├── ui.py            # UI helpers (controls, overlays, layout)
├── requirements.txt # Project dependencies
└── README.md        # Project documentation

```

---

## 🧠 Mathematical Model

### Thomas Attractor Equations

[
\frac{dx}{dt} = \sin(y) - b x
]
[
\frac{dy}{dt} = \sin(z) - b y
]
[
\frac{dz}{dt} = \sin(x) - b z
]

Where:

* (x, y, z) → state variables
* (b) → system parameter controlling chaos

---

## ⚙️ Numerical Approach

* Method: **Euler Integration**
* Discrete update:

[
x_{t+1} = x_t + \Delta t (\sin(y_t) - b x_t)
]

* Multiple particles simulate trajectory evolution

---

## 🎯 Gesture Mapping

### ✋ Hand Interaction

| Gesture               | Action                |
| --------------------- | --------------------- |
| Pinch (thumb + index) | Activate control      |
| Left Hand             | Zoom + Rotation       |
| Right Hand            | Control parameter `b` |
| No gesture            | UI slider fallback    |

---

## 🔄 3D Transformation

Rotation matrices:

[
R = R_y \cdot R_x
]

Projection:

[
x' = x \cdot scale + center_x
]
[
y' = y \cdot scale + center_y
]

---

## 🎨 Rendering Pipeline

1. Simulate attractor points
2. Apply rotation matrix
3. Project to 2D screen
4. Accumulate intensity
5. Apply logarithmic glow

[
I' = \frac{\log(1 + kI)}{\log(1 + k)}
]

---

## 🏗️ System Architecture

```id="y8x2lp"
Webcam Input
     ↓
MediaPipe Hand Tracking
     ↓
Gesture Processing (Pinch Detection)
     ↓
Parameter Mapping (b, zoom, rotation)
     ↓
Attractor Simulation (Numerical Integration)
     ↓
3D Transformation & Projection
     ↓
Rendering (Glow + Particles)
     ↓
Frame Composition (Camera + Model)
     ↓
Streamlit WebRTC Output
```

---

## ▶️ Run the Application

### 1️⃣ Clone repository

```bash id="f2x9ka"
git clone <your-repo-url>
cd gesture-attractor
```

### 2️⃣ Install dependencies

```bash id="d8k2pl"
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit app

```bash id="k92mza"
streamlit run app.py
```

---

## ⚙️ Controls (UI)

* **b parameter slider**
* **Zoom slider**
* **Particle count**
* **Toggle hand control**
* **Show finger tracking**

---

## 📊 Model Details

* System Type: **Nonlinear Dynamical System**
* Simulation: **Multi-particle system**
* Rendering: **Glow-based accumulation**
* Interaction: **Gesture-driven real-time control**

---

## ⚡ Performance

* Resolution: 640×480
* Frame Rate: ~15 FPS
* Optimized using:

  * NumPy vectorization
  * EMA smoothing
  * Efficient rendering loops

---

## 🔁 Future Enhancements

* GPU acceleration (OpenGL / CUDA)
* Gesture classification (ML-based)
* Multi-user interaction
* Advanced visual effects
* Mobile deployment

---

## 🌍 Deployment

This project can be deployed on:

* Hugging Face Spaces
* Streamlit Cloud

---

## 👤 Author

Developed by **Rishi Raj Chaurasiya**
B.Tech in Artificial Intelligence & Machine Learning

---

## 📜 License

This project is open-source and available under the MIT License.
