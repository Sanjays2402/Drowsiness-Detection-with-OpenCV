# 🚗💤 Drowsiness Detection with OpenCV

> **Real-time driver drowsiness detection** using computer vision and the Eye Aspect Ratio (EAR) algorithm.

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/9532758"><img src="https://img.shields.io/badge/IEEE-Published%20Paper-00629B?style=for-the-badge&logo=ieee&logoColor=white" alt="IEEE Paper"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/dlib-face__recognition-green?style=flat-square">
  <img src="https://img.shields.io/badge/SciPy-EAR%20Computation-8CAAE6?style=flat-square&logo=scipy&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square">
</p>

---

## 📄 Publication

This project is published in **IEEE Xplore**:

> **S. Santhanam et al.**, "Real-Time Drowsiness Detection using Computer Vision," *2021 5th International Conference on Computing Methodologies and Communication (ICCMC)*, 2021.
>
> 🔗 **[Read the paper on IEEE Xplore →](https://ieeexplore.ieee.org/document/9532758)**

---

## 🧠 How It Works

The system uses the **Eye Aspect Ratio (EAR)** to detect drowsiness in real-time:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

```
       p2    p3
      •--------•
     /          \
p1 •            • p4    ← horizontal axis
     \          /
      •--------•
       p6    p5
```

- **Eyes open** → EAR ≈ 0.3
- **Eyes closed** → EAR ≈ 0.05
- If EAR stays below a threshold for enough consecutive frames → **🚨 ALARM**

### Architecture

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌───────────┐
│  Webcam   │───▶│ Face Landmark │───▶│ Compute   │───▶│  Trigger  │
│  Stream   │    │  Detection    │    │   EAR     │    │  Alarm?   │
└──────────┘    └──────────────┘    └───────────┘    └───────────┘
                  face_recognition      scipy           playsound
```

---

## ✨ Features

- 🎥 Real-time webcam-based detection
- 👁️ Eye Aspect Ratio (EAR) algorithm
- 🔊 Audio alarm on drowsiness detection
- 📊 Live EAR display overlay
- ⚙️ Configurable thresholds via CLI args

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Webcam
- CMake (for dlib): `brew install cmake` (macOS) or `apt install cmake` (Linux)

### Installation

```bash
git clone https://github.com/Sanjays2402/Drowsiness-Detection-with-OpenCV.git
cd Drowsiness-Detection-with-OpenCV
pip install -r requirements.txt
```

### Usage

```bash
# Default settings
python drowsiness_detection.py

# Custom thresholds
python drowsiness_detection.py --ear-threshold 0.22 --frame-count 48

# Different video source
python drowsiness_detection.py --video-source 1
```

Press **`q`** to quit.

---

## ⚙️ Configuration

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| EAR Threshold | `--ear-threshold` | `0.25` | EAR below this = eyes closed |
| Frame Count | `--frame-count` | `60` | Consecutive frames before alarm |
| Video Source | `--video-source` | `0` | Webcam index |
| Alarm Sound | `--alarm-sound` | `assets/alert1.mp3` | Path to alert audio |

---

## 🎬 Demo

> *Add a GIF or screenshot here!*
>
> ![Demo placeholder](https://via.placeholder.com/600x300?text=Add+Demo+GIF+Here)

---

## 📁 Project Structure

```
├── drowsiness_detection.py   # Main detection script
├── assets/
│   └── alert1.mp3            # Alarm sound
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

- **[OpenCV](https://opencv.org/)** — Video capture & frame processing
- **[face_recognition](https://github.com/ageitgey/face_recognition)** — Facial landmark detection (dlib-based)
- **[SciPy](https://scipy.org/)** — Euclidean distance for EAR
- **[imutils](https://github.com/PyImageSearch/imutils)** — Video stream utilities
- **[playsound](https://github.com/TaylorSMarks/playsound)** — Audio alarm playback

---

## 📜 License

[MIT](LICENSE) © Sanjay Santhanam

## 👤 Author

**Sanjay Santhanam** — [GitHub](https://github.com/Sanjays2402)
