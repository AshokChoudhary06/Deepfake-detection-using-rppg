# 🧠 Deepfake Detection using rPPG (Remote Photoplethysmography)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-orange)
![PyVHR](https://img.shields.io/badge/Library-PyVHR-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

### 🎥 Overview

> **Deepfakes** are AI-generated fake videos that manipulate facial expressions and identities.  
> This project detects such **deepfake videos** by analyzing **biological signals** using **rPPG (remote photoplethysmography)** — a method that captures **subtle color changes in the face** caused by heartbeats.

💡 **Core Concept:**  
Real human faces exhibit consistent rPPG patterns, while deepfakes do not.  
Using **PyVHR**, **Digital Signal Processing (DSP)**, and **Deep Learning**, we extract, filter, and classify these patterns to distinguish real vs. fake videos.

---

## 🧩 Table of Contents
- [🎯 Objectives](#-objectives)
- [🧰 Tech Stack](#-tech-stack)
- [🧪 Role of PyVHR](#-role-of-pyvhr)
- [⚙️ Signal Processing Techniques](#️-signal-processing-techniques)
- [📂 Folder Structure](#-folder-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Running the Web App](#-running-the-web-app)
- [📊 Results](#-results)
- [📚 References](#-references)
- [🧑‍💻 Author](#-author)

---

## 🎯 Objectives
- Detect **deepfake videos** using physiological-based rPPG signals.  
- Extract pulse waveforms from facial regions using **PyVHR**.  
- Apply **DSP techniques** like ZCR, Petrosian FD, Kurtosis, etc.  
- Build and deploy a **Flask-based web app** for real-time demo and visualization.

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Framework** | Flask |
| **rPPG Extraction** | [PyVHR](https://github.com/phuselab/pyVHR) |
| **Computer Vision** | OpenCV, Dlib, Mediapipe |
| **Signal Processing** | NumPy, SciPy, HeartPy |
| **Deep Learning** | PyTorch / TensorFlow |
| **Visualization** | Matplotlib, Seaborn |
| **Frontend** | HTML, CSS, JS (via `templates` & `static`) |

---

## 🧪 Role of PyVHR

![rppg Extraction Flow](static/pyvhr.png)

**PyVHR (Python Video-based Heart Rate)** is an open-source library designed to extract physiological signals from facial videos.  
In this project, PyVHR helps to:
✅ Detect facial regions and skin pixels dynamically.
✅ Extract RGB signals from video frames average them.
✅ Apply rPPG algorithms (CHROM)  
✅ Compute **heart rate** and raw **rPPG waveform or Bvp **  
✅ Provide a smooth, noise-free signal of **BVP** for DSP-based feature extraction.

## Signal Processing Techniques
We use **Digital Signal Processing (DSP)** to extract features from the rPPG signal:

| Technique | Description | Feature Type |
|------------|--------------|---------------|
| **ZCR (Zero Crossing Rate)** | Measures sign changes in the signal — reflects smoothness. | Temporal |
| **Petrosian Fractal Dimension (PFD)** | Quantifies complexity of waveform. | Nonlinear |
| **Kurtosis** | Detects spikiness — higher in fake signals. | Statistical |
| **Skewness** | Identifies asymmetry — fake signals often skewed. | Statistical |
| **Spectral Entropy** | Measures randomness in frequency content. | Frequency |
| **FFT Analysis** | Extracts dominant heart rhythm frequency. | Frequency |

These features form a **feature vector** that’s classified by the deep learning model.

## ⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/AshokChoudhary06/Deepfake-detection-using-rppg.

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Mac/Linux
venv\Scripts\activate       # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🚀 Running the Web App.
▶️ Start the Flask server
python app.py

## 📚 References

Qi et al., “DeepRhythm: Exposing DeepFakes with Attentional Visual Heartbeat Rhythms,” ACM MM 2020

PyVHR GitHub Repository

FaceForensics++ Dataset

Celeb-DF Dataset

## 🧑‍💻 Author

👋 Ashok Choudhary
🎓 Data Science Enthusiast | AI, Computer Vision & Signal Processing Learner


