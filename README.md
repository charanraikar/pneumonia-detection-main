# 🏥 AI-Powered Pneumonia Detection System

An advanced Deep Learning-based Pneumonia Detection System that analyzes Chest X-ray images and provides accurate predictions along with Explainable AI visualizations and clinical risk assessment.

---

## 🚀 Features

### 🩻 Automated Chest X-ray Analysis
- Detects Pneumonia from Chest X-ray images.
- Binary Classification:
  - NORMAL
  - PNEUMONIA

### 🤖 Deep Learning Architecture
- ConvNeXt-Based Neural Network
- Transfer Learning with Pretrained Weights
- Optimized for Medical Imaging Tasks

### 🔥 Explainable AI (XAI)
- Grad-CAM Visualization
- Heatmap Generation
- Model Decision Interpretation
- Region-of-Interest Localization

### 📊 Clinical Risk Assessment
- Risk Score Calculation
- Risk Level Classification
  - LOW
  - MODERATE
  - HIGH
  - CRITICAL
- Clinical Recommendation Generation

### 🌊 Synthetic Data Generation
- DDPM (Denoising Diffusion Probabilistic Model)
- Synthetic Chest X-ray Generation
- Data Augmentation Support

### ⚡ High Performance
- CUDA GPU Acceleration
- Mixed Precision Training
- Optimized Memory Usage
- Fast Inference Pipeline

### 🌐 Web Application
- FastAPI Backend
- REST API Endpoints
- Interactive Dashboard
- Real-Time Predictions

---

# 🛠️ Tech Stack

## Artificial Intelligence & Deep Learning
- PyTorch
- ConvNeXt
- Grad-CAM
- DDPM

## Backend
- FastAPI
- Python

## Data Processing
- NumPy
- Pandas
- OpenCV
- Albumentations

## Visualization
- Matplotlib
- Grad-CAM Heatmaps

## Deployment
- FastAPI
- CUDA
- NVIDIA GPU Support

---

# 📂 Project Structure

```bash
pneumonia-detection/
│
├── app.py
├── config.py
├── dataset.py
├── model.py
├── gradcam.py
├── ddpm.py
│
├── checkpoints/
├── data/
├── logs/
├── results/
│
└── requirements.txt
```

---

# 🧠 Model Pipeline

```text
Chest X-ray
      │
      ▼
Image Enhancement
      │
      ▼
ConvNeXt Model
      │
      ├────────► Pneumonia Probability
      │
      └────────► Grad-CAM Heatmap
                    │
                    ▼
            Risk Assessment Engine
                    │
                    ▼
            Clinical Recommendation
```

---

# 📊 Outputs

The system generates:

✅ Prediction Label

✅ Confidence Score

✅ Grad-CAM Heatmap

✅ Risk Score

✅ Clinical Recommendation

✅ Risk Dashboard

---

# 🎯 Risk Assessment Levels

| Risk Score | Level |
|------------|--------|
| 0-25% | LOW |
| 25-50% | MODERATE |
| 50-75% | HIGH |
| 75-100% | CRITICAL |

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/charanraikar/pneumonia-detection-main.git
cd pneumonia-detection-main
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
python app.py
```

or

```bash
uvicorn app:app --reload
```

---

# 📸 Sample Features

- Chest X-ray Upload
- Pneumonia Prediction
- Explainable AI Heatmaps
- Clinical Risk Dashboard
- Synthetic X-ray Generation

---

# 📈 Future Improvements

- Multi-Disease Detection
- Mobile Application
- Cloud Deployment
- Real-Time Hospital Integration
- Federated Learning Support

---

# 👨‍💻 Author

### Charan Raikar

AI & Machine Learning Engineer

📧 raikarcharan64@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/charanraikar?utm_source=share_via&utm_content=profile&utm_medium=member_android

---

## ⭐ If you found this project useful, please consider giving it a star.
