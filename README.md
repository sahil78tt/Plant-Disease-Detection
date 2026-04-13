<p align="center">
  <img src="https://img.icons8.com/color/512/plant-under-sun.png" alt="Plant Disease Detection Logo" width="140" />
</p>

<h1 align="center">Plant Disease Detection</h1>

<p align="center">
<strong>AI-powered plant leaf disease detection using Deep Learning and Transfer Learning</strong>
</p>

<p align="center">
Upload a plant leaf image and get instant disease predictions, confidence scores, and remedy suggestions through a modern AI-powered web application.
</p>

<p align="center">
<a href="#features">Features</a> •
<a href="#screenshots">Screenshots</a> •
<a href="#installation">Installation</a> •
<a href="#tech-stack">Tech Stack</a> •
<a href="#dataset">Dataset</a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-DeepLearning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Keras-NeuralNetworks-D00000?style=for-the-badge&logo=keras&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/MobileNet-TransferLearning-4CAF50?style=for-the-badge"/>
</p>

---

# Why This Project?

Plant diseases significantly impact agricultural productivity. Early detection can help farmers prevent crop losses.

This project uses **Computer Vision and Deep Learning** to automatically identify plant diseases from leaf images.

The system uses **Transfer Learning with MobileNet**, a lightweight convolutional neural network trained on ImageNet.

Users can upload a leaf image through a **Streamlit web interface** and receive:

- Disease prediction
- Confidence score
- Disease description
- Suggested remedies

---

# Features

<table>
<tr>
<td width="50%">

### AI Disease Detection

Detect plant diseases from leaf images using a deep learning model.

</td>
<td width="50%">

### Instant Predictions

Upload an image and get results within seconds.

</td>
</tr>

<tr>
<td>

### Confidence Visualization

Prediction confidence chart for all detected classes.

</td>
<td>

### Remedy Suggestions

Get disease descriptions and treatment recommendations.

</td>
</tr>

<tr>
<td>

### Modern Web Interface

Interactive UI built with Streamlit.

</td>
<td>

### Transfer Learning

Uses MobileNet pretrained on ImageNet.

</td>
</tr>
</table>

---

# Supported Plants & Diseases

| Plant       | Diseases                                                      |
| ----------- | ------------------------------------------------------------- |
| Tomato      | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Healthy |
| Potato      | Early Blight, Late Blight, Healthy                            |
| Pepper Bell | Bacterial Spot, Healthy                                       |

Total Classes: **10**

---

# Tech Stack

### Machine Learning

```
TensorFlow
Keras
MobileNet
Transfer Learning
```

### Backend / Processing

```
Python
NumPy
OpenCV
Pillow
Matplotlib
```

### Web Application

```
Streamlit
```

---

# Architecture

```
User uploads leaf image
        │
        ▼
Streamlit Web Interface
        │
        ▼
Image Preprocessing
        │
        ▼
Deep Learning Model (MobileNet)
        │
        ▼
Prediction + Confidence Score
        │
        ▼
Disease Information + Remedies
```

---

# Project Structure

```
plant-disease-detection/
│
├── dataset/                # Plant disease images
│
├── models/                 # Trained models
│   ├── plant_disease_model.h5
│   ├── class_indices.json
│   └── training_history.png
│
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── predict.py              # CLI prediction script
├── utils.py                # Utility functions
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/sahil78tt/Plant-Disease-Detection.git
cd plant-disease-detection
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Dataset

This project uses the **PlantVillage Dataset**.

Download from Kaggle

https://www.kaggle.com/datasets/emmarex/plantdisease

Extract the following folders inside `dataset/`

```
Pepper__bell___Bacterial_spot
Pepper__bell___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Tomato_Bacterial_spot
Tomato_Early_blight
Tomato_Late_blight
Tomato_Leaf_Mold
Tomato_healthy
```

---

# Train the Model

Train using real dataset

```bash
python train_model.py
```

Quick test using synthetic dataset

```bash
python train_model.py --create-sample
```

---

# Run the Web Application

Start the Streamlit app

```bash
streamlit run app.py
```

Open browser

```
http://localhost:8501
```

---

# Command Line Prediction

Predict disease from terminal

```bash
python predict.py path/to/image.jpg
```

Or auto test

```bash
python predict.py
```

---

# Troubleshooting

### TensorFlow Installation Error

```
pip install tensorflow
```

### Model Not Found

Train the model first

```
python train_model.py --create-sample
```

### Virtual Environment Issue

```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

---

# Future Improvements

- Add more plant species
- Improve model accuracy
- Add Grad-CAM visualization
- Deploy using Docker
- Mobile app using TensorFlow Lite
- REST API using FastAPI

---

# References

PlantVillage Dataset  
https://www.kaggle.com/datasets/emmarex/plantdisease

MobileNet Paper  
https://arxiv.org/abs/1704.04861

TensorFlow Transfer Learning  
https://www.tensorflow.org/tutorials/images/transfer_learning

Streamlit Documentation  
https://docs.streamlit.io

---

# Author

Sahil Vishwakarma

Built with **Python, TensorFlow, and Streamlit**

---

<p align="center">
© 2026 Sahil Vishwakarma
</p>
